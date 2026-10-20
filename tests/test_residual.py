"""Train-only offsets, support checks, and replay without any experiment runs."""
import numpy as np
import pytest
import torch
from torch.nn import functional as F

from fluvius_cgan.models import Generator
from fluvius_cgan.residual import ResidualGenerator, fit_mean_profiles


@pytest.mark.parametrize("width", [24, 96])
def test_fit_uses_only_supplied_training_subset_and_float64_accumulation(width):
    # The caller selected these four rows; unrelated full-data rows are not input.
    selected = np.repeat(np.array([[1e8], [1.], [-1e8], [7.]], dtype=np.float32), width, axis=1)
    means = fit_mean_profiles(selected, np.array([0, 0, 0, 8]), 64)
    assert means.dtype == torch.float32 and not means.requires_grad
    torch.testing.assert_close(means[0], torch.full((width,), 1 / 3))
    torch.testing.assert_close(means[8], torch.full((width,), 7.))
    assert torch.isnan(means[[1, 7, 63]]).all()


@pytest.mark.parametrize("width", [24, 96])
def test_wrapper_adds_fitted_means_without_reinitializing_or_clipping(width):
    base = Generator(latent_dim=4, condition_dim=64, hidden_dim=4, output_dim=width)
    original = {name: value.clone() for name, value in base.state_dict().items()}
    means = fit_mean_profiles(torch.tensor([[-20.] * width, [30.] * width]), torch.tensor([0, 8]), 64)
    generator = ResidualGenerator(base, means)
    conditions = F.one_hot(torch.tensor([8, 0, 8]), 64).float()
    noise = torch.randn(3, 4)
    torch.testing.assert_close(generator(noise, conditions), base(noise, conditions) + means[[8, 0, 8]])
    for name, value in base.state_dict().items():
        torch.testing.assert_close(value, original[name], rtol=0, atol=0)
    assert torch.isfinite(generator.mean_profiles).all()
    assert generator(noise, conditions)[0].min() > 1
    assert generator(noise, conditions)[1].max() < 0
    means[0].fill_(999.)
    assert generator.mean_profiles[0, 0] == -20  # The caller cannot mutate the buffer.
    with pytest.raises(ValueError, match="unsupported"):
        generator(noise[:1], F.one_hot(torch.tensor([1]), 64).float())


def test_unconditional_mean_is_global_and_does_not_receive_condition_information():
    x = torch.tensor([[1.] * 24, [3.] * 24, [11.] * 24])
    first = fit_mean_profiles(x, torch.tensor([0, 0, 8]), 0)
    relabeled = fit_mean_profiles(x, torch.tensor([8, 12, 12]), 0)
    torch.testing.assert_close(first, torch.full((1, 24), 5.))
    torch.testing.assert_close(first, relabeled)
    base = Generator(latent_dim=4, condition_dim=0, hidden_dim=4, output_dim=24)
    generator = ResidualGenerator(base, first)
    noise, conditions = torch.randn(3, 4), torch.empty(3, 0)
    torch.testing.assert_close(generator(noise, conditions), base(noise, conditions) + 5.)
    with pytest.raises(ValueError, match="shape"):
        generator(noise, F.one_hot(torch.tensor([0, 0, 8]), 64).float())


@pytest.mark.parametrize("condition_dim", [0, 64])
def test_fixed_buffers_follow_dtype_and_base_gradients_and_replay(tmp_path, condition_dim):
    base = Generator(latent_dim=4, condition_dim=condition_dim, hidden_dim=4, output_dim=24)
    means = fit_mean_profiles(torch.randn(6, 24), torch.tensor([0, 8, 0, 8, 0, 8]), condition_dim)
    generator = ResidualGenerator(base, means).double()
    conditions = (F.one_hot(torch.tensor([0, 8]), 64).double()
                  if condition_dim else torch.empty(2, 0, dtype=torch.float64))
    noise = torch.randn(2, 4, dtype=torch.float64)
    before = generator.mean_profiles.clone()
    output = generator(noise, conditions)
    output.square().mean().backward()
    assert output.dtype == torch.float64
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in base.parameters())
    assert generator.mean_profiles.grad is None
    assert not generator.mean_profiles.requires_grad
    assert set(dict(generator.named_buffers())) == {"mean_profiles", "supported_conditions"}
    assert all(name.startswith("base.") for name, _ in generator.named_parameters())
    torch.testing.assert_close(generator.mean_profiles, before, rtol=0, atol=0)
    path = tmp_path / "residual.pt"
    torch.save(generator.state_dict(), path)
    saved = torch.load(path, weights_only=True)
    replay = ResidualGenerator(
        Generator(latent_dim=4, condition_dim=condition_dim, hidden_dim=4, output_dim=24),
        saved["mean_profiles"],
    ).double()
    replay.load_state_dict(saved, strict=True)
    torch.testing.assert_close(replay(noise, conditions), output, rtol=0, atol=0)
    if condition_dim:
        with pytest.raises(ValueError, match="unsupported"):
            replay(noise[:1], F.one_hot(torch.tensor([1]), 64).double())


@pytest.mark.parametrize("change", ["empty", "nonfinite", "label_count", "label_float", "label_negative", "label_outside"])
def test_invalid_fitting_data_rejected(change):
    x, labels = torch.ones(2, 24), torch.tensor([0, 8])
    if change == "empty":
        x, labels = x[:0], labels[:0]
    elif change == "nonfinite":
        x[0, 0] = float("inf")
    elif change == "label_count":
        labels = labels[:1]
    elif change == "label_float":
        labels = labels.float()
    elif change == "label_negative":
        labels[0] = -1
    else:
        labels[0] = 64
    with pytest.raises(ValueError):
        fit_mean_profiles(x, labels, 64)


@pytest.mark.parametrize("condition_dim", [-1, True, 1.5])
def test_invalid_condition_dimension_rejected(condition_dim):
    with pytest.raises(ValueError):
        fit_mean_profiles(torch.ones(2, 24), torch.tensor([0, 0]), condition_dim)


@pytest.mark.parametrize("invalid", ["shape", "partial_nan", "infinite", "all_missing"])
def test_invalid_mean_table_rejected(invalid):
    base = Generator(latent_dim=4, condition_dim=64, hidden_dim=4, output_dim=24)
    means = torch.zeros(64, 24)
    if invalid == "shape":
        means = means[:, :23]
    elif invalid == "partial_nan":
        means[0, 0] = float("nan")
    elif invalid == "infinite":
        means[0, 0] = float("inf")
    else:
        means.fill_(float("nan"))
    with pytest.raises(ValueError):
        ResidualGenerator(base, means)


@pytest.mark.parametrize("invalid", ["zero", "soft", "multiple", "nonfinite"])
def test_invalid_condition_vectors_rejected(invalid):
    base = Generator(latent_dim=4, condition_dim=64, hidden_dim=4, output_dim=24)
    generator = ResidualGenerator(base, torch.zeros(64, 24))
    condition = torch.zeros(1, 64)
    if invalid == "soft":
        condition[0, :2] = .5
    elif invalid == "multiple":
        condition[0, :2] = 1
    elif invalid == "nonfinite":
        condition[0, 0] = float("nan")
    with pytest.raises(ValueError, match="one-hot"):
        generator(torch.randn(1, 4), condition)
