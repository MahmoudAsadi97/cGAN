"""Synthetic mathematical checks for fixed signed-derivative loss coordinates."""
import json
import math

import numpy as np
import pytest
import torch

from fluvius_cgan.temporal_loss import fit_derivative_scale, derivative_sliced_wasserstein_loss


def ramp_profiles(slopes, width=24, dtype=np.float32):
    return np.asarray(slopes, dtype=dtype)[:, None] * np.arange(width, dtype=dtype)


def first_direction(width=24, dtype=torch.float64):
    direction = torch.zeros(width - 1, 1, dtype=dtype)
    direction[0] = 1.
    return direction


def test_scale_is_condition_macro_rms_without_centering_or_wraparound():
    values = ramp_profiles([1., 3., 3., 3.]) + np.array([10., -20., 50., 4.])[:, None]
    report = fit_derivative_scale(values, np.array([0, 9, 9, 9]))
    assert report["estimate_float64"] == pytest.approx(math.sqrt(5.))
    assert report["estimate_float64"] != pytest.approx(math.sqrt(7.))  # Natural row weighting.
    assert report["used_float32"] == float(np.float32(math.sqrt(5.)))
    assert report["condition_counts"] == {"0": 1, "9": 3}
    assert report["condition_mean_squared_differences"] == {"0": 1., "9": 9.}
    assert report["n_profiles"] == 4 and report["n_features"] == 24 and report["n_differences"] == 23
    assert report["centering"] is report["wraparound"] is False
    assert json.loads(json.dumps(report, allow_nan=False)) == report


@pytest.mark.parametrize("width", [24, 96])
@pytest.mark.parametrize("batch_size", [1, 4, 4096])
def test_chunked_fit_uses_every_row_and_slot_including_final_partial_batch(width, batch_size):
    rng = np.random.default_rng(29)
    values = rng.normal(size=(11, width)).astype(np.float32)
    values[-1, -1] = 150.
    labels = np.array([0, 4, 4, 0, 4, 4, 4, 0, 4, 4, 4])
    derivatives = np.diff(values.astype(np.float64), axis=1)
    means = {str(label): float(np.mean(derivatives[labels == label] ** 2)) for label in (0, 4)}
    expected = math.sqrt(sum(means.values()) / 2)
    report = fit_derivative_scale(values, labels, batch_size=batch_size)
    assert report["estimate_float64"] == pytest.approx(expected, rel=1e-14)
    assert report["condition_mean_squared_differences"] == pytest.approx(means, rel=1e-14)
    assert sum(report["condition_counts"].values()) == len(values)
    assert report["accumulation_batch_size"] == batch_size


def test_repeating_one_condition_does_not_change_its_macro_weight():
    values = ramp_profiles([0., 2.])
    first = fit_derivative_scale(values, np.array([0, 63]))
    repeated = fit_derivative_scale(values[[0, 1, 1, 1, 1, 1]], np.array([0, 63, 63, 63, 63, 63]))
    assert first["estimate_float64"] == repeated["estimate_float64"] == math.sqrt(2)
    assert first["condition_mean_squared_differences"]["0"] == 0.


def test_cpu_tensor_fit_is_detached_deterministic_and_does_not_consume_rng_or_mutate_inputs():
    values = torch.tensor(ramp_profiles([1., -2., 3.]), requires_grad=True)
    labels = torch.tensor([0, 4, 4])
    before = values.detach().clone()
    torch_state = torch.random.get_rng_state().clone()
    numpy_state = np.random.get_state()
    first = fit_derivative_scale(values, labels, batch_size=2)
    second = fit_derivative_scale(values.detach().numpy(), labels.numpy(), batch_size=2)
    assert first == second
    assert torch.equal(torch_state, torch.random.get_rng_state())
    after_numpy = np.random.get_state()
    assert numpy_state[0] == after_numpy[0] and numpy_state[2:] == after_numpy[2:]
    np.testing.assert_array_equal(numpy_state[1], after_numpy[1])
    torch.testing.assert_close(values, before)
    assert values.grad is None


@pytest.mark.parametrize("values", [
    np.empty((0, 24)), np.zeros(24), np.zeros((2, 23)), np.zeros((2, 25)),
    np.zeros((2, 24), dtype=np.int64), np.zeros((2, 24), dtype=complex),
    np.full((2, 24), np.nan), np.full((2, 24), np.inf),
])
def test_fit_rejects_invalid_profiles(values):
    with pytest.raises(ValueError):
        fit_derivative_scale(values, np.zeros(len(values), dtype=np.int64))


@pytest.mark.parametrize("labels", [np.array([0]), np.array([[0], [0]]), np.array([0., 0.]),
                                    np.array([True, False]), np.array([-1, 0]), np.array([0, 64])])
def test_fit_rejects_invalid_labels(labels):
    with pytest.raises(ValueError, match="labels"):
        fit_derivative_scale(ramp_profiles([1., 2.]), labels)


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5])
def test_fit_rejects_invalid_chunk_size(batch_size):
    with pytest.raises(ValueError, match="batch_size"):
        fit_derivative_scale(ramp_profiles([1.]), np.array([0]), batch_size=batch_size)


@pytest.mark.parametrize("magnitude", [0., 1e-50, 1e40, 1e200])
def test_fit_rejects_zero_or_unrepresentable_float64_and_float32_scales(magnitude):
    values = ramp_profiles([magnitude], dtype=np.float64)
    with pytest.raises(ValueError, match="scale|mean squares"):
        fit_derivative_scale(values, np.array([0]))


@pytest.mark.parametrize("width", [24, 96])
def test_loss_has_equal_condition_weights_and_expected_profile_gradients(width):
    real = torch.zeros(6, width, dtype=torch.float64)
    generated = torch.tensor(ramp_profiles([1., 1., 3., 3., 3., 3.], width, np.float64), requires_grad=True)
    labels = torch.tensor([0, 0, 9, 9, 9, 9])
    loss = derivative_sliced_wasserstein_loss(real, generated, labels, labels, 2.,
        min_samples_per_condition=2, projections=first_direction(width))
    assert loss.item() == 1.25  # SW2 squared: ((1/2)**2 + (3/2)**2)/2, no outer root.
    loss.backward()
    expected = torch.zeros_like(generated)
    slopes = torch.tensor([.125, .125, .1875, .1875, .1875, .1875], dtype=generated.dtype)
    expected[:, 0], expected[:, 1] = -slopes, slopes
    torch.testing.assert_close(generated.grad, expected, rtol=0, atol=0)


def test_derivatives_keep_their_sign_and_nonzero_mean_without_a_boundary_wrap():
    real = torch.tensor(ramp_profiles([1., 1.], dtype=np.float64))
    generated = -real
    labels = torch.zeros(2, dtype=torch.int64)
    directions = torch.ones(23, 1, dtype=torch.float64)
    loss = derivative_sliced_wasserstein_loss(real, generated, labels, labels, 1.,
        min_samples_per_condition=2, projections=directions)
    assert loss.item() == pytest.approx(4 * 23)
    # Absolute or centered derivatives would make this discrepancy vanish.
    with pytest.raises(ValueError, match="projections"):
        derivative_sliced_wasserstein_loss(real, generated, labels, labels, 1.,
            min_samples_per_condition=2, projections=torch.ones(24, 1, dtype=torch.float64))


def test_profile_offsets_and_common_scale_change_leave_derivative_loss_invariant():
    real = torch.tensor(ramp_profiles([1., 2., 3., 4.], dtype=np.float64))
    generated = real * 1.5
    labels = torch.tensor([0, 0, 4, 4])
    directions = first_direction()
    base = derivative_sliced_wasserstein_loss(real, generated, labels, labels, 2.,
        min_samples_per_condition=2, projections=directions)
    for a, b, scale in [(real * 7., generated * 7., 14.),
                         (real + torch.arange(4)[:, None] * 100, generated - 50., 2.)]:
        actual = derivative_sliced_wasserstein_loss(a, b, labels, labels, scale,
            min_samples_per_condition=2, projections=directions)
        torch.testing.assert_close(actual, base, rtol=0, atol=0)


def test_loss_replays_with_dedicated_rng_without_using_global_rng():
    real = torch.tensor(ramp_profiles([1., 2., 3., 4.], dtype=np.float64))
    generated, labels = real.flip(1), torch.zeros(4, dtype=torch.int64)
    before = torch.random.get_rng_state().clone()
    kwargs = {"min_samples_per_condition": 2, "num_projections": 19}
    first = derivative_sliced_wasserstein_loss(real, generated, labels, labels, 2.,
        generator=torch.Generator().manual_seed(17), **kwargs)
    second = derivative_sliced_wasserstein_loss(real, generated, labels, labels, 2.,
        generator=torch.Generator().manual_seed(17), **kwargs)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert torch.equal(before, torch.random.get_rng_state())


@pytest.mark.parametrize("scale", [0., -1., float("nan"), float("inf"), True,
                                   torch.tensor(1.), torch.tensor(1., requires_grad=True), [1.]])
def test_loss_rejects_invalid_or_trainable_scale(scale):
    values = torch.zeros(2, 24)
    labels = torch.zeros(2, dtype=torch.int64)
    with pytest.raises(ValueError, match="scale"):
        derivative_sliced_wasserstein_loss(values, values, labels, labels, scale,
                                           min_samples_per_condition=2)


@pytest.mark.parametrize("scale", [1e-50, 1e100])
def test_loss_rejects_scales_that_underflow_or_overflow_the_profile_dtype(scale):
    real = torch.zeros(2, 24, dtype=torch.float32)
    generated = torch.tensor(ramp_profiles([1., 1.]))
    labels = torch.zeros(2, dtype=torch.int64)
    with pytest.raises(ValueError, match="scale.*profiles' dtype"):
        derivative_sliced_wasserstein_loss(real, generated, labels, labels, scale,
            min_samples_per_condition=2, projections=first_direction(dtype=torch.float32))


@pytest.mark.parametrize("labels", [torch.tensor([0, 64]), torch.tensor([-1, 0]),
                                     torch.tensor([0., 0.]), torch.tensor([0j, 0j])])
def test_loss_rejects_invalid_labels(labels):
    values = torch.zeros(2, 24)
    with pytest.raises(ValueError, match="labels"):
        derivative_sliced_wasserstein_loss(values, values, labels, labels, 1., min_samples_per_condition=2)


def test_loss_retains_condition_support_counts_and_default_sample_floor_guards():
    values = torch.zeros(4, 24)
    labels = torch.tensor([0, 0, 4, 4])
    for other, message in [(torch.tensor([0, 0, 9, 9]), "support"),
                           (torch.tensor([0, 4, 4, 4]), "counts")]:
        with pytest.raises(ValueError, match=message):
            derivative_sliced_wasserstein_loss(values, values, labels, other, 1., min_samples_per_condition=2)
    with pytest.raises(ValueError, match="at least 32"):
        derivative_sliced_wasserstein_loss(values, values, labels, labels, 1.)


@pytest.mark.parametrize("values", [torch.zeros(2, 23), torch.zeros(2, 25), torch.empty(0, 24),
                                     torch.zeros(2, 24, dtype=torch.int64),
                                     torch.full((2, 24), float("nan"))])
def test_loss_rejects_invalid_profile_coordinates(values):
    labels = torch.zeros(len(values), dtype=torch.int64)
    with pytest.raises(ValueError):
        derivative_sliced_wasserstein_loss(values, values, labels, labels, 1., min_samples_per_condition=2)


def test_derivative_loss_passes_double_precision_autograd_check():
    values = np.random.default_rng(31).normal(size=(4, 24))
    real = torch.tensor(values, dtype=torch.float64)
    generated = torch.tensor(values + np.linspace(-.3, .4, 24), dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([0, 0, 4, 4])
    directions = torch.tensor(np.random.default_rng(32).normal(size=(23, 3)), dtype=torch.float64)
    function = lambda x: derivative_sliced_wasserstein_loss(real, x, labels, labels, 1.7,
        min_samples_per_condition=2, projections=directions)
    assert torch.autograd.gradcheck(function, (generated,))
