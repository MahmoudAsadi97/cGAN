"""Mathematical and conditional safeguards for the optional distribution loss."""
import pytest
import torch

from fluvius_cgan.distribution_loss import (
    conditional_sliced_wasserstein_loss,
    sliced_wasserstein_loss,
)


@pytest.mark.parametrize("p,expected,gradient", [
    (1, 2., [1 / 3, 1 / 3, 1 / 3]),
    (2, 14 / 3, [2., 2 / 3, 4 / 3]),
])
def test_exact_one_dimensional_transport_and_sample_gradients(p, expected, gradient):
    real = torch.tensor([[0.], [2.], [4.]], dtype=torch.float64)
    generated = torch.tensor([[7.], [1.], [4.]], dtype=torch.float64, requires_grad=True)
    loss = sliced_wasserstein_loss(real, generated, p=p, projections=torch.tensor([[3.]], dtype=torch.float64))
    assert loss.item() == pytest.approx(expected)
    loss.backward()
    torch.testing.assert_close(generated.grad[:, 0], torch.tensor(gradient, dtype=torch.float64))


@pytest.mark.parametrize("p", [1, 2])
def test_permutation_symmetry_translation_and_projection_scale_invariance(p):
    real = torch.tensor([[0., 1.], [2., 3.], [4., 2.]])
    generated = torch.tensor([[1., 0.], [4., 4.], [2., 1.]])
    directions = torch.tensor([[1., 2., -1.], [0., 1., 3.]])
    loss = sliced_wasserstein_loss(real, generated, p=p, projections=directions)
    for a, b, projections in [
        (generated, real, directions),
        (real[[2, 0, 1]], generated[[1, 2, 0]], directions),
        (real + 4, generated + 4, directions),
        (real, generated, directions * torch.tensor([1e30, 1e-30, -7.])),
    ]:
        torch.testing.assert_close(sliced_wasserstein_loss(a, b, p=p, projections=projections), loss)
    assert sliced_wasserstein_loss(real, real[[1, 0, 2]], p=p, projections=directions).item() == 0


def test_random_directions_replay_with_local_rng_without_using_global_rng():
    real = torch.arange(24, dtype=torch.float64).reshape(8, 3)
    generated = real.flip(1)
    state = torch.random.get_rng_state().clone()
    first_rng = torch.Generator().manual_seed(71)
    first = sliced_wasserstein_loss(real, generated, num_projections=19, generator=first_rng)
    second = sliced_wasserstein_loss(real, generated, num_projections=19,
                                     generator=torch.Generator().manual_seed(71))
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert torch.equal(state, torch.random.get_rng_state())
    assert not torch.equal(first, sliced_wasserstein_loss(real, generated, num_projections=19, generator=first_rng))


def test_condition_swap_cannot_hide_behind_a_correct_pooled_distribution():
    real = torch.tensor([[-2.], [-2.], [2.], [2.]])
    generated = -real
    labels = torch.tensor([0, 0, 9, 9])
    directions = torch.ones(1, 1)
    assert sliced_wasserstein_loss(real, generated, projections=directions).item() == 0
    loss = conditional_sliced_wasserstein_loss(real, generated, labels, labels,
                min_samples_per_condition=2, projections=directions)
    assert loss.item() == 16


def test_macro_weighting_and_independent_sample_order_within_conditions():
    real = torch.zeros(6, 1)
    generated = torch.tensor([[1.], [1.], [3.], [3.], [3.], [3.]])
    labels = torch.tensor([0, 0, 9, 9, 9, 9])
    order = torch.tensor([5, 0, 4, 1, 2, 3])
    loss = conditional_sliced_wasserstein_loss(real, generated[order], labels, labels[order],
                min_samples_per_condition=2, projections=torch.ones(1, 1))
    assert loss.item() == 5  # Equal class weights: (1**2 + 3**2) / 2.


def test_conditional_loss_passes_double_precision_autograd_check():
    real = torch.tensor([[0., 1.], [2., 3.], [1., -1.], [3., 2.]], dtype=torch.float64)
    generated = torch.tensor([[.3, .7], [1.4, 2.8], [1.6, -.7], [3.1, 1.2]],
                             dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1])
    directions = torch.tensor([[1., .2], [.3, 1.]], dtype=torch.float64)
    function = lambda x: conditional_sliced_wasserstein_loss(
        real, x, labels, labels, min_samples_per_condition=2, projections=directions)
    assert torch.autograd.gradcheck(function, (generated,))


@pytest.mark.parametrize("generated_labels,message", [
    ([0, 0, 2, 2], "support"),
    ([0, 1, 1, 1], "counts"),
])
def test_conditional_support_and_counts_fail_closed(generated_labels, message):
    x = torch.zeros(4, 2)
    with pytest.raises(ValueError, match=message):
        conditional_sliced_wasserstein_loss(x, x, torch.tensor([0, 0, 1, 1]),
            torch.tensor(generated_labels), min_samples_per_condition=2)


def test_sparse_classes_rejected_without_silently_dropping_them():
    x = torch.zeros(512, 2)
    labels = torch.arange(64).repeat_interleave(8)
    with pytest.raises(ValueError, match="at least 32"):
        conditional_sliced_wasserstein_loss(x, x, labels, labels)
    # A homogeneous batch is supported; missing classes are the caller's responsibility.
    assert conditional_sliced_wasserstein_loss(x, x, torch.zeros(512, dtype=torch.int64),
        torch.zeros(512, dtype=torch.int64), generator=torch.Generator().manual_seed(1)).item() == 0


@pytest.mark.parametrize("kwargs", [
    {"p": 0}, {"p": True}, {"p": 2.0}, {"num_projections": 0},
    {"num_projections": True}, {"projections": torch.zeros(2, 1)},
    {"projections": torch.ones(1, 2)}, {"projections": torch.full((2, 1), float("nan"))},
    {"projections": torch.ones(2, 1, dtype=torch.float64)},
    {"projections": torch.ones(2, 1), "generator": torch.Generator()},
])
def test_invalid_loss_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        sliced_wasserstein_loss(torch.zeros(3, 2), torch.ones(3, 2), **kwargs)


@pytest.mark.parametrize("real,generated", [
    (torch.empty(0, 2), torch.empty(0, 2)),
    (torch.zeros(2, 2, dtype=torch.int64), torch.zeros(2, 2)),
    (torch.zeros(2, 2), torch.zeros(3, 2)),
    (torch.zeros(2, 2), torch.zeros(2, 3)),
    (torch.zeros(2, 2), torch.zeros(2, 2, dtype=torch.float64)),
    (torch.zeros(2, 2), torch.full((2, 2), float("inf"))),
])
def test_invalid_profiles_rejected(real, generated):
    with pytest.raises(ValueError):
        sliced_wasserstein_loss(real, generated)


@pytest.mark.parametrize("labels", [torch.zeros(4), torch.zeros(4, dtype=torch.bool),
                                     torch.tensor([-1, -1, 0, 0]), torch.zeros(4, 1, dtype=torch.int64)])
def test_invalid_condition_labels_rejected(labels):
    x = torch.zeros(4, 2)
    with pytest.raises(ValueError):
        conditional_sliced_wasserstein_loss(x, x, labels, labels, min_samples_per_condition=2)


@pytest.mark.parametrize("minimum", [0, 1, True, 2.0])
def test_condition_sample_floor_is_explicit_and_validated(minimum):
    x = torch.zeros(4, 2)
    labels = torch.zeros(4, dtype=torch.int64)
    with pytest.raises(ValueError, match="min_samples_per_condition"):
        conditional_sliced_wasserstein_loss(x, x, labels, labels, min_samples_per_condition=minimum)
