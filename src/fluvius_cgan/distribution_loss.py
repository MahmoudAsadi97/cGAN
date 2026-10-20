"""Differentiable empirical sliced-Wasserstein losses for research ablations.

Random unit projections and sorted one-dimensional transport follow Deshpande,
Zhang and Schwing, CVPR 2018, https://arxiv.org/abs/1803.11188 . This module is
an implementation of established ideas, not a new method or a training policy.

The returned quantity is SW_p**p: mean absolute projected differences for p=1,
or mean squared projected differences for p=2 (WITHOUT a square root). It is an
empirical minibatch objective, not an unbiased population-distance estimator.
Small independently sampled batches can favor reduced dispersion. Increasing
projection count does not fix inadequate samples per condition. See Lezama et
al., ICML 2021, https://proceedings.mlr.press/v139/lezama21a.html .

Inputs must use the same train-fitted coordinates. No fitting, clipping, data
loading, or training integration occurs here. Callers must separately declare
condition sampling, sample/update budgets, auxiliary weight and evaluation rules.
"""
from numbers import Integral

import torch


def _positive_int(name, value, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _validate_profiles(real, generated):
    for name, values in (("real", real), ("generated", generated)):
        if (not isinstance(values, torch.Tensor) or values.ndim != 2 or
                not values.shape[0] or not values.shape[1] or
                not values.is_floating_point()):
            raise ValueError(f"{name} must be a nonempty floating tensor of shape (N, D)")
        if not torch.isfinite(values).all():
            raise ValueError(f"{name} must contain finite values")
    if real.shape != generated.shape:
        raise ValueError("real and generated must have equal sample counts and feature dimensions")
    if real.dtype != generated.dtype or real.device != generated.device:
        raise ValueError("real and generated must have the same dtype and device")


def _directions(real, num_projections, projections, generator):
    count = _positive_int("num_projections", num_projections)
    if projections is None:
        projections = torch.randn((real.shape[1], count), dtype=real.dtype,
                                  device=real.device, generator=generator)
    else:
        if generator is not None:
            raise ValueError("supply either explicit projections or a random generator, not both")
        if (not isinstance(projections, torch.Tensor) or projections.ndim != 2 or
                projections.shape[0] != real.shape[1] or not projections.shape[1] or
                projections.dtype != real.dtype or projections.device != real.device):
            raise ValueError("projections must have shape (D, K), K >= 1, and the profiles' dtype/device")
    projections = projections.detach()
    if not torch.isfinite(projections).all():
        raise ValueError("projections must contain finite values")
    # First rescale each column to avoid overflow/underflow while normalizing.
    scales = projections.abs().amax(dim=0, keepdim=True)
    if (scales == 0).any():
        raise ValueError("projection directions must be nonzero")
    directions = projections / scales
    return directions / torch.linalg.vector_norm(directions, dim=0, keepdim=True)


def _order(p):
    if isinstance(p, bool) or not isinstance(p, Integral) or p not in (1, 2):
        raise ValueError("p must be 1 or 2")
    return int(p)


def _projected_loss(real, generated, directions, p):
    real_sorted = (real @ directions).sort(dim=0).values
    generated_sorted = (generated @ directions).sort(dim=0).values
    differences = real_sorted - generated_sorted
    loss = differences.abs().mean() if p == 1 else differences.square().mean()
    if not torch.isfinite(loss):
        raise ValueError("sliced-Wasserstein loss is nonfinite; check input scale")
    return loss


def sliced_wasserstein_loss(real: torch.Tensor, generated: torch.Tensor, *,
                            num_projections: int = 64, p: int = 2,
                            projections: torch.Tensor | None = None,
                            generator: torch.Generator | None = None) -> torch.Tensor:
    """Return empirical SW_p**p for equally sized, equally weighted samples.

    Gaussian directions are normalized to the unit sphere and sampled each call.
    A dedicated ``generator`` controls their RNG independently of latent noise;
    it must belong to the input device. Alternatively, supply constant direction
    columns of shape (D, K); K then determines the projection count. Their length
    is normalized, and gradients do not flow into directions. Sorting preserves
    sample gradients almost everywhere; ties have nonunique subgradients.

    Gradients can flow to either input. For conditional generation, do not pool
    conditions here: use ``conditional_sliced_wasserstein_loss`` or provide a
    homogeneous condition batch. Finite directions yield a Monte Carlo estimate
    and do not guarantee that every distributional discrepancy is detected.
    """
    _validate_profiles(real, generated)
    p = _order(p)
    directions = _directions(real, num_projections, projections, generator)
    return _projected_loss(real, generated, directions, p)


def conditional_sliced_wasserstein_loss(
        real: torch.Tensor, generated: torch.Tensor,
        real_labels: torch.Tensor, generated_labels: torch.Tensor, *,
        num_projections: int = 64, p: int = 2,
        min_samples_per_condition: int = 32,
        projections: torch.Tensor | None = None,
        generator: torch.Generator | None = None) -> torch.Tensor:
    """Equally average empirical SW_p**p across the batch's observed conditions.

    Real/fake condition support and each condition's sample count must agree.
    Conditions are never silently dropped or pooled. Counts may differ BETWEEN
    conditions, but every condition receives equal weight. Conditions absent
    from both batches contribute nothing; the caller controls their sampling
    across updates. Thus uniform condition sampling targets a condition-macro
    objective, not the natural condition-prevalence-weighted objective.

    The default floor32 rejects the usual sparse 64-class mixed minibatch; it is
    an explicit engineering guard, not a statistical guarantee. For this study,
    128 or256 samples per condition are preferable when affordable. A lower
    floor (>=2) is available for controlled tests and documented sensitivity
    experiments. Real/fake share the same random directions within each call.
    """
    _validate_profiles(real, generated)
    p = _order(p)
    minimum = _positive_int("min_samples_per_condition", min_samples_per_condition, 2)
    integer_dtypes = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
    for name, labels in (("real_labels", real_labels), ("generated_labels", generated_labels)):
        if (not isinstance(labels, torch.Tensor) or labels.shape != (real.shape[0],) or
                labels.dtype not in integer_dtypes or labels.device != real.device or
                (labels < 0).any()):
            raise ValueError(f"{name} must be nonnegative integer labels on the profiles' device")
    real_support, real_counts = real_labels.unique(sorted=True, return_counts=True)
    generated_support, generated_counts = generated_labels.unique(sorted=True, return_counts=True)
    if not torch.equal(real_support, generated_support):
        raise ValueError("real and generated condition support must agree")
    if not torch.equal(real_counts, generated_counts):
        raise ValueError("real and generated sample counts must agree within each condition")
    if (real_counts < minimum).any():
        raise ValueError(f"every observed condition needs at least {minimum} samples")
    directions = _directions(real, num_projections, projections, generator)
    losses = [_projected_loss(real[real_labels == label],
                              generated[generated_labels == label], directions, p)
              for label in real_support]
    return torch.stack(losses).mean()
