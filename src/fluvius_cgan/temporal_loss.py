"""Fixed train-only derivative coordinates for a conditional SW2-squared loss.

The caller supplies the actual selected training rows in the same normalized
coordinates used for training. The scale is the square root of the equally
weighted condition means of squared, signed adjacent differences. It is one
scalar for every condition and slot, fitted once without sampling or centering.

The loss uses the existing conditional sliced-Wasserstein objective on those
differences divided by the fixed scalar. Differences do not wrap from the last
slot to the first. This is an auxiliary minibatch distribution objective; its
sample-size limitations are the same as those documented in distribution_loss.
"""
from __future__ import annotations

import math
from numbers import Integral, Real

import numpy as np
import torch

from .distribution_loss import conditional_sliced_wasserstein_loss


def _fixed_scale(scale) -> float:
    if (isinstance(scale, (bool, np.bool_)) or not isinstance(scale, Real)
            or not math.isfinite(scale) or scale <= 0):
        raise ValueError("scale must be a fixed positive finite real scalar, not a tensor")
    return float(scale)


def fit_derivative_scale(normalized_profiles, labels, *, batch_size=4096) -> dict:
    """Estimate one condition-macro RMS derivative scale from all supplied rows.

    NumPy arrays and CPU tensors are accepted; profiles must be floating-point
    with 24 or 96 slots, and labels must be integers in [0, 63]. Float64 chunked
    accumulation bounds temporary memory without subsampling. The JSON-safe
    result records both the float64 estimate and its positive finite float32
    representation, which the caller should freeze for float32 training.

    Every observed condition has equal weight, irrespective of row count.
    A condition with zero derivatives is allowed, but an overall zero scale
    is rejected. No epsilon, clipping, centering, or fallback alters the fit.
    """
    if (isinstance(batch_size, (bool, np.bool_)) or not isinstance(batch_size, Integral)
            or batch_size < 1):
        raise ValueError("batch_size must be a positive integer")
    batch_size = int(batch_size)
    if isinstance(normalized_profiles, torch.Tensor):
        if normalized_profiles.device.type != "cpu":
            raise ValueError("normalized_profiles must be on CPU for scale fitting")
        values = normalized_profiles.detach()
        floating = values.is_floating_point()
    else:
        try:
            values = np.asarray(normalized_profiles)
        except (TypeError, ValueError) as exc:
            raise ValueError("normalized_profiles must be a floating array") from exc
        floating = values.dtype.kind == "f"
    if (not floating or values.ndim != 2 or not values.shape[0]
            or values.shape[1] not in (24, 96)):
        raise ValueError("normalized_profiles must be a nonempty floating array with 24 or 96 features")
    if isinstance(labels, torch.Tensor):
        if labels.device.type != "cpu":
            raise ValueError("labels must be on CPU for scale fitting")
        labels = labels.detach()
        if labels.is_floating_point() or labels.is_complex() or labels.dtype == torch.bool:
            raise ValueError("labels must be an integer vector matching the profiles")
        labels = labels.numpy()
    try:
        classes = np.asarray(labels)
    except (TypeError, ValueError) as exc:
        raise ValueError("labels must be an integer vector matching the profiles") from exc
    if (classes.ndim != 1 or len(classes) != len(values) or classes.dtype.kind not in "iu"
            or np.any(classes < 0) or np.any(classes >= 64)):
        raise ValueError("labels must be an integer vector in [0, 63] matching the profiles")
    classes = classes.astype(np.int64, copy=False)
    counts = np.bincount(classes, minlength=64)
    sums = np.zeros(64, dtype=np.float64)
    n_features = int(values.shape[1])
    for start in range(0, len(values), batch_size):
        block = values[start:start + batch_size]
        if isinstance(block, torch.Tensor):
            block = block.to(dtype=torch.float64).numpy()
        else:
            block = np.asarray(block, dtype=np.float64)
        if not np.isfinite(block).all():
            raise ValueError("normalized_profiles must contain finite values")
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            differences = np.diff(block, axis=1)
            row_sums = np.square(differences).sum(axis=1, dtype=np.float64)
            chunk_labels = classes[start:start + batch_size]
            for label in np.unique(chunk_labels):
                sums[label] += row_sums[chunk_labels == label].sum(dtype=np.float64)
        if not np.isfinite(sums).all():
            raise ValueError("derivative mean squares must be finite in float64")
    support = np.flatnonzero(counts)
    means = sums[support] / (counts[support].astype(np.float64) * (n_features - 1))
    estimate = math.sqrt(math.fsum(float(value) / len(support) for value in means))
    _fixed_scale(estimate)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        used = float(np.float32(estimate))
    if not math.isfinite(used) or used <= 0:
        raise ValueError("derivative scale must remain positive and finite in float32")
    return {
        "estimate_float64": estimate,
        "used_float32": used,
        "n_profiles": int(len(values)),
        "n_features": n_features,
        "n_differences": n_features - 1,
        "condition_counts": {str(label): int(counts[label]) for label in support},
        "condition_mean_squared_differences": {str(label): float(mean) for label, mean in zip(support, means)},
        "estimator": "sqrt_of_equal_condition_mean_squared_adjacent_differences",
        "condition_weighting": "equal_observed_conditions",
        "difference_definition": "z[:, 1:] - z[:, :-1]",
        "centering": False,
        "wraparound": False,
        "accumulation_dtype": "float64",
        "accumulation_batch_size": batch_size,
        "usage_dtype": "float32",
    }


def derivative_sliced_wasserstein_loss(
        real: torch.Tensor, generated: torch.Tensor,
        real_labels: torch.Tensor, generated_labels: torch.Tensor, scale, *,
        num_projections: int = 128, min_samples_per_condition: int = 32,
        projections: torch.Tensor | None = None,
        generator: torch.Generator | None = None) -> torch.Tensor:
    """Return condition-macro empirical SW2**2 of signed adjacent differences.

    ``scale`` is a positive finite non-tensor scalar fitted before training and
    must remain positive and finite in the profiles' floating-point dtype.
    Gradients flow through profile differences, never through the fixed scale.
    Explicit projections have shape (D-1, K); otherwise use a dedicated torch
    generator for projection draws independent of latent sampling. The existing
    conditional objective enforces matched condition support, counts, dtype and
    device, with at least ``min_samples_per_condition`` rows per condition.
    """
    scale = _fixed_scale(scale)
    for name, values in (("real", real), ("generated", generated)):
        if (not isinstance(values, torch.Tensor) or values.ndim != 2 or not values.shape[0]
                or values.shape[1] not in (24, 96) or not values.is_floating_point()):
            raise ValueError(f"{name} must be a nonempty floating tensor with 24 or 96 features")
        if not torch.isfinite(values).all():
            raise ValueError(f"{name} must contain finite values")
    represented_scale = real.new_tensor(scale)
    if not torch.isfinite(represented_scale) or represented_scale <= 0:
        raise ValueError("scale must remain positive and finite in the profiles' dtype")
    for name, labels in (("real_labels", real_labels), ("generated_labels", generated_labels)):
        if isinstance(labels, torch.Tensor):
            if labels.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
                raise ValueError(f"{name} must contain integer condition labels in [0, 63]")
            if (labels >= 64).any():
                raise ValueError(f"{name} must contain condition labels in [0, 63]")
    return conditional_sliced_wasserstein_loss(
        torch.diff(real, dim=1) / scale, torch.diff(generated, dim=1) / scale,
        real_labels, generated_labels, num_projections=num_projections, p=2,
        min_samples_per_condition=min_samples_per_condition,
        projections=projections, generator=generator,
    )
