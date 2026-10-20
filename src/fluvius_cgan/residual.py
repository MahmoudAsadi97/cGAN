"""Fixed train-mean offsets for an otherwise unchanged, unbounded generator.

The caller must pass only the actual selected training profiles, already in the
training normalization coordinates. This module never opens prepared data or
fits anything during generation. Means accumulate in float64 and are stored as
float32, matching the normalized training tensors.
"""
from numbers import Integral

import torch
from torch import nn


def fit_mean_profiles(x_normalized, labels, condition_dim: int) -> torch.Tensor:
    """Fit an actual-training-subset mean table, without consulting held-out data.

    Conditional tables have ``condition_dim`` rows. Entirely NaN rows mark labels
    absent from the supplied data; ``ResidualGenerator`` converts those rows to
    finite zeros and retains a separate support mask. With ``condition_dim=0``,
    return one global mean row, irrespective of labels or their frequencies.
    The returned float32 tensor is detached and on the profile tensor's device.
    """
    if isinstance(condition_dim, bool) or not isinstance(condition_dim, Integral) or condition_dim < 0:
        raise ValueError("condition_dim must be a nonnegative integer")
    condition_dim = int(condition_dim)
    try:
        values = torch.as_tensor(x_normalized)
        classes = torch.as_tensor(labels)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError("profiles and labels must be numeric arrays") from exc
    if values.ndim != 2 or not values.shape[0] or not values.shape[1]:
        raise ValueError("profiles must be a nonempty rank-2 array")
    if values.dtype == torch.bool or values.is_complex() or not torch.isfinite(values).all():
        raise ValueError("profiles must contain finite real numeric values")
    if classes.ndim != 1 or len(classes) != len(values) or classes.dtype not in (
        torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
    ):
        raise ValueError("labels must be an integer vector matching the profiles")
    if (classes < 0).any() or (condition_dim and (classes >= condition_dim).any()):
        raise ValueError("labels must be nonnegative and within condition_dim")
    values = values.detach().to(dtype=torch.float64)
    classes = classes.detach().to(device=values.device, dtype=torch.int64)
    means = torch.full((condition_dim or 1, values.shape[1]), float("nan"),
                       dtype=torch.float32, device=values.device)
    if not condition_dim:
        means[0] = values.mean(dim=0).to(dtype=torch.float32)
        supported = torch.ones(1, dtype=torch.bool, device=values.device)
    else:
        supported = torch.zeros(condition_dim, dtype=torch.bool, device=values.device)
        for label in torch.unique(classes):
            means[label] = values[classes == label].mean(dim=0).to(dtype=torch.float32)
            supported[label] = True
    if not torch.isfinite(means[supported]).all():
        raise ValueError("training means cannot be represented as finite float32 values")
    return means


class ResidualGenerator(nn.Module):
    """Add fixed train means to a base generator; keep its initialization intact.

    Conditional inputs must be one-hot and supported by the fitted training data.
    The unconditional case uses only the single global mean and an empty condition
    matrix. Buffers follow module device/dtype moves and survive state-dict replay.
    No clipping or output activation restricts the residual or resulting profile.
    """

    def __init__(self, base: nn.Module, mean_profiles: torch.Tensor):
        super().__init__()
        self.base = base
        self.latent_dim = base.latent_dim
        self.condition_dim = base.condition_dim
        self.output_dim = base.output_dim
        try:
            means = torch.as_tensor(mean_profiles).detach().clone()
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError("mean_profiles must be a numeric tensor") from exc
        if means.shape != (self.condition_dim or 1, self.output_dim):
            raise ValueError("mean_profiles shape disagrees with the base generator")
        if not means.is_floating_point() or means.is_complex():
            raise ValueError("mean_profiles must be floating-point values")
        supported = torch.isfinite(means).all(dim=1)
        missing = torch.isnan(means).all(dim=1)
        if not supported.any() or not (supported | missing).all():
            raise ValueError("mean rows must be finite or entirely NaN for unsupported labels")
        if not self.condition_dim and not supported.all():
            raise ValueError("the unconditional global mean must be finite")
        parameter = next(base.parameters())
        means[missing] = 0
        means = means.to(device=parameter.device, dtype=parameter.dtype)
        if not torch.isfinite(means).all():
            raise ValueError("mean_profiles cannot be represented in the base generator dtype")
        self.register_buffer("mean_profiles", means)
        self.register_buffer("supported_conditions", supported.to(device=parameter.device))

    def forward(self, noise: torch.Tensor, onehot: torch.Tensor) -> torch.Tensor:
        if onehot.ndim != 2 or onehot.shape != (len(noise), self.condition_dim):
            raise ValueError(f"condition must have shape (N, {self.condition_dim})")
        if self.condition_dim:
            if not (((onehot == 0) | (onehot == 1)).all() and (onehot.sum(dim=1) == 1).all()):
                raise ValueError("conditions must be one-hot labels")
            if (onehot[:, ~self.supported_conditions] != 0).any():
                raise ValueError("unsupported condition absent from the fitted training subset")
        residual = self.base(noise, onehot)
        if self.condition_dim:
            offset = onehot.to(dtype=residual.dtype) @ self.mean_profiles
        else:
            offset = self.mean_profiles
        return residual + offset
