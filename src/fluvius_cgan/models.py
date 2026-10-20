"""MLPs adapted from malvin0130/cGAN_EB (MIT; see docs/UPSTREAM_LICENSE.txt).

The discriminator returns raw scores for BCE or spectral-normalized hinge loss.
The legacy generator's linear output permits values outside the training range.
An opt-in fixed quantile wrapper interprets that same head as logits instead.
"""
import math
from numbers import Integral, Real

import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm

from .quantile_decoder import ConditionalQuantileDecoder


def _dimension(name: str, value: int, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < (0 if allow_zero else 1):
        raise ValueError(f"{name} must be an integer >= {0 if allow_zero else 1}")
    return int(value)


def _join(data: torch.Tensor, condition: torch.Tensor, data_dim: int, condition_dim: int) -> torch.Tensor:
    if data.ndim != 2 or data.shape[1] != data_dim:
        raise ValueError(f"data must have shape (N, {data_dim})")
    if condition.ndim != 2 or condition.shape != (data.shape[0], condition_dim):
        raise ValueError(f"condition must have shape (N, {condition_dim})")
    return torch.cat((data, condition.to(dtype=data.dtype)), dim=1)


class Generator(nn.Module):
    def __init__(self, latent_dim: int = 24, condition_dim: int = 64,
                 hidden_dim: int = 64, output_dim: int = 96):
        super().__init__()
        self.latent_dim = _dimension("latent_dim", latent_dim)
        self.condition_dim = _dimension("condition_dim", condition_dim, allow_zero=True)
        hidden_dim = _dimension("hidden_dim", hidden_dim)
        self.output_dim = _dimension("output_dim", output_dim)
        self.model = nn.Sequential(
            nn.Linear(self.latent_dim + self.condition_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, self.output_dim),
        )

    def forward(self, noise: torch.Tensor, onehot: torch.Tensor) -> torch.Tensor:
        return self.model(_join(noise, onehot, self.latent_dim, self.condition_dim))


class QuantileDecodedGenerator(nn.Module):
    """Learn logits, decode physical values, then normalize for physical losses.

    Float32 base logits are promoted before sigmoid and fixed-quantile
    interpolation; physical output is rounded once to float32. ``forward`` then
    uses the same float64 scalar affine transform as real-profile normalization.
    Inference calls ``decode_physical`` directly, avoiding an affine roundtrip.
    This wrapper adds fixed buffers and no trainable parameters.
    """

    def __init__(self, base: Generator, decoder: ConditionalQuantileDecoder, *, offset, scale):
        super().__init__()
        if not isinstance(decoder, ConditionalQuantileDecoder):
            raise ValueError("quantile generator requires a fixed conditional decoder")
        if (base.condition_dim < 1 or base.condition_dim != decoder.condition_dim
                or base.output_dim != decoder.output_dim):
            raise ValueError("base generator and decoder conditional/output dimensions disagree")
        if any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) for v in (offset, scale)) or scale <= 0:
            raise ValueError("quantile generator requires a finite offset and positive effective scale")
        parameter = next(base.parameters())
        if (parameter.dtype not in (torch.float32, torch.float64)
                or parameter.device != decoder.quantiles.device or decoder.quantiles.dtype != torch.float64):
            raise ValueError("base and canonical float64 decoder must share a device")
        self.base, self.decoder = base, decoder
        self.latent_dim, self.condition_dim, self.output_dim = base.latent_dim, base.condition_dim, base.output_dim
        self.register_buffer("normalization_offset", torch.tensor(float(offset), dtype=torch.float64, device=parameter.device))
        self.register_buffer("normalization_scale", torch.tensor(float(scale), dtype=torch.float64, device=parameter.device))

    def decode_with_probabilities(self, noise, onehot):
        parameter = next(self.base.parameters())
        if (not isinstance(noise, torch.Tensor) or noise.dtype != parameter.dtype or noise.device != parameter.device
                or not isinstance(onehot, torch.Tensor) or onehot.device != noise.device
                or onehot.shape != (len(noise), self.condition_dim)
                or not ((onehot == 0) | (onehot == 1)).all() or not (onehot.sum(dim=1) == 1).all()):
            raise ValueError("quantile generator needs matching noise and one-hot conditional rows")
        logits = self.base(noise, onehot)
        return self.decoder.decode_with_probabilities(logits, onehot.argmax(dim=1))

    def decode_physical(self, noise, onehot):
        return self.decode_with_probabilities(noise, onehot)[0]

    def forward(self, noise, onehot):
        physical = self.decode_physical(noise, onehot)
        if (self.normalization_offset.dtype != torch.float64 or self.normalization_scale.dtype != torch.float64
                or not torch.isfinite(self.normalization_offset) or not torch.isfinite(self.normalization_scale)
                or self.normalization_scale <= 0):
            raise ValueError("quantile generator normalization must retain finite canonical float64 buffers")
        normalized = ((physical.to(torch.float64) - self.normalization_offset) / self.normalization_scale).to(physical.dtype)
        if not torch.isfinite(normalized).all():
            raise ValueError("quantile generator produced nonfinite normalized profiles")
        return normalized


class Discriminator(nn.Module):
    def __init__(self, input_dim: int = 96, condition_dim: int = 64,
                 hidden_dim: int = 64, *, objective: str = "bce"):
        super().__init__()
        if objective not in ("bce", "hinge_sn"):
            raise ValueError("objective must be bce or hinge_sn")
        self.input_dim = _dimension("input_dim", input_dim)
        self.condition_dim = _dimension("condition_dim", condition_dim, allow_zero=True)
        hidden_dim = _dimension("hidden_dim", hidden_dim)
        self.model = nn.Sequential(
            nn.Linear(self.input_dim + self.condition_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.0 if objective == "hinge_sn" else 0.4),
            nn.Linear(hidden_dim * 4, 1),
        )
        if objective == "hinge_sn":
            for layer in self.model:
                if isinstance(layer, nn.Linear):
                    spectral_norm(layer)

    def forward(self, data: torch.Tensor, onehot: torch.Tensor) -> torch.Tensor:
        return self.model(_join(data, onehot, self.input_dim, self.condition_dim))
