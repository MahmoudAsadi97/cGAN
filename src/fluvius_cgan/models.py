"""MLPs adapted from malvin0130/cGAN_EB (MIT; see docs/UPSTREAM_LICENSE.txt).

The discriminator returns raw scores for BCE or spectral-normalized hinge loss.
The generator's linear output deliberately permits values outside the training
min/max range.
"""
from numbers import Integral

import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm


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
