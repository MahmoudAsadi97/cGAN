"""Dense conditional GAN matching the cGAN_EB baseline architecture.

The discriminator emits logits; training uses BCEWithLogitsLoss.  A linear
generator output is the reproduction baseline.  Softplus is an explicitly
different, nonnegative-output variant, paired with zero-fixed scaling.
"""

import torch
from torch import nn
from torch.nn import functional as F


def _condition(labels: torch.Tensor, num_classes: int, dtype: torch.dtype) -> torch.Tensor:
    if labels.ndim != 1 or labels.dtype not in (torch.int32, torch.int64):
        raise ValueError("labels must be a one-dimensional integer tensor")
    if labels.numel() and (labels.min() < 0 or labels.max() >= num_classes):
        raise ValueError("labels are outside the configured class range")
    return F.one_hot(labels.long(), num_classes=num_classes).to(dtype=dtype)


class Generator(nn.Module):
    def __init__(self, profile_length: int, num_classes: int, latent_dim: int = 24,
                 hidden_dim: int = 64, output_activation: str = "linear"):
        super().__init__()
        if min(profile_length, num_classes, latent_dim, hidden_dim) < 1:
            raise ValueError("model dimensions must be positive")
        if output_activation not in {"linear", "softplus"}:
            raise ValueError("output_activation must be linear or softplus")
        self.profile_length = profile_length
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        layers = [
            nn.Linear(latent_dim + num_classes, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim * 2), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, profile_length),
        ]
        if output_activation == "softplus":
            layers.append(nn.Softplus())
        self.network = nn.Sequential(*layers)

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if noise.ndim != 2 or noise.shape != (labels.shape[0], self.latent_dim):
            raise ValueError("noise must have shape (number of labels, latent_dim)")
        condition = _condition(labels, self.num_classes, noise.dtype)
        return self.network(torch.cat((noise, condition), dim=1))


class Discriminator(nn.Module):
    def __init__(self, profile_length: int, num_classes: int, hidden_dim: int = 64):
        super().__init__()
        if min(profile_length, num_classes, hidden_dim) < 1:
            raise ValueError("model dimensions must be positive")
        self.profile_length = profile_length
        self.num_classes = num_classes
        self.network = nn.Sequential(
            nn.Linear(profile_length + num_classes, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim * 2), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim * 4), nn.LeakyReLU(0.2),
            nn.Dropout(0.4), nn.Linear(hidden_dim * 4, 1),
        )

    def forward(self, profiles: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if profiles.ndim != 2 or profiles.shape != (labels.shape[0], self.profile_length):
            raise ValueError("profiles must have shape (number of labels, profile_length)")
        condition = _condition(labels, self.num_classes, profiles.dtype)
        return self.network(torch.cat((profiles, condition), dim=1))
