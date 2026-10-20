"""Reference Fluvius GAN topology (upstream cGAN.py, MIT licensed).

The discriminator returns logits so the original binary cross entropy can be
evaluated stably with BCEWithLogitsLoss. No output transform is added to G.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _inputs(values, labels, dimension, classes):
    if values.ndim != 2 or values.shape[1] != dimension:
        raise ValueError(f"Expected input shape [N,{dimension}]")
    if not classes:
        return values
    if labels is None or labels.ndim != 1 or len(labels) != len(values):
        raise ValueError("labels must be an integer vector aligned with inputs")
    if labels.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise TypeError("labels must have integer dtype")
    if torch.any((labels < 0) | (labels >= classes)):
        raise ValueError(f"labels must be in [0,{classes})")
    condition = F.one_hot(labels.to(device=values.device, dtype=torch.long), classes)
    return torch.cat((values, condition.to(values.dtype)), dim=1)


def _dimensions(profile_dim, num_classes, latent_dim=24):
    for name, value in (("profile_dim", profile_dim), ("latent_dim", latent_dim)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if type(num_classes) is not int or num_classes < 0:
        raise ValueError("num_classes must be a nonnegative integer")


class Generator(nn.Module):
    def __init__(self, profile_dim=24, num_classes=64, latent_dim=24):
        super().__init__()
        _dimensions(profile_dim, num_classes, latent_dim)
        self.profile_dim, self.num_classes, self.latent_dim = profile_dim, num_classes, latent_dim
        self.model = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 64), nn.LeakyReLU(.2, inplace=True),
            nn.Linear(64, 128), nn.LeakyReLU(.2, inplace=True),
            nn.Linear(128, 64), nn.LeakyReLU(.2, inplace=True), nn.Linear(64, profile_dim),
        )

    def forward(self, noise, labels=None):
        return self.model(_inputs(noise, labels, self.latent_dim, self.num_classes))


class Discriminator(nn.Module):
    def __init__(self, profile_dim=24, num_classes=64):
        super().__init__()
        _dimensions(profile_dim, num_classes)
        self.profile_dim, self.num_classes = profile_dim, num_classes
        self.model = nn.Sequential(
            nn.Linear(profile_dim + num_classes, 64), nn.LeakyReLU(.2, inplace=True),
            nn.Linear(64, 128), nn.LeakyReLU(.2, inplace=True),
            nn.Linear(128, 256), nn.LeakyReLU(.2, inplace=True),
            nn.Dropout(.4), nn.Linear(256, 1),
        )

    def forward(self, profiles, labels=None):
        return self.model(_inputs(profiles, labels, self.profile_dim, self.num_classes))
