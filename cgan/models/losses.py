"""Adversarial losses.

``bce``         reference cGAN_EB: sigmoid followed by ``F.binary_cross_entropy``
                (reproduces the clamped log of ``nn.BCELoss``); D loss is the
                mean of the real and fake terms as in cGAN.py:97.
``bce_logits``  same objective computed with ``binary_cross_entropy_with_logits``.
``wgan_gp``     Wasserstein loss with gradient penalty (Gulrajani et al. 2017).
``hinge``       hinge loss (Lim & Ye 2017; Miyato et al. 2018).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def d_loss(kind: str, real_logit: torch.Tensor, fake_logit: torch.Tensor) -> torch.Tensor:
    if kind == "bce":
        real = F.binary_cross_entropy(torch.sigmoid(real_logit), torch.ones_like(real_logit))
        fake = F.binary_cross_entropy(torch.sigmoid(fake_logit), torch.zeros_like(fake_logit))
        return (real + fake) / 2
    if kind == "bce_logits":
        real = F.binary_cross_entropy_with_logits(real_logit, torch.ones_like(real_logit))
        fake = F.binary_cross_entropy_with_logits(fake_logit, torch.zeros_like(fake_logit))
        return (real + fake) / 2
    if kind == "wgan_gp":
        return fake_logit.mean() - real_logit.mean()
    if kind == "hinge":
        return F.relu(1 - real_logit).mean() + F.relu(1 + fake_logit).mean()
    raise ValueError(f"unknown loss {kind!r}")


def g_loss(kind: str, fake_logit: torch.Tensor) -> torch.Tensor:
    if kind == "bce":
        return F.binary_cross_entropy(torch.sigmoid(fake_logit), torch.ones_like(fake_logit))
    if kind == "bce_logits":
        return F.binary_cross_entropy_with_logits(fake_logit, torch.ones_like(fake_logit))
    if kind in ("wgan_gp", "hinge"):
        return -fake_logit.mean()
    raise ValueError(f"unknown loss {kind!r}")


def gradient_penalty(critic, real: torch.Tensor, fake: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    """Two-sided penalty E[(||grad_x D(x_hat, y)||_2 - 1)^2] on random interpolates."""
    eps = torch.rand(real.shape[0], 1, device=real.device, dtype=real.dtype)
    x_hat = (eps * real + (1 - eps) * fake).requires_grad_(True)
    out = critic(x_hat, label)
    (grad,) = torch.autograd.grad(out.sum(), x_hat, create_graph=True)
    return ((grad.norm(2, dim=1) - 1) ** 2).mean()
