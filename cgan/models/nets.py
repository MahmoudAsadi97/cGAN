"""Generator / discriminator networks.

``RefGenerator`` and ``RefDiscriminator`` reproduce the cGAN_EB architecture
(Li, Dong & Qiu 2025, Table 2 and cGAN.py) exactly: 24-d Gaussian noise
concatenated with a 64-d one-hot label, hidden widths 64-128-64 for G and
64-128-256 for D, LeakyReLU(0.2), Dropout(0.4) before the D output, linear G
output.  D returns logits; the sigmoid of the reference is applied in the
loss (see ``cgan.models.losses``) so that the faithful configuration can
reproduce ``Sigmoid -> BCELoss`` exactly while other configurations use the
numerically stable logit form.

``MLPGenerator`` / ``ProjectionCritic`` form the improved model: wider MLPs,
factorised condition embeddings and a projection discriminator
(Miyato & Koyama 2018) trained with WGAN-GP.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.profiles import N_DAYTYPES, N_GROUPS, N_LABELS, N_SEASONS


def decode_label_t(label: torch.Tensor):
    per_daytype = N_GROUPS * N_SEASONS
    daytype = label // per_daytype
    group = (label // N_SEASONS) % N_GROUPS
    season = label % N_SEASONS
    return daytype, group, season


class OneHotCond(nn.Module):
    """Flat one-hot encoding of the joint label (reference conditioning)."""

    def __init__(self, n_labels: int = N_LABELS):
        super().__init__()
        self.n_labels = n_labels
        self.dim = n_labels

    def forward(self, label: torch.Tensor) -> torch.Tensor:
        return F.one_hot(label, self.n_labels).float()


class FactorCond(nn.Module):
    """Concatenated embeddings of PV, heat pump, EV, day type, season and the joint label.

    The per-factor embeddings let labels that share a factor share
    parameters; the joint-label embedding keeps capacity for interactions.
    """

    def __init__(self, emb_dim: int = 8, joint_dim: int = 16):
        super().__init__()
        self.pv = nn.Embedding(2, emb_dim)
        self.hp = nn.Embedding(2, emb_dim)
        self.ev = nn.Embedding(2, emb_dim)
        self.daytype = nn.Embedding(N_DAYTYPES, emb_dim)
        self.season = nn.Embedding(N_SEASONS, emb_dim)
        self.joint = nn.Embedding(N_LABELS, joint_dim) if joint_dim > 0 else None
        self.dim = 5 * emb_dim + max(joint_dim, 0)

    def forward(self, label: torch.Tensor) -> torch.Tensor:
        daytype, group, season = decode_label_t(label)
        parts = [
            self.pv(group & 1),
            self.hp((group >> 1) & 1),
            self.ev((group >> 2) & 1),
            self.daytype(daytype),
            self.season(season),
        ]
        if self.joint is not None:
            parts.append(self.joint(label))
        return torch.cat(parts, dim=1)


def make_cond(kind: str, **kw) -> nn.Module:
    if kind == "onehot":
        return OneHotCond()  # embedding options do not apply
    if kind == "factor":
        return FactorCond(**kw)
    raise ValueError(f"unknown conditioning {kind!r}")


class RefGenerator(nn.Module):
    def __init__(self, noise_dim: int = 24, out_dim: int = 24, hidden: int = 64, cond: str = "onehot", cond_kw=None):
        super().__init__()
        self.noise_dim = noise_dim
        self.cond = make_cond(cond, **(cond_kw or {}))
        self.model = nn.Sequential(
            nn.Linear(noise_dim + self.cond.dim, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, hidden * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden * 2, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, noise: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        return self.model(torch.cat([noise, self.cond(label)], dim=1))


class RefDiscriminator(nn.Module):
    def __init__(self, in_dim: int = 24, hidden: int = 64, dropout: float = 0.4, cond: str = "onehot", cond_kw=None):
        super().__init__()
        self.cond = make_cond(cond, **(cond_kw or {}))
        self.model = nn.Sequential(
            nn.Linear(in_dim + self.cond.dim, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, hidden * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden * 2, hidden * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden * 4, 1),
        )

    def forward(self, x: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        return self.model(torch.cat([x, self.cond(label)], dim=1))


def _mlp(dims, act_slope=0.2, norm: str | None = None):
    layers = []
    for a, b in zip(dims[:-1], dims[1:]):
        layers.append(nn.Linear(a, b))
        if norm == "layer":
            layers.append(nn.LayerNorm(b))
        layers.append(nn.LeakyReLU(act_slope, inplace=True))
    return nn.Sequential(*layers)


class BoundedOutput(nn.Module):
    """Maps a linear output into [lo, hi] with a sigmoid (identity when no range is given).

    With ``lo_nopv`` (the scaled 0 kWh) the output of labels without PV is
    passed through a smooth barrier  y -> lo_nopv + softplus(beta (y - lo_nopv)) / beta,
    which is close to the identity above zero net load and never below it:
    households without PV cannot export.  A barrier keeps gradients alive near
    zero, where most of their load lies (a sigmoid with lower bound 0 would
    saturate there).
    """

    def __init__(self, out_range=None, lo_nopv=None, beta: float = 8.0):
        super().__init__()
        self.bounded = out_range is not None
        self.pv_aware = self.bounded and lo_nopv is not None
        self.beta = beta
        lo, hi = out_range if self.bounded else (0.0, 1.0)
        # Not persistent: the range is part of the saved model config.
        self.register_buffer("lo", torch.tensor(float(lo)), persistent=False)
        self.register_buffer("span", torch.tensor(float(hi) - float(lo)), persistent=False)
        self.register_buffer("lo_nopv", torch.tensor(float(lo_nopv if lo_nopv is not None else lo)),
                             persistent=False)

    def forward(self, h: torch.Tensor, label: torch.Tensor | None = None) -> torch.Tensor:
        if not self.bounded:
            return h
        y = self.lo + self.span * torch.sigmoid(h)
        if self.pv_aware:
            _, group, _ = decode_label_t(label)
            barrier = self.lo_nopv + nn.functional.softplus(self.beta * (y - self.lo_nopv)) / self.beta
            y = torch.where((group & 1).bool()[:, None], y, barrier)
        return y


class MLPGenerator(nn.Module):
    """Conditional MLP generator; the condition is fed to the input layer.

    With ``out_range`` the output is confined to [lo, hi] in the scaled space
    (training range plus a margin), which prevents rare extreme samples from
    exploding through the inverse of a compressive value transform.
    """

    def __init__(self, noise_dim: int = 32, out_dim: int = 24, hidden: int = 256, depth: int = 3,
                 cond: str = "factor", cond_kw=None, out_range=None, lo_nopv=None):
        super().__init__()
        self.noise_dim = noise_dim
        self.cond = make_cond(cond, **(cond_kw or {}))
        self.body = _mlp([noise_dim + self.cond.dim] + [hidden] * depth)
        self.out = nn.Linear(hidden, out_dim)
        self.bound = BoundedOutput(out_range, lo_nopv)

    def forward(self, noise: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        return self.bound(self.out(self.body(torch.cat([noise, self.cond(label)], dim=1))), label)


class ProjectionCritic(nn.Module):
    """MLP critic with an optional projection term <embed(y), phi(x)>.

    No batch normalisation (incompatible with the per-sample gradient penalty);
    optional LayerNorm.
    """

    def __init__(self, in_dim: int = 24, hidden: int = 256, depth: int = 3, cond: str = "factor",
                 cond_kw=None, projection: bool = True, concat: bool = True, norm: str | None = None):
        super().__init__()
        self.cond = make_cond(cond, **(cond_kw or {}))
        self.concat = concat
        self.body = _mlp([in_dim + (self.cond.dim if concat else 0)] + [hidden] * depth, norm=norm)
        self.out = nn.Linear(hidden, 1)
        self.proj = nn.Linear(self.cond.dim, hidden, bias=False) if projection else None

    def forward(self, x: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        c = self.cond(label)
        h = self.body(torch.cat([x, c], dim=1) if self.concat else x)
        out = self.out(h)
        if self.proj is not None:
            out = out + (self.proj(c) * h).sum(dim=1, keepdim=True)
        return out


def build_nets(cfg: dict, data_dim: int):
    """Construct (G, D) from the ``model`` section of a run config."""
    arch = cfg.get("arch", "ref")
    cond = cfg.get("cond", "onehot")
    cond_kw = cfg.get("cond_kw") or {}
    if arch == "ref":
        if cfg.get("out_range") is not None or cfg.get("out_bound") is not None:
            raise ValueError("bounded output is not available for arch 'ref' (reference linear output)")
        g = RefGenerator(cfg.get("noise_dim", 24), data_dim, cfg.get("hidden", 64), cond, cond_kw)
        d = RefDiscriminator(data_dim, cfg.get("hidden", 64), cfg.get("dropout", 0.4), cond, cond_kw)
    elif arch == "mlp":
        g = MLPGenerator(cfg.get("noise_dim", 32), data_dim, cfg.get("g_hidden", 256), cfg.get("g_depth", 3),
                         cond, cond_kw, cfg.get("out_range"), cfg.get("lo_nopv"))
        d = ProjectionCritic(data_dim, cfg.get("d_hidden", 256), cfg.get("d_depth", 3), cond, cond_kw,
                             cfg.get("projection", True), cfg.get("concat", True), cfg.get("d_norm"))
    else:
        raise ValueError(f"unknown arch {arch!r}")
    return g, d
