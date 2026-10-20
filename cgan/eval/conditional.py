"""Conditional fidelity and utility via a label classifier.

A small MLP predicts the joint label (64 classes) from a standardised daily
profile; factor accuracies (PV, heat pump, EV, day type, season) are obtained
by decoding the predicted label.

label recovery (GAN-test, Shmelkov et al. 2018)
    classifier trained on REAL training profiles, evaluated on synthetic
    profiles with their intended labels; compare with its accuracy on the real
    test profiles.
TSTR / CAS (Esteban et al. 2017; Ravuri & Vinyals 2019)
    the same classifier architecture and budget trained on SYNTHETIC profiles,
    evaluated on the real test profiles; compare with TRTR (trained on real).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..data.profiles import N_GROUPS, N_LABELS, N_SEASONS

FACTORS = ("label", "group", "pv", "hp", "ev", "daytype", "season")


def decode_factors(label: np.ndarray) -> dict[str, np.ndarray]:
    label = np.asarray(label)
    group = (label // N_SEASONS) % N_GROUPS
    return {
        "label": label,
        "group": group,
        "pv": group & 1,
        "hp": (group >> 1) & 1,
        "ev": (group >> 2) & 1,
        "daytype": label // (N_GROUPS * N_SEASONS),
        "season": label % N_SEASONS,
    }


def factor_accuracy(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    """Balanced accuracy (mean per-class recall) of each factor; chance = 1 / number of classes."""
    p, t = decode_factors(pred), decode_factors(true)
    out = {}
    for f in FACTORS:
        classes = np.unique(t[f])
        out[f] = float(np.mean([np.mean(p[f][t[f] == c] == c) for c in classes]))
    return out


class LabelClassifier:
    def __init__(self, in_dim: int, hidden: int = 256, epochs: int = 20, batch: int = 512, lr: float = 1e-3,
                 weight_decay: float = 1e-4, seed: int = 0, device: str = "cpu"):
        self.cfg = dict(epochs=epochs, batch=batch, lr=lr, weight_decay=weight_decay, seed=seed)
        self.device = device
        g = torch.Generator().manual_seed(seed)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, N_LABELS)
        )
        with torch.no_grad():
            for p in self.net.parameters():
                p.copy_(torch.empty_like(p).uniform_(-1, 1, generator=g) * (1.0 / max(p.shape[-1], 1) ** 0.5))
        self.net.to(device)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LabelClassifier":
        c = self.cfg
        Xt = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        yt = torch.as_tensor(y, dtype=torch.long, device=self.device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
        gen = torch.Generator(device=self.device).manual_seed(c["seed"])
        loss_fn = nn.CrossEntropyLoss()
        self.net.train()
        for _ in range(c["epochs"]):
            perm = torch.randperm(len(Xt), device=self.device, generator=gen)
            for s in range(0, len(Xt), c["batch"]):
                idx = perm[s:s + c["batch"]]
                opt.zero_grad(set_to_none=True)
                loss_fn(self.net(Xt[idx]), yt[idx]).backward()
                opt.step()
        self.net.eval()
        return self

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        Xt = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        return torch.cat([self.net(Xt[s:s + 65536]).argmax(1) for s in range(0, len(Xt), 65536)]).cpu().numpy()


def balanced_subsample(label: np.ndarray, n_per_label: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = []
    for lab in np.unique(label):
        cand = np.flatnonzero(label == lab)
        idx.append(rng.choice(cand, min(n_per_label, len(cand)), replace=False))
    return np.sort(np.concatenate(idx))
