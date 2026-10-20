"""Cheap validation score used for checkpoint selection during training.

Score = mean over labels of the relative Frechet distance between Gaussian fits
of real (validation split) and generated 24-h profiles in kWh:

    rFD_k = FD(N(mu_r, S_r), N(mu_g, S_g)) / tr(S_r)

Dividing by tr(S_r) makes labels with large loads (heat pump winter, EV) and
small loads comparable.  Generation uses a fixed noise seed so scores of
different checkpoints are computed with common random numbers.
"""
from __future__ import annotations

import numpy as np
import torch


def frechet_distance(mu1, s1, mu2, s2) -> float:
    """||mu1 - mu2||^2 + tr(S1 + S2 - 2 (S1 S2)^{1/2}) for symmetric PSD S1, S2."""
    diff = np.asarray(mu1, np.float64) - np.asarray(mu2, np.float64)
    s1 = np.atleast_2d(np.asarray(s1, np.float64))
    s2 = np.atleast_2d(np.asarray(s2, np.float64))
    # Eigenvalues of S1 S2 are real and non-negative for PSD S1, S2.
    ev = np.linalg.eigvals(s1 @ s2).real
    tr_sqrt = np.sqrt(np.clip(ev, 0.0, None)).sum()
    return float(diff @ diff + np.trace(s1) + np.trace(s2) - 2.0 * tr_sqrt)


def gaussian_stats(x: np.ndarray):
    x = np.asarray(x, np.float64)
    return x.mean(0), np.cov(x, rowvar=False)


@torch.no_grad()
def generate_scaled(G, labels: torch.Tensor, generator: torch.Generator, chunk: int = 65536) -> torch.Tensor:
    was_training = G.training
    G.eval()
    out = []
    for s in range(0, labels.shape[0], chunk):
        y = labels[s:s + chunk]
        z = torch.randn(y.shape[0], G.noise_dim, device=y.device, generator=generator)
        out.append(G(z, y))
    G.train(was_training)
    return torch.cat(out)


class ValScorer:
    def __init__(self, split, scaler, n_max: int = 1000, seed: int = 12345, device: str = "cpu"):
        self.scaler = scaler
        self.seed = seed
        self.device = device
        self.labels, counts = np.unique(split.label, return_counts=True)
        self.n = np.minimum(counts, n_max)
        self.stats = {}
        for lab in self.labels:
            mu, cov = gaussian_stats(split.X[split.label == lab])
            self.stats[int(lab)] = (mu, cov, float(np.trace(cov)))
        self.y = torch.from_numpy(np.repeat(self.labels, self.n)).to(device)

    def per_label(self, G) -> dict[int, float]:
        gen = torch.Generator(device=self.device).manual_seed(self.seed)
        x = self.scaler.inverse_torch(generate_scaled(G, self.y, gen)).double().cpu().numpy()
        out, start = {}, 0
        for lab, n in zip(self.labels, self.n):
            mu_g, cov_g = gaussian_stats(x[start:start + n])
            mu_r, cov_r, tr = self.stats[int(lab)]
            out[int(lab)] = frechet_distance(mu_r, cov_r, mu_g, cov_g) / tr
            start += n
        return out

    def score(self, G) -> float:
        vals = np.array(list(self.per_label(G).values()))
        return float(np.mean(vals)) if np.isfinite(vals).all() else float("inf")
