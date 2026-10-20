"""Train-only conditional empirical and Gaussian-copula comparators."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import rankdata


def _labels(labels, length=None):
    value = np.asarray(labels)
    if value.ndim != 1 or (length is not None and len(value) != length):
        raise ValueError("condition labels must be a vector matching the profiles")
    try:
        valid = np.isfinite(value).all() and np.all(value == np.floor(value))
    except TypeError as error:
        raise ValueError("condition labels must be nonnegative integers") from error
    if not valid or np.any(value < 0):
        raise ValueError("condition labels must be nonnegative integers")
    return value.astype(np.int64)


@dataclass
class ConditionalBaseline:
    """A fitted comparator; all arrays originate exclusively from training."""

    kind: str
    profile_dim: int
    groups: dict

    def sample(self, labels, seed):
        labels = _labels(labels)
        missing = set(labels.tolist()) - self.groups.keys()
        if missing:
            raise ValueError(f"unsupported condition labels: {sorted(missing)}")
        rng = np.random.default_rng(seed)
        result = np.empty((len(labels), self.profile_dim), dtype=np.float64)
        for label in np.unique(labels):
            indices = np.flatnonzero(labels == label)
            group = self.groups[int(label)]
            if self.kind == "mean":
                result[indices] = group["mean"]
            elif self.kind == "resample":
                values = group["values"]
                result[indices] = values[rng.integers(len(values), size=len(indices))]
            else:
                ordered = group["ordered"]
                result[indices] = ordered[0]
                active = group["active"]
                if len(active):
                    normal = rng.standard_normal((len(indices), len(active)))
                    normal = normal @ group["factor"].T
                    probability = ndtr(normal)
                    ranks = np.clip(np.ceil(len(ordered) * probability).astype(int) - 1,
                                    0, len(ordered) - 1)
                    result[np.ix_(indices, active)] = ordered[ranks, active]
        return result


def fit_baseline(kind, x, labels):
    """Fit mean, complete-day resampling or the frozen Gaussian copula.

    Copula marginals use average ranks and empirical step inverse quantiles.
    Constant slots are fixed. Nonconstant normal-score correlations use
    0.01 identity shrinkage, then a 1e-8 eigenvalue floor and renormalization.
    """
    if kind not in {"mean", "resample", "copula"}:
        raise ValueError("kind must be mean, resample or copula")
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or not all(x.shape) or not np.isfinite(x).all():
        raise ValueError("profiles must be a nonempty finite matrix")
    labels = _labels(labels, len(x))
    groups = {}
    for label in np.unique(labels):
        values = x[labels == label].copy()
        if kind == "mean":
            group = {"mean": values.mean(axis=0)}
        elif kind == "resample":
            group = {"values": values}
        else:
            active = np.flatnonzero(np.ptp(values, axis=0) > 0)
            group = {"ordered": np.sort(values, axis=0), "active": active}
            if len(active):
                ranks = rankdata(values[:, active], axis=0, method="average")
                scores = ndtri((ranks - 0.5) / len(values))
                correlation = np.atleast_2d(np.corrcoef(scores, rowvar=False))
                correlation = 0.99 * correlation + 0.01 * np.eye(len(active))
                eigenvalues, eigenvectors = np.linalg.eigh(correlation)
                correlation = (eigenvectors * np.maximum(eigenvalues, 1e-8)) @ eigenvectors.T
                scale = np.sqrt(np.diag(correlation))
                correlation /= scale[:, None] * scale[None, :]
                group["factor"] = np.linalg.cholesky(correlation)
            else:
                group["factor"] = np.empty((0, 0))
        groups[int(label)] = group
    return ConditionalBaseline(kind, x.shape[1], groups)
