"""Train-only conditional Gaussian copula with empirical marginal quantiles.

This is a statistical fresh-sample comparator, not a GAN or a row bootstrap.
Fit separately within each supplied training label: transform average ranks to
normal scores, estimate their correlation, and shrink it toward the identity.
Generation draws a joint Gaussian, applies the normal CDF, then interpolates
each training marginal in the original physical units, including signed values.

The default 257 uniformly spaced quantile knots bound model memory. They
approximate the full empirical linear quantile function and can noticeably
smooth sparse tails. ``quantile_knots=None`` retains every sorted observation.
Both modes use linear interpolation (including across ties); neither models
values outside the observed per-condition, per-feature training range. Midranks
are a pragmatic approximation for tied/discrete margins. Gaussian dependence
does not represent arbitrary tail dependence or multimodal joint structure.

Method references (the implementation does not require statsmodels):
https://www.statsmodels.org/stable/generated/statsmodels.distributions.copula.api.GaussianCopula.html
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rankdata.html
https://numpy.org/doc/stable/reference/generated/numpy.quantile.html
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from numbers import Integral, Real
from pathlib import Path

import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import rankdata


def _positive_integer(value, name: str, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _settings(condition_dim, quantile_knots, shrinkage):
    condition_dim = _positive_integer(condition_dim, "condition_dim")
    if quantile_knots is not None:
        quantile_knots = _positive_integer(quantile_knots, "quantile_knots", 2)
    if (isinstance(shrinkage, (bool, np.bool_)) or not isinstance(shrinkage, Real)
            or not np.isfinite(shrinkage) or not 0 <= shrinkage <= 1):
        raise ValueError("shrinkage must be finite and between 0 and 1")
    return condition_dim, quantile_knots, float(shrinkage)


def _labels(labels, condition_dim: int, n: int | None = None) -> np.ndarray:
    try:
        labels = np.asarray(labels)
    except (TypeError, ValueError) as exc:
        raise ValueError("labels must be an integer vector") from exc
    if labels.ndim != 1 or labels.dtype.kind not in "iu" or (n is not None and len(labels) != n):
        raise ValueError("labels must be an integer vector matching the profiles")
    if np.any(labels < 0) or np.any(labels >= condition_dim):
        raise ValueError("labels must be nonnegative and within condition_dim")
    return labels.astype(np.int64, copy=False)


def _normal_scores(values: np.ndarray) -> np.ndarray:
    """Finite plotting positions (midrank - 0.5) / n, even for tied extrema."""
    return ndtri((rankdata(values, method="average", axis=0) - 0.5) / len(values))


def _repair_correlation(correlation: np.ndarray, eigenvalue_floor: float = 1e-10) -> np.ndarray:
    """Symmetrize, floor eigenvalues, and renormalize the diagonal to one.

    This numerical PSD repair is not a nearest-correlation optimization. The
    fixed identity shrinkage is applied separately before this repair.
    """
    correlation = np.asarray(correlation, dtype=np.float64)
    if (correlation.ndim != 2 or correlation.shape[0] != correlation.shape[1]
            or not correlation.shape[0] or not np.isfinite(correlation).all()):
        raise ValueError("correlation must be a finite nonempty square matrix")
    if not np.isfinite(eigenvalue_floor) or eigenvalue_floor <= 0:
        raise ValueError("eigenvalue_floor must be finite and positive")
    symmetric = 0.5 * correlation + 0.5 * correlation.T
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    repaired = (eigenvectors * np.maximum(eigenvalues, eigenvalue_floor)) @ eigenvectors.T
    scale = np.sqrt(np.diag(repaired))
    repaired /= scale[:, None] * scale[None, :]
    repaired = 0.5 * repaired + 0.5 * repaired.T
    np.fill_diagonal(repaired, 1.0)
    return repaired


def _inverse_quantiles(uniform: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    """Monotone linear inverse with exact ties/endpoints and bounded probabilities.

    Anchor same-sign intervals at the endpoint nearest zero. This avoids the
    rounding variation of two weighted copies of an equal value, and avoids
    cancellation near zero. Opposite-sign intervals use the weighted form so
    their endpoint difference never overflows. Each fixed interval formula is
    monotone in the probability; no output clipping or cumulative repair is
    needed.
    """
    positions = np.clip(uniform, 0.0, 1.0) * (len(quantiles) - 1)
    lower = np.minimum(positions.astype(np.int64), len(quantiles) - 2)
    weight = positions - lower
    columns = np.arange(quantiles.shape[1])
    left, right = quantiles[lower, columns], quantiles[lower + 1, columns]
    nonnegative = left >= 0
    nonpositive = (right <= 0) & ~nonnegative
    crossing = ~(nonnegative | nonpositive)
    values = np.empty_like(weight)
    # Mask before arithmetic: np.where would still evaluate overflowing
    # right-left differences on intervals spanning both float64 extremes.
    values[nonnegative] = (left[nonnegative]
                          + weight[nonnegative] * (right[nonnegative] - left[nonnegative]))
    values[nonpositive] = (right[nonpositive]
                          - (1.0 - weight[nonpositive]) * (right[nonpositive] - left[nonpositive]))
    values[crossing] = ((1.0 - weight[crossing]) * left[crossing]
                        + weight[crossing] * right[crossing])
    np.copyto(values, left, where=weight == 0)
    np.copyto(values, right, where=weight == 1)
    return values


@dataclass(frozen=True)
class ConditionCopula:
    """Fitted training count, uniform marginal knots, and latent dependence."""

    train_count: int
    quantiles: np.ndarray
    correlation: np.ndarray
    cholesky: np.ndarray


@dataclass(frozen=True)
class ConditionalGaussianCopula:
    """A fitted model that never accesses training or held-out data on generation.

    ``generate`` preserves requested label order and starts its own seeded RNG.
    Identical parameters, labels, seed and batch size replay exactly in the same
    numerical environment. Different batch sizes use the same latent draws but
    may differ by floating-point roundoff in matrix multiplication. Generation
    scratch memory is bounded by batch size; the returned array is materialized.
    """

    condition_dim: int
    n_features: int
    quantile_knots: int | None
    shrinkage: float
    models: dict[int, ConditionCopula]

    @property
    def supported_labels(self) -> np.ndarray:
        return np.array(sorted(self.models), dtype=np.int64)

    @property
    def train_counts(self) -> dict[int, int]:
        return {label: fit.train_count for label, fit in self.models.items()}

    def generate(self, labels, *, seed: int = 0, batch_size: int = 4096) -> np.ndarray:
        """Draw fresh float64 profiles in training units, without tail extrapolation."""
        labels = _labels(labels, self.condition_dim)
        batch_size = _positive_integer(batch_size, "batch_size")
        seed = _positive_integer(seed, "seed", 0)
        unsupported = np.setdiff1d(np.unique(labels), self.supported_labels)
        if len(unsupported):
            raise ValueError(f"unsupported conditions absent from training: {unsupported.tolist()}")
        rng = np.random.default_rng(seed)
        output = np.empty((len(labels), self.n_features), dtype=np.float64)
        for start in range(0, len(labels), batch_size):
            stop = min(start + batch_size, len(labels))
            batch_labels = labels[start:stop]
            noise = rng.standard_normal((stop - start, self.n_features))
            for label in np.unique(batch_labels):
                selected = batch_labels == label
                fit = self.models[int(label)]
                uniform = ndtr(noise[selected] @ fit.cholesky.T)
                output[start:stop][selected] = _inverse_quantiles(uniform, fit.quantiles)
        return output

    def save(self, path: str | Path) -> None:
        """Store fitted numeric parameters in a versioned NPZ without pickle."""
        arrays = {
            "metadata": np.asarray(json.dumps({
                "schema_version": 1, "condition_dim": self.condition_dim,
                "n_features": self.n_features, "quantile_knots": self.quantile_knots,
                "shrinkage": self.shrinkage,
            })),
            "labels": self.supported_labels,
            "counts": np.asarray([self.models[int(k)].train_count for k in self.supported_labels],
                                 dtype=np.int64),
        }
        for label, fit in self.models.items():
            arrays[f"quantiles_{label}"] = fit.quantiles
            arrays[f"correlation_{label}"] = fit.correlation
            arrays[f"cholesky_{label}"] = fit.cholesky
        # Passing a handle avoids NumPy silently appending a filename suffix.
        with Path(path).open("wb") as stream:
            np.savez_compressed(stream, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> ConditionalGaussianCopula:
        """Load a saved fitted model; reject malformed or nonfinite parameters."""
        with np.load(path, allow_pickle=False) as saved:
            try:
                metadata = json.loads(str(saved["metadata"].item()))
                if metadata["schema_version"] != 1:
                    raise ValueError("unsupported copula schema_version")
                condition_dim, knots, shrinkage = _settings(
                    metadata["condition_dim"], metadata["quantile_knots"], metadata["shrinkage"])
                width = _positive_integer(metadata["n_features"], "n_features")
                labels = _labels(saved["labels"], condition_dim)
                counts = saved["counts"]
                if (not len(labels) or len(np.unique(labels)) != len(labels)
                        or counts.shape != labels.shape or counts.dtype.kind not in "iu"
                        or np.any(counts < 1)):
                    raise ValueError("invalid saved labels or training counts")
                models = {}
                for label, count in zip(labels, counts):
                    raw_arrays = tuple(saved[f"{key}_{label}"]
                                       for key in ("quantiles", "correlation", "cholesky"))
                    if any(array.dtype.kind not in "iuf" for array in raw_arrays):
                        raise ValueError("invalid saved copula parameters: expected real numeric arrays")
                    quantiles, correlation, cholesky = (
                        np.array(array, dtype=np.float64, copy=True) for array in raw_arrays)
                    expected_knots = max(2, min(int(count), knots) if knots is not None else int(count))
                    if (quantiles.shape != (expected_knots, width)
                            or correlation.shape != (width, width) or cholesky.shape != (width, width)
                            or not all(np.isfinite(x).all() for x in (quantiles, correlation, cholesky))
                            or np.any(quantiles[1:] < quantiles[:-1])
                            or not np.allclose(correlation, correlation.T, rtol=0, atol=1e-10)
                            or not np.allclose(np.diag(correlation), 1, rtol=0, atol=1e-10)
                            or not np.array_equal(cholesky, np.tril(cholesky))
                            or np.any(np.diag(cholesky) <= 0)
                            or not np.allclose(cholesky @ cholesky.T, correlation, rtol=1e-9, atol=1e-10)):
                        raise ValueError("invalid saved copula parameters")
                    models[int(label)] = ConditionCopula(int(count), quantiles, correlation, cholesky)
            except (KeyError, TypeError, OverflowError) as exc:
                raise ValueError("invalid copula archive") from exc
        return cls(condition_dim, width, knots, shrinkage, models)


def fit_conditional_copula(x_train, labels, *, condition_dim: int = 64,
                           quantile_knots: int | None = 257,
                           shrinkage: float = 0.01) -> ConditionalGaussianCopula:
    """Fit only the supplied training rows, separately for each observed label.

    The caller is responsible for passing the actual selected training subset.
    No file or held-out array is consulted. Defaults are fixed a priori, not
    chosen from validation/test targets. The full joint feature correlation is
    fitted (96 x 96 for quarter-hour profiles); features are not independent.
    Constant features have zero latent cross-correlations and unit variances;
    their empirical inverse remains constant. Singleton conditions are allowed.

    Runtime is dominated by per-column ranking/sorting and a d x d Gram matrix:
    O(sum_c n_c*d*log(n_c) + N*d*d + C*d*d*d). At 64 conditions, 96 features,
    257 knots, stored float64 parameters occupy about 22 MiB. Exact margins
    additionally scale with N*d. Fitting scratch is per condition.
    """
    condition_dim, quantile_knots, shrinkage = _settings(condition_dim, quantile_knots, shrinkage)
    try:
        values = np.asarray(x_train)
    except (TypeError, ValueError) as exc:
        raise ValueError("profiles must be a finite real numeric array") from exc
    if values.ndim != 2 or not values.shape[0] or not values.shape[1]:
        raise ValueError("profiles must be a nonempty rank-2 array")
    if values.dtype.kind not in "iuf" or not np.isfinite(values).all():
        raise ValueError("profiles must contain finite real numeric values")
    labels = _labels(labels, condition_dim, len(values))
    models = {}
    width = values.shape[1]
    for label in np.unique(labels):
        selected = np.asarray(values[labels == label], dtype=np.float64)
        if not np.isfinite(selected).all():
            raise ValueError("profiles cannot be represented as finite float64 values")
        scores = _normal_scores(selected)
        scores -= scores.mean(axis=0)
        scale = np.linalg.norm(scores, axis=0)
        np.divide(scores, scale, out=scores, where=scale > 0)
        correlation = scores.T @ scores
        np.fill_diagonal(correlation, 1.0)
        correlation = (1.0 - shrinkage) * correlation + shrinkage * np.eye(width)
        correlation = _repair_correlation(correlation)
        cholesky = np.linalg.cholesky(correlation)
        sorted_values = np.sort(selected, axis=0)
        count = len(selected)
        n_knots = max(2, min(count, quantile_knots) if quantile_knots is not None else count)
        if count == 1:
            quantiles = np.repeat(sorted_values, 2, axis=0)
        elif n_knots == count:
            quantiles = sorted_values
        else:
            uniform = np.broadcast_to(np.linspace(0.0, 1.0, n_knots)[:, None], (n_knots, width))
            quantiles = _inverse_quantiles(uniform, sorted_values)
        models[int(label)] = ConditionCopula(count, quantiles, correlation, cholesky)
    return ConditionalGaussianCopula(condition_dim, width, quantile_knots, shrinkage, models)
