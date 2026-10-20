"""Fixed linear next-interval utility diagnostic for daily electricity profiles.

This is train-on-synthetic/test-on-real (TSTR) within-day autoregression, not
day-ahead forecasting, future-year validation, or evidence of utility for every
application. At each held-out target interval, the predictor receives the actual
observed preceding intervals from that same held-out day. It does not recursively
feed predictions back as observations. Negative signed net energy is preserved.

The fixed model uses chronological lagged kWh, target-clock sine/cosine features
at harmonics one and two, 64 condition indicators, and an intercept. Ridge alpha
is one in the SUM-of-squared-errors objective, penalizing every coefficient except
the intercept. There is no fitted feature scaling or hyperparameter selection.
Sufficient statistics and evaluation errors are accumulated in profile batches;
no all-dataset window/design matrix is materialized. No files are read here.
"""
from __future__ import annotations

from numbers import Integral

import numpy as np


RIDGE_ALPHA = 1.0
CONDITION_COUNT = 64


def _positive_int(name, value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _profiles(values, name, batch_profiles):
    values = np.asarray(values)
    if (values.ndim != 2 or not values.shape[0] or values.shape[1] not in (24, 96) or
            values.dtype.kind not in "fiu"):
        raise ValueError(f"{name} must be nonempty numeric profiles of shape (N, 24) or (N, 96)")
    # Validate in bounded blocks and retain the source dtype until feature creation.
    for start in range(0, len(values), batch_profiles):
        if not np.isfinite(values[start:start + batch_profiles]).all():
            raise ValueError(f"{name} must contain finite values")
    return values


def _labels(values, count, name="labels"):
    values = np.asarray(values)
    if (values.shape != (count,) or values.dtype.kind not in "iu" or
            np.any(values < 0) or np.any(values >= CONDITION_COUNT)):
        raise ValueError(f"{name} must contain {count} integer condition IDs in [0, 63]")
    return values.astype(np.int64, copy=False)


def _feature_batch(profiles, labels, lags):
    """Build one bounded block in profile-major, then target-slot order.

    The first target in each row is slot ``lags``. Feature columns are the lags
    oldest-to-newest, sin1/cos1/sin2/cos2 at the TARGET slot's known clock time,
    condition0..condition63, and intercept. Only within-row windows are formed.
    """
    profiles = np.asarray(profiles, dtype=np.float64)
    resolution = profiles.shape[1]
    windows = np.lib.stride_tricks.sliding_window_view(profiles, lags + 1, axis=1)
    per_profile = resolution - lags
    count = len(profiles) * per_profile
    features = np.zeros((count, lags + 4 + CONDITION_COUNT + 1), dtype=np.float64)
    features[:, :lags] = windows[:, :, :-1].reshape(count, lags)
    phase = 2 * np.pi * np.arange(lags, resolution, dtype=np.float64) / resolution
    clock = np.column_stack((np.sin(phase), np.cos(phase),
                             np.sin(2 * phase), np.cos(2 * phase)))
    features[:, lags:lags + 4] = np.tile(clock, (len(profiles), 1))
    feature_labels = np.repeat(labels, per_profile)
    features[np.arange(count), lags + 4 + feature_labels] = 1.0
    features[:, -1] = 1.0
    targets = windows[:, :, -1].reshape(count)
    return features, targets


def _feature_names(lags):
    return ([f"lag_{lag}_intervals" for lag in range(lags, 0, -1)] +
            ["clock_sin_1", "clock_cos_1", "clock_sin_2", "clock_cos_2"] +
            [f"condition_{label}" for label in range(CONDITION_COUNT)] + ["intercept"])


def _error_report(squared, absolute, profile_counts, windows_per_profile):
    support = np.flatnonzero(profile_counts)
    per_condition = {}
    for label in support:
        windows = int(profile_counts[label]) * windows_per_profile
        per_condition[str(label)] = {
            "rmse_kwh": float(np.sqrt(squared[label] / windows)),
            "mae_kwh": float(absolute[label] / windows),
            "n_profiles": int(profile_counts[label]),
            "n_windows": windows,
        }
    total_windows = int(profile_counts.sum()) * windows_per_profile
    return {
        "overall": {
            "rmse_kwh": float(np.sqrt(squared.sum() / total_windows)),
            "mae_kwh": float(absolute.sum() / total_windows),
            "n_profiles": int(profile_counts.sum()),
            "n_windows": total_windows,
        },
        "macro": {
            name: float(np.mean([row[name] for row in per_condition.values()]))
            for name in ("rmse_kwh", "mae_kwh")
        },
        "per_condition": per_condition,
        "n_conditions": len(per_condition),
    }


class RidgeProfileForecaster:
    """Fixed ridge AR diagnostic; ``fit`` uses only its two supplied arrays.

    ``lags=None`` chooses three hours of context: 12 intervals at quarter-hourly
    resolution96, or3 at hourly resolution24. ``batch_profiles`` affects memory
    and floating-point accumulation order, not the statistical objective.
    """

    def __init__(self, lags: int | None = None, batch_profiles: int = 256):
        self.lags = None if lags is None else _positive_int("lags", lags)
        self.batch_profiles = _positive_int("batch_profiles", batch_profiles)
        self.coef_ = None

    def fit(self, x, y) -> "RidgeProfileForecaster":
        """Fit sum-of-squares ridge regression from within-day training windows.

        ``y`` contains profile condition IDs, not next-interval target values.
        Targets are obtained exclusively from ``x``. The fitted state is replaced
        only after input validation and a successful finite linear solve.
        """
        profiles = _profiles(x, "x", self.batch_profiles)
        labels = _labels(y, len(profiles))
        resolution = profiles.shape[1]
        lags = self.lags if self.lags is not None else resolution // 8
        if lags >= resolution:
            raise ValueError("lags must be smaller than the profile resolution")
        width = lags + 4 + CONDITION_COUNT + 1
        xtx = np.zeros((width, width), dtype=np.float64)
        xty = np.zeros(width, dtype=np.float64)
        try:
            with np.errstate(over="raise", invalid="raise"):
                for start in range(0, len(profiles), self.batch_profiles):
                    stop = start + self.batch_profiles
                    features, targets = _feature_batch(profiles[start:stop], labels[start:stop], lags)
                    xtx += features.T @ features
                    xty += features.T @ targets
                # Penalize all lag, clock and indicator coefficients; intercept is last.
                xtx.flat[::width + 1] += RIDGE_ALPHA
                xtx[-1, -1] -= RIDGE_ALPHA
                if not np.isfinite(xtx).all() or not np.isfinite(xty).all():
                    raise ValueError("nonfinite ridge sufficient statistics; check input scale")
                coefficients = np.linalg.solve(xtx, xty)
        except (FloatingPointError, np.linalg.LinAlgError) as exc:
            raise ValueError("ridge fitting failed; check input scale and conditioning") from exc
        if not np.isfinite(coefficients).all():
            raise ValueError("ridge fitting produced nonfinite coefficients")
        support, counts = np.unique(labels, return_counts=True)
        self.coef_ = coefficients
        self.resolution_ = resolution
        self.lags_ = lags
        self.condition_support_ = support
        self.feature_names_ = _feature_names(lags)
        self.fit_summary_ = {
            "n_profiles": len(profiles),
            "n_windows": len(profiles) * (resolution - lags),
            "n_features": width,
            "resolution": resolution,
            "lags": lags,
            "ridge_alpha": RIDGE_ALPHA,
            "ridge_objective": "sum_squared_error_plus_alpha_nonintercept_squared_norm",
            "feature_scaling": "none; lagged energy remains in native kWh/interval",
            "condition_counts": {str(label): int(count) for label, count in zip(support, counts)},
        }
        return self

    def _evaluation_inputs(self, x, y):
        if self.coef_ is None:
            raise ValueError("forecaster must be fitted before prediction or evaluation")
        profiles = _profiles(x, "x", self.batch_profiles)
        labels = _labels(y, len(profiles))
        if profiles.shape[1] != self.resolution_:
            raise ValueError("evaluation resolution must match the fitted profiles")
        unsupported = np.setdiff1d(np.unique(labels), self.condition_support_)
        if len(unsupported):
            raise ValueError(f"unsupported evaluation conditions absent from fitting: {unsupported.tolist()}")
        return profiles, labels

    def predict(self, x, y) -> np.ndarray:
        """Predict slots ``lags_..resolution_-1`` using observed within-day history.

        Returns shape (N, resolution-lags), in signed kWh/interval. No target's
        own value or later interval enters its prediction. Previous target values
        are valid observed inputs for subsequent one-step predictions.
        """
        profiles, labels = self._evaluation_inputs(x, y)
        result = np.empty((len(profiles), self.resolution_ - self.lags_), dtype=np.float64)
        for start in range(0, len(profiles), self.batch_profiles):
            stop = min(start + self.batch_profiles, len(profiles))
            features, _ = _feature_batch(profiles[start:stop], labels[start:stop], self.lags_)
            result[start:stop] = (features @ self.coef_).reshape(stop - start, -1)
        if not np.isfinite(result).all():
            raise ValueError("ridge prediction produced nonfinite values")
        return result

    def evaluate(self, x, y) -> dict:
        """Report pooled and condition-macro errors, plus same-window persistence.

        Macro RMSE is the arithmetic mean of each condition's RMSE, not the
        square root of the average condition MSE. Every observed condition is
        included. Counts expose the number of profiles and within-day windows.
        Calling this method never refits or changes the fitted coefficients.
        """
        profiles, labels = self._evaluation_inputs(x, y)
        squared = np.zeros(CONDITION_COUNT, dtype=np.float64)
        absolute = np.zeros_like(squared)
        persistence_squared = np.zeros_like(squared)
        persistence_absolute = np.zeros_like(squared)
        per_profile = self.resolution_ - self.lags_
        try:
            with np.errstate(over="raise", invalid="raise"):
                for start in range(0, len(profiles), self.batch_profiles):
                    stop = start + self.batch_profiles
                    block = np.asarray(profiles[start:stop], dtype=np.float64)
                    block_labels = labels[start:stop]
                    features, targets = _feature_batch(block, block_labels, self.lags_)
                    errors = (features @ self.coef_ - targets).reshape(len(block), per_profile)
                    persistence_errors = block[:, self.lags_ - 1:-1] - block[:, self.lags_:]
                    for sums, values in ((squared, np.square(errors)),
                                         (absolute, np.abs(errors)),
                                         (persistence_squared, persistence_errors ** 2),
                                         (persistence_absolute, np.abs(persistence_errors))):
                        sums += np.bincount(block_labels, weights=values.sum(axis=1), minlength=CONDITION_COUNT)
        except FloatingPointError as exc:
            raise ValueError("nonfinite forecasting errors; check input scale") from exc
        if not all(np.isfinite(values).all() for values in
                   (squared, absolute, persistence_squared, persistence_absolute)):
            raise ValueError("nonfinite forecasting errors; check input scale")
        counts = np.bincount(labels, minlength=CONDITION_COUNT)
        result = _error_report(squared, absolute, counts, per_profile)
        result["persistence"] = _error_report(persistence_squared, persistence_absolute, counts, per_profile)
        return result


def utility_comparison(real_train, train_labels, synthetic, synthetic_labels,
                       heldout, heldout_labels, *, lags: int | None = None,
                       batch_profiles: int = 256) -> dict:
    """Compare real-trained and synthetic-trained fixed ridge on identical holdout.

    The caller owns the train/held-out meter split and synthetic provenance. This
    array-only diagnostic cannot verify household identity, test sealing, or that
    synthetic data were generated solely from a training-fitted generator. It
    never uses held-out data to fit, scale, tune or choose either model.
    """
    real_model = RidgeProfileForecaster(lags=lags, batch_profiles=batch_profiles).fit(real_train, train_labels)
    synthetic_model = RidgeProfileForecaster(lags=lags, batch_profiles=batch_profiles).fit(synthetic, synthetic_labels)
    if real_model.resolution_ != synthetic_model.resolution_:
        raise ValueError("real and synthetic training resolutions must agree")
    real_report = real_model.evaluate(heldout, heldout_labels)
    synthetic_report = synthetic_model.evaluate(heldout, heldout_labels)
    persistence = real_report.pop("persistence")
    synthetic_report.pop("persistence")
    return {
        "task": "within_day_observed_history_next_interval_linear_ridge",
        "interpretation": (
            "TSTR temporal utility diagnostic with observed real held-out lag inputs; "
            "not recursive, day-ahead, future-year, or general application validation"
        ),
        "units": "kWh/interval",
        "resolution": real_model.resolution_,
        "forecast_horizon_intervals": 1,
        "forecast_horizon_hours": 24 / real_model.resolution_,
        "lags": real_model.lags_,
        "target_slots_inclusive": [real_model.lags_, real_model.resolution_ - 1],
        "training_condition_counts_match": (
            real_model.fit_summary_["condition_counts"] == synthetic_model.fit_summary_["condition_counts"]
        ),
        "feature_names": real_model.feature_names_,
        "training": {"real": real_model.fit_summary_, "synthetic": synthetic_model.fit_summary_},
        "methods": {"train_real": real_report, "train_synthetic": synthetic_report,
                    "persistence": persistence},
    }
