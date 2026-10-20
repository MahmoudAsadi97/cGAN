"""Paired real-meter uncertainty for frozen observed-history forecasting errors.

Inputs are prediction-minus-observation errors on identical real rows/windows.
Nothing here fits predictors, generates profiles, selects data, or opens files.
"""
from __future__ import annotations

import numpy as np

from .uncertainty import _meter_groups, _stratified_meter_groups


def paired_utility_meter_bootstrap(
    candidate_errors, comparator_errors, labels, meter_ids, *,
    n_bootstrap=500, confidence=.95, seed=42027,
):
    """Bootstrap candidate-minus-comparator macro RMSE and MAE in kWh.

    Resample the original number of meters with replacement within technology
    strata (condition // 8). One meter draw applies to both methods and every
    condition; every selected meter contributes all its selected days/windows.
    Compute RMSE within each condition, then average conditions equally. This
    differs from taking the square root of condition-macro MSE. The estimand
    and original condition support are fixed throughout the bootstrap.

    Fitted predictors, feature construction, training data and real evaluation
    rows are frozen. Percentile intervals describe real-meter sampling only;
    they exclude fitting, generation, training-seed, calibration, selection and
    future-period uncertainty. They assume independent meters within technology
    strata, which trace deduplication alone does not establish. Intervals are
    descriptive, not simultaneous or a formal noninferiority test. If any draw
    omits an observed condition, all interval bounds are withheld rather than
    silently changing the estimand or rejecting that draw.
    """
    errors = []
    for name, values in (("candidate_errors", candidate_errors),
                         ("comparator_errors", comparator_errors)):
        values = np.asarray(values)
        if (values.ndim != 2 or not all(values.shape) or
                values.dtype.kind not in "fiu" or not np.isfinite(values).all()):
            raise ValueError(f"{name} must be a nonempty finite numeric 2D array")
        errors.append(values.astype(np.float64, copy=False))
    if errors[0].shape != errors[1].shape:
        raise ValueError("methods must share identical rows and prediction-window counts")
    rows, windows = errors[0].shape
    labels = np.asarray(labels)
    if (labels.shape != (rows,) or labels.dtype.kind not in "iu" or
            np.any(labels < 0) or np.any(labels >= 64)):
        raise ValueError("labels must match rows and contain integer condition IDs in [0, 63]")
    if (isinstance(n_bootstrap, (bool, np.bool_)) or
            not isinstance(n_bootstrap, (int, np.integer)) or n_bootstrap < 2):
        raise ValueError("n_bootstrap must be an integer of at least two")
    if (isinstance(seed, (bool, np.bool_)) or
            not isinstance(seed, (int, np.integer)) or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    if isinstance(confidence, (bool, np.bool_)):
        raise ValueError("confidence must be strictly between zero and one")
    try:
        confidence = float(confidence)
    except (ValueError, TypeError) as exc:
        raise ValueError("confidence must be strictly between zero and one") from exc
    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be strictly between zero and one")

    meters, inverse = _meter_groups(meter_ids, rows)
    groups, stratum_counts = _stratified_meter_groups(labels // 8, inverse, len(meters))
    support = np.unique(labels)
    # Sum error statistics by condition and meter; retain whole-meter weighting
    # without copying repeated profiles in each replicate.
    counts = np.zeros((len(support), len(meters)), dtype=np.int64)
    totals = np.zeros((2, 2, len(support), len(meters)), dtype=np.float64)
    try:
        with np.errstate(over="raise", invalid="raise"):
            for ci, label in enumerate(support):
                mask = labels == label
                counts[ci] = np.bincount(inverse[mask], minlength=len(meters))
                for method, values in enumerate(errors):
                    block = values[mask]
                    for metric, statistics in enumerate((np.square(block), np.abs(block))):
                        totals[method, metric, ci] = np.bincount(
                            inverse[mask], weights=statistics.sum(axis=1), minlength=len(meters))
    except FloatingPointError as exc:
        raise ValueError("nonfinite utility error statistics; check error scale") from exc
    if not np.isfinite(totals).all():
        raise ValueError("nonfinite utility error statistics; check error scale")

    def score(multiplicities):
        real_counts = counts @ multiplicities
        missing = real_counts == 0
        if missing.any():
            return None, missing
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                values = (totals @ multiplicities) / (real_counts * windows)
                values[:, 0] = np.sqrt(values[:, 0])
        except FloatingPointError as exc:
            raise ValueError("nonfinite utility error statistics in a meter draw; check error scale") from exc
        if not np.isfinite(values).all():
            raise ValueError("nonfinite utility error statistics in a meter draw; check error scale")
        return values, missing

    condition_points, _ = score(np.ones(len(meters), dtype=np.int64))
    points = condition_points.mean(axis=-1)
    rng = np.random.default_rng(seed)
    draws, missing_counts = [], np.zeros(len(support), dtype=np.int64)
    for _ in range(n_bootstrap):
        selected = np.concatenate([group[rng.integers(0, len(group), size=len(group))]
                                   for group in groups])
        values, missing = score(np.bincount(selected, minlength=len(meters)))
        missing_counts += missing
        if values is not None:
            macro = values.mean(axis=-1)
            draws.append(macro[0] - macro[1])
    invalid = int(n_bootstrap - len(draws))
    bounds = (np.quantile(draws, [(1 - confidence) / 2, (1 + confidence) / 2], axis=0)
              if invalid == 0 else None)
    metrics = ("rmse_kwh", "mae_kwh")
    return {
        "schema_version": 1,
        "estimand": "candidate_minus_comparator_equal_condition_mean_forecasting_error",
        "condition_weighting": "equal_over_original_observed_support",
        "resampling_scheme": "within_technology_real_meter",
        "confidence": confidence, "seed": int(seed), "n_bootstrap": int(n_bootstrap),
        "n_valid_bootstrap": len(draws), "n_invalid_bootstrap": invalid,
        "interval_status": "complete" if invalid == 0 else "withheld_missing_conditions",
        "stratum_meter_counts": stratum_counts,
        "n_meters": len(meters), "n_profiles": rows, "windows_per_profile": windows,
        "condition_counts": {str(label): int(counts[ci].sum()) for ci, label in enumerate(support)},
        "missing_condition_draws": {str(label): int(missing_counts[ci]) for ci, label in enumerate(support)},
        "point_estimates": {
            method: {metric: float(points[mi, ki]) for ki, metric in enumerate(metrics)}
            for mi, method in enumerate(("candidate", "comparator"))},
        "per_condition_point_estimates": {
            str(label): {method: {metric: float(condition_points[mi, ki, ci])
                                 for ki, metric in enumerate(metrics)}
                         for mi, method in enumerate(("candidate", "comparator"))}
            for ci, label in enumerate(support)},
        "intervals": {metric: {
            "estimate": float(points[0, ki] - points[1, ki]),
            "lower": None if bounds is None else float(bounds[0, ki]),
            "upper": None if bounds is None else float(bounds[1, ki]),
        } for ki, metric in enumerate(metrics)},
        "interpretation": (
            "Descriptive paired percentile intervals for real-meter sampling, conditional on frozen "
            "predictors, evaluation windows and observed support. Negative differences favor candidate. "
            "Independent meters within technology strata are assumed, not established by trace deduplication. "
            "Nominal coverage is not guaranteed; intervals are not simultaneous, significance tests or formal "
            "noninferiority tests. Fitting, synthetic generation, training-seed, calibration, model-selection "
            "and future-period uncertainty are excluded."),
    }
