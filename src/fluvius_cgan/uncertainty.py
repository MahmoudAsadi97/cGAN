"""Paired, condition-macro uncertainty for frozen synthetic profile comparisons.

Only real meters are resampled. A synthetic profile's condition is a distribution
label, not evidence of correspondence to any particular real day or meter.
"""

from __future__ import annotations

import numpy as np

from .evaluation import _acf, _interval_hours, _labels, _matching_width, _profiles


_METRICS = (
    "pointwise_wasserstein_kwh",
    "daily_energy_wasserstein_kwh",
    "ramp_wasserstein_kw",
)
_SUPPORTED_METRICS = (*_METRICS, "peak_power_wasserstein_kw", "lag_one_acf_wasserstein")


def _metric_features(profiles, metric, hours):
    """Return columns defining one W1 diagnostic and rows contributed per day."""
    if metric == "pointwise_wasserstein_kwh":
        return profiles, 1
    if metric == "daily_energy_wasserstein_kwh":
        return profiles.sum(axis=1, keepdims=True), 1
    if metric == "ramp_wasserstein_kw":
        return (np.diff(profiles, axis=1) / hours).reshape(-1, 1), profiles.shape[1] - 1
    if metric == "peak_power_wasserstein_kw":
        return profiles.max(axis=1, keepdims=True) / hours, 1
    if metric == "lag_one_acf_wasserstein":
        return _acf(profiles, 1)[:, None], 1
    raise ValueError(f"unsupported metric: {metric}")


class _FixedReferenceWasserstein:
    """Cache a merged empirical CDF grid; vary only real meter multiplicities.

    Each column is a separate one-dimensional W1, averaged over columns. Repeated
    observations and tied values need no special case: zero-width CDF intervals
    contribute zero. This is exactly the empirical distribution distance, without
    sorting or materializing repeated daily profiles in each bootstrap draw.
    """

    def __init__(self, real, reference, real_meters, n_meters, repeats=1):
        values = np.concatenate((real, reference), axis=0)
        order = np.argsort(values, axis=0, kind="stable")
        sorted_values = np.take_along_axis(values, order, axis=0)
        is_real = order < len(real)
        # A zero-weight sentinel represents each fixed reference observation.
        meter_lookup = np.full(len(values), n_meters, dtype=np.intp)
        meter_lookup[:len(real)] = real_meters
        self.meters = meter_lookup[order]
        self.widths = np.diff(sorted_values, axis=0)
        self.reference_cdf = np.cumsum(~is_real, axis=0)[:-1] / len(reference)
        self.repeats = repeats

    def distance(self, multiplicities, n_real):
        cdf = np.cumsum(multiplicities[self.meters], axis=0)[:-1]
        cdf = cdf / (n_real * self.repeats)
        return float(np.mean(np.sum(np.abs(cdf - self.reference_cdf) * self.widths, axis=0)))


def _meter_groups(real_meter_ids, count):
    ids = np.asarray(real_meter_ids)
    if ids.ndim != 1 or len(ids) != count or ids.dtype.kind not in "iuSU":
        raise ValueError("real_meter_ids must be a rank-1 string/integer array matching real rows")
    if ids.dtype.kind in "SU" and (np.char.str_len(np.char.strip(ids.astype(str))) == 0).any():
        raise ValueError("real_meter_ids cannot contain empty identifiers")
    meters, inverse = np.unique(ids, return_inverse=True)
    if len(meters) < 2:
        raise ValueError("meter bootstrap requires at least two distinct real meters")
    return meters, inverse


def _stratified_meter_groups(real_meter_strata, real_meters, n_meters):
    """Validate row-level strata and return distinct-meter groups to resample."""
    if real_meter_strata is None:
        return [np.arange(n_meters)], {"all": n_meters}
    values = np.asarray(real_meter_strata)
    if values.ndim != 1 or len(values) != len(real_meters) or values.dtype.kind not in "iuSU":
        raise ValueError("real_meter_strata must be a rank-1 string/integer array matching real rows")
    if values.dtype.kind in "SU" and (np.char.str_len(np.char.strip(values.astype(str))) == 0).any():
        raise ValueError("real_meter_strata cannot contain empty strata")
    strata, row_strata = np.unique(values, return_inverse=True)
    meter_strata = np.empty(n_meters, dtype=np.int64)
    for meter in range(n_meters):
        observed = np.unique(row_strata[real_meters == meter])
        if len(observed) != 1:
            raise ValueError("each real meter must belong to exactly one stratum")
        meter_strata[meter] = observed[0]
    groups = [np.flatnonzero(meter_strata == stratum) for stratum in range(len(strata))]
    counts = {str(stratum): len(group) for stratum, group in zip(strata, groups)}
    if any(count < 2 for count in counts.values()):
        raise ValueError("stratified meter bootstrap requires at least two distinct meters per stratum")
    return groups, counts


def paired_condition_macro_meter_bootstrap(
    real,
    real_labels,
    generated,
    generated_labels,
    comparator,
    comparator_labels,
    real_meter_ids,
    interval_hours: float = .25,
    *,
    n_bootstrap: int = 200,
    confidence: float = .95,
    seed: int = 42,
    real_meter_strata=None,
    metrics=None,
    return_per_condition: bool = False,
) -> dict:
    """Compare fixed generated and comparator samples with paired percentile CIs.

    The estimand is the equally weighted mean over the original observed condition
    support of W1(real, generated) - W1(real, comparator). Negative differences
    favor generated. Metric definitions and units match ``evaluate_conditional``:
    interval W1 averages across slots (kWh), daily-energy W1 compares row sums
    (kWh), and ramp W1 pools signed within-row power changes (kW). At least two
    intervals are required. No clipping or between-day ramp construction occurs.

    Each replicate samples the original number of distinct real meters uniformly
    with replacement. Every selected meter contributes all its real days; repeated
    selections repeat the whole meter. The identical real draw feeds both methods
    and every condition. Within each condition, all fixed generated/comparator rows
    remain available with their original empirical weights. Their row counts can
    differ from one another and from real data. There is no synthetic-to-real day
    pairing, synthetic meter assignment, or separate resampling by condition.

    Optional ``real_meter_strata`` assigns one string/integer stratum to each real
    row, with one consistent stratum per meter and at least two distinct meters
    per stratum. The original meter count is then sampled with replacement within
    each stratum, retaining the sample's stratum allocation. Without strata, meters
    are sampled globally. Meter multiplicities remain paired across both methods.

    Condition supports must match exactly. A draw missing any original condition
    is undefined, never averaged over fewer conditions. All draws are attempted
    and missing-condition counts are reported. If even one draw is undefined,
    all CI bounds are None and ``interval_status`` explains why. This fail-closed
    policy avoids silently conditioning intervals on draws with complete support.
    Point estimates remain available. A new explicitly justified sampling design
    or estimand is needed when rare conditions make this bootstrap unsuitable.

    These are descriptive marginal percentile intervals, not simultaneous or
    bias-corrected intervals. Their nominal coverage is not guaranteed: in
    particular W1 is nonsmooth at coincident CDFs, where ordinary bootstrap
    inference can be nonregular. They do not establish statistical significance.
    They assume meters are independent sampling units (within each specified
    stratum); all within-meter dependence is retained. They describe real-meter sampling only,
    conditional on these frozen models, generated arrays, comparator arrays, and
    observed condition support. They exclude training-seed, synthetic Monte Carlo,
    preprocessing-fit, model-selection, and future-period uncertainty. More draws
    improve quantile precision, but do not cover these excluded uncertainties.

    ``metrics=None`` preserves the original interval/daily/ramp diagnostics.
    An explicit nonempty unique sequence may also request peak-power W1 and
    ``lag_one_acf_wasserstein``. ACF follows evaluation.py's full-row biased ACF,
    with ACF=0 for constant profiles. ``return_per_condition=True`` adds marginal
    condition intervals computed from the same globally paired meter draws, not
    separate condition-specific resampling. All intervals remain withheld if any
    global draw lacks an original condition; these are not simultaneous intervals.
    """
    if not isinstance(return_per_condition, (bool, np.bool_)):
        raise ValueError("return_per_condition must be boolean")
    if metrics is None:
        metrics = _METRICS
    if (not isinstance(metrics, (tuple, list)) or not metrics
            or any(not isinstance(metric, str) or metric not in _SUPPORTED_METRICS for metric in metrics)
            or len(set(metrics)) != len(metrics)):
        raise ValueError("metrics must be a nonempty sequence of unique supported metric names")
    metrics = tuple(metrics)
    real = _profiles(real, "real")
    generated = _profiles(generated, "generated")
    comparator = _profiles(comparator, "comparator")
    _matching_width(real, generated, comparator)
    if real.shape[1] < 2:
        raise ValueError("at least two intervals are required for ramp Wasserstein")
    real_labels = _labels(real_labels, "real_labels", len(real))
    generated_labels = _labels(generated_labels, "generated_labels", len(generated))
    comparator_labels = _labels(comparator_labels, "comparator_labels", len(comparator))
    hours = _interval_hours(interval_hours)
    support = np.unique(real_labels)
    for name, labels in (("generated", generated_labels), ("comparator", comparator_labels)):
        other = np.unique(labels)
        if not np.array_equal(support, other):
            raise ValueError(
                f"condition support differs for {name}; "
                f"missing {name} conditions: {np.setdiff1d(support, other).tolist()}; "
                f"extra {name} conditions: {np.setdiff1d(other, support).tolist()}"
            )
    meters, real_meters = _meter_groups(real_meter_ids, len(real))
    n_meters = len(meters)
    sampling_groups, stratum_counts = _stratified_meter_groups(real_meter_strata, real_meters, n_meters)
    if (isinstance(n_bootstrap, (bool, np.bool_))
            or not isinstance(n_bootstrap, (int, np.integer)) or n_bootstrap < 2):
        raise ValueError("n_bootstrap must be an integer of at least two")
    if isinstance(confidence, (bool, np.bool_)):
        raise ValueError("confidence must be strictly between zero and one")
    try:
        confidence = float(confidence)
    except (TypeError, ValueError) as exc:
        raise ValueError("confidence must be strictly between zero and one") from exc
    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be strictly between zero and one")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")

    counts = {"real": {}, "generated": {}, "comparator": {}}
    condition_meter_counts = []
    grids = []
    for condition in support:
        mask = real_labels == condition
        real_part, part_meters = real[mask], real_meters[mask]
        condition_meter_counts.append(np.bincount(part_meters, minlength=n_meters))
        counts["real"][str(condition)] = len(real_part)
        real_features = [_metric_features(real_part, metric, hours) for metric in metrics]
        by_method = []
        for name, profiles, labels in (
            ("generated", generated, generated_labels),
            ("comparator", comparator, comparator_labels),
        ):
            part = profiles[labels == condition]
            counts[name][str(condition)] = len(part)
            features = [_metric_features(part, metric, hours)[0] for metric in metrics]
            by_method.append([
                _FixedReferenceWasserstein(
                    x, y,
                    part_meters if repeats == 1 else np.repeat(part_meters, repeats),
                    n_meters,
                    repeats=repeats,
                )
                for (x, repeats), y in zip(real_features, features)
            ])
        grids.append(by_method)
    condition_meter_counts = np.asarray(condition_meter_counts)

    def condition_values(multiplicities, real_counts):
        # The final sentinel has zero multiplicity for every fixed reference row.
        weights = np.append(multiplicities, 0)
        values = np.empty((len(support), 2, len(metrics)))
        for condition, (n_real, methods) in enumerate(zip(real_counts, grids)):
            for method, method_metrics in enumerate(methods):
                for metric, grid in enumerate(method_metrics):
                    values[condition, method, metric] = grid.distance(weights, n_real)
        return values

    condition_estimate = condition_values(np.ones(n_meters, dtype=np.int64), condition_meter_counts.sum(axis=1))
    estimate = condition_estimate.mean(axis=0)
    rng = np.random.default_rng(seed)
    draws = []
    condition_draws = []
    missing_counts = np.zeros(len(support), dtype=np.int64)
    n_invalid = 0
    for _ in range(n_bootstrap):
        selected = np.concatenate([
            group[rng.integers(0, len(group), size=len(group))] for group in sampling_groups
        ])
        multiplicities = np.bincount(selected, minlength=n_meters)
        real_counts = condition_meter_counts @ multiplicities
        missing = real_counts == 0
        if missing.any():
            missing_counts += missing
            n_invalid += 1
            continue
        values = condition_values(multiplicities, real_counts)
        macro = values.mean(axis=0)
        draws.append(macro[0] - macro[1])
        if return_per_condition:
            condition_draws.append(values[:, 0] - values[:, 1])
    alpha = (1 - confidence) / 2
    bounds = None if n_invalid else np.quantile(draws, [alpha, 1 - alpha], axis=0)
    result = {
        "method": "paired_percentile_cluster_bootstrap",
        "resampling_unit": "real_meter",
        "resampling_scheme": "global_real_meter" if real_meter_strata is None else "within_stratum_real_meter",
        "stratum_meter_counts": stratum_counts,
        "coverage_interpretation": (
            "Descriptive marginal percentile intervals; nominal coverage is not guaranteed, "
            "particularly near coincident CDFs where W1 is nonsmooth. Not a significance claim."
        ),
        "condition_weighting": "equal_over_original_observed_support",
        "generated_sample_policy": "fixed_condition_reference",
        "comparator_sample_policy": "fixed_condition_reference",
        "comparison": "generated_minus_comparator",
        "negative_difference_favors": "generated",
        "uncertainty_scope": (
            "real-meter sampling conditional on frozen models and reference arrays; "
            "excludes training-seed, generated-sample, comparator-sample, preprocessing-fit, "
            "model-selection, and future-period uncertainty"
        ),
        "interval_status": "unavailable_missing_condition_draws" if n_invalid else "complete",
        "missing_condition_policy": "withhold_all_intervals_if_any_draw_lacks_original_support",
        "missing_condition_draw_counts": {
            str(condition): int(count) for condition, count in zip(support, missing_counts)
        },
        "n_meters": n_meters,
        "n_conditions": len(support),
        "condition_support": support.tolist(),
        "counts": counts,
        "interval_hours": hours,
        "n_bootstrap": int(n_bootstrap),
        "n_valid_bootstrap": len(draws),
        "n_invalid_bootstrap": n_invalid,
        "confidence": confidence,
        "seed": int(seed),
        "point_estimates": {
            name: {metric: float(estimate[i, j]) for j, metric in enumerate(metrics)}
            for i, name in enumerate(("generated", "comparator"))
        },
        "intervals": {
            metric: {
                "estimate": float(estimate[0, i] - estimate[1, i]),
                "lower": None if bounds is None else float(bounds[0, i]),
                "upper": None if bounds is None else float(bounds[1, i]),
            }
            for i, metric in enumerate(metrics)
        },
    }
    if return_per_condition:
        condition_bounds = None if n_invalid else np.quantile(condition_draws, [alpha, 1 - alpha], axis=0)
        result["per_condition"] = {
            str(condition): {
                "interval_status": result["interval_status"],
                "point_estimates": {
                    name: {metric: float(condition_estimate[c, i, j]) for j, metric in enumerate(metrics)}
                    for i, name in enumerate(("generated", "comparator"))
                },
                "intervals": {
                    metric: {
                        "estimate": float(condition_estimate[c, 0, j] - condition_estimate[c, 1, j]),
                        "lower": None if condition_bounds is None else float(condition_bounds[0, c, j]),
                        "upper": None if condition_bounds is None else float(condition_bounds[1, c, j]),
                    }
                    for j, metric in enumerate(metrics)
                },
            }
            for c, condition in enumerate(support)
        }
    return result
