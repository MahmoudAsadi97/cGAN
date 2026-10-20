"""Paired fixed-draw inference and conditional training-proximity diagnostics.

Profiles contain signed physical interval energy in kWh. Meter-cluster intervals
condition on fitted models, generation draws and the observed 2024 calendar;
they do not include training-seed or synthetic-sampling variability.
"""

from __future__ import annotations

import hashlib

import numpy as np
from scipy.spatial import cKDTree

from .evaluation import _condition_labels, _profiles, projection_directions
from .uncertainty import _metadata


_METRICS = ('daily_energy_w1_kwh', 'profile_sw1_kwh')


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return int(value)


def _directions(values, width):
    if np.iscomplexobj(values):
        raise ValueError('directions must be finite real unit vectors')
    try:
        values = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError('directions must be finite unit vectors') from exc
    if (values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != width
            or not np.isfinite(values).all()
            or not np.allclose(np.linalg.norm(values, axis=1), 1., rtol=1e-7, atol=1e-10)):
        raise ValueError('directions must be nonempty unit vectors matching profile width')
    return values


class _PairedW1Cache:
    """Exact weighted empirical W1 to two fixed, unweighted samples.

    Sort and merge once, then evaluate integrals of absolute CDF differences.
    CDF indices retain repeated observations and support unequal sample sizes.
    Only a small block of bootstrap weights is materialized at evaluation time.
    """

    def __init__(self, real, fake_a, fake_b):
        values = [np.asarray(v, dtype=np.float64) for v in (real, fake_a, fake_b)]
        if any(v.ndim != 1 or not len(v) or not np.isfinite(v).all() for v in values):
            raise ValueError('W1 supports must be nonempty finite vectors')
        real, fake_a, fake_b = values
        self.real_order = np.argsort(real, kind='stable')
        ordered_real = real[self.real_order]
        support = np.unique(np.concatenate(values))
        self.widths = np.diff(support)
        if not np.isfinite(self.widths).all():
            raise ValueError('profile magnitudes exceed finite metric arithmetic')
        left = support[:-1]
        # Prefix array has a leading zero; index 0 means no mass at this point.
        self.real_cdf_indices = np.searchsorted(ordered_real, left, side='right')
        self.a_cdf = np.searchsorted(np.sort(fake_a), left, side='right') / len(fake_a)
        self.b_cdf = np.searchsorted(np.sort(fake_b), left, side='right') / len(fake_b)

    def distances(self, weights):
        weights = np.asarray(weights, dtype=np.float64)
        if (weights.ndim != 2 or weights.shape[1] != len(self.real_order)
                or not np.isfinite(weights).all() or np.any(weights < 0)):
            raise ValueError('weights must be finite nonnegative rows matching real support')
        prefix = np.empty((len(weights), weights.shape[1] + 1), dtype=np.float64)
        prefix[:, 0] = 0.
        np.cumsum(weights[:, self.real_order], axis=1, out=prefix[:, 1:])
        if np.any(prefix[:, -1] <= 0) or not np.isfinite(prefix).all():
            raise ValueError('each weight row must have finite positive total mass')
        cdf = prefix[:, self.real_cdf_indices] / prefix[:, -1:]
        a = np.einsum('ij,j->i', np.abs(cdf - self.a_cdf), self.widths)
        b = np.einsum('ij,j->i', np.abs(cdf - self.b_cdf), self.widths)
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('profile magnitudes exceed finite metric arithmetic')
        return a, b


def _cluster_multiplicities(meter_ids, technology, n_rows, n_bootstrap, rng):
    """Draw meter multiplicities using the uncertainty primitive's RNG order."""
    meters = _metadata(meter_ids, 'meter_ids')
    technology = _metadata(technology, 'technology')
    if len(meters) != n_rows or len(technology) != n_rows:
        raise ValueError('meter_ids and technology must have one entry per real profile')
    identities, strata, meter_strata = {}, {}, {}
    row_meter = np.empty(n_rows, dtype=np.int64)
    for i, (meter, stratum) in enumerate(zip(meters, technology)):
        if meter not in identities:
            identities[meter] = len(identities)
            meter_strata[meter] = stratum
            strata.setdefault(stratum, []).append(identities[meter])
        elif meter_strata[meter] != stratum:
            raise ValueError('each meter must have one stable technology')
        row_meter[i] = identities[meter]
    weights = np.zeros((n_bootstrap + 1, len(identities)), dtype=np.float64)
    weights[0] = 1.  # original empirical sample, not a bootstrap draw
    for replicate in range(1, n_bootstrap + 1):
        for members in strata.values():
            draws = rng.integers(0, len(members), size=len(members))
            weights[replicate, members] = np.bincount(draws, minlength=len(members))
    return row_meter, weights


def paired_cluster_comparison(real, fake_a, fake_b, labels, meter_ids, technology,
                              *, directions=None, n_bootstrap=499, seed=20261020):
    """Compare condition-macro physical W1 errors, A minus B (positive=worse).

    All three arrays must have identical shape: generated rows share the supplied
    condition labels with real rows, giving equal generated counts per condition.
    There is no day-to-day pairing of generated profiles. Synthetic samples stay
    fixed while whole real meter histories are resampled within technology.

    Return percentile 95% intervals over the same bootstrap draws for both
    methods. If a draw loses any original condition, withhold all intervals and
    report the count: discarding unsupported draws would condition the bootstrap
    on an extra event. Observed condition-macro differences remain available.

    Projections and ordered support/CDF maps are built once per condition and
    direction, then reused for all replicates. Auxiliary memory is linear in
    sample count and bootstrap-count times meter-count, never sample-count squared.
    """
    real, fake_a = _profiles(real, fake_a)
    _, fake_b = _profiles(real, fake_b)
    if real.shape != fake_a.shape or real.shape != fake_b.shape:
        raise ValueError('real, fake_a and fake_b must have identical aligned shapes')
    keys, identities = _condition_labels(labels, len(real), 'labels')
    n_bootstrap = _positive_integer(n_bootstrap, 'n_bootstrap')
    default_directions = directions is None
    directions = _directions(projection_directions(real.shape[1]) if default_directions else directions,
                             real.shape[1])
    rng = np.random.default_rng(seed)
    row_meter, meter_weights = _cluster_multiplicities(
        meter_ids, technology, len(real), n_bootstrap, rng)
    conditions = sorted(identities)
    condition_rows = {key: np.flatnonzero(keys == key) for key in conditions}
    complete = np.ones(n_bootstrap + 1, dtype=bool)
    for rows in condition_rows.values():
        meter_counts = np.bincount(row_meter[rows], minlength=meter_weights.shape[1])
        complete &= meter_weights @ meter_counts > 0
    missing = int(np.count_nonzero(~complete[1:]))
    valid_weights = meter_weights[complete]
    accum = {name: np.zeros(len(valid_weights)) for name in _METRICS}
    per_condition = {}
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            for key, rows in condition_rows.items():
                r, a, b = real[rows], fake_a[rows], fake_b[rows]
                condition_observed = {}
                # A direction cache is released before the next direction. This
                # bounds memory for large or imbalanced condition memberships.
                for metric in _METRICS:
                    condition_distances = np.zeros((len(valid_weights), 2))
                    vectors = [None] if metric == _METRICS[0] else directions
                    for direction in vectors:
                        projected = ([v.sum(axis=1) for v in (r, a, b)] if direction is None
                                     else [v @ direction for v in (r, a, b)])
                        cache = _PairedW1Cache(*projected)
                        for start in range(0, len(valid_weights), 16):
                            stop = min(start + 16, len(valid_weights))
                            weights = valid_weights[start:stop, row_meter[rows]]
                            distance_a, distance_b = cache.distances(weights)
                            condition_distances[start:stop, 0] += distance_a / len(vectors)
                            condition_distances[start:stop, 1] += distance_b / len(vectors)
                    accum[metric] += (condition_distances[:, 0] - condition_distances[:, 1]) / len(conditions)
                    condition_observed[metric] = {
                        'a': float(condition_distances[0, 0]),
                        'b': float(condition_distances[0, 1]),
                        'difference': float(condition_distances[0, 0] - condition_distances[0, 1]),
                    }
                per_condition[key] = {'n_real': len(rows), 'n_fake_a': len(rows),
                                      'n_fake_b': len(rows), 'metrics': condition_observed}
    except FloatingPointError as exc:
        raise ValueError('profile magnitudes exceed finite metric arithmetic') from exc
    ci95 = {name: (None if missing else np.percentile(values[1:], [2.5, 97.5]).tolist())
            for name, values in accum.items()}
    return {
        'observed_difference': {name: float(values[0]) for name, values in accum.items()},
        'ci95': ci95,
        'per_condition': per_condition,
        'metadata': {
            'n_bootstrap': n_bootstrap, 'seed': int(seed),
            'bootstrap_unit': 'whole real meter history, stratified within technology',
            'interval_method': 'percentile 95%, linear quantiles, paired A-minus-B draws',
            'positive_difference': 'A worse than B',
            'aggregation': 'equal weight per original condition',
            'projection_count': len(directions),
            'projection_seed': 2026 if default_directions else None,
            'projection_sha256': hashlib.sha256(np.asarray(directions, dtype='<f8').tobytes()).hexdigest(),
            'metric_definitions': {
                'daily_energy_w1_kwh': 'empirical W1 of signed per-day sums of interval energy in kWh',
                'profile_sw1_kwh': 'mean empirical W1 of signed physical kWh profiles over recorded unit projections',
            },
            'uncertainty_scope': 'conditional on fitted models, fixed generation draws and shared 2024 calendar/weather; '
                                 'excludes training-seed, synthetic-sampling and future-year/weather uncertainty',
            'missing_condition_replicates': missing,
            'n_complete_bootstrap': int(complete[1:].sum()),
            'ci_unavailable_reason': 'bootstrap draws missing original conditions' if missing else None,
        },
    }


def nearest_neighbor_diagnostic(train, train_labels, heldout, heldout_labels,
                                generated, generated_labels, *,
                                max_queries_per_condition=256, seed=41001):
    """Compare conditional nearest-training distances for generated/held-out days.

    Use every training day as a reference. Independently sample at most the
    declared query cap per condition and sample type without replacement, then
    query a condition-specific cKDTree. Distances are Euclidean/sqrt(slots), i.e.
    RMSE in signed physical interval kWh. This is a proximity diagnostic only;
    an exact match or a larger distance supplies no privacy guarantee.
    """
    train, heldout = _profiles(train, heldout)
    _, generated = _profiles(train, generated)
    cap = _positive_integer(max_queries_per_condition, 'max_queries_per_condition')
    train_keys, identities = _condition_labels(train_labels, len(train), 'train_labels')
    heldout_keys, heldout_identities = _condition_labels(heldout_labels, len(heldout), 'heldout_labels')
    generated_keys, generated_identities = _condition_labels(generated_labels, len(generated), 'generated_labels')
    if identities != heldout_identities or identities != generated_identities:
        raise ValueError('train, heldout and generated must support exactly the same conditions and label types')
    rng = np.random.default_rng(seed)
    tolerance = 1e-8
    per_condition = {}
    for key in sorted(identities):
        reference = train[train_keys == key]
        tree = cKDTree(reference)
        entry = {'n_train': len(reference)}
        for name, sample, keys in [('generated_to_train', generated, generated_keys),
                                   ('heldout_to_train', heldout, heldout_keys)]:
            rows = np.flatnonzero(keys == key)
            available = len(rows)
            if available > cap:
                rows = rng.choice(rows, size=cap, replace=False)
            distances, _ = tree.query(sample[rows], k=1, workers=1)
            distances = distances / np.sqrt(train.shape[1])
            if not np.isfinite(distances).all():
                raise ValueError('profile magnitudes exceed finite nearest-neighbor arithmetic')
            quantiles = np.quantile(distances, [.05, .25, .5, .75, .95])
            entry[name] = {
                'n_available': available, 'n_queries': len(rows),
                'median_rmse_kwh': float(quantiles[2]),
                'quantiles_rmse_kwh': dict(zip(('0.05', '0.25', '0.50', '0.75', '0.95'), quantiles.tolist())),
                'exact_match_rate': float(np.mean(distances == 0.)),
                'tolerance_match_rate': float(np.mean(distances <= tolerance)),
            }
        per_condition[key] = entry
    return {'per_condition': per_condition, 'metadata': {
        'max_queries_per_condition': cap, 'seed': int(seed),
        'metric': 'nearest conditional training-profile RMSE in signed physical interval kWh',
        'tolerance_rmse_kwh': tolerance,
        'query_sampling': 'independently without replacement per condition and query sample; all training references',
        'quantile_method': 'linear',
        'interpretation': 'training proximity diagnostic; no privacy guarantee',
    }}
