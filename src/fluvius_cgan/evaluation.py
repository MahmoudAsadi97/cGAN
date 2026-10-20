"""Empirical distribution metrics for physical interval-energy profiles.

Inputs are rows of profiles and columns of interval energy in kWh. No metric
fits a transform or uses a validation/test range. In particular profile SW1 is
in physical kWh; selection code may divide it by a *training-only* range.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import wasserstein_distance


_DIAGNOSTICS = {'real_constant_profiles', 'fake_constant_profiles', 'real_zero_diversity'}


def _profiles(real, fake):
    arrays = []
    for name, values in [('real', real), ('fake', fake)]:
        if np.iscomplexobj(values):
            raise ValueError(f'{name} must contain real finite values')
        try:
            array = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'{name} must be a finite 2D array') from exc
        if array.ndim != 2 or 0 in array.shape or not np.isfinite(array).all():
            raise ValueError(f'{name} must be a nonempty finite 2D array')
        arrays.append(array)
    if arrays[0].shape[1] != arrays[1].shape[1]:
        raise ValueError('real and fake must have the same profile width')
    return arrays


def projection_directions(dimension, count=128, seed=2026):
    """Return reproducible Gaussian unit directions, one direction per row."""
    for name, value in [('dimension', dimension), ('count', count)]:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(count, dimension))
    norms = np.linalg.norm(vectors, axis=1)
    # A zero Gaussian vector is extremely unlikely, but cannot be normalized.
    while np.any(norms == 0):
        zero = norms == 0
        vectors[zero] = rng.normal(size=(int(zero.sum()), dimension))
        norms[zero] = np.linalg.norm(vectors[zero], axis=1)
    return vectors / norms[:, None]


def sliced_wasserstein(real, fake, directions):
    """Mean empirical W1 over unit projections, supporting unequal row counts.

    Each direction is processed separately. Auxiliary memory is linear in the
    two sample counts, independent of the number of directions; no pairwise
    distance matrix or sample-size matching is used.
    """
    real, fake = _profiles(real, fake)
    if np.iscomplexobj(directions):
        raise ValueError('directions must be finite real unit vectors')
    try:
        directions = np.asarray(directions, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError('directions must be finite unit vectors') from exc
    if (directions.ndim != 2 or directions.shape[0] == 0
            or directions.shape[1] != real.shape[1]
            or not np.isfinite(directions).all()
            or not np.allclose(np.linalg.norm(directions, axis=1), 1., rtol=1e-7, atol=1e-10)):
        raise ValueError('directions must be nonempty unit vectors matching profile width')
    distances = []
    try:
        with np.errstate(over='raise', invalid='raise'):
            for direction in directions:
                distances.append(wasserstein_distance(real @ direction, fake @ direction))
            result = float(np.mean(distances))
    except FloatingPointError as exc:
        raise ValueError('profile magnitudes exceed finite metric arithmetic') from exc
    if not np.isfinite(result):
        raise ValueError('profile magnitudes exceed finite metric arithmetic')
    return result


def _acf(centered, denominator, lag):
    result = np.zeros(len(centered), dtype=np.float64)
    if centered.shape[1] > lag:
        numerator = np.sum(centered[:, :-lag] * centered[:, lag:], axis=1)
        np.divide(numerator, denominator, out=result, where=denominator != 0)
    return result


def profile_metrics(real, fake, interval_hours=1.0, directions=None):
    """Compare profile distributions in physical units.

    Slot W1 is the mean W1 over slots; daily energy is the row sum. Peak is
    max(energy / interval_hours). Ramps pool *signed* adjacent power changes,
    without a last-to-first wrap. A single-slot profile has zero ramp and ACF.
    Population slot variances use ddof=0. Diversity is mean fake slot variance
    divided by mean real slot variance: both zero gives 1, while a zero real
    denominator with positive fake variance gives None (JSON null). The
    real_zero_diversity flag identifies both cases; no infinity is manufactured.

    Per-row ACF subtracts the full-row mean and divides the lagged dot product
    by the full centered sum of squares (the biased convention). Constants
    have ACF=0 and explicit counts. Lag 6 is included only for hourly profiles
    with at least seven slots. Other distribution metrics must remain finite.
    """
    real, fake = _profiles(real, fake)
    try:
        interval_hours = float(interval_hours)
    except (TypeError, ValueError) as exc:
        raise ValueError('interval_hours must be finite and positive') from exc
    if not np.isfinite(interval_hours) or interval_hours <= 0:
        raise ValueError('interval_hours must be finite and positive')
    if directions is None:
        directions = projection_directions(real.shape[1])
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            # Remove a shared offset before reduction so repeated decimal
            # constants do not acquire variance from mean-rounding error.
            real_variance = (real - real[:1]).var(axis=0)
            fake_variance = (fake - fake[:1]).var(axis=0)
            real_diversity = float(real_variance.mean())
            fake_diversity = float(fake_variance.mean())
            if real_diversity == 0:
                diversity_ratio = 1. if fake_diversity == 0 else None
            else:
                diversity_ratio = fake_diversity / real_diversity
            centered_real = real - real[:, :1]
            centered_real -= centered_real.mean(axis=1, keepdims=True)
            centered_fake = fake - fake[:, :1]
            centered_fake -= centered_fake.mean(axis=1, keepdims=True)
            denom_real = np.sum(centered_real ** 2, axis=1)
            denom_fake = np.sum(centered_fake ** 2, axis=1)
            metrics = {
                'slot_w1_kwh': float(np.mean([
                    wasserstein_distance(real[:, j], fake[:, j]) for j in range(real.shape[1])
                ])),
                'daily_energy_w1_kwh': float(wasserstein_distance(real.sum(axis=1), fake.sum(axis=1))),
                'peak_w1_kw': float(wasserstein_distance(real.max(axis=1) / interval_hours,
                                                        fake.max(axis=1) / interval_hours)),
                'ramp_w1_kw': float(wasserstein_distance(
                    (np.diff(real, axis=1) / interval_hours).ravel(),
                    (np.diff(fake, axis=1) / interval_hours).ravel(),
                )) if real.shape[1] > 1 else 0.,
                'mean_profile_rmse_kwh': float(np.sqrt(np.mean((real.mean(axis=0) - fake.mean(axis=0)) ** 2))),
                'variance_mae_kwh2': float(np.mean(np.abs(real_variance - fake_variance))),
                'diversity_ratio': diversity_ratio,
                'acf_lag1_w1': float(wasserstein_distance(_acf(centered_real, denom_real, 1),
                                                         _acf(centered_fake, denom_fake, 1))),
                'profile_sw1_kwh': sliced_wasserstein(real, fake, directions),
                'real_constant_profiles': int(np.count_nonzero(denom_real == 0)),
                'fake_constant_profiles': int(np.count_nonzero(denom_fake == 0)),
                'real_zero_diversity': real_diversity == 0,
            }
            if interval_hours == 1.0 and real.shape[1] >= 7:
                metrics['acf_lag6_w1'] = float(wasserstein_distance(
                    _acf(centered_real, denom_real, 6), _acf(centered_fake, denom_fake, 6)))
    except FloatingPointError as exc:
        raise ValueError('profile magnitudes exceed finite metric arithmetic') from exc
    if any(value is not None and not np.isfinite(value) for value in metrics.values()):
        raise ValueError('profile magnitudes exceed finite metric arithmetic')
    return metrics


def _condition_labels(labels, count, name):
    labels = np.asarray(labels, dtype=object)
    if labels.ndim != 1 or len(labels) != count:
        raise ValueError(f'{name} must be one-dimensional with one label per profile')
    keys = []
    identities = {}
    for value in labels:
        if isinstance(value, np.generic):
            value = value.item()
        if not isinstance(value, (str, int, float, bool)):
            raise ValueError(f'{name} contains a missing or unsupported label')
        if isinstance(value, str):
            if not value.strip():
                raise ValueError(f'{name} contains a missing label')
        elif not np.isfinite(value):
            raise ValueError(f'{name} contains a nonfinite label')
        key = str(value)
        identity = (type(value), value)
        if key in identities and identities[key] != identity:
            raise ValueError(f'{name} contains labels with ambiguous string representations')
        identities[key] = identity
        keys.append(key)
    return np.array(keys), identities


def conditional_metrics(real, fake, real_labels, fake_labels, interval_hours=1.0, directions=None):
    """Return per-condition metrics and macro / real-count-weighted summaries.

    Both samples must support exactly the same nonmissing conditions. Keys are
    string labels. Counts and diagnostic flags stay in per_condition; they are
    never treated as distribution errors. Undefined diversity ratios are
    skipped and weights renormalized over valid conditions. An all-undefined
    summary is None; metric_valid_conditions exposes the support of each metric.
    """
    real, fake = _profiles(real, fake)
    real_keys, real_identities = _condition_labels(real_labels, len(real), 'real_labels')
    fake_keys, fake_identities = _condition_labels(fake_labels, len(fake), 'fake_labels')
    if real_identities != fake_identities:
        raise ValueError('real and fake must support exactly the same conditions and label types')
    if directions is None:
        directions = projection_directions(real.shape[1])
    per_condition = {}
    for key in sorted(real_identities):
        real_mask, fake_mask = real_keys == key, fake_keys == key
        per_condition[key] = {
            'n_real': int(real_mask.sum()), 'n_fake': int(fake_mask.sum()),
            'metrics': profile_metrics(real[real_mask], fake[fake_mask], interval_hours, directions),
        }
    metric_names = [name for name in next(iter(per_condition.values()))['metrics'] if name not in _DIAGNOSTICS]
    macro, weighted, valid_counts = {}, {}, {}
    for name in metric_names:
        valid = [entry for entry in per_condition.values() if entry['metrics'][name] is not None]
        valid_counts[name] = len(valid)
        macro[name] = float(np.mean([entry['metrics'][name] for entry in valid])) if valid else None
        weighted[name] = float(np.average(
            [entry['metrics'][name] for entry in valid], weights=[entry['n_real'] for entry in valid]
        )) if valid else None
    return {'per_condition': per_condition, 'macro': macro, 'weighted': weighted,
            'metric_valid_conditions': valid_counts}
