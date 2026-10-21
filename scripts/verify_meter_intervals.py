#!/usr/bin/env python3
"""Synthetic verification of fixed-score meter intervals; never reads real data.

The scalar profile construction permits independent analytic risk calculations
and fast scoring. Coverage here concerns independent simulated meters only;
it cannot validate postselection intervals on reused Fluvius validation data.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import scipy
from scipy.special import gamma, hyp1f1, ndtr, ndtri
from threadpoolctl import threadpool_limits

from fluvius_cgan.meter_uncertainty import stratified_studentized_bootstrap


ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = ('scripts/verify_meter_intervals.py', 'src/fluvius_cgan/meter_uncertainty.py',
                'src/fluvius_cgan/proper_scores.py')
SCENARIO_IDS = ['weak_normal', 'strong_normal', 'skewed_mixture', 'small_difference']
SCORE_NAMES = ('energy_score', 'variogram_score')
ALL_PAIR_WEIGHT = sum(1. / (k - h) for h in range(24) for k in range(h + 1, 24))
ODD_PAIR_WEIGHT = sum(1. / (k - h) for h in range(24) for k in range(h + 1, 24)
                      if (k - h) % 2) / ALL_PAIR_WEIGHT
SCOPE = ('Synthetic independent-meter verification of fixed finite-ensemble score functions. '
         'Does not establish new-meter coverage on reused/postselection Fluvius validation, '
         'future-calendar coverage, or unconditional generator-law coverage.')


def calendar_counts(calendar):
    """Derive retained calendar bins from dates, without reading any dataset."""
    if calendar.get('year') != 2024:
        raise ValueError('This simulation declares the 2024 calendar')
    excluded = {date.fromisoformat(value) for value in calendar['excluded_dates']}
    if excluded != {date(2024, 3, 31), date(2024, 10, 27)}:
        raise ValueError('The two declared Brussels DST dates must be excluded')
    counts, current = [0] * 8, date(2024, 1, 1)
    while current.year == 2024:
        if current not in excluded:
            season = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1,
                      6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}[current.month]
            counts[4 * (current.weekday() >= 5) + season] += 1
        current += timedelta(days=1)
    return counts


def _finite_vector(values, length=None, positive=False):
    values = np.asarray(values)
    if (values.ndim != 1 or not len(values) or values.dtype.kind not in 'fiu'
            or not np.isfinite(values).all() or (length is not None and len(values) != length)
            or (positive and np.any(values <= 0))):
        raise ValueError('Expected a finite numeric vector of the declared size and support')
    return values.astype(np.float64)


def validate_plan(plan):
    """Validate the declared design; execution additionally verifies source hashes."""
    if type(plan.get('schema_version')) is not int or plan['schema_version'] != 1:
        raise ValueError('Expected schema_version 1')
    if plan.get('real_data_access') is not False:
        raise ValueError('Synthetic verification cannot access real data')
    for key, minimum in [('cohorts_per_scenario', 2), ('bootstrap_replicates', 2),
                         ('bootstrap_block_size', 1), ('fixed_ensemble_size', 2)]:
        if type(plan.get(key)) is not int or plan[key] < minimum:
            raise ValueError(f'{key} must be an integer of at least {minimum}')
    if type(plan.get('level')) not in (float, int) or not 0 < plan['level'] < 1:
        raise ValueError('level must be a probability')
    sizes = plan.get('stratum_sizes')
    if not isinstance(sizes, list) or len(sizes) != 8 or any(type(n) is not int or n < 2 for n in sizes):
        raise ValueError('All eight strata require at least two meters')
    if calendar_counts(plan['calendar']) != plan['calendar']['bin_counts']:
        raise ValueError('Calendar counts differ from the actual declared dates')
    for source, name, positive in [(plan, 'stratum_locations', False), (plan, 'stratum_scales', True),
                                    (plan['calendar'], 'locations', False), (plan['calendar'], 'scales', True)]:
        _finite_vector(source[name], 8, positive)
    if [entry['id'] for entry in plan['scenarios']] != SCENARIO_IDS:
        raise ValueError('Keep every declared fixed-ensemble scenario in its original order')
    omission = plan['generation_omission']
    if omission['id'] != 'refreshed_ensemble_omission' or type(omission['ensemble_size']) is not int or omission['ensemble_size'] < 2:
        raise ValueError('Declare the separate refreshed-ensemble demonstration')
    for scenario in [*plan['scenarios'], omission]:
        rho = scenario['rho']
        if type(rho) not in (int, float) or not np.isfinite(rho) or not 0 <= rho < 1:
            raise ValueError('rho must be finite in [0,1)')
        effect = scenario['meter_effect']
        weights = _finite_vector(effect['weights'], positive=True)
        means = _finite_vector(effect['means'], len(weights))
        sds = _finite_vector(effect['sds'], len(weights), positive=True)
        if (not np.isclose(weights.sum(), 1., rtol=0, atol=1e-12)
                or not np.isclose(weights @ means, 0., rtol=0, atol=1e-12)
                or not np.isclose(weights @ (sds**2 + means**2), 1., rtol=0, atol=1e-12)):
            raise ValueError('Meter effect must have probabilities summing to one, mean zero and variance one')
        for key in ('ensemble_seed', 'cohort_seed', 'bootstrap_seed'):
            if type(scenario[key]) is not int or not 0 <= scenario[key] < 2**32:
                raise ValueError('Seeds must be unsigned 32-bit integers')
        for key in ('forecast_a', 'forecast_b'):
            _finite_vector([scenario[key]['offset'], scenario[key]['sd_factor']])
            if scenario[key]['sd_factor'] <= 0:
                raise ValueError('Forecast standard deviations must be positive')
    weak, strong = plan['scenarios'][:2]
    for key in ('meter_effect', 'forecast_a', 'forecast_b', 'ensemble_seed'):
        if weak[key] != strong[key]:
            raise ValueError('Weak/strong scenarios must share marginals and saved ensembles')
    if omission['meter_effect'] != {'weights': [1.0], 'means': [0.0], 'sds': [1.0]}:
        raise ValueError('The generation-omission demonstration has a normal DGP')
    return plan


def normal_absolute_moment(mean, sd, power):
    """E|N(mean,sd**2)|**power for power .5 or 1; verified against quadrature."""
    mean, sd = np.broadcast_arrays(np.asarray(mean, dtype=float), np.asarray(sd, dtype=float))
    if power not in (.5, 1.) or not np.isfinite(mean).all() or not np.isfinite(sd).all() or np.any(sd <= 0):
        raise ValueError('Normal moments require finite means, positive SDs and power .5 or 1')
    ratio = mean / sd
    if power == 1:
        result = sd * np.sqrt(2. / np.pi) * np.exp(-.5 * ratio**2) + mean * (2 * ndtr(ratio) - 1)
    else:
        result = sd**power * 2**(power / 2.) * gamma((power + 1) / 2.) / np.sqrt(np.pi)
        result = result * hyp1f1(-power / 2., .5, -.5 * ratio**2)
    if not np.isfinite(result).all() or np.any(result < 0):
        raise FloatingPointError('Nonfinite or negative analytic absolute moment')
    return result


@dataclass(frozen=True)
class ScalarEnsemble:
    values: np.ndarray
    prefix: np.ndarray
    pair_mean: float
    half_mean: float
    half_variance_over_count: float


def fit_scalar_ensemble(values):
    values = np.sort(_finite_vector(values))
    count = len(values)
    if count < 2:
        raise ValueError('An ensemble requires at least two IID members')
    prefix = np.r_[0., np.cumsum(values)]
    pair_mean = 2. * float(np.dot(2 * np.arange(count) - count + 1, values)) / (count * (count - 1))
    features = np.sqrt(np.abs(values))
    values.flags.writeable = prefix.flags.writeable = False
    return ScalarEnsemble(values, prefix, pair_mean, float(features.mean()), float(features.var(ddof=1) / count))


def _mean_absolute_distance(observations, ensemble):
    count = len(ensemble.values)
    positions = np.searchsorted(ensemble.values, observations, side='right')
    return ((2 * positions - count) * observations + ensemble.prefix[-1] - 2 * ensemble.prefix[positions]) / count


def scalar_score_difference(observations, first, second):
    observations = _finite_vector(observations)
    energy = (_mean_absolute_distance(observations, first) - _mean_absolute_distance(observations, second)
              - .5 * (first.pair_mean - second.pair_mean)) / np.sqrt(2.)
    variogram = ODD_PAIR_WEIGHT * (
        -2 * np.sqrt(np.abs(observations)) * (first.half_mean - second.half_mean)
        + first.half_mean**2 - second.half_mean**2
        - first.half_variance_over_count + second.half_variance_over_count)
    return np.column_stack([energy, variogram])


def conditional_score_truth(first, second, components):
    """Integrate fixed score functions analytically over a normal-mixture Y law."""
    expected_absolute = sum(weight * normal_absolute_moment(mean, sd, 1.) for weight, mean, sd in components)
    expected_half = sum(weight * normal_absolute_moment(mean, sd, .5) for weight, mean, sd in components)
    scores = []
    for ensemble in (first, second):
        cross = sum(weight * normal_absolute_moment(ensemble.values - mean, sd, 1.).mean()
                    for weight, mean, sd in components)
        energy = (cross - .5 * ensemble.pair_mean) / np.sqrt(2.)
        variogram = ODD_PAIR_WEIGHT * (expected_absolute - 2 * ensemble.half_mean * expected_half
                                     + ensemble.half_mean**2 - ensemble.half_variance_over_count)
        scores.append(np.array([energy, variogram]))
    return scores[0] - scores[1]


def generator_score_truth(first, second, components):
    """Population scores for two normal generator laws, without ensemble offsets."""
    expected_absolute = sum(weight * normal_absolute_moment(mean, sd, 1.) for weight, mean, sd in components)
    expected_half = sum(weight * normal_absolute_moment(mean, sd, .5) for weight, mean, sd in components)
    scores = []
    for mean, sd in (first, second):
        cross = sum(weight * normal_absolute_moment(mean - location, np.sqrt(sd**2 + scale**2), 1.)
                    for weight, location, scale in components)
        energy = (cross - sd / np.sqrt(np.pi)) / np.sqrt(2.)
        half = normal_absolute_moment(mean, sd, .5)
        scores.append(np.array([energy, ODD_PAIR_WEIGHT * (expected_absolute - 2 * half * expected_half + half**2)]))
    return scores[0] - scores[1]


def _condition(plan, scenario, condition):
    group, calendar_bin = divmod(condition, 8)
    location = plan['stratum_locations'][group] + plan['calendar']['locations'][calendar_bin]
    scale = plan['stratum_scales'][group] * plan['calendar']['scales'][calendar_bin]
    rho, effect = scenario['rho'], scenario['meter_effect']
    components = [(weight, location + scale * np.sqrt(rho) * mean,
                   scale * np.sqrt(rho * sd**2 + 1 - rho))
                  for weight, mean, sd in zip(effect['weights'], effect['means'], effect['sds'])]
    forecasts = tuple((location + scale * scenario[key]['offset'], scale * scenario[key]['sd_factor'])
                      for key in ('forecast_a', 'forecast_b'))
    return location, scale, components, forecasts


def make_ensembles(plan, scenario, cohort_index=None):
    count = plan['fixed_ensemble_size'] if cohort_index is None else scenario['ensemble_size']
    refresh_index = 0 if cohort_index is None else cohort_index + 1
    ensembles = []
    for condition in range(64):
        rng = np.random.default_rng(np.random.SeedSequence([scenario['ensemble_seed'], refresh_index, condition]))
        base = rng.standard_normal(count)
        forecasts = _condition(plan, scenario, condition)[3]
        ensembles.append(tuple(fit_scalar_ensemble(mean + sd * base) for mean, sd in forecasts))
    return ensembles


def ensembles_digest(ensembles):
    digest = hashlib.sha256()
    for pair in ensembles:
        for ensemble in pair:
            digest.update(np.asarray(ensemble.values, dtype='<f8').tobytes())
    return digest.hexdigest()


def scenario_truth(plan, scenario, ensembles=None):
    results = []
    for condition in range(64):
        _, _, components, forecasts = _condition(plan, scenario, condition)
        results.append(generator_score_truth(*forecasts, components) if ensembles is None
                       else conditional_score_truth(*ensembles[condition], components))
    return np.mean(results, axis=0)


def score_calendar_bins(observation_bins, ensemble_pairs):
    if len(observation_bins) != 8 or len(ensemble_pairs) != 8:
        raise ValueError('Exactly eight calendar bins are required')
    count = len(observation_bins[0])
    values = np.zeros((count, 3))
    for observations, pair in zip(observation_bins, ensemble_pairs):
        observations = np.asarray(observations)
        if observations.ndim != 2 or len(observations) != count or not observations.shape[1]:
            raise ValueError('Calendar bins must contain the same meters and at least one day')
        scores = scalar_score_difference(observations.ravel(), *pair).reshape(*observations.shape, 2)
        values[:, :2] += scores.mean(axis=1) / 8.
    return values  # Third column is an exactly identical A-minus-A score function.


def _simulate_cohort(plan, scenario, ensembles, index):
    meter_rng = np.random.default_rng(np.random.SeedSequence([scenario['cohort_seed'], index, 0]))
    daily_rng = np.random.default_rng(np.random.SeedSequence([scenario['cohort_seed'], index, 1]))
    effect, rho = scenario['meter_effect'], scenario['rho']
    values, strata = [], []
    for group, count in enumerate(plan['stratum_sizes']):
        component = meter_rng.choice(len(effect['weights']), size=count, p=effect['weights'])
        means, sds = np.asarray(effect['means']), np.asarray(effect['sds'])
        meter_effect = means[component] + sds[component] * meter_rng.standard_normal(count)
        bins = []
        for calendar_bin, days in enumerate(plan['calendar']['bin_counts']):
            location, scale, _, _ = _condition(plan, scenario, group * 8 + calendar_bin)
            bins.append(location + scale * (np.sqrt(rho) * meter_effect[:, None]
                        + np.sqrt(1 - rho) * daily_rng.standard_normal((count, days))))
        values.append(score_calendar_bins(bins, ensembles[group * 8:(group + 1) * 8]))
        strata.extend([group] * count)
    return np.concatenate(values), np.asarray(strata, dtype=np.int64)


def _wilson(successes, count, level=.95):
    if not count:
        return None
    z = float(ndtri(.5 + level / 2.))
    proportion, denominator = successes / count, 1 + z**2 / count
    center = (proportion + z**2 / (2 * count)) / denominator
    half = z * np.sqrt(proportion * (1 - proportion) / count + z**2 / (4 * count**2)) / denominator
    return [float(center - half), float(center + half)]


def summarize_records(records, truth_key):
    result = {}
    for column, name in enumerate(SCORE_NAMES):
        errors = np.asarray([row['estimate'][column] - row[truth_key][column] for row in records])
        valid = [row for row in records if row['status'][column] == 'ok' and row['interval'][column] is not None]
        covered = sum(row['interval'][column][0] <= row[truth_key][column] <= row['interval'][column][1] for row in valid)
        count, total = len(valid), len(records)
        coverage = covered / count if count else None
        widths = [row['interval'][column][1] - row['interval'][column][0] for row in valid]
        result[name] = {
            'estimate_count': total, 'bias': float(errors.mean()),
            'bias_mcse': float(errors.std(ddof=1) / np.sqrt(total)) if total > 1 else None,
            'available_intervals': count, 'failed_intervals': total - count,
            'interval_status_counts': dict(Counter(row['status'][column] for row in records)),
            'covered_intervals': int(covered), 'coverage': coverage, 'coverage_denominator': count,
            'coverage_scope': 'conditional on an interval being available; inspect availability separately',
            'coverage_wilson_interval': _wilson(covered, count),
            'coverage_mcse': float(np.sqrt(coverage * (1 - coverage) / count)) if count else None,
            'joint_available_and_covers_rate': covered / total,
            'mean_interval_width': float(np.mean(widths)) if widths else None,
            'median_interval_width': float(np.median(widths)) if widths else None,
            'minimum_interval_width': float(np.min(widths)) if widths else None,
            'maximum_interval_width': float(np.max(widths)) if widths else None,
        }
    return result


def run_scenario(plan, scenario, refresh=False, progress=None):
    """One complete declared scenario; callback preserves per-cohort audit records."""
    ensembles = None if refresh else make_ensembles(plan, scenario)
    fixed_truth = None if refresh else scenario_truth(plan, scenario, ensembles)
    generator_truth = scenario_truth(plan, scenario) if refresh else None
    records = []
    for index in range(plan['cohorts_per_scenario']):
        if refresh:
            ensembles = make_ensembles(plan, scenario, index)
        truth = scenario_truth(plan, scenario, ensembles) if refresh else fixed_truth
        values, strata = _simulate_cohort(plan, scenario, ensembles, index)
        seed = int(np.random.SeedSequence([scenario['bootstrap_seed'], index]).generate_state(1)[0])
        interval = stratified_studentized_bootstrap(values, strata, seed=seed,
            replicates=plan['bootstrap_replicates'], level=plan['level'], block_size=plan['bootstrap_block_size'])
        record = {'cohort_index': index, 'bootstrap_seed': seed, 'estimate': interval['estimate'][:2],
            'standard_error': interval['standard_error'][:2], 'interval': interval['interval'][:2],
            'status': interval['status'][:2], 'conditional_truth': truth.tolist(),
            'identical_status': interval['status'][2], 'identical_estimate': interval['estimate'][2],
            'identical_interval': interval['interval'][2], 'ensemble_sha256': ensembles_digest(ensembles),
            'multiplicity_sha256': interval['multiplicity_sha256'],
            'bootstrap_zero_standard_error_counts': interval['bootstrap_zero_standard_error_counts'],
            'variance_roundoff_clip_counts': interval['variance_roundoff_clip_counts']}
        if refresh:
            record['generator_truth'] = generator_truth.tolist()
        records.append(record)
        if progress is not None:
            progress(record)
    result = {'id': scenario['id'], 'description': scenario['description'], 'cohorts': len(records),
        'ensemble_size': scenario['ensemble_size'] if refresh else plan['fixed_ensemble_size'],
        'ensemble_refreshed_per_cohort': refresh, 'scope': SCOPE,
        'conditional_fixed_ensemble': summarize_records(records, 'conditional_truth'),
        'identical_function_check': {
            'expected_degenerate_count': sum(row['identical_status'] == 'degenerate_observed_variance' for row in records),
            'status_counts': dict(Counter(row['identical_status'] for row in records)),
            'all_estimates_zero': all(row['identical_estimate'] == 0. for row in records),
            'all_intervals_unavailable': all(row['identical_interval'] is None for row in records),
            'coverage': None, 'interpretation': 'Identical score functions have zero variance; no interval is defined.'}}
    if refresh:
        result['underlying_generator_law'] = summarize_records(records, 'generator_truth')
        result['omission_interpretation'] = ('These same meter-only intervals omit refreshed ensemble uncertainty. '
            'Conditional-target coverage here is averaged over regenerated ensembles; it does not establish '
            'coverage for each individual ensemble. Contrasting targets demonstrates the scope difference; '
            'no undercoverage outcome is guaranteed.')
    return result, records


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_plan(path):
    plan = validate_plan(json.loads(Path(path).read_text()))
    sources = plan.get('execution_sources')
    if not isinstance(sources, dict) or set(sources) != set(SOURCE_FILES):
        raise ValueError('The plan must bind the verifier and both numerical core source hashes')
    for source, expected in sources.items():
        if not isinstance(expected, str) or len(expected) != 64 or _sha256(ROOT / source) != expected:
            raise ValueError(f'Execution source hash differs: {source}')
    return plan


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    plan = load_plan(args.plan)
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    manifest = {'schema_version': 1, 'status': 'running', 'scope': SCOPE,
        'started_utc': datetime.now(timezone.utc).isoformat(), 'real_data_access': False,
        'plan_sha256': _sha256(args.plan), 'source_hashes': plan['execution_sources'],
        'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__},
        'cpu_threads': 1, 'completed_scenarios': []}
    _write_json(args.output / 'manifest.json', manifest)
    (args.output / 'plan.json').write_bytes(args.plan.read_bytes())
    summary = {'schema_version': 1, 'scope': SCOPE, 'scenarios': [],
        'coverage_level': plan['level'], 'bootstrap_replicates': plan['bootstrap_replicates'],
        'coverage_wilson_confidence_level': .95, 'calendar_bin_counts': calendar_counts(plan['calendar']),
        'stratum_sizes': plan['stratum_sizes'],
        'verification_limits': 'Scalar-profile synthetic families and independent simulated meters only. '
            'Finite-ensemble truth is analytic; synthetic coverage never establishes postselection Fluvius coverage.'}
    try:
        with threadpool_limits(limits=1):
            for scenario in [*plan['scenarios'], plan['generation_omission']]:
                path = args.output / f"{scenario['id']}_cohorts.jsonl"
                with path.open('x') as stream:
                    def progress(record):
                        stream.write(json.dumps(record, allow_nan=False) + '\n')
                        stream.flush()
                        if (record['cohort_index'] + 1) % 100 == 0:
                            print(f"{scenario['id']}: {record['cohort_index'] + 1}/{plan['cohorts_per_scenario']} cohorts", flush=True)
                    result, _ = run_scenario(plan, scenario, refresh=scenario is plan['generation_omission'], progress=progress)
                summary['scenarios'].append(result)
                manifest['completed_scenarios'].append(scenario['id'])
                _write_json(args.output / 'summary.json', summary)
                _write_json(args.output / 'manifest.json', manifest)
        manifest['status'] = 'completed'
    except Exception as error:
        manifest.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        manifest['elapsed_seconds'] = time.monotonic() - start
        manifest['finished_utc'] = datetime.now(timezone.utc).isoformat()
        manifest['files'] = {path.name: _sha256(path) for path in sorted(args.output.iterdir())
                             if path.is_file() and path.name != 'manifest.json'}
        _write_json(args.output / 'manifest.json', manifest)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
