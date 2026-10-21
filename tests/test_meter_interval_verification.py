"""Independent scalar-score/truth checks; tiny simulations test accounting only."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'reports/phase1-development/METER_INTERVAL_SIMULATION_PLAN.json'


def verifier():
    path = ROOT / 'scripts/verify_meter_intervals.py'
    assert path.is_file(), 'The synthetic interval verifier has not been implemented'
    name = 'meter_interval_verifier_test_module'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def plan():
    return json.loads(PLAN.read_text())


def integrate(function, mean, sd, knots=()):
    boundaries = [-np.inf, *sorted(set([0., *knots])), np.inf]
    return sum(quad(lambda x: function(x) * norm.pdf(x, loc=mean, scale=sd),
                    lower, upper, epsabs=2e-10, epsrel=2e-10)[0]
               for lower, upper in zip(boundaries[:-1], boundaries[1:]))


def test_scalar_shortcut_matches_full_profile_score_core_for_signed_tied_values():
    from fluvius_cgan.proper_scores import fit_ensemble_scores
    module = verifier()
    first, second = np.array([-2., 0., 0., .3, 4.]), np.array([-1.2, -.2, .4, 1., 2.])
    observations = np.array([-3., -.5, 0., .01, 2., 5.])
    direction = np.tile([0., 1.], 12)
    a = fit_ensemble_scores(first[:, None] * direction, block_size=3).score(observations[:, None] * direction)
    b = fit_ensemble_scores(second[:, None] * direction, block_size=3).score(observations[:, None] * direction)
    actual = module.scalar_score_difference(observations, module.fit_scalar_ensemble(first),
                                             module.fit_scalar_ensemble(second))
    expected = np.column_stack([a[key] - b[key] for key in ('energy_score', 'variogram_score')])
    np.testing.assert_allclose(actual, expected, atol=3e-14, rtol=3e-13)


@pytest.mark.parametrize('mean,sd', [(-2., .3), (0., 1.), (.7, 2.), (4., .7)])
@pytest.mark.parametrize('power', [.5, 1.])
def test_normal_absolute_moments_agree_with_independent_quadrature(mean, sd, power):
    actual = verifier().normal_absolute_moment(mean, sd, power)
    expected = integrate(lambda x: abs(x) ** power, mean, sd)
    assert actual == pytest.approx(expected, rel=2e-10, abs=2e-10)


def test_conditional_fixed_ensemble_truth_matches_full_score_core_quadrature():
    from fluvius_cgan.proper_scores import fit_ensemble_scores
    module = verifier()
    direction = np.tile([0., 1.], 12)
    za, zb = np.array([-1.4, -.1, .4, 2.]), np.array([-2., -.3, .5, 1.])
    a, b = module.fit_scalar_ensemble(za), module.fit_scalar_ensemble(zb)
    components = [(0.8, -.3, .7), (0.2, 1.1, 1.2)]
    actual = module.conditional_score_truth(a, b, components)
    fa, fb = fit_ensemble_scores(za[:, None] * direction), fit_ensemble_scores(zb[:, None] * direction)
    for column, key in enumerate(('energy_score', 'variogram_score')):
        def difference(y):
            row = (direction * y)[None]
            return float(fa.score(row)[key][0] - fb.score(row)[key][0])
        expected = sum(weight * integrate(difference, mean, sd, knots=np.r_[za, zb])
                       for weight, mean, sd in components)
        assert actual[column] == pytest.approx(expected, abs=2e-8)


def test_generator_law_truth_agrees_with_independent_normal_integrals():
    module = verifier()
    qa, qb, components = (0.2, 1.1), (-0.3, .8), [(1., .4, .9)]
    actual = module.generator_score_truth(qa, qb, components)
    weight_sum = sum((1 / (k-h)) for h in range(24) for k in range(h+1,24) if (k-h) % 2)
    weight_sum /= sum(1 / (k-h) for h in range(24) for k in range(h+1,24))
    half_y = integrate(lambda x: abs(x) ** .5, .4, .9)
    absolute_y = integrate(abs, .4, .9)
    scores = []
    for mean, sd in (qa, qb):
        cross = integrate(abs, mean - .4, np.sqrt(sd**2 + .9**2))
        energy = (cross - sd / np.sqrt(np.pi)) / np.sqrt(2)
        half_z = integrate(lambda x: abs(x) ** .5, mean, sd)
        scores.append([energy, weight_sum * (absolute_y - 2 * half_y * half_z + half_z**2)])
    np.testing.assert_allclose(actual, np.asarray(scores[0]) - scores[1], atol=3e-10)


def test_actual_calendar_counts_and_weak_strong_fixed_targets_are_identical():
    module, settings = verifier(), plan()
    assert module.calendar_counts(settings['calendar']) == [66,66,65,65,25,25,27,25]
    assert sum(module.calendar_counts(settings['calendar'])) == 364
    first = module.make_ensembles(settings, settings['scenarios'][0])
    second = module.make_ensembles(settings, settings['scenarios'][1])
    assert module.ensembles_digest(first) == module.ensembles_digest(second)
    ta = module.scenario_truth(settings, settings['scenarios'][0], first)
    tb = module.scenario_truth(settings, settings['scenarios'][1], second)
    np.testing.assert_allclose(ta, tb, atol=2e-15)


def test_calendar_score_aggregation_matches_real_meter_core_with_unequal_bin_days():
    from fluvius_cgan.meter_uncertainty import meter_contributions
    module = verifier()
    bins = [np.arange(2 * (index + 1)).reshape(2, index + 1) / 3. - 1. for index in range(8)]
    pairs = [(module.fit_scalar_ensemble([-1., .2, 1.]), module.fit_scalar_ensemble([-.5, .4, 2.]))] * 8
    actual = module.score_calendar_bins(bins, pairs)
    differences, meters, conditions = [], [], []
    for index, values in enumerate(bins):
        for meter in range(2):
            difference = module.scalar_score_difference(values[meter], *pairs[index])
            differences.extend(difference)
            meters.extend([meter] * len(values[meter]))
            conditions.extend([index] * len(values[meter]))
    expected = meter_contributions(differences, meters, conditions)['values']
    np.testing.assert_allclose(actual[:, :2], expected, atol=3e-15)
    np.testing.assert_array_equal(actual[:, 2], 0.)


def tiny_plan():
    settings = plan()
    settings.update(cohorts_per_scenario=3, bootstrap_replicates=19, bootstrap_block_size=19,
                    stratum_sizes=[3,4,5,6,7,8,9,10])
    return settings


@pytest.mark.parametrize('refresh', [False, True])
def test_tiny_simulation_accounts_for_all_cohorts_targets_and_degenerate_column(refresh):
    module, settings = verifier(), tiny_plan()
    scenario = settings['generation_omission'] if refresh else settings['scenarios'][0]
    result, records = module.run_scenario(settings, scenario, refresh=refresh)
    assert len(records) == result['cohorts'] == 3
    assert result['identical_function_check']['expected_degenerate_count'] == 3
    assert result['identical_function_check']['coverage'] is None
    assert all(row['identical_status'] == 'degenerate_observed_variance' for row in records)
    assert len({row['ensemble_sha256'] for row in records}) == (3 if refresh else 1)
    for key in ('energy_score', 'variogram_score'):
        metric = result['conditional_fixed_ensemble'][key]
        assert metric['estimate_count'] == 3
        assert metric['available_intervals'] + metric['failed_intervals'] == 3
        assert 0 <= metric['covered_intervals'] <= metric['available_intervals']
        assert np.isfinite(metric['bias_mcse'])
        assert metric['coverage_denominator'] == metric['available_intervals']
    assert ('underlying_generator_law' in result) is refresh
    if refresh:
        assert 'averaged over regenerated ensembles' in result['omission_interpretation']
        assert 'each individual ensemble' in result['omission_interpretation']


def test_summary_does_not_count_missing_intervals_as_successful_coverage():
    module = verifier()
    rows = [dict(estimate=[1., 2.], conditional_truth=[0., 0.],
                 interval=[[0., 2.], None], status=['ok', 'failed_studentization']),
            dict(estimate=[-1., 3.], conditional_truth=[0., 0.],
                 interval=[None, None], status=['failed_studentization', 'degenerate_observed_variance'])]
    summary = module.summarize_records(rows, 'conditional_truth')
    energy, variogram = summary['energy_score'], summary['variogram_score']
    assert energy['coverage'] == 1. and energy['coverage_denominator'] == 1
    assert energy['joint_available_and_covers_rate'] == .5
    assert energy['bias'] == 0. and energy['bias_mcse'] == pytest.approx(1.)
    assert variogram['coverage'] is None and variogram['coverage_wilson_interval'] is None
    assert variogram['failed_intervals'] == 2


@pytest.mark.parametrize('change', ['calendar', 'rho', 'probability', 'count', 'seed', 'real_access'])
def test_invalid_simulation_settings_fail_before_execution(change):
    module, settings = verifier(), deepcopy(plan())
    if change == 'calendar': settings['calendar']['bin_counts'][0] += 1
    elif change == 'rho': settings['scenarios'][0]['rho'] = 1.1
    elif change == 'probability': settings['scenarios'][0]['meter_effect']['weights'] = [.8]
    elif change == 'count': settings['cohorts_per_scenario'] = True
    elif change == 'seed': settings['scenarios'][0]['ensemble_seed'] = -1
    else: settings['real_data_access'] = True
    with pytest.raises(ValueError): module.validate_plan(settings)


def bound_tiny_plan(module):
    settings = tiny_plan()
    settings.update(cohorts_per_scenario=2, bootstrap_replicates=9, bootstrap_block_size=9)
    settings['execution_sources'] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                     for name in module.SOURCE_FILES}
    return settings


def test_cli_runs_only_tiny_synthetic_design_and_preserves_complete_receipts(tmp_path):
    module = verifier()
    settings = bound_tiny_plan(module)
    source, output = tmp_path / 'plan.json', tmp_path / 'synthetic'
    source.write_text(json.dumps(settings))
    assert module.main(['--plan', str(source), '--output', str(output)]) == 0
    manifest = json.loads((output / 'manifest.json').read_text())
    summary = json.loads((output / 'summary.json').read_text())
    assert manifest['status'] == 'completed' and manifest['real_data_access'] is False
    assert len(manifest['completed_scenarios']) == len(summary['scenarios']) == 5
    assert manifest['plan_sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert (output / 'plan.json').read_bytes() == source.read_bytes()
    for name, digest in manifest['files'].items():
        assert digest == hashlib.sha256((output / name).read_bytes()).hexdigest()
    for scenario in summary['scenarios']:
        rows = (output / f"{scenario['id']}_cohorts.jsonl").read_text().splitlines()
        assert len(rows) == scenario['cohorts'] == 2
        assert 'postselection' in scenario['scope']
    with pytest.raises(FileExistsError):
        module.main(['--plan', str(source), '--output', str(output)])


def test_source_mismatch_rejected_before_any_output_is_created(tmp_path):
    module = verifier()
    settings = bound_tiny_plan(module)
    settings['execution_sources'][module.SOURCE_FILES[0]] = '0' * 64
    source, output = tmp_path / 'plan.json', tmp_path / 'must_not_exist'
    source.write_text(json.dumps(settings))
    with pytest.raises(ValueError, match='source hash differs'):
        module.main(['--plan', str(source), '--output', str(output)])
    assert not output.exists()


def test_fatal_numerical_failure_keeps_hashes_of_partial_output_receipts(tmp_path):
    module = verifier()
    settings = bound_tiny_plan(module)
    for scenario in settings['scenarios'][:2]:
        scenario['forecast_a']['offset'] = 1e308
    source, output = tmp_path / 'plan.json', tmp_path / 'failed'
    source.write_text(json.dumps(settings))
    with np.errstate(over='raise', invalid='raise'), pytest.raises(FloatingPointError):
        module.main(['--plan', str(source), '--output', str(output)])
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['status'] == 'failed'
    assert manifest['error_type'] == 'FloatingPointError'
    assert set(manifest['files']) == {'plan.json', 'weak_normal_cohorts.jsonl'}
    for name, digest in manifest['files'].items():
        assert digest == hashlib.sha256((output / name).read_bytes()).hexdigest()
