"""Meter aggregation and paired studentization preserve declared sampling units."""
import hashlib
import json

import numpy as np
import pytest


def core():
    from fluvius_cgan import meter_uncertainty
    return meter_uncertainty


def small_values():
    strata, rows = [], []
    for group in range(8):
        for index in range(group + 2):
            value = index - (group + 1) / 2.
            rows.append([value + group / 10., value ** 2 - group / 3.])
            strata.append(group)
    return np.asarray(rows), np.asarray(strata)


def multiplicities(strata, seed, replicates):
    weights = np.zeros((replicates, len(strata)), dtype=np.int64)
    for group, child in enumerate(np.random.SeedSequence(seed).spawn(8)):
        positions = np.flatnonzero(strata == group)
        rng = np.random.default_rng(child)
        weights[:, positions] = rng.multinomial(len(positions), np.full(len(positions), 1. / len(positions)),
                                                size=replicates)
    return weights


def expanded_reference(values, strata, weights, level):
    means, variances = [], []
    for group in range(8):
        members = values[strata == group]
        means.append(members.mean(axis=0))
        variances.append(members.var(axis=0, ddof=1) / len(members))
    estimate = np.mean(means, axis=0)
    standard_error = np.sqrt(np.sum(variances, axis=0) / 64.)
    statistics = []
    for draw in weights:
        draw_means, draw_variances = [], []
        for group in range(8):
            selected = strata == group
            expanded = np.repeat(values[selected], draw[selected], axis=0)
            draw_means.append(expanded.mean(axis=0))
            draw_variances.append(expanded.var(axis=0, ddof=1) / len(expanded))
        draw_se = np.sqrt(np.sum(draw_variances, axis=0) / 64.)
        assert np.all(draw_se > 0)
        statistics.append((np.mean(draw_means, axis=0) - estimate) / draw_se)
    alpha = (1. - level) / 2.
    quantiles = np.quantile(statistics, [alpha, 1 - alpha], axis=0, method='linear')
    intervals = np.column_stack([estimate - quantiles[1] * standard_error,
                                 estimate - quantiles[0] * standard_error])
    return estimate, standard_error, quantiles.T, intervals


def test_meter_means_weight_calendar_bins_equally_instead_of_pooling_days():
    differences, identifiers, conditions = [], [], []
    for calendar_bin in range(8):
        for _ in range(calendar_bin + 1):
            differences.append([calendar_bin, calendar_bin ** 2])
            identifiers.append('a')
            conditions.append(16 + calendar_bin)
    for calendar_bin in range(8):
        differences.append([100., -100.])
        identifiers.append(17)
        conditions.append(calendar_bin)
    result = core().meter_contributions(differences, identifiers, conditions)
    assert result['meter_ids'] == ('a', 17)
    np.testing.assert_array_equal(result['strata'], [2, 0])
    np.testing.assert_array_equal(result['values'], [[3.5, 17.5], [100., -100.]])
    np.testing.assert_array_equal(result['bin_counts'], [np.arange(1, 9), np.ones(8, dtype=int)])
    np.testing.assert_array_equal(result['bin_sums'][0, :, 0], np.arange(8) * np.arange(1, 9))
    np.testing.assert_allclose((result['bin_sums'] / result['bin_counts'][:, :, None]).mean(axis=1),
                              result['values'])
    assert result['values'][0, 0] != np.mean(np.asarray(differences)[:36, 0])
    assert result['input_was_vector'] is False
    assert not result['values'].flags.writeable and not result['bin_counts'].flags.writeable


def test_meter_ids_keep_first_appearance_and_distinguish_strings_from_numbers():
    differences = np.r_[np.arange(8.), np.arange(8.) + 10.]
    identifiers = ['1'] * 8 + [1.] * 8
    conditions = list(range(8)) * 2
    result = core().meter_contributions(differences, identifiers, conditions)
    assert result['meter_ids'] == ('1', 1.)
    assert result['input_was_vector'] is True
    np.testing.assert_array_equal(result['values'], [[3.5], [13.5]])
    differences[:] = 999
    np.testing.assert_array_equal(result['values'], [[3.5], [13.5]])
    numeric = core().meter_contributions(np.arange(8.), [1, 1., 1, 1., 1, 1., 1, 1.], list(range(8)))
    assert numeric['meter_ids'] == (1,)


def test_inconsistent_equipment_or_missing_calendar_bin_is_an_explicit_failure():
    with pytest.raises(ValueError, match='equipment'):
        core().meter_contributions(np.arange(8.), ['a'] * 8, [0, 1, 2, 3, 4, 5, 6, 15])
    with pytest.raises(ValueError, match='calendar'):
        core().meter_contributions(np.arange(7.), ['a'] * 7, list(range(7)))


def test_bootstrap_matches_expanded_paired_meter_resamples_and_hash():
    values, strata = small_values()
    seed, replicates, level = 20261028, 67, .9
    weights = multiplicities(strata, seed, replicates)
    expected = expanded_reference(values, strata, weights, level)
    result = core().stratified_studentized_bootstrap(values, strata, seed, replicates, level, block_size=7)
    assert result['status'] == ['ok', 'ok']
    assert result['bootstrap_zero_standard_error_counts'] == [0, 0]
    np.testing.assert_allclose(result['estimate'], expected[0], rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(result['standard_error'], expected[1], rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(result['studentized_quantiles'], expected[2], rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(result['interval'], expected[3], rtol=1e-13, atol=1e-13)
    assert result['multiplicity_sha256'] == hashlib.sha256(weights.astype('<i8').tobytes()).hexdigest()
    assert result['stratum_sizes'] == list(range(2, 10))
    assert result['quantile_method'] == 'linear'
    json.dumps(result, allow_nan=False)


def test_columns_share_draws_and_block_size_only_changes_working_memory():
    values, strata = small_values()
    first = values[:, 0]
    paired = np.column_stack([first, 2 * first, 5 - first])
    result = core().stratified_studentized_bootstrap(paired, strata, 19, replicates=83, block_size=3)
    other = core().stratified_studentized_bootstrap(paired, strata, 19, replicates=83, block_size=17)
    assert result['multiplicity_sha256'] == other['multiplicity_sha256']
    for name in ('estimate', 'standard_error', 'interval', 'studentized_quantiles'):
        np.testing.assert_array_equal(result[name], other[name])
    np.testing.assert_allclose(result['interval'][1], 2 * np.array(result['interval'][0]), rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(result['interval'][2], 5 - np.array(result['interval'][0])[::-1], rtol=1e-13, atol=1e-13)
    single = core().stratified_studentized_bootstrap(first, strata, 19, replicates=83, block_size=3)
    assert single['input_was_vector'] is True
    assert single['multiplicity_sha256'] == result['multiplicity_sha256']
    np.testing.assert_allclose(single['interval'][0], result['interval'][0], rtol=1e-13, atol=1e-13)


def test_degenerate_and_failed_studentization_columns_do_not_hide_resamples():
    values, strata = small_values()
    local = np.where(strata == 0, values[:, 0], 0.)
    constant_in_each_stratum = strata.astype(float) + .1
    columns = np.column_stack([local, values[:, 0], constant_in_each_stratum])
    result = core().stratified_studentized_bootstrap(columns, strata, 8, replicates=67, block_size=9)
    assert result['status'] == ['failed_studentization', 'ok', 'degenerate_observed_variance']
    assert result['interval'][0] is None and result['studentized_quantiles'][0] is None
    assert result['interval'][2] is None and result['studentized_quantiles'][2] is None
    assert result['interval'][1] is not None
    weights = multiplicities(strata, 8, 67)
    expected_bad = int(np.count_nonzero(np.any(weights[:, strata == 0] == 2, axis=1)))
    assert result['bootstrap_zero_standard_error_counts'] == [expected_bad, 0, 67]
    assert result['standard_error'][2] == 0.
    json.dumps(result, allow_nan=False)


def test_large_shared_offsets_preserve_studentization_and_local_rng_state():
    values, strata = small_values()
    state = np.random.get_state()
    first = core().stratified_studentized_bootstrap(values, strata, 24, replicates=31, block_size=4)
    after = np.random.get_state()
    assert state[0] == after[0] and state[2:] == after[2:]
    np.testing.assert_array_equal(state[1], after[1])
    # Use exactly representable offsets and observations to avoid input rounding.
    integer_values = np.round(values * 12)
    unshifted = core().stratified_studentized_bootstrap(integer_values, strata, 24, replicates=31, block_size=4)
    shifted = core().stratified_studentized_bootstrap(integer_values + 2. ** 40, strata, 24, replicates=31, block_size=4)
    np.testing.assert_array_equal(shifted['standard_error'], unshifted['standard_error'])
    np.testing.assert_array_equal(shifted['studentized_quantiles'], unshifted['studentized_quantiles'])
    np.testing.assert_allclose(np.array(shifted['estimate']) - 2. ** 40, unshifted['estimate'], atol=2. ** -12, rtol=0.)
    assert first['scope'].startswith('Conditional fixed score functions')


@pytest.mark.parametrize('differences,ids,conditions', [
    (np.arange(8.), ['a'] * 7, list(range(8))),
    (np.r_[np.nan, np.arange(7.)], ['a'] * 8, list(range(8))),
    (np.arange(8.), [None] * 8, list(range(8))),
    (np.arange(8.), [True] * 8, list(range(8))),
    (np.arange(8.), [np.inf] * 8, list(range(8))),
    (np.arange(8.), [''] * 8, list(range(8))),
    (np.arange(8.), ['a'] * 8, [64] * 8),
    (np.arange(8.), ['a'] * 8, [.5] * 8),
    (np.empty((8, 0)), ['a'] * 8, list(range(8))),
])
def test_invalid_profile_inputs_fail(differences, ids, conditions):
    with pytest.raises(ValueError):
        core().meter_contributions(differences, ids, conditions)


@pytest.mark.parametrize('setting', [
    {'seed': -1}, {'seed': True}, {'seed': 2.5}, {'replicates': 1}, {'replicates': 3.5},
    {'level': 0.}, {'level': 1.}, {'level': np.nan}, {'level': True},
    {'block_size': 0}, {'block_size': True},
])
def test_invalid_bootstrap_settings_fail(setting):
    values, strata = small_values()
    arguments = dict(seed=1, replicates=3, level=.95, block_size=2)
    arguments.update(setting)
    with pytest.raises(ValueError):
        core().stratified_studentized_bootstrap(values, strata, **arguments)


@pytest.mark.parametrize('case', ['missing_stratum', 'single_meter', 'bad_labels', 'bool_labels',
                                  'nonfinite', 'no_columns', 'wrong_length'])
def test_invalid_meter_inputs_fail(case):
    values, strata = small_values()
    if case == 'missing_stratum':
        values, strata = values[strata != 7], strata[strata != 7]
    elif case == 'single_meter':
        values, strata = values[1:], strata[1:]
    elif case == 'bad_labels':
        strata = strata.astype(float)
        strata[0] = .5
    elif case == 'bool_labels':
        strata = strata.astype(bool)
    elif case == 'nonfinite':
        values[0, 0] = np.inf
    elif case == 'no_columns':
        values = values[:, :0]
    elif case == 'wrong_length':
        strata = strata[:-1]
    with pytest.raises(ValueError):
        core().stratified_studentized_bootstrap(values, strata, 1, replicates=3)


def test_finite_inputs_that_overflow_derived_moments_fail_explicitly():
    values, strata = small_values()
    values[:2, 0] = [-1e308, 1e308]
    with pytest.raises(FloatingPointError):
        core().stratified_studentized_bootstrap(values, strata, 1, replicates=3)
