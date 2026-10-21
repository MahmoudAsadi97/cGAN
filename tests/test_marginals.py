"""Conditional TRAIN marginals retain signed atoms and exact empirical quantiles."""
import hashlib
import json

import numpy as np
import pytest


def core():
    from fluvius_cgan import marginals
    return marginals


def example():
    return np.array([[-2., 7.], [0., 7.], [0., 7.], [3., 7.]]), np.zeros(4, dtype=np.int64)


def test_intervals_partition_empirical_mass_without_arbitrary_tie_ranks():
    # Assigning separate ranks to identical readings would break equal intervals.
    training, labels = example()
    fitted = core().fit_marginals(training, labels)
    lower, upper = fitted.intervals(training, labels)
    np.testing.assert_array_equal(lower, [[0., 0.], [.25, 0.], [.25, 0.], [.75, 0.]])
    np.testing.assert_array_equal(upper, [[.25, 1.], [.75, 1.], [.75, 1.], [1., 1.]])
    assert lower.dtype == upper.dtype == np.float64
    distinct = [0, 1, 3]
    np.testing.assert_array_equal(upper[distinct[:-1], 0], lower[distinct[1:], 0])
    np.testing.assert_array_equal(upper[distinct, 0] - lower[distinct, 0], [.25, .5, .25])
    assert sum(upper[distinct, 0] - lower[distinct, 0]) == 1.


def test_intervals_of_unseen_values_are_frozen_cdf_gaps_not_reranked():
    training, labels = example()
    fitted = core().fit_marginals(training, labels)
    query = np.array([[-3., 7.], [-1., 6.], [1., 8.], [4., 7.]])
    lower, upper = fitted.intervals(query, labels)
    np.testing.assert_array_equal(lower, [[0., 0.], [.25, 0.], [.75, 1.], [1., 0.]])
    np.testing.assert_array_equal(upper, [[0., 1.], [.25, 0.], [.75, 1.], [1., 1.]])
    for i in range(len(query)):
        single = fitted.intervals(query[i:i+1], labels[i:i+1])
        np.testing.assert_array_equal(single[0], lower[i:i+1])
        np.testing.assert_array_equal(single[1], upper[i:i+1])


def test_randomized_training_intervals_invert_to_original_atoms():
    # Interpolating target quantiles would erase zeros and change these values.
    training, labels = example()
    fitting = np.concatenate([training, training + 50.])
    fitting_labels = np.concatenate([labels, labels + 8])
    fitted = core().fit_marginals(fitting, fitting_labels)
    lower, upper = fitted.intervals(fitting, fitting_labels)
    jitter = np.random.default_rng(2026).random(lower.shape)
    assert np.all((jitter > 0) & (jitter < 1))
    actual = fitted.inverse(lower + jitter * (upper - lower), fitting_labels)
    np.testing.assert_array_equal(actual, fitting)
    assert fitted.profile_dim == 2
    assert fitted.conditions == (0, 8)


def test_step_inverse_retains_exact_boundary_convention_and_observed_extrema():
    training, labels = example()
    fitted = core().fit_marginals(training, labels)
    probability = np.array([[0., 0.], [.25, .25], [.250001, .5], [.75, .75], [1., 1.]])
    actual = fitted.inverse(probability, [0] * 5)
    np.testing.assert_array_equal(actual, [[-2., 7.], [-2., 7.], [0., 7.], [0., 7.], [3., 7.]])
    assert actual.dtype == np.float64
    # Exactly zero jitter at a positive lower bound takes the preceding atom.
    lower, _ = fitted.intervals(training, labels)
    np.testing.assert_array_equal(fitted.inverse(lower, labels)[:, 0], [-2., -2., -2., 0.])


def test_inverse_is_condition_specific_batch_invariant_and_owns_training_arrays():
    training, labels = example()
    values = np.concatenate([training, training + 50.])
    conditions = np.concatenate([labels, labels + 8])
    fitted = core().fit_marginals(values, conditions)
    probability = np.array([[.9, .2], [.4, .8], [.1, .7], [.4, .3]])
    requested = np.array([8, 0, 8, 8])
    original = probability.copy()
    actual = fitted.inverse(probability, requested)
    np.testing.assert_array_equal(actual, [[53., 57.], [0., 7.], [48., 57.], [50., 57.]])
    values[:] = 999.
    np.testing.assert_array_equal(fitted.inverse(probability, requested), actual)
    np.testing.assert_array_equal(probability, original)
    pieces = np.concatenate([fitted.inverse(probability[i:i+1], requested[i:i+1]) for i in range(4)])
    np.testing.assert_array_equal(pieces, actual)
    permutation = [3, 1, 0, 2]
    np.testing.assert_array_equal(fitted.inverse(probability[permutation], requested[permutation]),
                                  actual[permutation])


def test_summary_counts_atoms_constants_and_returns_independent_json_data():
    training, labels = example()
    fitted = core().fit_marginals(training, labels)
    summary = fitted.summary()
    json.dumps(summary, allow_nan=False)
    assert summary['profile_dim'] == 2 and summary['conditions'] == [0]
    group = summary['per_condition']['0']
    assert group == {'training_profiles': 4, 'constant_hours': [1],
                     'duplicate_counts_by_hour': [1, 3], 'zero_counts': [2, 0],
                     'negative_counts': [1, 0], 'min_counts': [1, 4], 'max_counts': [1, 4],
                     'minimum': [-2., 7.], 'maximum': [3., 7.]}
    group['zero_counts'][0] = 999
    assert fitted.summary()['per_condition']['0']['zero_counts'] == [2, 0]


@pytest.mark.parametrize('training,labels', [
    ([[np.nan]], [0]), ([[np.inf]], [0]), ([[1. + 2.j]], [0]),
    (np.empty((0, 2)), []), ([1., 2.], [0, 0]),
    ([[0.]], [64]), ([[0.]], [-1]), ([[0.]], [.5]), ([[0.]], [True]),
    ([[0.]], ['0']), ([[0.]], [0, 0]),
])
def test_invalid_fitting_inputs_fail(training, labels):
    with pytest.raises(ValueError):
        core().fit_marginals(training, labels)


@pytest.mark.parametrize('values,labels', [
    ([[np.inf, .5]], [0]), ([[np.nan, .5]], [0]),
    ([[0.]], [0]), (np.empty((0, 2)), []),
    ([[0., .5]], [8]), ([[0., .5]], [0, 0]),
    ([[0., .5]], [False]), ([[0., .5]], [64]),
])
@pytest.mark.parametrize('method', ['intervals', 'inverse'])
def test_invalid_application_inputs_fail(method, values, labels):
    training, conditions = example()
    fitted = core().fit_marginals(training, conditions)
    with pytest.raises(ValueError):
        getattr(fitted, method)(values, labels)


@pytest.mark.parametrize('value', [-1e-16, np.nextafter(1., np.inf)])
def test_inverse_never_clips_probabilities_outside_unit_interval(value):
    training, conditions = example()
    fitted = core().fit_marginals(training, conditions)
    with pytest.raises(ValueError, match=r'\[0,1\]'):
        fitted.inverse([[value, .5]], [0])


def test_save_load_requires_hash_and_never_overwrites(tmp_path):
    training, labels = example()
    fitted = core().fit_marginals(training, labels)
    path = tmp_path / 'marginals.npz'
    fitted.save(path)
    contents = path.read_bytes()
    digest = hashlib.sha256(contents).hexdigest()
    restored = core().load_marginals(path, digest)
    assert restored.summary() == fitted.summary()
    np.testing.assert_array_equal(restored.intervals(training, labels), fitted.intervals(training, labels))
    np.testing.assert_array_equal(restored.inverse([[0., 1.]], [0]), fitted.inverse([[0., 1.]], [0]))
    with np.load(path, allow_pickle=False) as arrays:
        assert set(arrays.files) == {'metadata', 'target_0'}
        assert arrays['target_0'].dtype == np.float64
    with pytest.raises(FileExistsError):
        fitted.save(path)
    assert path.read_bytes() == contents
    with pytest.raises(ValueError, match='SHA-256'):
        core().load_marginals(path, None)
    with pytest.raises(ValueError, match='SHA-256'):
        core().load_marginals(path, '0' * 64)
    path.write_bytes(contents + b'tamper')
    with pytest.raises(ValueError, match='SHA-256'):
        core().load_marginals(path, digest)


@pytest.mark.parametrize('change', ['unsorted', 'nonfinite', 'empty', 'missing', 'extra',
                                    'wrong_shape', 'wrong_schema', 'wrong_labels', 'object_dtype'])
def test_invalid_saved_schema_fails_even_with_matching_hash(tmp_path, change):
    training, labels = example()
    path = tmp_path / 'marginals.npz'
    core().fit_marginals(training, labels).save(path)
    with np.load(path, allow_pickle=False) as stored:
        arrays = {name: stored[name].copy() for name in stored.files}
    metadata = json.loads(arrays['metadata'].item())
    if change == 'unsorted':
        arrays['target_0'] = arrays['target_0'][::-1]
    elif change == 'nonfinite':
        arrays['target_0'][0, 0] = np.nan
    elif change == 'empty':
        arrays['target_0'] = np.empty((0, 2), dtype=np.float64)
    elif change == 'missing':
        arrays.pop('target_0')
    elif change == 'extra':
        arrays['target_8'] = training
    elif change == 'wrong_shape':
        arrays['target_0'] = arrays['target_0'][:, :1]
    elif change == 'wrong_schema':
        metadata['schema_version'] = 999
    elif change == 'wrong_labels':
        metadata['conditions'] = [64]
    elif change == 'object_dtype':
        arrays['target_0'] = arrays['target_0'].astype(object)
    arrays['metadata'] = np.asarray(json.dumps(metadata))
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError):
        core().load_marginals(path, hashlib.sha256(path.read_bytes()).hexdigest())
