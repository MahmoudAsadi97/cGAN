"""Frozen marginal maps must not invent diversity or rerank evaluation batches."""
import hashlib
import json

import numpy as np
import pytest


def core():
    from fluvius_cgan import calibration
    return calibration


def example():
    target = np.array([[-.25, 4.], [0., 3.], [0., 2.], [1., 1.], [2., 0.]])
    source = np.array([[10., 40.], [20., 30.], [20., 20.], [30., 10.], [40., 0.]])
    return target, source, np.zeros(5, dtype=np.int64)


def test_inverse_monotone_distortion_recovers_empirical_values_and_zero_atom():
    # A linear/interpolated target inverse would fail the atom assertions.
    target, source, labels = example()
    fitted = core().fit_calibration(target, labels, source, labels)
    actual, diagnostics = fitted.apply(source, labels)
    np.testing.assert_array_equal(actual, target)
    assert actual.dtype == np.float64
    assert np.count_nonzero(actual[:, 0] == 0) == 2
    assert diagnostics['per_condition']['0']['mapped_target_zero'] == [2, 1]
    assert diagnostics['per_condition']['0']['training_zero_counts'] == [2, 1]
    assert diagnostics['per_condition']['0']['training_negative_counts'] == [1, 0]
    assert diagnostics['per_condition']['0']['training_min_counts'] == [1, 1]
    assert diagnostics['per_condition']['0']['training_max_counts'] == [1, 1]
    json.dumps(diagnostics, allow_nan=False)
    diagnostics['per_condition']['0']['training_zero_counts'][0] = 999
    assert fitted.apply(source, labels)[1]['per_condition']['0']['training_zero_counts'] == [2, 1]


def test_midrank_ties_and_boundaries_preserve_rare_signed_support():
    # Arbitrary tie ranks or a nonnegative clamp would change these outputs.
    target = np.array([[-.25], [0.], [0.], [1.], [2.]])
    source = np.array([[10.], [20.], [20.], [30.], [40.]])
    fitted = core().fit_calibration(target, [0] * 5, source, [0] * 5)
    queries = np.array([[9.], [10.], [15.], [20.], [25.], [30.], [40.], [41.]])
    actual, diagnostic = fitted.apply(queries, [0] * 8)
    np.testing.assert_array_equal(actual[:, 0], [-.25, -.25, -.25, 0., 0., 1., 2., 2.])
    assert np.all(np.diff(actual[:, 0]) >= 0)
    group = diagnostic['per_condition']['0']
    assert group['below_source_min'] == [1]
    assert group['above_source_max'] == [1]
    assert group['equal_source_min'] == [1]
    assert group['equal_source_max'] == [1]
    assert group['source_tie_matches'] == [1]
    assert group['source_value_matches'] == [4]
    assert group['source_duplicate_counts_by_hour'] == [1]
    assert group['probability_zero'] == [1]
    assert group['probability_one'] == [1]
    assert group['mapped_target_min'] == [3]
    assert group['mapped_target_max'] == [2]
    assert group['mapped_target_negative'] == [3]
    assert group['probability_decile_counts'] == [[1, 1, 1, 0, 1, 1, 1, 0, 0, 2]]
    assert group['probability_mean'] == [0.4875]
    assert diagnostic['probability_decile_edges'] == np.linspace(0, 1, 11).tolist()


def test_map_is_condition_specific_and_independent_of_evaluation_batch():
    # Recomputing ranks from queries would break singleton/chunk invariance.
    target, source, labels = example()
    training = np.concatenate([target, target + 100])
    generated = np.concatenate([source, source - 50])
    conditions = np.concatenate([labels, labels + 8])
    fitted = core().fit_calibration(training, conditions, generated, conditions)
    queries = np.array([[25., 5.], [-25., -45.], [10., 40.], [-40., -10.]])
    requested = np.array([0, 8, 0, 8])
    whole, _ = fitted.apply(queries, requested)
    singles = np.concatenate([fitted.apply(queries[i:i+1], requested[i:i+1])[0]
                              for i in range(len(queries))])
    np.testing.assert_array_equal(whole, singles)
    np.testing.assert_array_equal(whole[0] + 100, whole[1])
    np.testing.assert_array_equal(whole[2] + 100, whole[3])
    permutation = np.array([3, 1, 0, 2])
    shuffled, _ = fitted.apply(queries[permutation], requested[permutation])
    np.testing.assert_array_equal(shuffled, whole[permutation])
    # Fitting owns its arrays and applying must not mutate queries.
    original = queries.copy()
    training[:] = -999
    generated[:] = 999
    np.testing.assert_array_equal(fitted.apply(queries, requested)[0], whole)
    np.testing.assert_array_equal(queries, original)


def test_constant_target_is_fixed_but_collapsed_source_cannot_invent_spread():
    fitted = core().fit_calibration([[3., 0.], [3., 0.]], [4, 4],
                                    [[1., -1.], [1., 1.]], [4, 4])
    actual, diagnostic = fitted.apply([[-999., 0.], [999., 5.]], [4, 4])
    np.testing.assert_array_equal(actual, [[3., 0.], [3., 0.]])
    assert diagnostic['per_condition']['4']['source_constant_hours'] == [0]
    assert diagnostic['per_condition']['4']['target_constant_hours'] == [0, 1]
    assert diagnostic['per_condition']['4']['training_zero_counts'] == [0, 2]
    assert diagnostic['per_condition']['4']['training_negative_counts'] == [0, 0]
    assert diagnostic['per_condition']['4']['training_min_counts'] == [2, 2]
    assert diagnostic['per_condition']['4']['training_max_counts'] == [2, 2]
    with pytest.raises(ValueError, match='constant.*varying'):
        core().fit_calibration([[0.], [1.]], [0, 0], [[5.], [5.]], [0, 0])


@pytest.mark.parametrize('training,train_labels,generated,generated_labels', [
    ([[np.nan]], [0], [[1.]], [0]),
    ([[0.]], [0], [[np.inf]], [0]),
    (np.empty((0, 2)), [], [[1., 2.]], [0]),
    ([[0.]], [0], [[1., 2.]], [0]),
    ([[0.]], [0], [[1.]], [8]),
    ([[0.]], [-1], [[1.]], [-1]),
    ([[0.]], [64], [[1.]], [64]),
    ([[0.]], [.5], [[1.]], [.5]),
    ([[0.]], [True], [[1.]], [True]),
    ([[0.]], ['0'], [[1.]], ['0']),
    ([[0.]], [0, 0], [[1.]], [0]),
])
def test_invalid_fitting_data_fail(training, train_labels, generated, generated_labels):
    with pytest.raises(ValueError):
        core().fit_calibration(training, train_labels, generated, generated_labels)


@pytest.mark.parametrize('values,labels', [
    ([[np.inf, 1.]], [0]), ([[1.]], [0]),
    ([[1., 2.]], [8]), ([[1., 2.]], [0, 0]),
    ([[1., 2.]], [False]), (np.empty((0, 2)), []),
])
def test_invalid_application_data_fail(values, labels):
    target, source, conditions = example()
    fitted = core().fit_calibration(target, conditions, source, conditions)
    with pytest.raises(ValueError):
        fitted.apply(values, labels)


def test_save_load_requires_matching_hash_and_never_overwrites(tmp_path):
    target, source, labels = example()
    fitted = core().fit_calibration(target, labels, source, labels)
    path = tmp_path / 'map.npz'
    fitted.save(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    restored = core().load_calibration(path, digest)
    actual, diagnostics = restored.apply(source, labels)
    np.testing.assert_array_equal(actual, fitted.apply(source, labels)[0])
    assert diagnostics == fitted.apply(source, labels)[1]
    with np.load(path, allow_pickle=False) as stored:
        assert set(stored.files) == {'metadata', 'source_0', 'target_0'}
        assert stored['source_0'].dtype == np.float64
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        fitted.save(path)
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match='SHA-256'):
        core().load_calibration(path, '0' * 64)
    with pytest.raises(ValueError, match='SHA-256'):
        core().load_calibration(path, None)
    path.write_bytes(original + b'tampering')
    with pytest.raises(ValueError, match='SHA-256'):
        core().load_calibration(path, digest)


@pytest.mark.parametrize('change', ['unsorted', 'nonfinite', 'missing', 'extra', 'wrong_shape',
                                    'wrong_schema', 'wrong_labels', 'object_dtype', 'degenerate'])
def test_loaded_map_rejects_invalid_array_schema_even_with_matching_hash(tmp_path, change):
    target, source, labels = example()
    path = tmp_path / 'map.npz'
    core().fit_calibration(target, labels, source, labels).save(path)
    with np.load(path, allow_pickle=False) as stored:
        arrays = {name: stored[name].copy() for name in stored.files}
    metadata = json.loads(arrays['metadata'].item())
    if change == 'unsorted':
        arrays['source_0'] = arrays['source_0'][::-1]
    elif change == 'nonfinite':
        arrays['target_0'][0, 0] = np.nan
    elif change == 'missing':
        arrays.pop('source_0')
    elif change == 'extra':
        arrays['source_8'] = source
    elif change == 'wrong_shape':
        arrays['target_0'] = arrays['target_0'][:, :1]
    elif change == 'wrong_schema':
        metadata['schema_version'] = 999
    elif change == 'wrong_labels':
        metadata['conditions'] = [64]
    elif change == 'object_dtype':
        arrays['source_0'] = arrays['source_0'].astype(object)
    elif change == 'degenerate':
        arrays['source_0'][:] = 0
    arrays['metadata'] = np.asarray(json.dumps(metadata))
    np.savez_compressed(path, **arrays)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        core().load_calibration(path, digest)
