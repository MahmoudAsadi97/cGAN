"""Conditional train-neighbor distance checks on synthetic profiles only."""

import numpy as np
import pytest


def _fit(training, labels):
    from fluvius_cgan.proximity import ConditionalNearestNeighbors

    return ConditionalNearestNeighbors(training, labels)


def _brute_force(training, training_labels, query, query_labels):
    return np.array([
        min(np.sqrt(np.mean((row - candidate) ** 2))
            for candidate in training[training_labels == label])
        for row, label in zip(query, query_labels, strict=True)
    ])


def test_known_translation_uses_rms_and_same_condition_only():
    training = np.array([[0.0, 0.0], [30.0, 40.0], [3.0, 4.0]])
    fitted = _fit(training, [0, 0, 1])
    query = np.array([[3.0, 4.0], [6.0, 8.0], [30.0, 40.0]])
    actual = fitted.distances(query, [0, 1, 0])
    np.testing.assert_allclose(actual, [5 / np.sqrt(2), 5 / np.sqrt(2), 0])
    assert actual.dtype == np.float64
    assert actual.shape == (3,)


def test_distances_equal_direct_brute_force_for_mixed_conditions():
    rng = np.random.default_rng(314)
    training = rng.normal(size=(31, 7))
    training_labels = np.resize([0, 7, 63], len(training))
    query = rng.normal(size=(23, 7))
    query_labels = np.resize([63, 0, 7], len(query))
    fitted = _fit(training, training_labels)
    np.testing.assert_allclose(
        fitted.distances(query, query_labels),
        _brute_force(training, training_labels, query, query_labels),
        rtol=1e-14, atol=0,
    )


def test_exact_training_resampling_has_zero_distance():
    rng = np.random.default_rng(15)
    training = rng.normal(size=(24, 5))
    labels = np.resize([0, 63], len(training))
    fitted = _fit(training, labels)
    indices = rng.integers(len(training), size=80)
    np.testing.assert_array_equal(
        fitted.distances(training[indices], labels[indices]), np.zeros(80)
    )


def test_distances_are_invariant_to_query_batches_and_order():
    rng = np.random.default_rng(60)
    training = rng.normal(size=(12, 3))
    training_labels = np.resize([2, 5], len(training))
    query = rng.normal(size=(303, 3))
    query_labels = np.resize([5, 2], len(query))
    fitted = _fit(training, training_labels)
    expected = fitted.distances(query, query_labels)
    singleton = np.array([
        fitted.distances(query[i:i + 1], query_labels[i:i + 1])[0]
        for i in range(len(query))
    ])
    np.testing.assert_array_equal(singleton, expected)
    permutation = rng.permutation(len(query))
    np.testing.assert_array_equal(
        fitted.distances(query[permutation], query_labels[permutation]),
        expected[permutation],
    )


def test_fitted_training_is_owned_and_query_inputs_are_unchanged():
    training = np.array([[0.0, 0.0], [10.0, 10.0]])
    training_labels = np.array([0, 1])
    fitted = _fit(training, training_labels)
    training[:] = -999
    training_labels[:] = 63
    query = np.array([[3.0, 4.0], [7.0, 6.0]])
    query_labels = np.array([0, 1])
    before_query, before_labels = query.copy(), query_labels.copy()
    query.setflags(write=False)
    query_labels.setflags(write=False)
    np.testing.assert_allclose(fitted.distances(query, query_labels),
                               [5 / np.sqrt(2)] * 2)
    np.testing.assert_array_equal(query, before_query)
    np.testing.assert_array_equal(query_labels, before_labels)


def test_training_metadata_and_integral_numeric_labels():
    fitted = _fit([[0, 1], [2, 3], [4, 5]], [63.0, 0.0, 63.0])
    assert fitted.training_rows == 3
    assert fitted.profile_dim == 2
    assert fitted.conditions == (0, 63)
    assert dict(fitted.counts) == {0: 1, 63: 2}
    np.testing.assert_array_equal(fitted.distances([[4, 5]], [63.0]), [0.0])


@pytest.mark.parametrize("profiles", [
    [1.0, 2.0], np.empty((0, 2)), np.empty((1, 0)),
    [[[1.0, 2.0]]], [[np.nan, 0.0]], [[np.inf, 0.0]],
    [[-np.inf, 0.0]], [[1 + 0j, 2 + 0j]], [["1", "2"]],
    [[True, False]], [[object(), object()]],
])
def test_invalid_training_profiles_are_rejected(profiles):
    with pytest.raises(ValueError, match="training"):
        _fit(profiles, [0])


@pytest.mark.parametrize("labels", [
    [0, 1], [], [[0]], 0, [-1], [64], [0.5], [np.nan],
    [np.inf], [True], ["0"], [0 + 0j], [object()],
])
def test_invalid_training_labels_are_rejected(labels):
    with pytest.raises(ValueError, match="condition labels"):
        _fit([[1.0, 2.0]], labels)


@pytest.mark.parametrize("query", [
    [1.0, 2.0], np.empty((0, 2)), np.empty((1, 0)),
    [[[1.0, 2.0]]], [[np.nan, 0.0]], [[np.inf, 0.0]],
    [[1 + 0j, 2 + 0j]], [["1", "2"]], [[True, False]],
])
def test_invalid_query_profiles_are_rejected(query):
    fitted = _fit([[1.0, 2.0]], [0])
    with pytest.raises(ValueError, match="query"):
        fitted.distances(query, [0])


@pytest.mark.parametrize("labels", [
    [0, 1], [], [[0]], 0, [-1], [64], [0.5], [np.nan],
    [np.inf], [True], ["0"], [0 + 0j], [object()],
])
def test_invalid_query_labels_are_rejected(labels):
    fitted = _fit([[1.0, 2.0]], [0])
    with pytest.raises(ValueError, match="condition labels"):
        fitted.distances([[1.0, 2.0]], labels)


def test_unsupported_query_condition_is_rejected():
    fitted = _fit([[1.0, 2.0]], [0])
    with pytest.raises(ValueError, match="unsupported condition labels.*63"):
        fitted.distances([[1.0, 2.0]], [63])


def test_query_profile_dimension_must_match_training():
    fitted = _fit([[1.0, 2.0]], [0])
    with pytest.raises(ValueError, match="dimension"):
        fitted.distances([[1.0, 2.0, 3.0]], [0])


def test_singleton_query_raises_when_every_training_distance_overflows():
    fitted = _fit([[1e308, 0.0], [1e308, 1.0]], [0, 0])
    with pytest.raises(FloatingPointError, match="nearest.*overflow"):
        fitted.distances([[-1e308, 0.0]], [0])


def test_singleton_query_uses_finite_nearest_despite_distant_overflow():
    fitted = _fit([[-1e308, 0.0], [1e308, 1.0]], [0, 0])
    np.testing.assert_allclose(
        fitted.distances([[1e308, 5.0]], [0]), [4 / np.sqrt(2)],
    )
