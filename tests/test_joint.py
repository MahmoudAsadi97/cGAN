"""Independent checks of the fixed whole-profile projection diagnostic."""
import hashlib
import json

import numpy as np
import pytest
from scipy.stats import wasserstein_distance

from fluvius_cgan.joint import conditional_projected_wasserstein


def directions(width):
    values = np.random.default_rng(9182).normal(size=(128, width))
    return values / np.linalg.norm(values, axis=1, keepdims=True)


def test_known_translation_matches_mean_absolute_projected_shift_and_frozen_protocol():
    rng = np.random.default_rng(73)
    real = rng.normal(size=(13, 96))
    shift = np.linspace(-.4, .2, 96)
    result = conditional_projected_wasserstein(real, np.zeros(13, dtype=int),
                                              real + shift, np.zeros(13, dtype=int))
    expected_directions = directions(96)
    expected = np.mean(np.abs(expected_directions @ shift))
    assert result["macro"] == pytest.approx(expected)
    assert result["per_condition"]["0"]["mean_projected_wasserstein_kwh"] == pytest.approx(expected)
    assert result["directions"]["seed"] == 9182
    assert result["directions"]["count"] == 128
    assert result["directions"]["n_features"] == 96
    assert result["directions"]["sha256"] == hashlib.sha256(expected_directions.astype("<f8").tobytes()).hexdigest()
    np.testing.assert_allclose(np.linalg.norm(expected_directions, axis=1), 1)
    json.dumps(result, allow_nan=False)


def test_deterministic_permutation_invariant_and_does_not_mutate_inputs():
    rng = np.random.default_rng(94)
    real = rng.normal(size=(12, 24))
    generated = rng.normal(size=(9, 24))
    labels = np.repeat([0, 8, 63], 4)
    synthetic_labels = np.repeat([0, 8, 63], 3)
    originals = [value.copy() for value in (real, labels, generated, synthetic_labels)]
    result = conditional_projected_wasserstein(real, labels, generated, synthetic_labels)
    assert result == conditional_projected_wasserstein(real, labels, generated, synthetic_labels)
    first, second = rng.permutation(12), rng.permutation(9)
    permuted = conditional_projected_wasserstein(real[first], labels[first], generated[second], synthetic_labels[second])
    assert result == permuted
    for original, value in zip(originals, (real, labels, generated, synthetic_labels)):
        np.testing.assert_array_equal(original, value)


def test_macro_weights_conditions_equally_and_all_conditions_share_directions():
    # Unequal condition counts and unequal generated counts must not change macro weighting.
    real = np.zeros((13, 3))
    labels = np.repeat([2, 8], [2, 11])
    generated = np.r_[np.tile([2., 0., 0.], (7, 1)), np.tile([0., 0., 0.], (3, 1))]
    generated_labels = np.repeat([2, 8], [7, 3])
    result = conditional_projected_wasserstein(real, labels, generated, generated_labels)
    expected = float(np.mean(np.abs(directions(3) @ np.array([2., 0., 0.]))))
    assert result["per_condition"]["2"]["mean_projected_wasserstein_kwh"] == pytest.approx(expected)
    assert result["per_condition"]["8"]["mean_projected_wasserstein_kwh"] == 0
    assert result["macro"] == pytest.approx(expected / 2)
    assert result["per_condition"]["2"]["n_real"] == 2
    assert result["per_condition"]["2"]["n_generated"] == 7
    assert result["per_condition"]["8"]["n_real"] == 11
    assert result["per_condition"]["8"]["n_generated"] == 3
    assert result["condition_weighting"] == "equal_over_observed_support"


def test_dependence_changes_are_visible_even_with_identical_coordinate_marginals():
    real = np.array([[-1., -1.], [1., 1.]])
    generated = np.array([[-1., 1.], [1., -1.]])
    assert all(wasserstein_distance(real[:, column], generated[:, column]) == 0 for column in range(2))
    result = conditional_projected_wasserstein(real, [0, 0], generated, [0, 0])
    projection = directions(2)
    manual = np.mean([wasserstein_distance(real @ row, generated @ row) for row in projection])
    assert result["macro"] == pytest.approx(manual)
    assert result["macro"] > .1
    assert "finite" in result["interpretation"].lower()
    assert "within" in result["comparability"].lower()


@pytest.mark.parametrize("real,labels,generated,generated_labels,match", [
    ([[0., 1.]], [0], [[0., 1.]], [1], "support"),
    ([[0., 1.]], [0], [[0., 1.], [1., 0.]], [0, 1], "support"),
    ([[0., 1.]], [0], [[0., 1., 2.]], [0], "same width"),
    ([[0., float('nan')]], [0], [[0., 1.]], [0], "finite"),
    ([[0., 1.]], [0, 1], [[0., 1.]], [0], "profile count"),
])
def test_invalid_arrays_or_mismatched_support_fail(real, labels, generated, generated_labels, match):
    with pytest.raises(ValueError, match=match):
        conditional_projected_wasserstein(real, labels, generated, generated_labels)
