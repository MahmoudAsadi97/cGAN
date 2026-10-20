"""Synthetic checks for fixed, train-only conditional marginal calibration."""

import hashlib
import json

import numpy as np
import pytest

from fluvius_cgan.calibration import ConditionalMarginalCalibrator, fit, load_verified_calibrator


def tied_example():
    source = np.array([[0.], [0.], [1.], [2.], [2.]])
    target = np.array([[-10.], [-5.], [0.], [5.], [10.]])
    labels = np.zeros(5, dtype=np.int64)
    return target, labels, source, labels.copy()


def test_plateau_midpoints_and_interpolation_use_adjacent_plateau_edges():
    model = fit(*tied_example(), quantile_knots=None)
    values = np.array([[-1.], [0.], [.5], [1.], [1.5], [2.], [3.]])
    # Source CDF probabilities are 0, .125, .375, .5, .625, .875, 1.
    # Interpolating between collapsed plateau midpoints would be incorrect.
    actual = model.transform(values, np.zeros(len(values), dtype=np.int64))
    expected = np.array([[-10.], [-7.5], [-2.5], [0.], [2.5], [7.5], [10.]])
    np.testing.assert_array_equal(actual, expected)
    assert np.all(np.diff(actual[:, 0]) >= 0)


def test_identical_marginals_preserve_in_range_values_even_between_tied_plateaus():
    _, labels, source, _ = tied_example()
    model = fit(source, labels, source, labels, quantile_knots=None)
    values = np.arange(0., 2.25, .25)[:, None]
    np.testing.assert_array_equal(model.transform(values, np.zeros(len(values), dtype=int)), values)


@pytest.mark.parametrize("knots", [None, 3, 1025])
def test_affine_quantile_transport_is_exact_and_bounded_with_signed_targets(knots):
    source = np.linspace(-3., 3., 17)[:, None]
    target = 2 * source - 4
    labels = np.zeros(len(source), dtype=int)
    model = fit(target, labels, source, labels, quantile_knots=knots)
    values = np.array([[-5.], [-3.], [-1.5], [0.], [1.5], [3.], [5.]])
    expected = np.clip(2 * values - 4, -10., 2.)
    actual = model.transform(values, np.zeros(len(values), dtype=int))
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
    assert np.isfinite(actual).all() and (actual < 0).any()


def test_conditional_mapping_is_fixed_and_independent_of_batch_composition():
    target, labels, source, _ = tied_example()
    train = np.vstack([target, target + 100.])
    calibration = np.vstack([source, source + 10.])
    conditions = np.r_[labels, np.full(len(labels), 8)]
    model = fit(train, conditions, calibration, conditions, quantile_knots=None)
    values = np.array([[10.5], [.5], [12.], [0.], [11.5], [1.]])
    requested = np.array([8, 0, 8, 0, 8, 0])
    expected = np.array([[97.5], [-2.5], [107.5], [-7.5], [102.5], [0.]])
    actual = model.transform(values, requested)
    np.testing.assert_array_equal(actual, expected)
    chunks = np.concatenate([model.transform(values[:1], requested[:1]),
                             model.transform(values[1:4], requested[1:4]),
                             model.transform(values[4:], requested[4:])])
    np.testing.assert_array_equal(chunks, actual)
    permutation = np.array([5, 2, 0, 3, 1, 4])
    np.testing.assert_array_equal(model.transform(values[permutation], requested[permutation]),
                                  actual[permutation])
    # Changing the reference rows for another condition cannot change this map.
    train[conditions == 8] += 500
    calibration[conditions == 8] -= 500
    other = fit(train, conditions, calibration, conditions, quantile_knots=None)
    np.testing.assert_array_equal(other.transform(values[requested == 0], requested[requested == 0]),
                                  actual[requested == 0])


def test_source_atom_mass_is_distinct_from_quantile_plateau_mass():
    model = fit(*tied_example(), quantile_knots=None)
    diagnostic = model.fitting_diagnostics()["per_condition"]["0"]
    assert diagnostic["train_count"] == 5
    assert diagnostic["calibration_count"] == 5
    slot = diagnostic["slots"][0]
    assert slot["source_tie_mass"] == .8
    assert slot["source_largest_atom_mass"] == .4
    assert slot["source_ties"] == [
        {"value": 0., "count": 2, "mass": .4},
        {"value": 2., "count": 2, "mass": .4},
    ]
    assert slot["source_knot_plateaus"] == [
        {"value": 0., "probability_lower": 0., "probability_upper": .25, "mass": .25},
        {"value": 2., "probability_lower": .75, "probability_upper": 1., "mass": .25},
    ]


def test_range_exceedance_and_endpoint_saturation_counts_are_explicit():
    model = fit(*tied_example(), quantile_knots=None)
    values = np.array([[-1.], [0.], [.5], [1.], [1.5], [2.], [3.]])
    labels = np.zeros(len(values), dtype=int)
    diagnostic = model.diagnostics(values, labels)
    assert diagnostic["n_profiles"] == 7
    condition = diagnostic["per_condition"]["0"]
    assert condition["count"] == 7
    slot = condition["slots"][0]
    for name in ("source_below", "source_above", "cdf_zero", "cdf_one",
                 "target_lower_endpoint", "target_upper_endpoint"):
        assert slot[f"{name}_count"] == 1
        assert slot[f"{name}_fraction"] == pytest.approx(1 / 7)


def test_target_atoms_can_saturate_endpoints_without_source_range_exceedance():
    source = np.arange(5.)[:, None]
    target = np.array([[0.], [0.], [1.], [2.], [2.]])
    labels = np.zeros(5, dtype=int)
    model = fit(target, labels, source, labels, quantile_knots=None)
    query = np.array([[.5], [3.5]])
    actual = model.transform(query, labels[:2])
    np.testing.assert_array_equal(actual, [[0.], [2.]])
    slot = model.diagnostics(query, labels[:2])["per_condition"]["0"]["slots"][0]
    for name in ("source_below", "source_above", "cdf_zero", "cdf_one"):
        assert slot[f"{name}_count"] == 0
    assert slot["target_lower_endpoint_count"] == 1
    assert slot["target_upper_endpoint_count"] == 1


def test_constant_target_columns_stay_constant_for_arbitrary_finite_queries():
    source = np.column_stack([np.arange(5.), np.full(5, 9.)])
    target = np.column_stack([np.full(5, -2.), np.full(5, 7.)])
    labels = np.zeros(5, dtype=int)
    model = fit(target, labels, source, labels, quantile_knots=None)
    query = np.array([[-100., -100.], [2., 9.], [100., 100.]])
    np.testing.assert_array_equal(model.transform(query, labels[:3]), [[-2., 7.]] * 3)


def test_constant_source_cannot_restore_a_nonconstant_target():
    labels = np.full(5, 8, dtype=int)
    with pytest.raises(ValueError, match=r"(?i)(condition.*8.*slot.*1|slot.*1.*condition.*8)"):
        fit(np.column_stack([np.arange(5.), np.arange(5.)]), labels,
            np.column_stack([np.arange(5.), np.full(5, 1.)]), labels)


@pytest.mark.parametrize("knots", [None, 3, 1025])
def test_saved_parameters_and_provenance_replay_without_reference_arrays(tmp_path, knots):
    target, train_labels, source, source_labels = tied_example()
    provenance = {"checkpoint_sha256": "a" * 64,
                  "reference": {"seed": 2027, "indices": [1, 4]}, "note": "synthetic"}
    original_provenance = json.loads(json.dumps(provenance))
    original_source, original_target = source.copy(), target.copy()
    model = fit(target, train_labels, source, source_labels,
                quantile_knots=knots, provenance=provenance)
    np.testing.assert_array_equal(source, original_source)
    np.testing.assert_array_equal(target, original_target)
    provenance["reference"]["indices"].append(99)
    provenance["checkpoint_sha256"] = "b" * 64
    assert model.provenance == original_provenance
    query = np.array([[-1.], [0.], [.5], [1.], [2.], [3.]])
    labels = np.zeros(len(query), dtype=int)
    expected = model.transform(query, labels)
    static, dynamic = model.fitting_diagnostics(), model.diagnostics(query, labels)
    source[:] = 500
    target[:] = -500
    np.testing.assert_array_equal(model.transform(query, labels), expected)
    path = tmp_path / "calibration.npz"
    model.save(path)
    restored = ConditionalMarginalCalibrator.load(path)
    np.testing.assert_array_equal(restored.transform(query, labels), expected)
    assert restored.provenance == original_provenance
    assert restored.fitting_diagnostics() == static
    assert restored.diagnostics(query, labels) == dynamic


def test_empty_transform_preserves_feature_dimension():
    model = fit(*tied_example())
    result = model.transform(np.empty((0, 1)), np.array([], dtype=int))
    assert result.shape == (0, 1)


def test_quarter_hour_profiles_are_batch_invariant_across_conditions():
    rng = np.random.default_rng(42)
    source = rng.normal(size=(120, 96))
    target = rng.normal(loc=-2., scale=3., size=(80, 96))
    source_labels = np.repeat([0, 8], 60)
    train_labels = np.repeat([0, 8], 40)
    model = fit(target, train_labels, source, source_labels, quantile_knots=17)
    query = rng.normal(size=(27, 96))
    labels = np.resize(np.array([8, 0, 8]), len(query))
    expected = model.transform(query, labels)
    assert expected.shape == (27, 96) and np.isfinite(expected).all()
    chunks = np.concatenate([
        model.transform(query[start:start + 4], labels[start:start + 4])
        for start in range(0, len(query), 4)
    ])
    np.testing.assert_array_equal(chunks, expected)
    permutation = rng.permutation(len(query))
    np.testing.assert_array_equal(model.transform(query[permutation], labels[permutation]),
                                  expected[permutation])


@pytest.mark.parametrize("malformation", ["nonfinite_source", "complex_target"])
def test_loading_rejects_invalid_quantiles_before_numeric_cast(tmp_path, malformation):
    path = tmp_path / "invalid.npz"
    fit(*tied_example(), quantile_knots=None).save(path)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    if malformation == "nonfinite_source":
        arrays["source_0"][0, 0] = np.nan
    else:
        arrays["target_0"] = arrays["target_0"].astype(complex)
        arrays["target_0"][0, 0] += 1j
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="quantile"):
        ConditionalMarginalCalibrator.load(path)


@pytest.mark.parametrize("target_labels,source_labels", [
    ([0, 0, 0], [0, 0, 8]),
    ([0, 0, 8], [0, 0, 0]),
    ([0, 0, 0], [8, 8, 8]),
])
def test_fitting_rejects_different_observed_condition_support(target_labels, source_labels):
    with pytest.raises(ValueError):
        fit(np.arange(3.)[:, None], target_labels, np.arange(3.)[:, None], source_labels)


@pytest.mark.parametrize("change", [
    "empty_target", "empty_source", "rank_one", "width", "nan_target", "inf_source",
    "complex", "boolean", "target_label_count", "source_label_count", "float_labels",
    "negative_label", "out_of_range_label", "boolean_labels",
])
def test_invalid_fitting_values_shapes_and_labels_are_rejected(change):
    target, target_labels, source, source_labels = tied_example()
    if change == "empty_target":
        target, target_labels = target[:0], target_labels[:0]
    elif change == "empty_source":
        source, source_labels = source[:0], source_labels[:0]
    elif change == "rank_one":
        target = target[:, 0]
    elif change == "width":
        source = np.repeat(source, 2, axis=1)
    elif change == "nan_target":
        target[0, 0] = np.nan
    elif change == "inf_source":
        source[0, 0] = np.inf
    elif change == "complex":
        source = source.astype(complex)
    elif change == "boolean":
        target = target.astype(bool)
    elif change == "target_label_count":
        target_labels = target_labels[:1]
    elif change == "source_label_count":
        source_labels = source_labels[:1]
    elif change == "float_labels":
        target_labels = target_labels.astype(float)
    elif change == "negative_label":
        target_labels[0] = -1
    elif change == "out_of_range_label":
        source_labels[0] = 64
    else:
        target_labels = target_labels.astype(bool)
    with pytest.raises(ValueError):
        fit(target, target_labels, source, source_labels)


@pytest.mark.parametrize("knots", [0, 1, True, 2.5])
def test_invalid_quantile_knot_counts_are_rejected(knots):
    with pytest.raises(ValueError):
        fit(*tied_example(), quantile_knots=knots)


@pytest.mark.parametrize("values,labels", [
    ([[0.]], [8]), ([[0.]], [64]), ([[0.]], [-1]), ([[0.]], [0.]),
    ([[0.]], [True]), ([[0.]], []), ([[0.]], [[0]]),
    ([[np.nan]], [0]), ([[np.inf]], [0]), ([[1j]], [0]), ([[True]], [0]),
    ([0.], [0]), ([[0., 0.]], [0]),
])
def test_invalid_application_inputs_are_rejected(values, labels):
    model = fit(*tied_example())
    with pytest.raises(ValueError):
        model.transform(values, labels)
    with pytest.raises(ValueError):
        model.diagnostics(values, labels)


@pytest.fixture
def saved_bound_calibrator(tmp_path):
    train_labels = np.array([0] * 5 + [8] * 3, dtype=np.int64)
    train = np.arange(16., dtype=float).reshape(8, 2) - 4
    calibration_labels = np.repeat([0, 8], 4).astype(np.int64)
    source = np.random.default_rng(7301).normal(size=(8, 2))
    data_hashes = {"train.npz": "c" * 64, "val.npz": "d" * 64,
                   "metadata.json": "e" * 64, "scaler.json": "f" * 64}
    provenance = {
        "checkpoint_sha256": "a" * 64,
        "train_indices_sha256": "b" * 64,
        "prepared_data_hashes": data_hashes,
        "calibration_seed": 7301,
        "calibration_samples_per_condition": 4,
        "calibration_profiles_total": len(source),
        "calibration_generated_profiles_sha256": hashlib.sha256(source.astype("<f8").tobytes()).hexdigest(),
        "calibration_labels_sha256": hashlib.sha256(calibration_labels.astype("<i8").tobytes()).hexdigest(),
        "generator_inference_batch_size": 1024,
        "quantile_knots": 3,
    }
    model = fit(train, train_labels, source, calibration_labels, quantile_knots=3, provenance=provenance)
    path = tmp_path / "bound-calibration.npz"
    model.save(path)
    arguments = {
        "checkpoint_sha256": provenance["checkpoint_sha256"],
        "train_indices_sha256": provenance["train_indices_sha256"],
        "prepared_data_hashes": data_hashes.copy(),
        "generation_seed": 2026,
        "n_features": 2,
        "train_labels": train_labels.copy(),
    }
    return path, model, arguments


def test_verified_calibrator_loading_checks_binding_and_replays(saved_bound_calibrator):
    path, original, arguments = saved_bound_calibrator
    loaded, details = load_verified_calibrator(path, **arguments)
    assert loaded.provenance == original.provenance
    assert details["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert {label: model.train_count for label, model in loaded.models.items()} == {0: 5, 8: 3}
    query = np.array([[-1., .5], [2., -1.], [.25, .75]])
    labels = np.array([8, 0, 8])
    np.testing.assert_array_equal(loaded.transform(query, labels), original.transform(query, labels))


@pytest.mark.parametrize("mismatch", [
    "checkpoint", "train_indices", "train_data", "val_data", "seed", "width",
    "train_counts", "train_support",
])
def test_verified_calibrator_rejects_mismatched_application_context(saved_bound_calibrator, mismatch):
    path, _, arguments = saved_bound_calibrator
    if mismatch == "checkpoint":
        arguments["checkpoint_sha256"] = "1" * 64
    elif mismatch == "train_indices":
        arguments["train_indices_sha256"] = "2" * 64
    elif mismatch == "train_data":
        arguments["prepared_data_hashes"]["train.npz"] = "3" * 64
    elif mismatch == "val_data":
        arguments["prepared_data_hashes"]["val.npz"] = "4" * 64
    elif mismatch == "seed":
        arguments["generation_seed"] = 7301
    elif mismatch == "width":
        arguments["n_features"] = 3
    elif mismatch == "train_counts":
        arguments["train_labels"][0] = 8
    else:
        arguments["train_labels"][:] = 0
    with pytest.raises(ValueError):
        load_verified_calibrator(path, **arguments)


@pytest.mark.parametrize("mismatch", [
    "missing_calibration_seed", "calibration_count", "calibration_total",
    "generated_hash", "label_hash", "inference_batch_size", "quantile_knots",
])
def test_verified_calibrator_rejects_inconsistent_saved_provenance(saved_bound_calibrator, mismatch):
    path, model, arguments = saved_bound_calibrator
    provenance = model.provenance
    if mismatch == "missing_calibration_seed":
        del provenance["calibration_seed"]
    elif mismatch == "calibration_count":
        provenance["calibration_samples_per_condition"] = 3
    elif mismatch == "calibration_total":
        provenance["calibration_profiles_total"] = 7
    elif mismatch == "generated_hash":
        provenance["calibration_generated_profiles_sha256"] = "not-a-hash"
    elif mismatch == "label_hash":
        provenance["calibration_labels_sha256"] = "0" * 64
    elif mismatch == "inference_batch_size":
        provenance["generator_inference_batch_size"] = 0
    else:
        provenance["quantile_knots"] = 1025
    model.save(path)
    with pytest.raises(ValueError):
        load_verified_calibrator(path, **arguments)


def test_verified_calibrator_rejects_file_changes_during_loading(saved_bound_calibrator, monkeypatch):
    path, _, arguments = saved_bound_calibrator
    original_load = ConditionalMarginalCalibrator.load

    def changed_load(cls, source):
        model = original_load(source)
        with path.open("ab") as stream:
            stream.write(b"changed during loading")
        return model

    monkeypatch.setattr(ConditionalMarginalCalibrator, "load", classmethod(changed_load))
    with pytest.raises(ValueError, match="(?i)(changed|hash|fingerprint)"):
        load_verified_calibrator(path, **arguments)
