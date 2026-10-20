"""Synthetic contracts for the reference runner; no research-data reads."""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/compare_reference_uncertainty.py"
SPEC = importlib.util.spec_from_file_location("reference_uncertainty", RUNNER)
comparison = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparison)


def _state(path, config, bundle, manifest):
    train, _, metadata, scaler, hashes = bundle
    state = {"format_version": 1, "epoch": 8, "config": config,
             "metadata": metadata, "scaler": scaler, "output_dim": metadata["resolution"],
             "data_hashes": {**hashes, manifest.name: comparison.sha256_file(manifest)},
             "supported_labels": np.unique(train["y"]).tolist(),
             "selection": {"train_profiles_available": len(train["x"]),
                           "train_profiles_used": len(train["x"]),
                           "train_indices_sha256": comparison._digest(np.arange(len(train["x"])), "<i8")},
             "provenance": {"git_commit": "1" * 40, "git_dirty": False,
                            "source_files_sha256": {name: "a" * 64 for name in comparison.SOURCE_CORE}}}
    torch.save(state, path)


@pytest.fixture
def toy(tmp_path, monkeypatch):
    """Real checkpoint binding, mocked already-verified loaders/generation only."""
    parent, child = tmp_path / "curated96", tmp_path / "derived24"
    parent.mkdir()
    child.mkdir()
    rng = np.random.default_rng(91)
    y = np.tile([0, 0, 4, 4], 4) + np.repeat([0, 0, 8, 8], 4)
    dates = np.tile(["2024-01-01", "2024-01-02", "2024-01-06", "2024-01-07"], 4)
    train = {"x": rng.normal(size=(16, 96)).astype(np.float32), "y": y,
             "meter_id": np.repeat(["t0", "t1", "t2", "t3"], 4), "date": dates}
    val = {"x": (rng.normal(size=(16, 96)) + 7).astype(np.float32), "y": y,
           "meter_id": np.repeat(["v0", "v1", "v2", "v3"], 4), "date": dates}
    bundles = {}
    for width, directory in ((96, parent), (24, child)):
        arrays = [copy.deepcopy(train), copy.deepcopy(val)]
        if width == 24:
            for data in arrays:
                data["x"] = comparison.hourly_sum(data["x"])
        for split, data in zip(("train", "val"), arrays):
            np.savez(directory / f"{split}.npz", **data)
        metadata = {"resolution": width, "target": "net", "units": "kWh/interval",
                    "source_sha256": "9" * 64, "curation": {"schema_version": 1},
                    "splits": {split: {"sha256": comparison.sha256_file(directory / f"{split}.npz")}
                               for split in ("train", "val")}}
        if width == 24:
            metadata["derived"] = {"schema_version": 1, "synthetic_fixture": True}
        scaler = {"minimum": float(arrays[0]["x"].min()), "maximum": float(arrays[0]["x"].max())}
        (directory / "metadata.json").write_text(json.dumps(metadata))
        (directory / "scaler.json").write_text(json.dumps(scaler))
        hashes = {name: comparison.sha256_file(directory / name)
                  for name in ("train.npz", "val.npz", "metadata.json", "scaler.json")}
        bundles[width] = (*arrays, metadata, scaler, hashes)
    curation, derivation = parent / "curation_manifest.json", child / "derivation_manifest.json"
    curation.write_text('{"synthetic":true}')
    derivation.write_text('{"synthetic":true}')
    manifests = {"parent_curation_manifest": {"path": str(curation), "sha256": comparison.sha256_file(curation)},
                 "hourly_derivation_manifest": {"path": str(derivation), "sha256": comparison.sha256_file(derivation)}}
    checkpoints = {}
    for name, width, kwargs in (("conditional", 96, {}), ("unconditional", 96, {"unconditional": True}),
                                ("hourly", 24, {}), ("swd10", 96, {"candidate_kind": "swd10"}),
                                ("capacity", 96, {"candidate_kind": "capacity"})):
        checkpoints[name] = tmp_path / f"{name}.pt"
        _state(checkpoints[name], comparison._declared_config(11, **kwargs), bundles[width],
               derivation if width == 24 else curation)
    calls = []

    def load(path):
        assert Path(path) == parent
        return bundles[96]

    def pair(a, b):
        assert (Path(a), Path(b)) == (parent, child)
        from compare_resolutions import _alignment
        return bundles[96], bundles[24], _alignment(bundles[96], bundles[24]), manifests

    def generated(path, details, labels, seed):
        calls.append((Path(path).stem, labels.copy(), seed))
        width = 24 if Path(path).stem == "hourly" else 96
        base = np.arange(width, dtype=np.float32)[None, :] / width
        return np.repeat(base, len(labels), axis=0) + labels[:, None] / 10

    monkeypatch.setattr(comparison, "_load_prepared", load)
    monkeypatch.setattr(comparison, "_training_curation_guard", lambda *a, **k: {
        "status": "verified_curation_v2", "pinned_release": True,
        "manifest_sha256": comparison.sha256_file(curation), "verified_file_hashes": bundles[96][4]})
    # Production protocol sizes are asserted separately; tiny fixtures retain
    # exactly the same guard and rejection paths without reading research data.
    monkeypatch.setattr(comparison, "TRAIN_PROFILES", 16)
    monkeypatch.setattr(comparison, "CONDITION_SUPPORT", (0, 4, 8, 12))
    monkeypatch.setattr(comparison, "_prepared_pair", pair)
    monkeypatch.setattr(comparison, "_generate_verified", generated)
    monkeypatch.setattr(comparison, "_provenance", lambda device: {"source_files_sha256": {}, "git_commit": "fixture"})
    return {"root": tmp_path, "parent": parent, "child": child, "bundles": bundles,
            "checkpoints": checkpoints, "calls": calls, "manifests": manifests}


def test_fixed_protocol_constants():
    assert comparison.TRAIN_PROFILES == 606060
    assert comparison.CONDITION_SUPPORT == tuple(range(64))
    assert (comparison.CAP, comparison.DRAW_SEED, comparison.REPLICATES) == (100, 2026, 500)


@pytest.mark.parametrize("change", ["train_count", "train_support", "val_support"])
def test_missing_declared_population_rejected_before_generation(toy, change):
    bundle = toy["bundles"][96]
    if change == "train_count":
        bundle[0]["x"] = bundle[0]["x"][:-1]
    else:
        split = bundle[0 if change == "train_support" else 1]
        split["y"] = np.where(split["y"] == 12, 8, split["y"])
    with pytest.raises(ValueError, match="TRAIN pool|all 64 conditions"):
        _call(toy)
    assert not toy["calls"]


def test_native_test_alias_rejected_before_any_content_read(tmp_path, monkeypatch):
    reserved = tmp_path / "test.npz"
    reserved.write_bytes(b"reserved synthetic sentinel")
    (tmp_path / "metadata.json").symlink_to(reserved)
    monkeypatch.setattr(comparison, "_training_curation_guard", lambda *a, **k: pytest.fail("content read"))
    monkeypatch.setattr(comparison, "_load_prepared", lambda *a: pytest.fail("content read"))
    with pytest.raises(ValueError, match="aliases reserved TEST"):
        comparison._native_prepared(tmp_path)


def test_unverified_curation_rejected_before_array_load(tmp_path, monkeypatch):
    monkeypatch.setattr(comparison, "_training_curation_guard", lambda *a, **k: {
        "status": "not_required_for_unpinned_source", "pinned_release": False})
    monkeypatch.setattr(comparison, "_load_prepared", lambda *a: pytest.fail("array read"))
    with pytest.raises(ValueError, match="verified curation v2"):
        comparison._native_prepared(tmp_path)


@pytest.mark.parametrize("alias_kind", ["literal", "symlink", "hardlink"])
@pytest.mark.parametrize("artifact_role", ["checkpoint", "map"])
def test_checkpoint_and_map_test_aliases_fail_before_content_reads(tmp_path, monkeypatch, alias_kind, artifact_role):
    reserved = tmp_path / "test.npz"
    reserved.write_bytes(b"reserved synthetic sentinel")
    alias = tmp_path / "artifact.bin"
    if alias_kind == "literal":
        alias = reserved
    elif alias_kind == "symlink":
        alias.symlink_to(reserved)
    else:
        alias.hardlink_to(reserved)
    monkeypatch.setattr(comparison, "_load_prepared", lambda *a: pytest.fail("content read"))
    monkeypatch.setattr(comparison, "_training_curation_guard", lambda *a, **k: pytest.fail("content read"))
    kwargs = {} if artifact_role == "checkpoint" else {
        "candidate_kind": "swd10", "candidate_map": alias, "comparator_map": "unused-map"}
    with pytest.raises(ValueError, match="input aliases reserved TEST"):
        comparison.compare(tmp_path, alias if artifact_role == "checkpoint" else "unused.pt", "other.pt",
                           contrast="conditioning" if artifact_role == "checkpoint" else "candidate_reference", **kwargs)


def _call(toy, contrast="conditioning", **kwargs):
    first = "hourly" if contrast.startswith("resolution_") else (kwargs.get("candidate_kind") or "conditional")
    second = "unconditional" if contrast == "conditioning" else "conditional"
    return comparison.compare(toy["parent"], toy["checkpoints"][first], toy["checkpoints"][second],
                              contrast=contrast,
                              **({"prepared24": toy["child"]} if contrast.startswith("resolution_") else {}),
                              **kwargs)


def _maps(toy, monkeypatch):
    paths = [toy["root"] / f"map-{i}.npz" for i in (0, 1)]
    for path in paths:
        path.write_bytes(b"synthetic map identity, deliberately not a fitted artifact")
    calls = []

    class Map:
        def transform(self, values, labels):
            calls.append(values.copy())
            # Deliberately nonlinear: mapping before/after aggregation differs.
            return values.astype(np.float64) ** 2 - .125

    def saved(path, details, bundle, seed):
        assert seed == 2026
        return Map(), {"sha256": comparison.sha256_file(path), "quantile_knots": 1025,
                       "calibration_seed": 7301, "checkpoint_sha256": details["sha256"]}

    monkeypatch.setattr(comparison, "_saved_map", saved)
    return paths, calls


def test_conditioning_five_metrics_shared_meter_draws_and_explicit_sign(toy):
    result = _call(toy)
    assert result["split"] == "val" and result["test_used"] is False
    assert result["negative_difference_favors"] == "raw_conditional_bce96"
    assert result["saved_maps"] == {} and result["replicates"] == 500
    assert result["training_seed"] == 11 and result["training_subsets_match"]
    assert result["validation_selection"]["samples_per_condition_cap"] == 100
    assert len(toy["calls"]) == 2
    np.testing.assert_array_equal(toy["calls"][0][1], toy["calls"][1][1])
    paired = result["paired_comparison"]
    assert paired["resampling_scheme"] == "within_stratum_real_meter"
    assert paired["stratum_meter_counts"] == {"0": 2, "1": 2}
    assert paired["n_bootstrap"] == 500 and paired["n_valid_bootstrap"] == 500
    assert set(paired["intervals"]) == set(comparison.METRICS)
    assert set(paired["per_condition"]) == {"0", "4", "8", "12"}
    # Identical frozen fake arrays yield identically zero paired differences,
    # including bounds, although each method has nonzero physical errors.
    assert all(value == 0 for row in paired["intervals"].values() for value in row.values())
    assert paired["point_estimates"]["generated"]["daily_energy_wasserstein_kwh"] > 0
    for label, row in paired["per_condition"].items():
        assert set(row["intervals"]) == set(comparison.METRICS)


@pytest.mark.parametrize("contrast,kwargs,error", [
    ("conditioning", {"candidate_map": "x"}, "raw contrasts forbid"),
    ("conditioning", {"prepared24": "x"}, "prepared24"),
    ("conditioning", {"candidate_kind": "capacity"}, "only valid"),
    ("candidate_reference", {}, "requires both"),
    ("candidate_reference", {"candidate_map": "a", "comparator_map": "b"}, "explicit candidate_kind"),
    ("resolution_calibrated", {"prepared24": "x", "candidate_map": "a"}, "requires both"),
    ("resolution_raw", {}, "prepared24"),
])
def test_invalid_contrast_contract_rejected_before_data_load(monkeypatch, contrast, kwargs, error):
    monkeypatch.setattr(comparison, "_load_prepared", lambda *a: pytest.fail("unexpected data load"))
    monkeypatch.setattr(comparison, "_prepared_pair", lambda *a: pytest.fail("unexpected data load"))
    with pytest.raises(ValueError, match=error):
        comparison.compare("unopened", "a.pt", "b.pt", contrast=contrast, **kwargs)


@pytest.mark.parametrize("field,value,error", [
    ("config.seed", 22, "share one declared"),
    ("config.unconditional", False, "declared conditioning"),
    ("config.lr_g", .0002, "declared conditioning"),
    ("selection.train_indices_sha256", "0" * 64, "exact training subset"),
    ("data_hashes.curation_manifest.json", "0" * 64, "curation_manifest"),
])
def test_checkpoint_and_config_bindings_fail_closed(toy, field, value, error):
    path = toy["checkpoints"]["unconditional"]
    state = torch.load(path, weights_only=True)
    section, key = field.split(".", 1)
    state[section][key] = value
    torch.save(state, path)
    with pytest.raises(ValueError, match=error):
        _call(toy)
    assert not toy["calls"]


def test_reconstructed_subsample_is_still_rejected(toy):
    path = toy["checkpoints"]["unconditional"]
    state = torch.load(path, weights_only=True)
    indices = np.sort(np.random.default_rng(11).choice(16, 8, replace=False))
    state["config"]["max_train_samples"] = 8
    state["selection"].update(train_profiles_used=8, train_indices_sha256=comparison._digest(indices, "<i8"))
    state["supported_labels"] = np.unique(toy["bundles"][96][0]["y"][indices]).tolist()
    torch.save(state, path)
    with pytest.raises(ValueError, match="full actual TRAIN"):
        _call(toy)


@pytest.mark.parametrize("kind", ["swd10", "capacity"])
def test_calibrated_candidate_packages_and_map_use(toy, monkeypatch, kind):
    paths, transforms = _maps(toy, monkeypatch)
    result = _call(toy, "candidate_reference", candidate_kind=kind,
                   candidate_map=paths[0], comparator_map=paths[1])
    assert result["negative_difference_favors"] == f"calibrated_{kind}96"
    assert len(transforms) == 2 and all(array.shape[1] == 96 for array in transforms)
    assert result["seeds"]["calibration_reference"] == 7301
    assert set(result["saved_maps"]) == {"candidate", "comparator"}


def test_resolution_maps_native96_before_hourly_sum_and_common_rows(toy, monkeypatch):
    paths, transforms = _maps(toy, monkeypatch)
    result = _call(toy, "resolution_calibrated", candidate_map=paths[0], comparator_map=paths[1])
    assert [values.shape[1] for values in transforms] == [24, 96]
    expected = comparison.hourly_sum(transforms[1].astype(np.float64) ** 2 - .125)
    assert result["frozen_samples"]["comparator"]["evaluated_profiles_sha256"] == comparison._digest(expected, "<f8")
    assert result["resolution"] == 24 and result["interval_hours"] == 1
    assert result["paired_comparison"]["interval_hours"] == 1
    assert all(row["same_order_and_values"] for row in result["row_bindings"].values())
    assert result["pipelines"]["comparator"]["order"] == "native generator -> fixed native map -> hourly sum"
    assert result["frozen_samples"]["candidate"]["evaluated_dtype"] == "float64"
    assert result["frozen_samples"]["comparator"]["evaluated_dtype"] == "float32"


def test_raw_resolution_never_loads_maps(toy, monkeypatch):
    monkeypatch.setattr(comparison, "_saved_map", lambda *a: pytest.fail("raw comparison requested a map"))
    result = _call(toy, "resolution_raw")
    assert result["saved_maps"] == {}
    assert result["negative_difference_favors"] == "raw_bce24"
    assert result["pipelines"]["comparator"]["order"] == "native generator -> hourly sum"


def test_saved_map_helper_enforces_declared_settings(monkeypatch):
    import compare_resolutions
    details = {"sha256": "c" * 64, "training_selection": {"train_indices_sha256": "d" * 64}}
    bundle = ({"y": np.array([0, 0])}, None, {"resolution": 96}, None, {"train.npz": "e" * 64})
    record = {"quantile_knots": 1025, "calibration_seed": 7301,
              "provenance": {"calibration_samples_per_condition": 4096}}
    for field, bad in (("quantile_knots", 257), ("calibration_seed", 7302),
                       ("calibration_samples_per_condition", 8192)):
        changed = copy.deepcopy(record)
        (changed["provenance"] if field == "calibration_samples_per_condition" else changed)[field] = bad
        monkeypatch.setattr(compare_resolutions, "load_verified_calibrator", lambda *a, **k: (object(), changed))
        with pytest.raises(ValueError, match="1025 knots"):
            comparison._saved_map("map", details, bundle, 2026)


def test_no_map_fit_and_no_research_test_access(toy, monkeypatch):
    import fluvius_cgan.calibration as calibration
    monkeypatch.setattr(calibration, "fit", lambda *a, **k: pytest.fail("map fitting is forbidden"))
    original = Path.open

    def opened(path, *args, **kwargs):
        assert path.name != "test.npz", "TEST may not be opened"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", opened)
    result = _call(toy)
    assert result["test_used"] is False


def test_end_of_run_checkpoint_mutation_rejected(toy, monkeypatch):
    real_bootstrap = comparison.paired_condition_macro_meter_bootstrap

    def mutated(*args, **kwargs):
        result = real_bootstrap(*args, **kwargs)
        with toy["checkpoints"]["conditional"].open("ab") as handle:
            handle.write(b"changed")
        return result

    monkeypatch.setattr(comparison, "paired_condition_macro_meter_bootstrap", mutated)
    with pytest.raises(ValueError, match="checkpoint changed during comparison"):
        _call(toy)


def test_output_guard_precedes_all_loading(tmp_path, monkeypatch):
    output = tmp_path / "existing.json"
    output.write_text("keep")
    monkeypatch.setattr(comparison, "compare", lambda *a, **k: pytest.fail("should not load"))
    with pytest.raises(FileExistsError, match="already exists"):
        comparison.main(["--contrast", "conditioning", "--prepared96", "x", "--candidate", "a",
                         "--comparator", "b", "--output", str(output)])
    assert output.read_text() == "keep"
