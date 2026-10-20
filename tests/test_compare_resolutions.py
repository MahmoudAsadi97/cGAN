"""Small synthetic common-hourly comparisons; no research-data access."""
import builtins
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
import pytest
import torch

from fluvius_cgan.calibration import ConditionalMarginalCalibrator, fit
from fluvius_cgan.curation import curate_prepared
from fluvius_cgan.models import Generator
from fluvius_cgan.release import RELEASE_SHA256
from fluvius_cgan.training import CHECKPOINT_FORMAT, _config, _load_prepared, generate

SPEC = importlib.util.spec_from_file_location(
    "compare_resolutions", Path(__file__).resolve().parents[1] / "scripts/compare_resolutions.py")
comparison = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparison)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_digest(values, dtype="<f8"):
    return hashlib.sha256(np.asarray(values, dtype=dtype).tobytes()).hexdigest()


@pytest.fixture(scope="module")
def experiment(tmp_path_factory):
    from fluvius_cgan.derived import derive_hourly

    root = tmp_path_factory.mktemp("resolution-comparison")
    original, parent, child = (root / name for name in ("original", "curated96", "derived24"))
    original.mkdir()
    dates = np.array(["2024-01-01", "2024-01-02", "2024-01-06", "2024-01-07"])
    metadata = {"resolution": 96, "target": "net", "units": "kWh/interval",
                "source_sha256": RELEASE_SHA256, "split_seed": 42, "splits": {}, "counts": {}}
    rng = np.random.default_rng(17)
    original_arrays = {}
    for number, split in enumerate(("train", "val", "test")):
        x = rng.normal(size=(16, 96)).astype(np.float32)
        x += number * 100  # Deliberately distinct VAL values: never fit controls/maps on them.
        ids = np.repeat([f"{split}-{i}" for i in range(4)], 4)
        # Labels' technology remains constant for each meter.
        y = np.concatenate([np.array([0, 0, 4, 4]) + 8 * (i % 2) for i in range(4)])
        order = np.arange(16)[::-1]
        arrays = {"x": x[order], "y": y[order], "meter_id": ids[order], "date": np.tile(dates, 4)[order]}
        original_arrays[split] = arrays
        np.savez_compressed(original / f"{split}.npz", **arrays)
        metadata["splits"][split] = {"profiles": 16, "meters": 4,
                                    "meter_ids": sorted(set(ids)), "sha256": digest(original / f"{split}.npz")}
    train_x = original_arrays["train"]["x"]
    (original / "scaler.json").write_text(json.dumps({"minimum": float(train_x.min()), "maximum": float(train_x.max())}))
    (original / "metadata.json").write_text(json.dumps(metadata))
    curate_prepared(original, parent, expected_dates=dates)
    derive_hourly(parent, child)
    bundles = {96: _load_prepared(parent), 24: _load_prepared(child)}
    checkpoints = {}
    config = _config({"seed": 11, "epochs": 26, "batch_size": 512,
                      "discriminator_hidden_dim": 64, "normalization": "minmax",
                      "objective": "bce", "batch_sampling": "balanced_conditions",
                      "conditions_per_batch": 4, "validation_selection": "conditional_w1"})
    for width, prepared, manifest in ((96, parent, "curation_manifest.json"),
                                      (24, child, "derivation_manifest.json")):
        train, _, meta, scaler, hashes = bundles[width]
        torch.manual_seed(3)
        state = {"format_version": CHECKPOINT_FORMAT, "epoch": 2, "config": config.copy(),
                 "metadata": meta, "scaler": scaler, "output_dim": width, "condition_dim": 64,
                 "data_hashes": {**hashes, manifest: digest(prepared / manifest)},
                 "selection": {"train_profiles_available": len(train["x"]), "train_profiles_used": len(train["x"]),
                               "train_indices_sha256": array_digest(np.arange(len(train["x"])), "<i8")},
                 "supported_labels": np.unique(train["y"]).tolist(),
                 "generator_state_dict": Generator(24, 64, 64, width).state_dict()}
        checkpoint = root / f"bce{width}.pt"
        torch.save(state, checkpoint)
        checkpoints[width] = checkpoint
    maps = {}
    for width in (96, 24):
        train, _, _, _, hashes = bundles[width]
        labels = np.repeat(np.unique(train["y"]), 4096)
        # A separate synthetic calibration reference, with nonconstant margins.
        source = np.random.default_rng(7301).normal(size=(len(labels), width)).astype(np.float32)
        provenance = {"checkpoint_sha256": digest(checkpoints[width]),
                      "train_indices_sha256": array_digest(np.arange(len(train["x"])), "<i8"),
                      "prepared_data_hashes": hashes, "calibration_seed": 7301,
                      "calibration_samples_per_condition": 4096, "calibration_profiles_total": len(labels),
                      "calibration_generated_profiles_sha256": array_digest(source),
                      "calibration_labels_sha256": array_digest(labels, "<i8"),
                      "generator_inference_batch_size": 1024, "quantile_knots": 1025}
        model = fit(train["x"], train["y"], source, labels, quantile_knots=1025, provenance=provenance)
        maps[width] = root / f"map{width}.npz"
        model.save(maps[width])
    return {"root": root, "parent": parent, "child": child, "bundles": bundles,
            "checkpoints": checkpoints, "maps": maps}


@pytest.fixture
def forbid_test_access(monkeypatch):
    original_open, path_open, numpy_load = builtins.open, Path.open, np.load

    def check(path):
        if isinstance(path, (str, bytes, Path)):
            path = Path(path.decode() if isinstance(path, bytes) else path)
            reserved = path.parent / "test.npz"
            assert (path.name != "test.npz" and path.resolve().name != "test.npz"
                    and not (path.exists() and reserved.exists() and path.samefile(reserved))), "reserved TEST accessed"

    def opened(path, *args, **kwargs):
        check(path)
        return original_open(path, *args, **kwargs)

    def path_opened(path, *args, **kwargs):
        check(path)
        return path_open(path, *args, **kwargs)

    def loaded(path, *args, **kwargs):
        check(path)
        return numpy_load(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", opened)
    monkeypatch.setattr(Path, "open", path_opened)
    monkeypatch.setattr(np, "load", loaded)


def arguments(experiment):
    return (experiment["parent"], experiment["child"], experiment["checkpoints"][96], experiment["checkpoints"][24])


def test_grouping_float64_accumulation_and_finite_rounding():
    x = np.tile([1e8, 1, -1e8, 1], (2, 24)).astype(np.float32)
    x[1, 4:8] = [1, 2, 3, 4]
    hourly = comparison.hourly_sum(x)
    assert hourly.dtype == np.float32 and hourly.shape == (2, 24)
    assert np.all(hourly[0] == 2) and hourly[1, 1] == 10
    assert hourly[1, 0] == hourly[1, 2] == 2
    for invalid in (np.ones((2, 24)), np.full((1, 96), np.inf), np.full((1, 96), np.finfo(np.float32).max)):
        with pytest.raises(ValueError):
            comparison.hourly_sum(invalid)


def test_child_preflight_and_explicit_parent_binding_precede_parent_load(monkeypatch):
    calls = []

    def rejected_child(path):
        calls.append(path)
        assert path == "child24"
        raise ValueError("development input aliases reserved TEST")

    monkeypatch.setattr(comparison, "_load_prepared", rejected_child)
    with pytest.raises(ValueError, match="aliases reserved TEST"):
        comparison._prepared_pair("supplied96", "child24")
    assert calls == ["child24"]

    def mismatched_child(path):
        assert path == "child24", "explicit parent was opened before its identity was bound"
        return None, None, {"derived": {"schema_version": 1, "parent": {
            "prepared_dir": "../different96", "path_policy": "relative_to_child_prepared_directory"}}}

    monkeypatch.setattr(comparison, "_load_prepared", mismatched_child)
    with pytest.raises(ValueError, match="supplied 96-slot parent"):
        comparison._prepared_pair("supplied96", "child24")


@pytest.mark.parametrize("location", ["parent_train", "child_metadata"])
def test_reserved_alias_rejected_before_any_loader_content_read(tmp_path, monkeypatch, location):
    parent, child = tmp_path / "parent96", tmp_path / "child24"
    parent.mkdir()
    child.mkdir()
    reserved = parent / "test.npz"
    reserved.write_bytes(b"synthetic reserved content must never be read")
    alias = parent / "train.npz" if location == "parent_train" else child / "metadata.json"
    alias.symlink_to(reserved)
    monkeypatch.setattr(comparison, "_load_prepared", lambda *a: pytest.fail("content read before TEST alias rejection"))
    with pytest.raises(ValueError, match="aliases reserved TEST"):
        comparison._prepared_pair(parent, child)


def test_relocated_pair_keeps_hashed_checkpoint_and_map_bindings(experiment, forbid_test_access, tmp_path):
    relocated = {}
    for width, key, manifest in ((96, "parent", "curation_manifest.json"),
                                 (24, "child", "derivation_manifest.json")):
        source = experiment[key]
        destination = tmp_path / source.name
        destination.mkdir()
        for name in ("train.npz", "val.npz", "metadata.json", "scaler.json", manifest):
            shutil.copy2(source / name, destination / name)
        relocated[width] = destination
    parent, child, row_bindings, manifests = comparison._prepared_pair(relocated[96], relocated[24])
    for width, bundle, manifest, manifest_key in (
        (96, parent, "curation_manifest.json", "parent_curation_manifest"),
        (24, child, "derivation_manifest.json", "hourly_derivation_manifest"),
    ):
        assert bundle[2:] == experiment["bundles"][width][2:]
        details = comparison._model(experiment["checkpoints"][width], bundle, manifest,
                                    manifests[manifest_key]["sha256"])
        _, map_details = comparison._saved_map(experiment["maps"][width], details, bundle, 2026)
        assert map_details["sha256"] == digest(experiment["maps"][width])
    assert child[2]["derived"]["parent"]["path_policy"] == "relative_to_child_prepared_directory"
    assert not Path(child[2]["derived"]["parent"]["prepared_dir"]).is_absolute()
    assert all(row["same_order_and_values"] for row in row_bindings.values())


def test_saved_maps_before_sum_full_train_controls_and_replay(experiment, forbid_test_access, monkeypatch):
    import fluvius_cgan.calibration as calibration

    def no_fit(*args, **kwargs):
        raise AssertionError("a saved calibration map must never be fitted during comparison")

    monkeypatch.setattr(calibration, "fit", no_fit)
    train = experiment["bundles"][24][0]
    original_fit = comparison.fit_conditional_copula
    fit_inputs = []

    def fitted(x, y, **kwargs):
        np.testing.assert_array_equal(x, train["x"])
        np.testing.assert_array_equal(y, train["y"])
        fit_inputs.append(kwargs)
        return original_fit(x, y, **kwargs)

    monkeypatch.setattr(comparison, "fit_conditional_copula", fitted)
    options = {"map96": experiment["maps"][96], "map24": experiment["maps"][24],
               "candidate96": experiment["checkpoints"][96], "candidate_map96": experiment["maps"][96],
               "samples_per_condition": 2, "seed": 2026}
    result = comparison.compare(*arguments(experiment), **options)
    replay = comparison.compare(*arguments(experiment), **options)
    assert result["methods"] == replay["methods"]
    assert result["frozen_samples"] == replay["frozen_samples"]
    assert result["validation_selection"] == replay["validation_selection"]
    assert len(fit_inputs) == 4 and all(v["quantile_knots"] == 1025 for v in fit_inputs)
    assert {v["shrinkage"] for v in fit_inputs} == {.01, 1.0}
    assert set(result["methods"]) == {"bce96_raw", "bce24_raw", "bce96_calibrated", "bce24_calibrated",
                                       "conditional_mean24", "conditional_resampling24", "gaussian_copula24",
                                       "independent_marginals24", "candidate96_raw", "candidate96_calibrated"}
    selected = np.array(result["validation_selection"]["indices"])
    labels = np.array(result["validation_selection"]["labels"])
    val = experiment["bundles"][96][1]
    np.testing.assert_array_equal(labels, val["y"][selected])
    assert result["validation_selection"]["meter_ids"] == val["meter_id"][selected].tolist()
    assert result["validation_selection"]["dates"] == val["date"][selected].tolist()
    expected_real = val["x"][selected].astype(np.float64).reshape(-1, 24, 4).sum(axis=2).astype(np.float32)
    assert result["frozen_samples"]["real_hourly_sha256"] == array_digest(expected_real)
    for width in (96, 24):
        raw = generate(experiment["checkpoints"][width], labels, seed=2026)
        model = ConditionalMarginalCalibrator.load(experiment["maps"][width])
        mapped = model.transform(raw, labels)
        expected = mapped.astype(np.float64).reshape(-1, 24, 4).sum(axis=2).astype(np.float32) if width == 96 else mapped
        method = f"bce{width}_calibrated"
        assert result["frozen_samples"]["profiles_sha256"][method] == array_digest(expected)
        assert result["methods"][method] == comparison.evaluate_conditional(expected_real, labels, expected, labels, interval_hours=1.)
    assert result["frozen_samples"]["profiles_sha256"]["bce96_raw"] != result["frozen_samples"]["profiles_sha256"]["bce96_calibrated"]
    for suffix in ("raw", "calibrated"):
        assert result["methods"][f"candidate96_{suffix}"] == result["methods"][f"bce96_{suffix}"]
    for name, expected in (
        ("conditional_mean24", comparison.mean_baseline(train["x"], train["y"], labels)),
        ("conditional_resampling24", comparison.resample_baseline(train["x"], train["y"], labels, seed=2026)),
    ):
        assert result["frozen_samples"]["profiles_sha256"][name] == array_digest(expected)
    for result_method in result["methods"].values():
        for row in result_method["per_condition"].values():
            assert {lag: value["lag_hours"] for lag, value in row["autocorrelation"].items()} == {"1": 1., "6": 6.}
    assert all(v["directions"]["n_features"] == 24 and v["directions"]["seed"] == 9182 for v in result["joint_diagnostics"].values())
    assert len({v["directions"]["sha256"] for v in result["joint_diagnostics"].values()}) == 1
    assert all(v["scenario"]["interval_hours"] == 1. for v in result["battery_diagnostics"].values())
    assert result["test_used"] is False and result["split"] == "val"
    assert result["training_profiles"]["controls"] == len(train["x"])


@pytest.mark.parametrize("key", ["y", "meter_id", "date", "x", "order"])
def test_loaded_row_or_profile_mismatch_rejected_before_generation(experiment, forbid_test_access, monkeypatch, key):
    parent = experiment["bundles"][96]
    child = copy.deepcopy(experiment["bundles"][24])
    if key == "order":
        for name in child[1]:
            child[1][name] = child[1][name][::-1]
    elif key == "x":
        child[1][key][0, 0] += 1
    elif key == "y":
        child[1][key][0] += 8
    else:
        child[1][key][0] = child[1][key][4 if key == "meter_id" else 1]
    monkeypatch.setattr(comparison, "_load_prepared", lambda path: parent if Path(path) == experiment["parent"] else child)
    monkeypatch.setattr(comparison, "_generate_verified", lambda *a, **k: pytest.fail("generation before row checks"))
    with pytest.raises(ValueError, match="row identities|four-interval sums"):
        comparison.compare(*arguments(experiment), samples_per_condition=2)


@pytest.mark.parametrize("mutation", ["subset", "seed", "objective", "manifest", "map"])
def test_mismatched_checkpoint_or_map_is_rejected(experiment, forbid_test_access, tmp_path, mutation):
    args = list(arguments(experiment))
    kwargs = {"samples_per_condition": 2}
    if mutation == "map":
        kwargs.update(map96=experiment["maps"][24], map24=experiment["maps"][24])
    else:
        state = torch.load(args[3], map_location="cpu", weights_only=True)
        if mutation == "subset":
            state["config"]["max_train_samples"] = 12
            state["selection"]["train_profiles_used"] = 12
            idx = np.sort(np.random.default_rng(11).choice(16, 12, replace=False))
            state["selection"]["train_indices_sha256"] = array_digest(idx, "<i8")
        elif mutation == "seed":
            state["config"]["seed"] = 22
        elif mutation == "objective":
            state["config"]["objective"] = "hinge_sn"
        else:
            state["data_hashes"]["derivation_manifest.json"] = "f" * 64
        changed = tmp_path / "changed.pt"
        torch.save(state, changed)
        args[3] = changed
    with pytest.raises(ValueError, match="full actual TRAIN|configurations|fingerprint|calibration"):
        comparison.compare(*args, **kwargs)


@pytest.mark.parametrize("kwargs", [{"samples_per_condition": True}, {"samples_per_condition": 0},
                                    {"seed": -1}, {"seed": 2**32}, {"map96": "x"},
                                    {"candidate_map96": "x"}])
def test_bad_options_fail_before_data_reads(kwargs, monkeypatch):
    monkeypatch.setattr(comparison, "_prepared_pair", lambda *a: pytest.fail("invalid options read prepared arrays"))
    with pytest.raises(ValueError):
        comparison.compare("parent", "child", "a.pt", "b.pt", **kwargs)


def test_cli_has_no_test_option_and_preserves_existing_output(tmp_path):
    with pytest.raises(SystemExit):
        comparison.main(["--split", "test"])
    output = tmp_path / "existing.json"
    output.write_text("keep")
    with pytest.raises(SystemExit):
        comparison.main(["--prepared96", "x", "--prepared24", "y", "--checkpoint96", "a",
                         "--checkpoint24", "b", "--output", str(output)])
    assert output.read_text() == "keep"
