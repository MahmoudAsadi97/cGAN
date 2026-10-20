"""Synthetic contracts for supplemental raw unconditional reference evaluation.

The completed-run contract uses the real declared counts and fingerprints. The
orchestration fixture replaces the verified full-data boundary with tiny arrays;
it never opens research data, trains a model, or fits a conditional calibrator.
"""

import builtins
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from fluvius_cgan import calibration
from fluvius_cgan.evaluation import evaluate_conditional, mean_baseline, resample_baseline
from fluvius_cgan.joint import conditional_projected_wasserstein


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "unconditional_reference", ROOT / "scripts" / "evaluate_unconditional_reference.py")
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)
_REAL_LOAD_RUN = evaluation._load_run


def _digest(values, dtype):
    return hashlib.sha256(np.asarray(values, dtype=dtype).tobytes(order="C")).hexdigest()


def _set_path(record, path, value):
    parts = path.split(".")
    for part in parts[:-1]:
        record = record[int(part)] if isinstance(record, list) else record[part]
    record[int(parts[-1]) if isinstance(record, list) else parts[-1]] = value


@pytest.fixture(scope="module")
def contract_template():
    config = evaluation.expected_config(11)
    history = [
        {"epoch": epoch, "loss_g": .7, "loss_d": 1.3,
         "validation_score": (abs(epoch - 8) + 1) / 10,
         "generator_updates": 1184, "discriminator_updates": 1184,
         "samples_seen": 606208, "loss_swd": 0., "loss_derivative_swd": 0.,
         "class_exposure_counts": {str(label): 9472 for label in range(64)}}
        for epoch in range(1, 27)
    ]
    common = {
        "condition_dim": 0, "validation_metric": evaluation.CONDITIONAL_VALIDATION_METRIC,
        "supported_labels": list(range(64)), "derivative_swd_scale": None,
        "best_epoch": 8, "best_validation_score": .1,
        "data_hashes": {name: "a" * 64 for name in (*evaluation.PREPARED_FILES, "curation_manifest.json")},
        "selection": {"train_profiles_available": 606060, "train_profiles_used": 606060,
                      "train_indices_sha256": _digest(np.arange(606060), "<i8"),
                      "validation_selection": "conditional_w1", "validation_samples_requested": 4096,
                      "batch_sampling": "balanced_conditions", "swd_conditioning": "pooled"},
        "scaler": {"minimum": -2., "maximum": 8.},
        "data_curation": {"status": "verified_curation_v2"},
        "provenance": {"git_commit": evaluation.SOURCE_COMMIT, "git_commit_tree": evaluation.SOURCE_TREE,
                       "git_dirty": False, "source_files_sha256": dict(evaluation.TRAINING_SOURCE_HASHES)},
    }

    def checkpoint(epoch):
        return {**copy.deepcopy(common), "config": copy.deepcopy(config), "epoch": epoch,
                "output_dim": 96, "validation_score": history[epoch - 1]["validation_score"],
                "generator_updates": epoch * 1184, "discriminator_updates": epoch * 1184,
                "samples_seen": epoch * 606208,
                "class_exposure_counts": {str(label): epoch * 9472 for label in range(64)},
                "epoch_metrics": copy.deepcopy(history[epoch - 1])}

    best, last = checkpoint(8), checkpoint(26)
    summary = {**copy.deepcopy(common), "resolution": 96, "target": "net", "units": "kWh/interval",
               "source_sha256": evaluation.RELEASE_SHA256, "epochs_completed": 26,
               "stopped_early": False, "test_used": False,
               "generator_updates": 30784, "discriminator_updates": 30784,
               "samples_seen": 15761408,
               "class_exposure_counts": {str(label): 246272 for label in range(64)},
               "last_epoch_metrics": copy.deepcopy(history[-1])}
    return {"config": config, "summary": summary, "history": history, "best": best, "last": last}


@pytest.fixture
def contract(contract_template):
    return copy.deepcopy(contract_template)


def _verify(contract, *, training_seed=11, train_count=606060):
    return evaluation._verify_run_contract(**contract, training_seed=training_seed, train_count=train_count)


def test_declared_source_counts_and_configuration_are_fixed():
    assert (evaluation.TRAIN_PROFILES, evaluation.EPOCHS, evaluation.UPDATES_PER_EPOCH, evaluation.UPDATES) == (606060, 26, 1184, 30784)
    assert evaluation.SOURCE_COMMIT == "1ca62d0fbe1d77f916f8904e812db7d720f8cf53"
    assert evaluation.SOURCE_TREE == "55b35f8204241eaabc58f5196cb14fff5b158cad"
    assert len(evaluation.TRAINING_SOURCE_HASHES) == 18
    assert evaluation.TRAINING_SOURCE_HASHES["src/fluvius_cgan/training.py"] == "1eabd6ee18c381ba16a2af659247400d0d08557c8fcf7a66dcd3c6fc16b5236a"
    for seed in (11, 22, 33, 44, 55):
        config = evaluation.expected_config(seed)
        assert config["seed"] == seed and config["unconditional"] is True
        assert config["objective"] == "bce" and config["normalization"] == "minmax"
        assert config["epochs"] == 26 and config["batch_size"] == 512
        assert config["latent_dim"] == 24 and config["hidden_dim"] == config["discriminator_hidden_dim"] == 64
        assert config["max_train_samples"] is None and config["patience"] is None
        assert config["allow_uncurated_data"] is False


def test_completed_budget_accepts_earlier_selected_best(contract):
    result = _verify(contract)
    assert result["epochs_completed"] == 26
    assert result["generator_updates"] == result["discriminator_updates"] == 30784
    assert result["best_epoch"] == 8 and result["selected_checkpoint_updates"] == 9472
    assert result["train_profiles"] == 606060 and result["full_train"] is True


@pytest.mark.parametrize("path,value", [
    ("config.unconditional", False), ("config.max_train_samples", 606060),
    ("config.allow_uncurated_data", True), ("best.config.objective", "hinge_sn"),
    ("last.config.seed", 22), ("summary.epochs_completed", 25),
    ("summary.stopped_early", True), ("summary.test_used", True),
    ("last.epoch", 25), ("history.3.epoch", 5),
    ("history.3.generator_updates", 1183), ("history.3.samples_seen", 606060),
    ("history.3.loss_g", float("nan")), ("history.3.loss_swd", .1),
    ("history.3.class_exposure_counts.0", 1),
    ("summary.best_epoch", 7), ("best.validation_score", .2),
    ("last.validation_score", .5), ("best.condition_dim", 64),
    ("last.output_dim", 24), ("summary.resolution", 24),
    ("summary.target", "import"), ("summary.units", "kW"),
    ("summary.source_sha256", "0" * 64),
    ("summary.supported_labels", list(range(63))),
    ("best.derivative_swd_scale", {"used_float32": 1.}),
    ("summary.generator_updates", 30783), ("last.discriminator_updates", 30783),
    ("best.generator_updates", 30784), ("summary.samples_seen", 15761407),
    ("last.class_exposure_counts.0", 0), ("best.epoch_metrics.loss_g", 3.),
    ("summary.last_epoch_metrics.loss_d", 5.),
    ("summary.data_hashes.train.npz", "0" * 64),
])
def test_completed_contract_rejects_drift(contract, path, value):
    # Filename keys contain dots and are handled explicitly below.
    if path == "summary.data_hashes.train.npz":
        contract["summary"]["data_hashes"]["train.npz"] = value
    else:
        _set_path(contract, path, value)
    with pytest.raises(ValueError):
        _verify(contract)


@pytest.mark.parametrize("field,value", [
    ("git_commit", "2" * 40), ("git_commit_tree", "3" * 40), ("git_dirty", True),
    ("source_files_sha256", {"src/fluvius_cgan/training.py": "a" * 64}),
])
def test_source_pin_cannot_be_rebound_consistently(contract, field, value):
    for name in ("summary", "best", "last"):
        contract[name]["provenance"][field] = copy.deepcopy(value)
    with pytest.raises(ValueError, match="source/tree/file"):
        _verify(contract)


@pytest.mark.parametrize("field,value", [
    ("train_profiles_used", 606059), ("train_indices_sha256", "0" * 64),
    ("validation_selection", "pooled_w1"), ("validation_samples_requested", 2048),
    ("batch_sampling", "shuffle"), ("swd_conditioning", "conditional"),
])
def test_selection_contract_cannot_be_rebound_consistently(contract, field, value):
    for name in ("summary", "best", "last"):
        contract[name]["selection"][field] = value
    with pytest.raises(ValueError):
        _verify(contract)


@pytest.mark.parametrize("filename", ["test.npz", "unexpected.json"])
def test_run_data_hashes_cannot_claim_test_or_extra_inputs(contract, filename):
    for name in ("summary", "best", "last"):
        contract[name]["data_hashes"][filename] = "f" * 64
    with pytest.raises(ValueError):
        _verify(contract)


@pytest.mark.parametrize("count", [606059, 606061, 606060., True])
def test_complete_training_count_is_not_a_nominal_claim(contract, count):
    with pytest.raises(ValueError, match="606060"):
        _verify(contract, train_count=count)


def test_tied_validation_minimum_selects_earliest_epoch(contract):
    contract["history"][6]["validation_score"] = .1
    with pytest.raises(ValueError, match="earliest minimum"):
        _verify(contract)


@pytest.fixture
def forbidden_operations(monkeypatch):
    path_open, builtin_open, numpy_load = Path.open, builtins.open, np.load

    def check(path):
        if isinstance(path, (str, bytes, Path)):
            name = Path(path.decode() if isinstance(path, bytes) else path).name
            assert name != "test.npz", "reserved TEST must never be read, loaded or hashed"

    def guarded_path_open(path, *args, **kwargs):
        check(path)
        return path_open(path, *args, **kwargs)

    def guarded_builtin_open(path, *args, **kwargs):
        check(path)
        return builtin_open(path, *args, **kwargs)

    def guarded_numpy_load(path, *args, **kwargs):
        check(path)
        return numpy_load(path, *args, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("conditional calibrator must not be fitted, loaded, transformed or saved")

    monkeypatch.setattr(Path, "open", guarded_path_open)
    monkeypatch.setattr(builtins, "open", guarded_builtin_open)
    monkeypatch.setattr(np, "load", guarded_numpy_load)
    monkeypatch.setattr(calibration, "fit", forbidden)
    for name in ("load", "transform", "save"):
        monkeypatch.setattr(calibration.ConditionalMarginalCalibrator, name, forbidden)
    import evaluate_calibration
    monkeypatch.setattr(evaluate_calibration, "fit", forbidden)


@pytest.fixture
def toy(tmp_path, monkeypatch):
    prepared, run = tmp_path / "prepared", tmp_path / "run"
    prepared.mkdir()
    run.mkdir()
    rng = np.random.default_rng(71)
    train_y = np.repeat(np.arange(64), 2)
    val_y = np.concatenate([np.zeros(101, dtype=np.int64), np.repeat(np.arange(1, 64), 2)])
    arrays = {}
    for split, labels in (("train", train_y), ("val", val_y)):
        arrays[split] = {"x": (rng.normal(size=(len(labels), 96)) + (10 if split == "val" else -3)).astype(np.float32),
                         "y": labels, "meter_id": np.array([f"{split}-{i}" for i in range(len(labels))]),
                         "date": np.repeat("2024-01-01", len(labels))}
        np.savez(prepared / f"{split}.npz", **arrays[split])
    (prepared / "test.npz").write_bytes(b"sealed synthetic TEST sentinel")
    metadata = {"resolution": 96, "target": "net", "units": "kWh/interval",
                "source_sha256": evaluation.RELEASE_SHA256, "curation": {"schema_version": 1},
                "splits": {name: {"sha256": evaluation.sha256_file(prepared / f"{name}.npz")}
                           for name in ("train", "val")}}
    scaler = {"minimum": float(arrays["train"]["x"].min()), "maximum": float(arrays["train"]["x"].max())}
    for name, values in (("metadata.json", metadata), ("scaler.json", scaler), ("curation_manifest.json", {"synthetic": True})):
        (prepared / name).write_text(json.dumps(values))
    hashes = {name: evaluation.sha256_file(prepared / name) for name in evaluation.PREPARED_FILES}
    guard = {"status": "verified_curation_v2", "pinned_release": True,
             "verified_file_hashes": dict(hashes),
             "manifest_sha256": evaluation.sha256_file(prepared / "curation_manifest.json")}
    for name in evaluation.RUN_FILES:
        (run / name).write_bytes(f"synthetic bound artifact: {name}".encode())
    artifacts = {name: evaluation.sha256_file(run / name) for name in evaluation.RUN_FILES}
    selection = {"train_profiles_used": len(train_y), "train_profiles_available": len(train_y),
                 "train_indices_sha256": _digest(np.arange(len(train_y)), "<i8")}
    details = {"sha256": artifacts["best.pt"], "training_selection": selection, "config": evaluation.expected_config(11)}
    evidence = {"full_train": True, "synthetic_verified_boundary": True,
                "primary_raw_report": {"sha256": artifacts["validation_report.json"], "superseded": False}}
    bundle = arrays["train"], arrays["val"], metadata, scaler, hashes
    calls = {"generate": [], "fits": [], "mean": [], "resample": []}
    generated = {}

    def load(path):
        assert Path(path) == prepared
        return bundle

    def load_run(checkpoint, train, val, md, scale, supplied_hashes, supplied_guard, training_seed):
        assert Path(checkpoint) == run / "best.pt" and training_seed == 11
        assert train is arrays["train"] and val is arrays["val"]
        assert md is metadata and scale is scaler and supplied_hashes == hashes and supplied_guard == guard
        return details, np.arange(len(train_y)), evidence, artifacts

    def raw(checkpoint, supplied_details, labels, seed):
        assert Path(checkpoint) == run / "best.pt" and supplied_details is details and seed == 2026
        calls["generate"].append(labels.copy())
        # Deliberately ignores labels, as an unconditional generator must.
        values = np.random.default_rng(seed).normal(size=(len(labels), 96)) - 4
        generated["raw_unconditional_cgan"] = values.copy()
        return values

    def mean(x, y, labels):
        calls["mean"].append((x.copy(), y.copy()))
        values = mean_baseline(x, y, labels)
        generated["conditional_mean"] = values.copy()
        return values

    def resample(x, y, labels, seed):
        calls["resample"].append((x.copy(), y.copy(), seed))
        values = resample_baseline(x, y, labels, seed=seed)
        generated["conditional_resampling"] = values.copy()
        return values

    def fit(x, y, *, quantile_knots, shrinkage):
        calls["fits"].append((x.copy(), y.copy(), quantile_knots, shrinkage))

        class Model:
            def generate(self, labels, seed):
                assert seed == 2026
                values = mean_baseline(x, y, labels) + np.random.default_rng(seed).normal(size=(len(labels), 96)) * shrinkage
                generated["gaussian_copula" if shrinkage == .01 else "independent_marginals"] = values.copy()
                return values

        return Model()

    monkeypatch.setattr(evaluation, "_load_prepared", load)
    monkeypatch.setattr(evaluation, "_load_run", load_run)
    monkeypatch.setattr(evaluation, "_training_curation_guard", lambda path, config: guard)
    monkeypatch.setattr(evaluation, "_generate_verified", raw)
    monkeypatch.setattr(evaluation, "mean_baseline", mean)
    monkeypatch.setattr(evaluation, "resample_baseline", resample)
    monkeypatch.setattr(evaluation, "fit_conditional_copula", fit)
    monkeypatch.setattr(evaluation, "_provenance", lambda device: {"synthetic": True})
    return {"prepared": prepared, "run": run, "arrays": arrays, "bundle": bundle,
            "guard": guard, "details": details, "evidence": evidence, "artifacts": artifacts,
            "calls": calls, "generated": generated}


def _call(toy, **kwargs):
    return evaluation.evaluate(toy["prepared"], toy["run"] / "best.pt", training_seed=11, **kwargs)


def _selected(toy, cap):
    labels = toy["arrays"]["val"]["y"]
    rng = np.random.default_rng(2026)
    return np.concatenate([rng.choice(np.flatnonzero(labels == label), min(cap, int(np.sum(labels == label))), replace=False)
                           for label in range(64)])


@pytest.mark.parametrize("cap", [100, 1000])
def test_raw_workflow_freezes_common_validation_rows_and_uses_only_train_controls(toy, forbidden_operations, cap):
    report = _call(toy, samples_per_condition=cap)
    selected = _selected(toy, cap)
    train, val = toy["arrays"]["train"], toy["arrays"]["val"]
    real, labels = val["x"][selected], val["y"][selected]
    assert len(toy["calls"]["generate"]) == 1
    np.testing.assert_array_equal(toy["calls"]["generate"][0], labels)
    for x, y, *rest in toy["calls"]["mean"] + toy["calls"]["resample"] + toy["calls"]["fits"]:
        np.testing.assert_array_equal(x, train["x"])
        np.testing.assert_array_equal(y, train["y"])
    assert [(row[2], row[3]) for row in toy["calls"]["fits"]] == [(1025, .01), (1025, 1.)]
    assert toy["calls"]["resample"][0][2] == 2026
    assert set(report["methods"]) == set(evaluation.METHODS) == set(toy["generated"])
    assert report["resolution"] == 96 and report["interval_hours"] == .25 and report["units"] == "kWh/interval"
    assert report["split"] == "val" and report["test_used"] is False
    assert report["conditional_calibrator_used"] is False and report["supersedes_primary_validation_report"] is False
    assert report["data_curation"] == toy["guard"]
    assert report["curation_manifest_sha256"] == toy["guard"]["manifest_sha256"]
    assert report["run_artifacts_sha256"] == toy["artifacts"]
    assert report["completed_training_run"]["primary_raw_report"]["superseded"] is False
    assert report["validation_selection"]["selected_indices_sha256"] == _digest(selected, "<i8")
    assert report["validation_selection"]["condition_counts"]["0"] == min(101, cap)
    assert report["validation_selection"]["selected_profiles"] == min(101, cap) + 126
    assert report["frozen_samples"]["profile_shape"] == [len(selected), 96]
    assert report["frozen_samples"]["labels_sha256"] == _digest(labels, "<i8")
    assert report["frozen_samples"]["real_profiles_sha256"] == _digest(real, "<f8")
    assert report["training_reference"]["profiles_sha256"] == _digest(train["x"], "<f8")
    assert report["training_reference"]["labels_sha256"] == _digest(train["y"], "<i8")
    assert report["training_profiles"] == {"prepared": 128, "model": 128, "baseline": 128}
    assert report["uncertainty"]["computed"] is False
    assert "current separately fingerprinted evaluator" in report["supplemental_numerical_version_disclosure"]
    assert "test.npz" not in report["prepared_data_hashes"]
    for method, values in toy["generated"].items():
        assert report["frozen_samples"]["profiles_sha256"][method] == _digest(values, "<f8")
        assert report["methods"][method] == evaluate_conditional(real, labels, values, labels, interval_hours=.25)
        assert report["joint_diagnostics"][method] == conditional_projected_wasserstein(real, labels, values, labels)
        manual_battery = evaluation._battery_diagnostics(evaluation.simulate_battery(real, .25), values, labels, .25)
        assert report["battery_diagnostics"][method] == manual_battery
    assert np.any(toy["generated"]["raw_unconditional_cgan"] < 0)
    assert len({row["directions"]["sha256"] for row in report["joint_diagnostics"].values()}) == 1
    assert len({row["scenario_sha256"] for row in report["battery_diagnostics"].values()}) == 1
    assert "copula.py" in " ".join(report["provenance"]["supplemental_evaluation_source_files_sha256"])
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("kwargs", [{"training_seed": 12}, {"training_seed": True}, {"seed": 2025},
                                    {"seed": 2026.}, {"samples_per_condition": 99},
                                    {"samples_per_condition": 100.}, {"samples_per_condition": True}])
def test_invalid_options_fail_before_any_data_access(monkeypatch, kwargs):
    monkeypatch.setattr(evaluation, "_training_curation_guard", lambda *a: pytest.fail("unexpected data access"))
    monkeypatch.setattr(evaluation, "_load_prepared", lambda *a: pytest.fail("unexpected data access"))
    with pytest.raises(ValueError):
        evaluation.evaluate("unopened", "unopened/best.pt", **{"training_seed": 11, **kwargs})


@pytest.mark.parametrize("field,value", [("resolution", 24), ("target", "import"), ("units", "kW"),
                                         ("source_sha256", "a" * 64), ("derived", {})])
def test_only_native_pinned_signed_net_data_accepted(toy, forbidden_operations, field, value):
    toy["bundle"][2][field] = value
    with pytest.raises(ValueError, match="native curated96"):
        _call(toy)
    assert not toy["calls"]["generate"] and not toy["calls"]["fits"]


@pytest.mark.parametrize("field,value", [("status", "uncurated_explicit_override"), ("pinned_release", False)])
def test_curation_override_cannot_qualify_reference(toy, forbidden_operations, field, value):
    toy["guard"][field] = value
    with pytest.raises(ValueError, match="identity-curated"):
        _call(toy)
    assert not toy["calls"]["generate"]


def test_changed_prepared_binding_rejected_before_generation(toy, forbidden_operations):
    toy["guard"]["verified_file_hashes"]["train.npz"] = "0" * 64
    with pytest.raises(ValueError, match="after curation"):
        _call(toy)
    assert not toy["calls"]["generate"]


@pytest.mark.parametrize("split", ["train", "val"])
def test_missing_condition_rejected_before_generation(toy, forbidden_operations, split):
    toy["arrays"][split]["y"][toy["arrays"][split]["y"] == 63] = 62
    with pytest.raises(ValueError, match="all64"):
        _call(toy)
    assert not toy["calls"]["generate"]


def test_reordered_train_control_selection_rejected(toy, forbidden_operations, monkeypatch):
    original = evaluation._load_run

    def reversed_rows(*args):
        details, indices, evidence, artifacts = original(*args)
        return details, indices[::-1], evidence, artifacts

    monkeypatch.setattr(evaluation, "_load_run", reversed_rows)
    with pytest.raises(ValueError, match="full ordered TRAIN"):
        _call(toy)
    assert not toy["calls"]["generate"]


@pytest.mark.parametrize("filename", ["train.npz", "curation_manifest.json", "validation_report.json", "best.pt"])
def test_artifact_changes_during_evaluation_fail_closed(toy, forbidden_operations, monkeypatch, filename):
    original = evaluation._generate_verified

    def mutate(*args):
        values = original(*args)
        directory = toy["prepared"] if filename in ("train.npz", "curation_manifest.json") else toy["run"]
        with (directory / filename).open("ab") as handle:
            handle.write(b"changed during synthetic evaluation")
        return values

    monkeypatch.setattr(evaluation, "_generate_verified", mutate)
    with pytest.raises(ValueError, match="changed during supplemental"):
        _call(toy)


@pytest.mark.parametrize("kind", ["symlink", "hardlink"])
def test_input_alias_to_reserved_test_is_rejected_without_reading(toy, forbidden_operations, kind):
    import os
    path = toy["prepared"] / "train.npz"
    path.unlink()
    if kind == "symlink":
        path.symlink_to(toy["prepared"] / "test.npz")
    else:
        os.link(toy["prepared"] / "test.npz", path)
    with pytest.raises(ValueError, match="aliases reserved TEST"):
        _call(toy)
    assert not toy["calls"]["generate"]


def test_cli_separate_output_no_overwrite_and_no_test_or_calibration_option(toy, forbidden_operations, monkeypatch, tmp_path):
    calls = []

    def evaluate(*args, **kwargs):
        calls.append((args, kwargs))
        return {"split": "val", "test_used": False, "conditional_calibrator_used": False}

    monkeypatch.setattr(evaluation, "evaluate", evaluate)
    output = tmp_path / "supplemental.json"
    args = ["--prepared", str(toy["prepared"]), "--checkpoint", str(toy["run"] / "best.pt"),
            "--training-seed", "11", "--output", str(output)]
    evaluation.main(args)
    original = output.read_bytes()
    assert calls[0][1] == {"training_seed": 11, "samples_per_condition": 100, "seed": 2026}
    with pytest.raises(FileExistsError):
        evaluation.main(args)
    assert output.read_bytes() == original and len(calls) == 1
    for extra in (["--split", "test"], ["--calibration-output", "map.npz"], ["--seed", "2025"]):
        with pytest.raises(SystemExit):
            evaluation.main(args + extra)
    for name in evaluation.RUN_FILES:
        with pytest.raises(ValueError, match="separately named"):
            evaluation.main(args[:-1] + [str(tmp_path / name)])
    assert len(calls) == 1


def test_cli_output_alias_to_test_is_rejected_before_evaluation(toy, forbidden_operations, monkeypatch, tmp_path):
    output = tmp_path / "supplemental.json"
    output.symlink_to(toy["prepared"] / "test.npz")
    monkeypatch.setattr(evaluation, "evaluate", lambda *a, **k: pytest.fail("unexpected evaluation"))
    with pytest.raises(ValueError, match="aliases reserved TEST"):
        evaluation.main(["--prepared", str(toy["prepared"]), "--checkpoint", str(toy["run"] / "best.pt"),
                         "--training-seed", "11", "--output", str(output)])


@pytest.fixture
def bound_run(toy, contract, monkeypatch):
    """Real checkpoint/VAL/report bindings; only the full-budget boundary is replaced."""
    train, val, metadata, scaler, hashes = toy["bundle"]
    selected = evaluation._validation_indices(val["y"], list(range(64)), 4096,
                                              np.random.default_rng(12), "conditional_w1")
    noise = torch.randn((len(selected), 24), generator=torch.Generator().manual_seed(13))
    selection = {**copy.deepcopy(contract["best"]["selection"]),
                 "train_profiles_available": len(train["x"]), "train_profiles_used": len(train["x"]),
                 "train_indices_sha256": _digest(np.arange(len(train["x"])), "<i8"),
                 "validation_indices": selected.tolist(),
                 "validation_labels_sha256": _digest(val["y"][selected], "<i8"),
                 "validation_noise_sha256": _digest(noise.numpy(), "<f4"),
                 "validation_profiles_available": len(val["x"]), "validation_profiles_eligible": len(val["x"]),
                 "validation_profiles_used": len(selected), "validation_unsupported_profiles": 0}
    for name in ("best", "last"):
        state = contract[name]
        state.update(format_version=1, metadata=metadata, scaler=scaler, selection=copy.deepcopy(selection),
                     data_hashes={**hashes, "curation_manifest.json": toy["guard"]["manifest_sha256"]},
                     data_curation=toy["guard"])
        torch.save(state, toy["run"] / f"{name}.pt")
    for name in ("config", "summary", "history"):
        (toy["run"] / f"{name}.json").write_text(json.dumps(contract[name], allow_nan=False))
    primary = {"split": "val", "seed": 2026, "samples_per_condition_cap": 100,
               "checkpoint_sha256": evaluation.sha256_file(toy["run"] / "best.pt"),
               "train_sha256": hashes["train.npz"], "evaluation_sha256": hashes["val.npz"],
               "units": "kWh/interval", "target": "net",
               "methods": {name: {} for name in ("cgan", "conditional_mean", "conditional_resampling")},
               "training_profiles": {name: 606060 for name in ("prepared", "model", "baselines")}}
    (toy["run"] / "validation_report.json").write_text(json.dumps(primary))
    budget_calls = []

    def verified_budget(config, summary, history, best, last, *, training_seed, train_count):
        budget_calls.append({"training_seed": training_seed, "train_count": train_count})
        assert training_seed == 11 and train_count == 128
        return {"synthetic_budget_boundary": True}

    monkeypatch.setattr(evaluation, "_verify_run_contract", verified_budget)
    return {**toy, "primary": primary, "budget_calls": budget_calls}


def _load_bound_run(bound_run):
    train, val, metadata, scaler, hashes = bound_run["bundle"]
    return _REAL_LOAD_RUN(bound_run["run"] / "best.pt", train, val, metadata, scaler, hashes,
                          bound_run["guard"], 11)


def test_real_run_loader_binds_selected_checkpoint_actual_val_and_original_report(bound_run, forbidden_operations):
    details, indices, evidence, artifacts = _load_bound_run(bound_run)
    assert details["sha256"] == evaluation.sha256_file(bound_run["run"] / "best.pt")
    assert details["data_hashes"]["curation_manifest.json"] == bound_run["guard"]["manifest_sha256"]
    np.testing.assert_array_equal(indices, np.arange(128))
    assert set(artifacts) == set(evaluation.RUN_FILES)
    assert artifacts["validation_report.json"] == evaluation.sha256_file(bound_run["run"] / "validation_report.json")
    assert evidence["primary_raw_report"]["sha256"] == artifacts["validation_report.json"]
    assert evidence["primary_raw_report"]["superseded"] is False
    assert "legacy raw report lacks its own evaluator source fingerprints" in evidence["primary_raw_report"]["attribution_scope"]
    assert bound_run["budget_calls"] == [{"training_seed": 11, "train_count": 128}]


@pytest.mark.parametrize("field,value", [("split", "test"), ("test_used", True), ("seed", 2025),
                                         ("samples_per_condition_cap", 1000), ("checkpoint_sha256", "0" * 64),
                                         ("train_sha256", "0" * 64), ("evaluation_sha256", "0" * 64),
                                         ("units", "kW"), ("target", "import"), ("methods", {}),
                                         ("training_profiles", {"prepared": 128, "model": 128, "baselines": 128})])
def test_original_primary_report_must_be_bound_and_retained(bound_run, forbidden_operations, field, value):
    bound_run["primary"][field] = value
    (bound_run["run"] / "validation_report.json").write_text(json.dumps(bound_run["primary"]))
    with pytest.raises(ValueError, match="original primary validation report"):
        _load_bound_run(bound_run)


@pytest.mark.parametrize("field,value,error", [
    ("selection.validation_indices", [], "actual validation rows"),
    ("selection.validation_labels_sha256", "0" * 64, "actual validation rows"),
    ("selection.validation_noise_sha256", "0" * 64, "noise fingerprint"),
    ("selection.validation_profiles_used", 1, "actual validation rows"),
    ("data_curation.manifest_sha256", "0" * 64, "curation verification"),
    ("data_hashes.curation_manifest.json", "0" * 64, "curation_manifest"),
    ("data_hashes.val.npz", "0" * 64, "val.npz"),
])
def test_run_loader_rejects_checkpoint_val_and_curation_drift(bound_run, forbidden_operations, field, value, error):
    path = bound_run["run"] / "best.pt"
    state = torch.load(path, weights_only=True)
    section, key = field.split(".", 1)
    state[section][key] = value
    torch.save(state, path)
    # Rebinding the primary checkpoint digest cannot disguise internal drift.
    bound_run["primary"]["checkpoint_sha256"] = evaluation.sha256_file(path)
    (bound_run["run"] / "validation_report.json").write_text(json.dumps(bound_run["primary"]))
    with pytest.raises(ValueError, match=error):
        _load_bound_run(bound_run)


def test_missing_primary_report_cannot_be_silently_reconstructed(bound_run, forbidden_operations):
    (bound_run["run"] / "validation_report.json").unlink()
    with pytest.raises(FileNotFoundError):
        _load_bound_run(bound_run)


def test_run_artifact_mutation_during_binding_is_detected(bound_run, forbidden_operations, monkeypatch):
    original = evaluation._verify_run_contract

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        with (bound_run["run"] / "summary.json").open("a") as handle:
            handle.write("\n")
        return result

    monkeypatch.setattr(evaluation, "_verify_run_contract", mutate)
    with pytest.raises(ValueError, match="run artifacts changed during verification"):
        _load_bound_run(bound_run)
