"""Small synthetic tests of the paired evaluator; no research data or training."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fluvius_cgan.calibration import ConditionalMarginalCalibrator, fit
from fluvius_cgan.evaluation import evaluate_conditional
from fluvius_cgan.utility import RidgeProfileForecaster
from test_evaluate_calibration import forbid_test_access

SPEC = importlib.util.spec_from_file_location("decoder_evaluation_test", Path(__file__).resolve().parents[1] / "scripts/evaluate_quantile_decoder.py")
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


def save_json(path, value):
    Path(path).write_text(json.dumps(value, allow_nan=False))


def reference_values(labels, seed):
    labels = np.asarray(labels)
    result = np.empty((len(labels), 96), dtype=np.float32)
    rng = np.random.default_rng(seed)
    for label in range(64):
        selected = np.flatnonzero(labels == label)
        template = (np.round(rng.normal(size=(8, 96)) * 16) / 16 + label / 64).astype(np.float32)
        result[selected] = template[np.arange(len(selected)) % 8]
    return result


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "TRAIN_ROWS", 128)
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    arrays, splits = {}, {}
    labels = np.repeat(np.arange(64), 2).astype(np.int64)
    for split, seed in (("train", 91), ("val", 92)):
        x = reference_values(labels, seed)
        arrays[split] = {"x": x, "y": labels.copy(), "meter_id": np.asarray([f"{split}_{c//8}_{i%2}" for i, c in enumerate(labels)]),
                         "date": np.asarray(["2024-01-01"] * 128)}
        np.savez(prepared / f"{split}.npz", **arrays[split])
        splits[split] = {"profiles": 128, "meters": 16, "meter_ids": np.unique(arrays[split]["meter_id"]).tolist(),
                         "sha256": evaluation.sha256_file(prepared / f"{split}.npz")}
    metadata = {"resolution": 96, "target": "net", "units": "kWh/interval", "source_sha256": "a" * 64,
                "splits": splits, "curation": {"synthetic": True}}
    tx, ty = arrays["train"]["x"], labels
    scaler = {"minimum": float(tx.min()), "maximum": float(tx.max())}
    save_json(prepared / "metadata.json", metadata)
    save_json(prepared / "scaler.json", scaler)
    save_json(prepared / "curation_manifest.json", {"synthetic": True})
    (prepared / "test.npz").write_bytes(b"reserved synthetic sentinel")
    hashes = {name: evaluation.sha256_file(prepared / name) for name in evaluation.PREPARED_FILES[:-1]}
    manifest = evaluation.sha256_file(prepared / "curation_manifest.json")
    guard = {"status": "verified_curation_v2", "pinned_release": True, "verified_file_hashes": hashes, "manifest_sha256": manifest}
    monkeypatch.setattr(evaluation.training, "_training_curation_guard", lambda *_: guard)
    monkeypatch.setattr(evaluation.training, "_load_prepared", lambda *_: (arrays["train"], arrays["val"], metadata, scaler, hashes))
    required = ("src/fluvius_cgan/training.py", "scripts/evaluate_quantile_decoder.py")
    monkeypatch.setattr(evaluation, "REQUIRED_SOURCES", required)
    source = {"git_commit": "1" * 40, "git_commit_tree": "2" * 40, "git_dirty": False,
              "source_files_sha256": {name: "3" * 64 for name in required}, "runner_sha256": "3" * 64}
    monkeypatch.setattr(evaluation, "_source", lambda *_: copy.deepcopy(source))
    config = evaluation.training._config({"epochs": 26, "batch_size": 512, "seed": 11, "objective": "hinge_sn",
        "normalization": "standard", "discriminator_hidden_dim": 128, "adam_beta1": .5, "swd_weight": 10.,
        "batch_sampling": "balanced_conditions", "conditions_per_batch": 4, "validation_selection": "conditional_w1"})
    fit_metadata = {"selected_profiles": 128, "selected_indices_sha256": evaluation._digest(np.arange(128), "<i8"),
                    "selected_profiles_sha256": evaluation._raw_digest(tx), "selected_labels_sha256": evaluation._digest(ty, "<i8")}
    q = np.stack([np.linspace(tx[ty == label].min(axis=0).astype(np.float64),
                             tx[ty == label].max(axis=0).astype(np.float64), 1025) for label in range(64)])
    decoder = SimpleNamespace(quantiles=torch.from_numpy(q), fit_metadata=fit_metadata)
    monkeypatch.setattr(evaluation.training, "load_quantile_decoded_generator", lambda *_a, **_k: SimpleNamespace(decoder=decoder))
    states, paths = {}, {}
    history = [{"epoch": i, "validation_score": float(27-i), "generator_updates": 1, "discriminator_updates": 1, "samples_seen": 512} for i in range(1, 27)]
    for arm in ("candidate", "reference"):
        directory = tmp_path / arm
        directory.mkdir()
        cfg = {**config, "conditional_quantile_decoder": arm == "candidate"}
        provenance = copy.deepcopy(source)
        if arm == "reference":
            provenance["git_dirty"] = True
            provenance["git_commit"] = "4" * 40
        state = {"format_version": evaluation.training.CHECKPOINT_FORMAT, "config": cfg, "epoch": 26,
            "metadata": metadata, "scaler": {**scaler, "kind": "standard", "mean": float(tx.astype(np.float64).mean()),
                                              "std": float(tx.astype(np.float64).std())}, "output_dim": 96, "condition_dim": 64,
            "supported_labels": list(range(64)), "data_hashes": {**hashes, "curation_manifest.json": manifest},
            "data_curation": guard, "selection": {"train_profiles_available": 128, "train_profiles_used": 128,
                "train_indices_sha256": evaluation._digest(np.arange(128), "<i8")}, "provenance": provenance,
            "validation_metric": evaluation.training.CONDITIONAL_VALIDATION_METRIC, "best_epoch": 26,
            "best_validation_score": 1., "validation_score": 1., "generator_updates": 26,
            "discriminator_updates": 26, "samples_seen": 26*512}
        if arm == "candidate":
            state["quantile_decoder"] = fit_metadata
        for filename in ("best.pt", "last.pt"):
            torch.save(state, directory / filename)
        summary = {**state, "test_used": False, "epochs_completed": 26, "stopped_early": False}
        save_json(directory / "summary.json", summary)
        save_json(directory / "config.json", cfg)
        save_json(directory / "history.json", history)
        states[arm], paths[arm] = state, directory / "best.pt"
    models, source_hash = {}, hashlib.sha256()
    for label in range(64):
        sy = np.full(4096, label, dtype=np.int64)
        sx = reference_values(sy, 7301)
        source_hash.update(np.asarray(sx, dtype="<f8").tobytes())
        fitted = fit(tx[ty == label], ty[ty == label], sx, sy, quantile_knots=1025)
        models[label] = fitted.models[label]
    sy = np.repeat(np.arange(64, dtype=np.int64), 4096)
    provenance = {"checkpoint_sha256": evaluation.sha256_file(paths["reference"]),
        "train_indices_sha256": evaluation._digest(np.arange(128), "<i8"), "prepared_data_hashes": hashes,
        "calibration_seed": 7301, "calibration_samples_per_condition": 4096, "calibration_profiles_total": len(sy),
        "calibration_generated_profiles_sha256": source_hash.hexdigest(), "calibration_labels_sha256": evaluation._digest(sy, "<i8"),
        "generator_inference_batch_size": 1024, "quantile_knots": 1025}
    fixed = ConditionalMarginalCalibrator(96, 1025, models, provenance)
    map_path = tmp_path / "reference" / "calibration.npz"
    fixed.save(map_path)
    binding = {"checkpoint_sha256": evaluation.sha256_file(paths["reference"]), "calibration_sha256": evaluation.sha256_file(map_path),
        "selected_epoch": 26, "config": states["reference"]["config"], "prepared_data_hashes": hashes,
        "checkpoint_data_hashes": states["reference"]["data_hashes"], "training_source": states["reference"]["provenance"]}
    bindings = tmp_path / "bindings.json"
    save_json(bindings, {"seeds": {"11": binding}})
    monkeypatch.setattr(evaluation, "BASELINE_BINDINGS", bindings)
    monkeypatch.setattr(evaluation, "BASELINE_BINDINGS_SHA256", evaluation.sha256_file(bindings))
    config_path, protocol_path = tmp_path / "config.json", tmp_path / "protocol.md"
    save_json(config_path, states["candidate"]["config"])
    protocol_path.write_text("Synthetic frozen protocol\n")
    monkeypatch.setattr(evaluation, "CONFIG_PATH", config_path)
    monkeypatch.setattr(evaluation, "PROTOCOL_PATH", protocol_path)
    declaration = {"schema_version": 1, "protocol_id": "conditional_physical_quantile_decoder_v1",
        "source_commit": source["git_commit"], "source_tree": source["git_commit_tree"], "source_files_sha256": source["source_files_sha256"],
        "config_sha256": evaluation.sha256_file(config_path), "protocol_sha256": evaluation.sha256_file(protocol_path),
        "baseline_bindings_sha256": evaluation.BASELINE_BINDINGS_SHA256, "seeds": [11, 22, 33],
        "prepared_data_hashes": hashes, "checkpoint_data_hashes": states["candidate"]["data_hashes"],
        "budget": {"train_profiles": 128, "epochs": 26, "updates_per_adversary": 26, "updates_per_epoch": 1},
        "evaluation": {"seed": 2026, "samples_per_condition": 100, "calibration_seed": 7301,
            "calibration_samples_per_condition": 4096, "quantile_knots": 1025, "candidate_posthoc_calibration": False,
            "baseline_calibration": "frozen_original_saved_map"}}
    declaration_path = tmp_path / "execution.json"
    save_json(declaration_path, declaration)
    calls = []
    def candidate_values(y, seed):
        p = np.random.default_rng(seed).uniform(size=(len(y), 96))
        x = ((1-p)*q[y, 0] + p*q[y, -1]).astype(np.float32)
        return x, p
    def generate(path, y, seed=42, batch_size=1024):
        calls.append((Path(path), len(y), seed, "ordinary"))
        return candidate_values(y, seed)[0] if Path(path) == paths["candidate"] else reference_values(y, seed)
    def diagnostics(path, y, seed=42, batch_size=1024):
        calls.append((Path(path), len(y), seed, "diagnostics"))
        x, p = candidate_values(y, seed)
        return x, p, fit_metadata
    monkeypatch.setattr(evaluation.training, "generate", generate)
    monkeypatch.setattr(evaluation.training, "generate_quantile_decoder_diagnostics", diagnostics)
    return {"prepared": prepared, "checkpoint": paths["candidate"], "reference_checkpoint": paths["reference"],
            "reference_calibrator": map_path, "declaration": declaration_path, "arrays": arrays, "fixed_map": fixed,
            "candidate_values": candidate_values, "calls": calls, "declaration_body": declaration}


def arguments(experiment):
    return {key: experiment[key] for key in ("prepared", "checkpoint", "reference_checkpoint", "reference_calibrator", "declaration")}


def test_physical_direct_decoder_shared_reference_controls_and_diagnostics(experiment, forbid_test_access, monkeypatch):
    monkeypatch.setattr(ConditionalMarginalCalibrator, "save", lambda *_: pytest.fail("evaluator saved/fitted another map"))
    report = evaluation.evaluate(**arguments(experiment), mode="physical")
    assert set(report["methods"]) == set(evaluation.PHYSICAL_METHODS)
    assert report["candidate_posthoc_calibration"] is report["reference_calibration_refitted"] is False
    assert report["input_sha256"] == report["post_input_sha256"]
    assert len(report["quantile_decoder"]["per_condition"]) == 64
    assert report["quantile_decoder"]["near_saturation_threshold"] == 1e-6
    assert report["frozen_samples"]["profile_dtypes"]["quantile_decoder_cgan"] == "float32"
    val = experiment["arrays"]["val"]
    rng = np.random.default_rng(2026)
    selected = np.concatenate([rng.choice(np.flatnonzero(val["y"] == c), 2, replace=False) for c in range(64)])
    real, labels = val["x"][selected], val["y"][selected]
    candidate = experiment["candidate_values"](labels, 2026)[0]
    reference = experiment["fixed_map"].transform(reference_values(labels, 2026), labels)
    for name, values in (("quantile_decoder_cgan", candidate), ("reference_calibrated_cgan", reference)):
        assert report["methods"][name] == evaluate_conditional(real, labels, values, labels, interval_hours=.25)
    assert [row[3] for row in experiment["calls"]] == ["diagnostics", "ordinary"]
    assert all(row[1:3] == (128, 2026) for row in experiment["calls"])


def test_utility_full_train_counts_fixed1025_shared_tasks_and_manual_scores(experiment, forbid_test_access, monkeypatch):
    monkeypatch.setattr(evaluation.training, "generate_quantile_decoder_diagnostics", lambda *_a, **_k: pytest.fail("utility allocated probabilities"))
    report = evaluation.evaluate(**arguments(experiment), mode="utility")
    assert report["copula"]["quantile_knots"] == 1025
    assert report["utility_training_selection"]["selected_profiles"] == 128
    assert report["utility_training_selection"]["cap_applied"] is False
    assert report["task"]["lags"] == 12 and report["task"]["windows_per_profile"] == 84
    assert len({row["labels_sha256"] for row in report["training"].values()}) == 1
    assert all(row["n_profiles"] == 128 for row in report["training"].values())
    train, val = experiment["arrays"]["train"], experiment["arrays"]["val"]
    rng = np.random.default_rng(2026)
    selected = np.concatenate([rng.choice(np.flatnonzero(val["y"] == c), 2, replace=False) for c in range(64)])
    values = experiment["candidate_values"](train["y"], 2026)[0]
    expected = RidgeProfileForecaster(batch_profiles=256).fit(values, train["y"]).evaluate(val["x"][selected], val["y"][selected])
    expected.pop("persistence")
    assert report["methods"]["training_quantile_decoder"] == expected
    assert len(experiment["calls"]) == 2 and all(row[1:] == (128, 2026, "ordinary") for row in experiment["calls"])


def test_declaration_test_alias_rejected_before_any_content_read(tmp_path, monkeypatch):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    sentinel = prepared / "test.npz"
    sentinel.write_bytes(b"sealed")
    alias = tmp_path / "declaration.json"
    alias.symlink_to(sentinel)
    monkeypatch.setattr(evaluation, "_json", lambda *_: pytest.fail("reserved declaration was read"))
    with pytest.raises(ValueError, match="TEST"):
        evaluation._declaration(alias, prepared)


@pytest.mark.parametrize("mismatch", ["scaler", "selection"])
def test_paired_training_records_reject_coherent_candidate_mismatch(experiment, forbid_test_access, mismatch):
    directory = experiment["checkpoint"].parent
    for filename in ("best.pt", "last.pt", "summary.json"):
        path = directory / filename
        state = json.loads(path.read_text()) if filename.endswith("json") else torch.load(path, weights_only=True)
        if mismatch == "scaler":
            state["scaler"]["mean"] += .25
        else:
            state["selection"]["validation_noise_sha256"] = "9" * 64
        if filename.endswith("json"):
            save_json(path, state)
        else:
            torch.save(state, path)
    with pytest.raises(ValueError, match="standardizers differ|fixed selector or sampling records differ"):
        evaluation.evaluate(**arguments(experiment), mode="physical")


def test_cli_refuses_existing_output_without_evaluating(tmp_path, monkeypatch):
    output = tmp_path / "already.json"
    output.write_text("{}")
    monkeypatch.setattr(evaluation, "evaluate", lambda *_a, **_k: pytest.fail("evaluated before output check"))
    with pytest.raises(SystemExit):
        evaluation.main(["--prepared", "unused", "--checkpoint", "unused", "--reference-checkpoint", "unused",
            "--reference-calibrator", "unused", "--declaration", "unused", "--mode", "physical", "--output", str(output)])


def test_production_scope_has_no_public_training_or_evaluation_cap_override():
    assert evaluation.TRAIN_ROWS == 606060 and evaluation.CAP == 100 and evaluation.SEED == 2026
    assert evaluation.SEEDS == (11, 22, 33) and evaluation.SATURATION_THRESHOLD == 1e-6
