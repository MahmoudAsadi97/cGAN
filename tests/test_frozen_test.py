"""Synthetic-only end-to-end frozen study tests; never read real prepared data."""
from contextlib import contextmanager
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from fluvius_cgan import frozen_test as ft
from fluvius_cgan.calibration import fit
from fluvius_cgan.curation import curate_prepared


DATES = np.array(["2024-01-01", "2024-01-06", "2024-04-01", "2024-04-06",
                  "2024-07-01", "2024-07-06", "2024-10-01", "2024-10-05"])


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, allow_nan=False))


def _fake_generate(checkpoint, labels, *, seed, batch_size=1024):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    rng = np.random.default_rng(seed + state["config"]["seed"])
    return rng.normal(size=(len(labels), 24)) + np.asarray(labels)[:, None] / 50


@contextmanager
def forbid_test_open(monkeypatch):
    original = Path.open

    def guarded(path, *args, **kwargs):
        if path.name == "test.npz":
            raise AssertionError("freeze opened or hashed TEST")
        return original(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", guarded)
        yield


def make_frozen_inputs(tmp_path, monkeypatch):
    """Real synthetic curation and saved-map formats, stubbed GAN inference only."""
    original = tmp_path / "original"
    original.mkdir()
    metadata = {"resolution": 24, "target": "net", "units": "kWh/interval",
                "source_sha256": "a" * 64, "split_seed": 42, "splits": {},
                "timezone_policy": "source local calendar", "counts": {"meters_kept": 48, "profiles_kept": 384}}
    rng = np.random.default_rng(100)
    for split in ("train", "val", "test"):
        tech = np.repeat(np.arange(8), 16)
        labels = tech * 8 + np.tile([0, 4, 1, 5, 2, 6, 3, 7], 16)
        ids = np.repeat([f"{split}_{i:02d}" for i in range(16)], 8)
        x = (rng.normal(size=(128, 24)) + labels[:, None] / 50).astype(np.float32)
        np.savez(original / f"{split}.npz", x=x, y=labels.astype(np.int64),
                 meter_id=ids, date=np.tile(DATES, 16))
        metadata["splits"][split] = {"profiles": 128, "meters": 16, "meter_ids": sorted(set(ids)),
                                    "sha256": ft.sha256_file(original / f"{split}.npz")}
        if split == "train":
            tx, ty = x, labels.astype(np.int64)
    _write_json(original / "metadata.json", metadata)
    _write_json(original / "scaler.json", {"minimum": float(tx.min()), "maximum": float(tx.max())})
    prepared = tmp_path / "prepared"
    curate_prepared(original, prepared, expected_dates=DATES)
    metadata = ft._json(prepared / "metadata.json")
    scaler = ft._json(prepared / "scaler.json")
    hashes = {name: ft.sha256_file(prepared / name) for name in ft.PREPARED}
    checkpoint_hashes = {**hashes, "curation_manifest.json": ft.sha256_file(prepared / "curation_manifest.json")}
    selection = {"train_profiles_available": len(tx), "train_profiles_used": len(tx),
                 "train_indices_sha256": ft._digest(np.arange(len(tx)), "<i8")}
    runs = []
    for seed in ft.SEEDS:
        directory = tmp_path / f"seed_{seed}"
        directory.mkdir()
        config = ft.training._config({"seed": seed, "epochs": 1, "num_threads": 1, "normalization": "standard"})
        model_scaler = {**scaler, "kind": "standard", "mean": float(tx.astype(np.float64).mean()),
                        "std": float(tx.astype(np.float64).std())}
        provenance = {"synthetic_fixture": True, "source_files_sha256": {
            f"src/fluvius_cgan/{name}": str(i) * 64 for i, name in enumerate(ft.TRAINING_CORE, start=1)}}
        state = {"format_version": ft.training.CHECKPOINT_FORMAT, "config": config,
                 "metadata": metadata, "output_dim": 24, "data_hashes": checkpoint_hashes,
                 "scaler": model_scaler, "supported_labels": ft.SUPPORT, "selection": selection,
                 "epoch": 1, "best_validation_score": 1., "provenance": provenance}
        checkpoint = directory / "best.pt"
        torch.save(state, checkpoint)
        _write_json(directory / "config.json", config)
        _write_json(directory / "summary.json", {"test_used": False, "epochs_completed": 1,
            "stopped_early": False, "best_epoch": 1, "best_validation_score": 1.,
            "data_hashes": checkpoint_hashes, "selection": selection, "provenance": provenance,
            "source_sha256": metadata["source_sha256"]})
        _write_json(directory / "history.json", [{"epoch": 1, "validation_score": 1.}])
        _write_json(directory / "calibration_validation.json", {"split": "val", "test_used": False,
            "scope": "Synthetic placeholder for immutable development evidence; not scored by freeze."})
        for name in ("comparison_gaussian_copula.json", "comparison_conditional_resampling.json", "calibration_sensitivity.json"):
            _write_json(directory / name, {"split": "val", "test_used": False, "synthetic_fixture": name})
        cy = np.repeat(np.arange(64, dtype=np.int64), 4096)
        # Tied synthetic references exercise true 4096-count maps while compressing well.
        source = np.round(_fake_generate(checkpoint, np.repeat(np.arange(64), 8), seed=7301) * 16) / 16
        cx = np.repeat(source, 512, axis=0)
        provenance = {"checkpoint_sha256": ft.sha256_file(checkpoint),
            "train_indices_sha256": selection["train_indices_sha256"], "prepared_data_hashes": hashes,
            "calibration_seed": 7301, "calibration_samples_per_condition": 4096, "calibration_profiles_total": len(cy),
            "calibration_generated_profiles_sha256": ft._digest(cx, "<f8"),
            "calibration_labels_sha256": ft._digest(cy, "<i8"),
            "generator_inference_batch_size": 1024, "quantile_knots": 1025}
        calibrator = directory / "calibration.npz"
        fit(tx, ty, cx, cy, provenance=provenance).save(calibrator)
        runs.append((checkpoint, calibrator))
    monkeypatch.setattr(ft.training, "generate", _fake_generate)
    protocol = tmp_path / "protocol.md"
    protocol.write_text("Synthetic frozen protocol; no real held-out files.\n")
    return {"prepared": prepared, "runs": runs, "protocol": protocol, "train": tx, "train_labels": ty}


def make_frozen_study(tmp_path, monkeypatch, *, utility=False, bootstrap_replicates=0):
    result = make_frozen_inputs(tmp_path, monkeypatch)
    bundle = tmp_path / "bundle"
    with forbid_test_open(monkeypatch):
        seal = ft.freeze_test(result["prepared"], result["runs"], bundle,
            decision_rationale="Synthetic fixture exercises the frozen five-seed contract.",
            protocol_files=[result["protocol"]], utility=utility, bootstrap_replicates=bootstrap_replicates)
    manifest = ft._json(bundle / "manifest.json")
    return {**result, "bundle": bundle, "seal": seal["seal_sha256"], "manifest": manifest,
            "test_binding": manifest["test_binding"]}


def test_freeze_never_opens_test_and_seals_train_only_evidence(tmp_path, monkeypatch):
    study = make_frozen_study(tmp_path, monkeypatch)
    manifest = study["manifest"]
    assert manifest["test_accessed_during_freeze"] is False
    assert list(manifest["training_data_hashes"]) == list(ft.PREPARED)
    assert "test.npz" not in manifest["training_data_hashes"]
    assert [r["seed"] for r in manifest["runs"]] == list(ft.SEEDS)
    assert all(len(r["development_evidence"]) == 7 for r in manifest["runs"])
    for run in manifest["runs"]:
        evidence_names = {Path(item["path_at_freeze"]).name for item in run["development_evidence"]}
        assert {"comparison_gaussian_copula.json", "comparison_conditional_resampling.json",
                "calibration_sensitivity.json"} <= evidence_names
        for item in run["development_evidence"]:
            assert ft.sha256_file(study["bundle"] / "artifacts" / item["artifact"]) == item["sha256"]
    assert manifest["decision"]["rationale"].startswith("Synthetic fixture")
    assert len(manifest["decision"]["protocol_references"]) == 1
    assert ft.sha256_file(study["bundle"] / "manifest.json") == study["seal"]
    assert not list((study["bundle"] / "attempts").iterdir())


def test_frozen_evaluation_matches_manual_metrics_and_never_fits(tmp_path, monkeypatch):
    study = make_frozen_study(tmp_path, monkeypatch, utility=True, bootstrap_replicates=3)
    def fail(*args, **kwargs):
        raise AssertionError("fitting/generation or TRAIN loading during evaluator")
    monkeypatch.setattr(ft.training, "generate", fail)
    monkeypatch.setattr(ft.training, "_load_prepared", fail)
    monkeypatch.setattr(ft, "fit_conditional_copula", fail)
    monkeypatch.setattr(ft.RidgeProfileForecaster, "fit", fail)
    report = ft.evaluate_frozen_test(study["bundle"], study["seal"], tmp_path / "evaluation")
    assert report["test_used"] and report["fitting_performed"] is False
    assert len(report["runs"]) == 5
    assert report["test_selection"]["selected_profiles"] == 128
    assert report["test_selection"]["condition_counts"] == {str(c): 2 for c in ft.SUPPORT}
    assert set(report["runs"][0]["paired_utility_uncertainty"]) == {"training_real", "training_resampling"}
    assert report["runs"][0]["calibration_diagnostics"]["data"]["application"]["n_profiles"] == 128
    assert report["runs"][0]["calibration_diagnostics"]["data"]["fitting"]["per_condition"]["0"]["calibration_count"] == 4096
    binding = study["test_binding"]
    data = ft._read_verified_test(binding)
    rng = np.random.default_rng(ft.PROTOCOL["selection_seed"])
    rows = np.concatenate([rng.choice(np.flatnonzero(data["y"] == c), 2, replace=False) for c in ft.SUPPORT])
    real, labels = data["x"][rows], data["y"][rows]
    frozen = study["manifest"]["runs"][0]
    values = ft._array(study["bundle"], frozen["arrays"]["calibrated_cgan"])
    manual = ft.evaluate_conditional(real, labels, values, labels, interval_hours=1.)
    assert report["runs"][0]["methods"]["calibrated_cgan"]["physical"] == manual
    model = ft._load_ridge(study["bundle"], frozen["utility"]["training_calibrated_cgan"])
    utility = model.evaluate(real, labels)
    utility.pop("persistence")
    assert report["runs"][0]["utility"]["training_calibrated_cgan"] == utility
    ramps = [r["methods"]["calibrated_cgan"]["physical"]["macro"]["ramp_wasserstein_kw"] for r in report["runs"]]
    summary = report["five_seed_summary"]["calibrated_cgan"]["physical_macro"]["ramp_wasserstein_kw"]
    assert summary["mean"] == pytest.approx(np.mean(ramps))
    assert summary["sample_sd"] == pytest.approx(np.std(ramps, ddof=1))
    rmses = [r["utility"]["training_calibrated_cgan"]["macro"]["rmse_kwh"] for r in report["runs"]]
    assert report["five_seed_summary"]["calibrated_cgan"]["utility_macro"]["rmse_kwh"]["mean"] == pytest.approx(np.mean(rmses))
    assert len(list((study["bundle"] / "attempts").glob("*.complete.json"))) == 1


def test_invalid_seed_set_records_freeze_failure_without_test_access(tmp_path, monkeypatch):
    study = make_frozen_inputs(tmp_path, monkeypatch)
    runs = study["runs"][:4] + study["runs"][:1]
    with forbid_test_open(monkeypatch), pytest.raises(ValueError, match="without duplicates"):
        ft.freeze_test(study["prepared"], runs, tmp_path / "failed", decision_rationale="Synthetic case",
            protocol_files=[study["protocol"]], utility=False, bootstrap_replicates=0)
    assert ft._json(tmp_path / "failed" / "freeze_failure.json")["test_accessed"] is False

    # A coherently rehashed metadata document must still describe actual development IDs.
    metadata_path = study["prepared"] / "metadata.json"
    curation_path = study["prepared"] / "curation_manifest.json"
    saved_metadata, saved_curation = metadata_path.read_bytes(), curation_path.read_bytes()
    metadata = ft._json(metadata_path)
    metadata["splits"]["train"]["meter_ids"][0] = "fictional_development_meter"
    _write_json(metadata_path, metadata)
    curation = ft._json(curation_path)
    curation["output_files_sha256"]["metadata.json"] = ft.sha256_file(metadata_path)
    _write_json(curation_path, curation)
    with forbid_test_open(monkeypatch), pytest.raises(ValueError, match="actual TRAIN/VAL meter membership"):
        ft.freeze_test(study["prepared"], study["runs"], tmp_path / "metadata_failure", decision_rationale="Synthetic case",
            protocol_files=[study["protocol"]], utility=False, bootstrap_replicates=0)
    metadata_path.write_bytes(saved_metadata)
    curation_path.write_bytes(saved_curation)

    # Equal settings cannot conceal a different core training implementation.
    checkpoint, calibrator = study["runs"][-1]
    state = torch.load(checkpoint, weights_only=True)
    state["provenance"]["source_files_sha256"]["src/fluvius_cgan/training.py"] = "f" * 64
    torch.save(state, checkpoint)
    summary_path = checkpoint.parent / "summary.json"
    summary = ft._json(summary_path)
    summary["provenance"] = state["provenance"]
    _write_json(summary_path, summary)
    from fluvius_cgan.calibration import ConditionalMarginalCalibrator
    model = ConditionalMarginalCalibrator.load(calibrator)
    model.provenance["checkpoint_sha256"] = ft.sha256_file(checkpoint)
    model.save(calibrator)
    with forbid_test_open(monkeypatch), pytest.raises(ValueError, match="training core source differs"):
        ft.freeze_test(study["prepared"], study["runs"], tmp_path / "core_failure", decision_rationale="Synthetic case",
            protocol_files=[study["protocol"]], utility=False, bootstrap_replicates=0)


def test_production_calibration_protocol_is_fixed():
    assert (ft.PROTOCOL["calibration_samples_per_condition"], ft.PROTOCOL["calibration_seed"],
            ft.PROTOCOL["quantile_knots"]) == (4096, 7301, 1025)
