"""Small independent schema/budget negatives; no prepared-data or TEST access."""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from fluvius_cgan import frozen_reference_study as fr
from fluvius_cgan import frozen_test as ft
from test_frozen_reference_study import paired_payload


@pytest.mark.parametrize("alias", [False, True])
def test_freeze_cli_rejects_test_named_specification_before_read(tmp_path, monkeypatch, alias):
    path = Path(__file__).resolve().parents[1] / "scripts" / "freeze_reference_study.py"
    spec = importlib.util.spec_from_file_location("freeze_reference_study_test_cli", path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    reserved = tmp_path / "unrelated" / "test.npz"
    reserved.parent.mkdir()
    reserved.write_bytes(b"must not be read")
    argument = reserved
    if alias:
        argument = tmp_path / "specification.json"
        argument.symlink_to(reserved)
    monkeypatch.setattr(cli, "_json", lambda *_: pytest.fail("TEST-named specification was read"))
    with pytest.raises(ValueError, match="TEST archive"):
        cli.main(["--prepared96", str(tmp_path / "parent"), "--prepared24", str(tmp_path / "child"),
                  "--specification", str(argument), "--output", str(tmp_path / "bundle")])


def _coverage():
    return {"native": {arm: {str(s): {k: f"{arm}_{s}_{k}.json" for k in fr.NATIVE_EVIDENCE} for s in fr.SEEDS} for arm in fr.ARMS},
        "common_hourly": {str(s): {k: f"hourly_{s}_{k}.json" for k in fr.HOURLY_EVIDENCE} for s in fr.SEEDS},
        "paired": {name: {str(s): f"{name}_{s}.json" for s in fr.SEEDS} for name in fr.CONTRASTS}, "additional": []}


@pytest.mark.parametrize("section", ["native", "common_hourly", "paired"])
def test_missing_seed_evidence_rejected(section):
    evidence = _coverage()
    if section == "native":
        del evidence[section]["bce24"]["55"]
    elif section == "common_hourly":
        del evidence[section]["55"]
    else:
        del evidence[section]["resolution_calibrated"]["55"]
    with pytest.raises(ValueError, match="five seeds"):
        fr._evidence_slots(evidence)


@pytest.mark.parametrize("role", fr.NATIVE_EVIDENCE)
def test_each_native_evidence_role_is_mandatory(role):
    evidence = _coverage()
    del evidence["native"]["candidate96"]["11"][role]
    with pytest.raises(ValueError, match="coverage"):
        fr._evidence_slots(evidence)


def _paired_fixture():
    hashes = {name: "a" * 64 for name in ft.PREPARED}
    hourly = {name: "b" * 64 for name in ft.PREPARED}
    runs = {(arm, 11): {"checkpoint_sha256": str(i + 1) * 64, "calibrator_sha256": str(i + 5) * 64,
            "calibration_provenance": {"synthetic_map_identity": arm}, "prepared_data_hashes": hourly if arm == "bce24" else hashes}
            for i, arm in enumerate(fr.ARMS)}
    roles = ("bce24", "bce96")
    report = {"split": "val", "test_used": False, "seed": 2026, "samples_per_condition_cap": 100,
        "candidate": {"sha256": runs[(roles[0], 11)]["checkpoint_sha256"]},
        "comparator": {"sha256": runs[(roles[1], 11)]["checkpoint_sha256"]},
        "prepared_data_hashes": {"parent96": hashes, "derived24": hourly}, "training_seed": 11,
        "replicates": 500, "metrics": list(ft.METRICS), "contrast": "resolution_calibrated",
        "resolution": 24, "interval_hours": 1., "paired_comparison": paired_payload(24),
        "saved_maps": {key: {"sha256": runs[(arm, 11)]["calibrator_sha256"], "provenance": runs[(arm, 11)]["calibration_provenance"]}
                       for key, arm in zip(("candidate", "comparator"), roles)}}
    return runs, roles, report


@pytest.mark.parametrize("mutation", ["missing", "condition", "metric", "estimate", "bounds", "draws", "hours"])
def test_paired_evidence_requires_complete_result_payload(mutation):
    value = paired_payload(96)
    fr._paired_evidence(value, 96)
    fr._paired_evidence(paired_payload(96, invalid=1), 96)
    if mutation == "missing":
        value = None
    elif mutation == "condition":
        del value["per_condition"]["63"]
    elif mutation == "metric":
        del value["per_condition"]["0"]["intervals"][ft.METRICS[0]]
    elif mutation == "estimate":
        value["intervals"][ft.METRICS[0]]["estimate"] = 3.
    elif mutation == "bounds":
        value["intervals"][ft.METRICS[0]]["lower"] = None
    elif mutation == "draws":
        value["n_valid_bootstrap"] = 499
    else:
        value["interval_hours"] = 1.
    with pytest.raises(ValueError, match="paired evidence"):
        fr._paired_evidence(value, 96)


@pytest.mark.parametrize("mutation", ["contrast", "order", "map", "prepared", "seed", "cap", "replicates", "split"])
def test_paired_evidence_cannot_substitute_another_pipeline(mutation):
    runs, roles, report = _paired_fixture()
    fr._validate_evidence_report(report, roles, 11, "paired", runs, "resolution_calibrated")
    if mutation == "contrast":
        report["contrast"] = "resolution_raw"
    elif mutation == "order":
        report["candidate"], report["comparator"] = report["comparator"], report["candidate"]
    elif mutation == "map":
        report["saved_maps"]["candidate"]["sha256"] = "f" * 64
    elif mutation == "prepared":
        report["prepared_data_hashes"]["derived24"] = {}
    elif mutation == "seed":
        report["seed"] = 7301
    elif mutation == "cap":
        report["samples_per_condition_cap"] = 1000
    elif mutation == "replicates":
        report["replicates"] = 50
    else:
        report["split"] = "test"
    with pytest.raises(ValueError):
        fr._validate_evidence_report(report, roles, 11, "paired", runs, "resolution_calibrated")


def test_report_types_are_distinct():
    physical_row = {**{k: 1. for k in ft.METRICS[:-1]}, "autocorrelation": {"1": {"wasserstein": .1}}}
    utility_row = {"rmse_kwh": 1., "mae_kwh": .5}
    physical = {"macro": physical_row, "per_condition": {str(c): physical_row for c in range(64)}}
    utility = {"macro": utility_row, "per_condition": {str(c): utility_row for c in range(64)}}
    fr._physical_evidence(physical)
    fr._utility_evidence(utility)
    with pytest.raises(ValueError, match="physical evidence"):
        fr._physical_evidence(utility)
    with pytest.raises(ValueError, match="utility evidence"):
        fr._utility_evidence(physical)


def test_meter_draws_match_across_physical_utility_and_resolution(monkeypatch):
    monkeypatch.setattr(fr, "BOOTSTRAP_REPLICATES", 2)
    labels = np.tile(np.arange(64, dtype=np.int64), 2)
    meters = np.asarray([f"t{label // 8}_m{row // 64}" for row, label in enumerate(labels)])
    real = np.arange(128*96, dtype=np.float64).reshape(128, 96) / 100
    original = np.random.default_rng
    recordings = []
    class Recorded:
        def __init__(self, seed):
            self.inner = original(seed)
            self.rows = []
            recordings.append(self.rows)
        def integers(self, *args, **kwargs):
            values = self.inner.integers(*args, **kwargs)
            self.rows.append(values.tolist())
            return values
    monkeypatch.setattr(np.random, "default_rng", Recorded)
    fr._physical_pair(real, labels, meters, real + .5, real + 1.)
    hourly = fr._sum4(real)
    fr._physical_pair(hourly, labels, meters, hourly + .5, hourly + 1.)
    fr._utility_pair(labels, meters, np.full((128, 84), .5), np.ones((128, 84)))
    fr._utility_pair(labels, meters, np.full((128, 21), .5), np.ones((128, 21)))
    assert len(recordings) == 4 and len(recordings[0]) == 16
    assert all(rows == recordings[0] for rows in recordings)


def _checkpoint_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(fr, "TRAIN_ROWS", 128)
    config = fr._expected_config("bce96", 11, "swd10")
    hashes = {name: "a" * 64 for name in ft.PREPARED}
    metadata = {"resolution": 96, "source_sha256": "b" * 64}
    scaler = {"minimum": 0., "maximum": 1.}
    data = {"train": {"x": np.zeros((128, 96), dtype=np.float32), "y": np.repeat(np.arange(64), 2)},
            "metadata": metadata, "scaler": scaler, "hashes": hashes, "manifest_name": "curation_manifest.json", "manifest_sha256": "c" * 64}
    status = {"status": "verified_curation_v2", "verified_file_hashes": hashes, "manifest_sha256": "c" * 64}
    selection = {"train_profiles_available": 128, "train_profiles_used": 128, "train_indices_sha256": ft._digest(np.arange(128), "<i8")}
    provenance = {"source_files_sha256": {f"src/fluvius_cgan/{name}": "d" * 64 for name in ft.TRAINING_CORE}}
    state = {"format_version": fr.training.CHECKPOINT_FORMAT, "config": config, "condition_dim": 64, "output_dim": 96,
        "metadata": metadata, "scaler": scaler, "data_hashes": {**hashes, "curation_manifest.json": "c" * 64},
        "data_curation": status, "selection": selection, "provenance": provenance, "supported_labels": list(range(64)),
        "epoch": 26, "validation_score": 1., "best_epoch": 26, "best_validation_score": 1., "generator_updates": 26,
        "discriminator_updates": 26, "samples_seen": 26*512, "validation_metric": fr.training.CONDITIONAL_VALIDATION_METRIC,
        "optimizer_g_state_dict": {"state": {0: {"step": torch.tensor(26.)}}},
        "optimizer_d_state_dict": {"state": {0: {"step": torch.tensor(26.)}}}}
    history = [{"epoch": i, "validation_score": float(27-i), "generator_updates": 1, "discriminator_updates": 1, "samples_seen": 512} for i in range(1, 27)]
    summary = {k: v for k, v in state.items() if k not in ("optimizer_g_state_dict", "optimizer_d_state_dict")}
    summary.update(test_used=False, epochs_completed=26, stopped_early=False, source_sha256=metadata["source_sha256"])
    for name, value in (("config.json", config), ("history.json", history), ("summary.json", summary)):
        (tmp_path / name).write_text(json.dumps(value))
    torch.save(state, tmp_path / "best.pt")
    torch.save(state, tmp_path / "last.pt")
    return data, state


@pytest.mark.parametrize("mutation", ["epoch", "optimizer_step", "last_source", "last_config", "last_samples"])
def test_last_checkpoint_budget_is_independently_verified(tmp_path, monkeypatch, mutation):
    data, state = _checkpoint_fixture(tmp_path, monkeypatch)
    fr._verify_run(tmp_path / "best.pt", "bce96", data, "swd10")
    last = copy.deepcopy(state)
    if mutation == "epoch":
        last["epoch"] = 25
    elif mutation == "optimizer_step":
        last["optimizer_g_state_dict"]["state"][0]["step"] = torch.tensor(25.)
    elif mutation == "last_source":
        last["provenance"]["source_files_sha256"]["src/fluvius_cgan/models.py"] = "e" * 64
    elif mutation == "last_config":
        last["config"]["latent_dim"] = 96
    else:
        last["samples_seen"] -= 1
    torch.save(last, tmp_path / "last.pt")
    with pytest.raises(ValueError):
        fr._verify_run(tmp_path / "best.pt", "bce96", data, "swd10")
