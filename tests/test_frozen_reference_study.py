"""Synthetic-only unified study contracts; never use research data or endpoints."""
from contextlib import contextmanager
import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from fluvius_cgan import frozen_reference_study as fr
from fluvius_cgan import frozen_test as ft
from fluvius_cgan.calibration import fit, ConditionalMarginalCalibrator
from fluvius_cgan.curation import curate_prepared
from fluvius_cgan.derived import derive_hourly
from fluvius_cgan.release import RELEASE_SHA256

DATES = np.array(["2024-01-01", "2024-01-06", "2024-04-01", "2024-04-06",
                  "2024-07-01", "2024-07-06", "2024-10-01", "2024-10-05"])


def write_json(path, value):
    Path(path).write_text(json.dumps(value, allow_nan=False))


@contextmanager
def forbid_test(monkeypatch):
    original = Path.open
    def guarded(path, *args, **kwargs):
        if path.name == "test.npz":
            pytest.fail("freeze opened/hashed reserved synthetic TEST")
        return original(path, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", guarded)
        yield


def fake_values(labels, width, seed):
    out = np.empty((len(labels), width), dtype=np.float32)
    rng = np.random.default_rng(seed)
    for label in range(64):
        rows = np.flatnonzero(labels == label)
        templates = np.round(rng.normal(size=(8, width)) * 16) / 16 + label / 64
        out[rows] = templates[np.arange(len(rows)) % 8]
    return out


def fake_generate(checkpoint, labels, *, seed, batch_size=1024):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    return fake_values(np.asarray(labels), state["output_dim"], seed)


def paired_payload(width, invalid=0):
    status = "unavailable_missing_condition_draws" if invalid else "complete"
    row = {"interval_status": status,
        "point_estimates": {name: {metric: value for metric in ft.METRICS} for name, value in (("generated", .25), ("comparator", .5))},
        "intervals": {metric: {"estimate": -.25, "lower": None if invalid else -.5,
                               "upper": None if invalid else 0.} for metric in ft.METRICS}}
    return {**row, "n_bootstrap": 500, "n_valid_bootstrap": 500-invalid, "n_invalid_bootstrap": invalid,
        "confidence": .95, "seed": 2026, "interval_hours": 24/width, "n_conditions": 64,
        "condition_support": list(range(64)), "resampling_unit": "real_meter",
        "resampling_scheme": "within_stratum_real_meter", "condition_weighting": "equal_over_original_observed_support",
        "comparison": "generated_minus_comparator",
        "missing_condition_policy": "withhold_all_intervals_if_any_draw_lacks_original_support",
        "per_condition": {str(c): copy.deepcopy(row) for c in range(64)}}


def make_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(fr, "TRAIN_ROWS", 128)
    monkeypatch.setattr(fr, "BOOTSTRAP_REPLICATES", 2)
    monkeypatch.setattr(fr.training, "generate", fake_generate)
    original, parent, child = (tmp_path / name for name in ("original", "curated96", "hourly24"))
    original.mkdir()
    metadata = {"resolution": 96, "target": "net", "units": "kWh/interval", "interval_hours": .25,
        "source_sha256": RELEASE_SHA256, "split_seed": 42, "splits": {},
        "timezone_policy": "source local calendar", "counts": {"meters_kept": 48, "profiles_kept": 384}}
    rng = np.random.default_rng(100)
    for split in ("train", "val", "test"):
        labels = np.repeat(np.arange(8), 16) * 8 + np.tile([0, 4, 1, 5, 2, 6, 3, 7], 16)
        ids = np.repeat([f"{split}_{i:02d}" for i in range(16)], 8)
        x = (np.round(rng.normal(size=(128, 96)) * 16) / 16 + labels[:, None] / 64).astype(np.float32)
        np.savez(original / f"{split}.npz", x=x, y=labels.astype(np.int64), meter_id=ids, date=np.tile(DATES, 16))
        metadata["splits"][split] = {"profiles": 128, "meters": 16, "meter_ids": sorted(set(ids)),
                                    "sha256": ft.sha256_file(original / f"{split}.npz")}
        if split == "train":
            train = x
    write_json(original / "metadata.json", metadata)
    write_json(original / "scaler.json", {"minimum": float(train.min()), "maximum": float(train.max())})
    curate_prepared(original, parent, expected_dates=DATES)
    with forbid_test(monkeypatch):
        derive_hourly(parent, child)
        data = fr._load_development(parent, child)
    templates = {}
    cy = np.repeat(np.arange(64, dtype=np.int64), 4096)
    for width in (96, 24):
        part = data[width]
        # Fit the exact separable per-condition maps in small blocks. This
        # preserves4096 genuine reference rows/class without a262144x96 buffer.
        models, source_digest = {}, hashlib.sha256()
        for label in range(64):
            local_labels = np.full(4096, label, dtype=np.int64)
            cx = fake_values(local_labels, width, 7301)
            source_digest.update(np.asarray(cx, dtype="<f8").tobytes(order="C"))
            selected = part["train"]["y"] == label
            local = fit(part["train"]["x"][selected], part["train"]["y"][selected], cx, local_labels)
            models[label] = local.models[label]
        provenance = {"calibration_seed": 7301, "calibration_samples_per_condition": 4096,
            "calibration_profiles_total": len(cy), "calibration_generated_profiles_sha256": source_digest.hexdigest(),
            "calibration_labels_sha256": ft._digest(cy, "<i8"), "generator_inference_batch_size": 1024, "quantile_knots": 1025}
        templates[width] = ConditionalMarginalCalibrator(width, 1025, models, provenance)
        del cx, local
    arms = {arm: [] for arm in fr.ARMS}
    digests, map_records = {}, {}
    for arm in fr.ARMS:
        width = 24 if arm == "bce24" else 96
        part = data[width]
        status = fr.training._training_curation_guard(part["directory"], {"allow_uncurated_data": False})
        for seed in fr.SEEDS:
            directory = tmp_path / f"{arm}_{seed}"
            directory.mkdir()
            config = fr._expected_config(arm, seed, "swd10")
            selection = {"train_profiles_available": 128, "train_profiles_used": 128,
                         "train_indices_sha256": ft._digest(np.arange(128), "<i8")}
            hashes = {**part["hashes"], part["manifest_name"]: part["manifest_sha256"]}
            provenance = {"git_commit": "1" * 40, "git_dirty": False, "source_files_sha256": {
                f"src/fluvius_cgan/{name}": str(i) * 64 for i, name in enumerate(ft.TRAINING_CORE, start=1)}}
            scaler = part["scaler"].copy()
            if arm == "candidate96":
                values = part["train"]["x"].astype(np.float64)
                scaler.update(kind="standard", mean=float(values.mean()), std=float(values.std()))
            history = [{"epoch": i, "validation_score": float(27-i), "generator_updates": 1,
                        "discriminator_updates": 1, "samples_seen": 512} for i in range(1, 27)]
            state = {"format_version": fr.training.CHECKPOINT_FORMAT, "config": config, "condition_dim": 0 if arm == "unconditional96" else 64,
                "output_dim": width, "metadata": part["metadata"], "scaler": scaler, "data_hashes": hashes,
                "data_curation": status, "supported_labels": ft.SUPPORT, "selection": selection, "epoch": 26,
                "best_validation_score": 1., "provenance": provenance,
                "validation_metric": fr.training.CONDITIONAL_VALIDATION_METRIC}
            checkpoint = directory / "best.pt"
            torch.save(state, checkpoint)
            last = {**state, "generator_updates": 26, "discriminator_updates": 26, "samples_seen": 512*26,
                    "best_epoch": 26, "validation_score": 1.}
            torch.save(last, directory / "last.pt")
            digests[(arm, seed)] = ft.sha256_file(checkpoint)
            write_json(directory / "config.json", config)
            write_json(directory / "history.json", history)
            write_json(directory / "summary.json", {"test_used": False, "epochs_completed": 26, "stopped_early": False,
                "best_epoch": 26, "best_validation_score": 1., "generator_updates": 26, "discriminator_updates": 26,
                "samples_seen": 512*26, "data_hashes": hashes, "data_curation": status, "selection": selection,
                "provenance": provenance, "source_sha256": RELEASE_SHA256,
                "validation_metric": fr.training.CONDITIONAL_VALIDATION_METRIC})
            calibrator = None
            if arm != "unconditional96":
                provenance = {**templates[width].provenance, "checkpoint_sha256": digests[(arm, seed)],
                    "train_indices_sha256": selection["train_indices_sha256"], "prepared_data_hashes": part["hashes"]}
                model = ConditionalMarginalCalibrator(width, 1025, templates[width].models, provenance)
                calibrator = directory / "calibration.npz"
                model.save(calibrator)
                map_records[(arm, seed)] = {"sha256": ft.sha256_file(calibrator), "provenance": provenance}
            arms[arm].append({"checkpoint": str(checkpoint), "calibrator": str(calibrator) if calibrator else None})
    evidence = {"native": {}, "common_hourly": {}, "paired": {}, "additional": []}
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    def report(key, roles, seed, kind, contrast=None):
        path = evidence_dir / (key + ".json")
        body = {"split": "val", "test_used": False, "seed": 2026,
                "samples_per_condition_cap": 1000 if kind in ("precision", "utility_precision") else 100,
                "synthetic_fixture": True}
        if len(roles) == 1:
            arm = roles[0]
            body["checkpoint"] = {"sha256": digests[(arm, seed)]}
            body["prepared_data_hashes"] = data[24 if arm == "bce24" else 96]["hashes"]
            body["methods"] = {name: {} for name in ("training_cgan", "training_real", "training_resampling")} if "utility" in kind else {"cgan": {}}
            if arm != "unconditional96" and kind != "raw_primary":
                body["calibration"] = {**map_records[(arm, seed)], "saved_model": {"sha256": map_records[(arm, seed)]["sha256"]}}
                body["methods"].update({"training_calibrated_cgan": {}} if "utility" in kind else {"raw_cgan": {}, "calibrated_cgan": {}})
            if arm == "unconditional96" and kind in ("primary", "precision"):
                body.update(report_kind="supplemental_native96_raw_unconditional_reference", conditional_calibrator_used=False)
                body["methods"] = {name: {} for name in ("raw_unconditional_cgan", "conditional_mean", "conditional_resampling", "gaussian_copula", "independent_marginals")}
        else:
            body["prepared_data_hashes"] = {"parent96": data[96]["hashes"], **({"derived24": data[24]["hashes"]} if "bce24" in roles else {})}
            if len(roles) == 3:
                body["models"] = {role: {"sha256": digests[(role, seed)]} for role in roles}
                body["saved_maps"] = {role: map_records[(role, seed)] for role in roles}
                body["methods"] = {f"{role}_{variant}": {} for role in roles for variant in ("raw", "calibrated")}
        if "utility" in kind:
            width = 24 if len(roles) == 3 or roles == ("bce24",) else 96
            body.update(task={"resolution": width, "lags": width // 8}, utility_training_selection={"selected_profiles": 128, "cap_applied": False})
        if kind == "raw_primary":
            body["methods"].update(conditional_mean={}, conditional_resampling={})
        if kind != "paired":
            metric = {"rmse_kwh": 1., "mae_kwh": .5} if "utility" in kind else {
                **{name: .25 for name in ft.METRICS[:-1]}, "autocorrelation": {"1": {"wasserstein": .25}}}
            payload = {"macro": metric, "per_condition": {str(c): metric for c in range(64)}}
            body["methods"] = {name: payload for name in body["methods"]}
        if kind == "paired":
            body.update(training_seed=seed, replicates=500, metrics=list(ft.METRICS), contrast=contrast)
            width = 24 if "bce24" in roles else 96
            body.update(resolution=width, interval_hours=24/width, paired_comparison=paired_payload(width))
            body.update(candidate={"sha256": digests[(roles[0], seed)]}, comparator={"sha256": digests[(roles[1], seed)]},
                saved_maps={name: map_records[(arm, seed)] for name, arm in zip(("candidate", "comparator"), roles)} if contrast in ("candidate_reference", "resolution_calibrated") else {})
        write_json(path, body)
        return str(path)
    for arm in fr.ARMS:
        evidence["native"][arm] = {str(seed): {kind: report(f"{arm}_{seed}_{kind}", (arm,), seed, kind)
            for kind in fr.NATIVE_EVIDENCE} for seed in fr.SEEDS}
    for seed in fr.SEEDS:
        evidence["common_hourly"][str(seed)] = {kind: report(f"hourly_{seed}_{kind}", ("candidate96", "bce96", "bce24"), seed, kind) for kind in fr.HOURLY_EVIDENCE}
    for contrast, (first, second) in fr._contrast_views().items():
        evidence["paired"][contrast] = {str(seed): report(f"{contrast}_{seed}", (first[0], second[0]), seed, "paired", contrast) for seed in fr.SEEDS}
    protocol, attempts = tmp_path / "protocol.md", tmp_path / "attempts.md"
    protocol.write_text("Synthetic declared fixed study.\n")
    attempts.write_text("Synthetic complete provided attempt ledger: twenty successful fixture runs.\n")
    return dict(parent=parent, child=child, arms=arms, evidence=evidence, protocol=protocol, attempts=attempts, data=data)


def freeze(inputs, output):
    return fr.freeze_reference_study(inputs["parent"], inputs["child"], inputs["arms"], output,
        candidate_kind="swd10", decision_rationale="Synthetic contract study, not research results.",
        protocol_files=[inputs["protocol"]], development_evidence=inputs["evidence"], attempt_history=[inputs["attempts"]])


def test_production_protocol_and_role_contracts():
    assert fr.TRAIN_ROWS == 606060 and fr.BOOTSTRAP_REPLICATES == 500
    assert (fr.TRAIN_ROWS + 511) // 512 == 1184
    assert 1184 * 26 == 30784
    assert len(fr._required_coverage()) == 140
    for arm in fr.ARMS:
        config = fr._expected_config(arm, 11, "capacity")
        assert config["unconditional"] == (arm == "unconditional96")
        assert config["latent_dim"] == (96 if arm == "candidate96" else 24)
    assert fr._protocol()["utility"] is True


def test_hourly_map_order_and_float64_accumulation():
    values = np.zeros((2, 96), dtype=np.float64)
    values[:, :4] = [2**24, 1.25, -(2**24), 1.25]
    result = fr._sum4(values)
    assert result.dtype == np.float32 and result[0, 0] == 2.5
    assert fr._sum4(values.astype(np.float32))[0, 0] == 2.5
    values[:, :4] = [2**24 + .25, 0., -(2**24), 0.]
    assert fr._sum4(values)[0, 0] == .25
    assert fr._sum4(values.astype(np.float32))[0, 0] == 0


@pytest.mark.parametrize("arm,calibrator", [("unconditional96", "map.npz"), ("bce24", None)])
def test_map_role_rejected_before_any_reads(arm, calibrator):
    arms = {name: [{"checkpoint": "unused.pt", "calibrator": None if name == "unconditional96" else "map.npz"} for _ in fr.SEEDS] for name in fr.ARMS}
    arms[arm][0]["calibrator"] = calibrator
    with pytest.raises(ValueError, match="raw-only"):
        fr._roster(arms)


def test_complete_bundle_scores_once_without_fit_or_generation(tmp_path, monkeypatch):
    inputs = make_inputs(tmp_path, monkeypatch)
    bundle = tmp_path / "bundle"
    with forbid_test(monkeypatch):
        sealed = freeze(inputs, bundle)
    manifest = ft._json(bundle / "manifest.json")
    assert manifest["schema_version"] == 2 and len(manifest["decision"]["coverage"]) == 140
    assert sum(len(rows) for rows in manifest["arms"].values()) == 20
    assert sum(r["calibrator"] is not None for rows in manifest["arms"].values() for r in rows) == 15
    assert set(manifest["controls"]["96"]) == set(fr.CONTROLS[96])
    assert all(r["calibrator"] is None for r in manifest["arms"]["unconditional96"])
    calls = []
    original = ft._read_verified_test
    def load_once(binding):
        calls.append(binding["sha256"])
        return original(binding)
    def fail(*args, **kwargs):
        pytest.fail("TEST evaluator attempted fitting/generation/map transformation/TRAIN load")
    monkeypatch.setattr(ft, "_read_verified_test", load_once)
    monkeypatch.setattr(fr.training, "generate", fail)
    monkeypatch.setattr(fr.training, "_load_prepared", fail)
    monkeypatch.setattr(ft, "fit_conditional_copula", fail)
    monkeypatch.setattr(ft.RidgeProfileForecaster, "fit", fail)
    monkeypatch.setattr(ConditionalMarginalCalibrator, "transform", fail)
    result = fr.evaluate_reference_study(bundle, sealed["seal_sha256"], tmp_path / "scored")
    assert len(calls) == 1
    assert result["split"] == "test" and result["fitting_performed"] is False
    assert set(result["paired_reference_comparisons"]) == set(fr.CONTRASTS)
    assert all(len(rows) == 5 for rows in result["paired_reference_comparisons"].values())
    assert all(len(rows) == 5 for rows in result["arms"].values())
    assert result["test_selection"]["selected_profiles"] == 128
    assert set(result["test_selection"]["profiles_sha256"]) == {"24", "96"}
    values = [row["methods"]["mapped96"]["physical"]["macro"]["ramp_wasserstein_kw"] for row in result["arms"]["candidate96"]]
    statistics = result["five_seed_summary"]["candidate96"]["mapped96"]["physical_macro"]["ramp_wasserstein_kw"]
    assert statistics["mean"] == pytest.approx(np.mean(values))
    assert statistics["sample_sd"] == pytest.approx(np.std(values, ddof=1))
    for contrast in fr.CONTRASTS:
        assert all(row["physical"]["n_bootstrap"] == 2 for row in result["paired_reference_comparisons"][contrast])
    with pytest.raises(FileExistsError):
        fr.evaluate_reference_study(bundle, sealed["seal_sha256"], tmp_path / "repeat")
    assert len(calls) == 1 and not (tmp_path / "repeat" / "report.json").exists()
    assert ft._json(tmp_path / "repeat" / "failure.json")["test_access_started"] is False


def test_implicit_evidence_test_alias_fails_before_content_access(tmp_path, monkeypatch):
    parent, child = tmp_path / "parent", tmp_path / "child"
    parent.mkdir(); child.mkdir()
    reserved = parent / "test.npz"
    reserved.write_bytes(b"synthetic reserved content")
    directory = tmp_path / "runs"
    directory.mkdir()
    os.link(reserved, directory / "summary.json")
    arms = {arm: [{"checkpoint": str(directory / "best.pt"), "calibrator": None if arm == "unconditional96" else "unused.npz"} for _ in fr.SEEDS] for arm in fr.ARMS}
    monkeypatch.setattr(fr, "_evidence_slots", lambda evidence: [])
    with forbid_test(monkeypatch), pytest.raises(ValueError, match="aliases reserved TEST"):
        fr.freeze_reference_study(parent, child, arms, tmp_path / "bad", candidate_kind="swd10",
            decision_rationale="Synthetic alias rejection", protocol_files=["unused.md"], development_evidence={"additional": []}, attempt_history=["unused.md"])
    assert ft._json(tmp_path / "bad" / "freeze_failure.json")["test_accessed"] is False
