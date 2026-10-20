"""Synthetic integration of fixed-map lineage and shared-meter comparisons."""
import builtins
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from fluvius_cgan.calibration import ConditionalMarginalCalibrator, fit
from fluvius_cgan.copula import fit_conditional_copula
from fluvius_cgan.evaluation import evaluate_conditional, resample_baseline
from fluvius_cgan.training import CHECKPOINT_FORMAT, _config
from fluvius_cgan.uncertainty import paired_condition_macro_meter_bootstrap


SPEC = importlib.util.spec_from_file_location(
    "compare_calibrated", Path(__file__).resolve().parents[1] / "scripts" / "compare_calibrated.py")
comparison = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparison)


def digest(values, dtype):
    return hashlib.sha256(np.asarray(values, dtype=dtype).tobytes(order="C")).hexdigest()


def generated(labels, seed):
    return np.random.default_rng(seed).normal(size=(len(labels), 24)) + np.asarray(labels)[:, None] / 5


@pytest.fixture(scope="module")
def experiment(tmp_path_factory):
    root = tmp_path_factory.mktemp("calibrated-comparison")
    prepared = root / "prepared"
    prepared.mkdir()
    rng = np.random.default_rng(61)
    arrays, manifest = {}, {}
    for split in ("train", "val"):
        arrays[split] = {
            "x": rng.normal(size=(16, 24)).astype(np.float32),
            "y": np.tile([0, 0, 4, 4], 4).astype(np.int64),
            "meter_id": np.repeat([f"{split}-{i}" for i in range(4)], 4),
            "date": np.tile(["2024-01-01", "2024-01-02", "2024-01-06", "2024-01-07"], 4),
        }
        path = prepared / f"{split}.npz"
        np.savez(path, **arrays[split])
        manifest[split] = {"profiles": 16, "meters": 4,
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                           "meter_ids": sorted(set(arrays[split]["meter_id"].tolist()))}
    (prepared / "test.npz").write_bytes(b"sealed synthetic test sentinel")
    scaler = {"minimum": float(arrays["train"]["x"].min()),
              "maximum": float(arrays["train"]["x"].max())}
    metadata = {"resolution": 24, "target": "net", "units": "kWh/interval",
                "source_sha256": "a" * 64, "split_seed": 42, "splits": manifest}
    (prepared / "scaler.json").write_text(json.dumps(scaler))
    (prepared / "metadata.json").write_text(json.dumps(metadata))
    selected = np.sort(np.random.default_rng(7).choice(16, 12, replace=False))
    hashes = {name: hashlib.sha256((prepared / name).read_bytes()).hexdigest()
              for name in ("train.npz", "val.npz", "metadata.json", "scaler.json")}
    checkpoint = root / "reference.pt"
    torch.save({
        "format_version": CHECKPOINT_FORMAT, "epoch": 1,
        "config": _config({"seed": 7, "max_train_samples": 12}),
        "metadata": metadata, "scaler": scaler, "output_dim": 24, "data_hashes": hashes,
        "selection": {"train_profiles_available": 16, "train_profiles_used": 12,
                      "train_indices_sha256": digest(selected, "<i8")},
        "supported_labels": np.unique(arrays["train"]["y"][selected]).tolist(),
    }, checkpoint)
    calibration_labels = np.repeat([0, 4], 17)
    source = generated(calibration_labels, 7301)
    model = fit(arrays["train"]["x"][selected], arrays["train"]["y"][selected],
                source, calibration_labels, quantile_knots=5, provenance={
                    "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    "train_indices_sha256": digest(selected, "<i8"), "prepared_data_hashes": hashes,
                    "calibration_seed": 7301, "calibration_samples_per_condition": 17,
                    "calibration_profiles_total": 34,
                    "calibration_generated_profiles_sha256": digest(source, "<f8"),
                    "calibration_labels_sha256": digest(calibration_labels, "<i8"),
                    "generator_inference_batch_size": 1024, "quantile_knots": 5,
                })
    calibrator = root / "calibration.npz"
    model.save(calibrator)
    return prepared, checkpoint, calibrator, arrays, selected


@pytest.fixture
def guards(monkeypatch):
    path_open, builtin_open, numpy_load = Path.open, builtins.open, np.load

    def check(path):
        if isinstance(path, (str, bytes, Path)):
            assert Path(path.decode() if isinstance(path, bytes) else path).name != "test.npz"

    def guarded_path_open(path, *args, **kwargs):
        check(path)
        return path_open(path, *args, **kwargs)

    def guarded_builtin_open(path, *args, **kwargs):
        check(path)
        return builtin_open(path, *args, **kwargs)

    def guarded_numpy_load(path, *args, **kwargs):
        check(path)
        return numpy_load(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_path_open)
    monkeypatch.setattr(builtins, "open", guarded_builtin_open)
    monkeypatch.setattr(np, "load", guarded_numpy_load)
    calls = []

    def generate(checkpoint, labels, seed):
        calls.append((np.array(labels), seed))
        return generated(labels, seed)

    monkeypatch.setattr(comparison, "generate", generate)
    return calls


@pytest.mark.parametrize("baseline", comparison.BASELINES)
def test_same_fixed_arrays_subset_margins_and_draws_match_manual_calculation(experiment, guards, baseline):
    prepared, checkpoint, calibrator, arrays, train_indices = experiment
    archive = calibrator.read_bytes()
    seed, replicates = 23, 5
    report = comparison.compare(prepared, checkpoint, calibrator, baseline=baseline,
                                seed=seed, samples_per_condition=100, replicates=replicates)
    rng = np.random.default_rng(seed)
    val_indices = np.concatenate([rng.choice(np.flatnonzero(arrays["val"]["y"] == label), 8, replace=False)
                                  for label in (0, 4)])
    real, labels, meters = (arrays["val"][key][val_indices] for key in ("x", "y", "meter_id"))
    tx, ty = (arrays["train"][key][train_indices] for key in ("x", "y"))
    raw = generated(labels, seed)
    model = ConditionalMarginalCalibrator.load(calibrator)
    candidate = model.transform(raw, labels)
    if baseline in ("gaussian_copula", "exact_marginal_copula"):
        knots = 5 if baseline == "gaussian_copula" else None
        reference = fit_conditional_copula(tx, ty, quantile_knots=knots, shrinkage=.01).generate(labels, seed=seed)
        assert report["comparator"]["quantile_knots"] == knots
        assert report["comparator"]["quantile_setting_matches_calibrator"] == (knots == 5)
    elif baseline == "conditional_resampling":
        reference = resample_baseline(tx, ty, labels, seed=seed)
    else:
        reference = raw
    manual = paired_condition_macro_meter_bootstrap(
        real, labels, candidate, labels, reference, labels, meters, interval_hours=1,
        n_bootstrap=replicates, seed=seed, real_meter_strata=labels // 8,
        metrics=comparison.METRICS, return_per_condition=True,
    )
    assert report["paired_comparison"] == manual
    assert report["methods"]["calibrated_cgan"] == evaluate_conditional(real, labels, candidate, labels, interval_hours=1)
    assert report["methods"][baseline] == evaluate_conditional(real, labels, reference, labels, interval_hours=1)
    assert report["validation_selection"]["selected_indices_sha256"] == digest(val_indices, "<i8")
    assert report["frozen_samples"]["candidate_profiles_sha256"] == digest(candidate, "<f8")
    assert report["calibrator"]["provenance"]["train_indices_sha256"] == digest(train_indices, "<i8")
    assert len(guards) == 1 and guards[0][1] == seed
    assert report["test_used"] is False and report["split"] == "val"
    assert "test.npz" not in report["prepared_data_hashes"]
    assert calibrator.read_bytes() == archive
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("key", ["checkpoint_sha256", "train_indices_sha256"])
def test_calibration_binding_rejects_different_checkpoint_or_subset_before_generation(experiment, guards, tmp_path, key):
    prepared, checkpoint, calibrator, _, _ = experiment
    model = ConditionalMarginalCalibrator.load(calibrator)
    model.provenance[key] = "0" * 64
    invalid = tmp_path / "invalid.npz"
    model.save(invalid)
    with pytest.raises(ValueError, match=key):
        comparison.compare(prepared, checkpoint, invalid, replicates=2)
    assert not guards


def test_rejects_bad_checkpoint_data_binding_before_generation(experiment, guards, tmp_path):
    prepared, checkpoint, calibrator, _, _ = experiment
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state["data_hashes"]["val.npz"] = "0" * 64
    invalid = tmp_path / "invalid.pt"
    torch.save(state, invalid)
    with pytest.raises(ValueError, match="fingerprint.*val.npz"):
        comparison.compare(prepared, invalid, calibrator, replicates=2)
    assert not guards


def test_forbids_calibration_reference_seed_as_fresh_generation(experiment, guards):
    prepared, checkpoint, calibrator, _, _ = experiment
    with pytest.raises(ValueError, match="distinct"):
        comparison.compare(prepared, checkpoint, calibrator, seed=7301, replicates=2)
    assert not guards


def test_cli_defaults_output_guard_and_no_test_option(experiment, guards, tmp_path):
    prepared, checkpoint, calibrator, _, _ = experiment
    output = tmp_path / "report.json"
    args = ["--prepared", str(prepared), "--checkpoint", str(checkpoint),
            "--calibrator", str(calibrator), "--output", str(output), "--replicates", "2"]
    comparison.main(args)
    original = output.read_bytes()
    report = json.loads(original)
    assert report["comparator"]["kind"] == "gaussian_copula"
    assert report["seed"] == 2026
    with pytest.raises(FileExistsError):
        comparison.main(args)
    assert output.read_bytes() == original
    with pytest.raises(SystemExit):
        comparison.main(args + ["--split", "test"])


@pytest.mark.parametrize("options", [{"replicates": 1}, {"samples_per_condition": 0},
                                      {"seed": 2 ** 32}, {"baseline": "unrecognized"}])
def test_invalid_options_fail_before_missing_data_are_read(options):
    with pytest.raises(ValueError):
        comparison.compare("missing", "missing", "missing", **options)
