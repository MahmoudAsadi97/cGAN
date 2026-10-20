"""Synthetic-only integration checks for the validation copula comparator.

Fixtures construct checkpoint provenance directly: no GAN training, generator
weights, or real prepared data are needed for the statistical baseline.
"""

import builtins
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from fluvius_cgan.copula import ConditionalGaussianCopula, fit_conditional_copula
from fluvius_cgan.evaluation import evaluate_conditional, nearest_neighbor_diagnostics
from fluvius_cgan.training import CHECKPOINT_FORMAT, _config


SPEC = importlib.util.spec_from_file_location(
    "evaluate_copula", Path(__file__).resolve().parents[1] / "scripts" / "evaluate_copula.py")
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


def digest(values, dtype):
    return hashlib.sha256(np.asarray(values, dtype=dtype).tobytes(order="C")).hexdigest()


@pytest.fixture(scope="module")
def experiment(tmp_path_factory):
    root = tmp_path_factory.mktemp("copula-evaluation")
    prepared = root / "prepared"
    prepared.mkdir()
    rng = np.random.default_rng(6)
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
        manifest[split] = {
            "profiles": 16, "meters": 4,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "meter_ids": sorted(set(arrays[split]["meter_id"].tolist())),
        }
    # This synthetic sentinel must never be loaded, opened, or hashed.
    (prepared / "test.npz").write_bytes(b"sealed synthetic test sentinel")
    scaler = {
        "minimum": float(arrays["train"]["x"].min()),
        "maximum": float(arrays["train"]["x"].max()),
    }
    metadata = {
        "resolution": 24, "target": "net", "units": "kWh/interval",
        "source_sha256": "a" * 64, "split_seed": 42, "splits": manifest,
    }
    (prepared / "scaler.json").write_text(json.dumps(scaler))
    (prepared / "metadata.json").write_text(json.dumps(metadata))
    selected_train = np.sort(np.random.default_rng(7).choice(16, 12, replace=False))
    state = {
        "format_version": CHECKPOINT_FORMAT, "epoch": 1,
        "config": _config({"seed": 7, "max_train_samples": 12}),
        "metadata": metadata, "scaler": scaler, "output_dim": 24,
        "data_hashes": {
            name: hashlib.sha256((prepared / name).read_bytes()).hexdigest()
            for name in ("train.npz", "val.npz", "scaler.json", "metadata.json")
        },
        "selection": {
            "train_profiles_available": 16, "train_profiles_used": 12,
            "train_indices_sha256": digest(selected_train, "<i8"),
        },
        "supported_labels": np.unique(arrays["train"]["y"][selected_train]).tolist(),
    }
    checkpoint = root / "reference.pt"
    torch.save(state, checkpoint)
    return prepared, checkpoint, arrays, selected_train


@pytest.fixture
def forbid_test_access(monkeypatch):
    path_open, builtin_open, numpy_load = Path.open, builtins.open, np.load

    def check(path):
        if isinstance(path, (str, bytes, Path)):
            name = Path(path.decode() if isinstance(path, bytes) else path).name
            assert name != "test.npz", "test split must not be opened, loaded, or hashed"

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


def selected_validation(arrays, seed, cap):
    rng = np.random.default_rng(seed)
    return np.concatenate([
        rng.choice(np.flatnonzero(arrays["val"]["y"] == label),
                   min(cap, int(np.sum(arrays["val"]["y"] == label))), replace=False)
        for label in np.unique(arrays["val"]["y"])
    ])


@pytest.mark.parametrize("knots", [None, 3])
def test_exact_checkpoint_subset_and_validation_rows_match_manual_metrics(
        experiment, forbid_test_access, knots):
    prepared, checkpoint, arrays, selected_train = experiment
    seed, cap = 23, 3
    report = evaluation.evaluate(prepared, checkpoint, seed=seed,
                                 samples_per_condition=cap, quantile_knots=knots)
    selected_val = selected_validation(arrays, seed, cap)
    tx, ty = (arrays["train"][key][selected_train] for key in ("x", "y"))
    vx, vy = (arrays["val"][key][selected_val] for key in ("x", "y"))
    fitted = fit_conditional_copula(tx, ty, quantile_knots=knots, shrinkage=.01)
    generated = fitted.generate(vy, seed=seed)
    manual = evaluate_conditional(vx, vy, generated, vy, interval_hours=1.)
    assert report["methods"]["conditional_gaussian_copula"] == manual
    assert report["nearest_neighbors"] == nearest_neighbor_diagnostics(tx, generated, vx, seed=seed)
    assert report["reference_checkpoint"]["training_selection"]["train_indices_sha256"] == digest(selected_train, "<i8")
    assert report["training_profiles"] == {"prepared": 16, "model": 12, "baseline": 12}
    assert report["validation_selection"]["selected_indices_sha256"] == digest(selected_val, "<i8")
    assert report["frozen_samples"]["real_profiles_sha256"] == digest(vx, "<f8")
    assert report["frozen_samples"]["generated_profiles_sha256"] == digest(generated, "<f8")
    assert report["frozen_samples"]["labels_sha256"] == digest(vy, "<i8")
    assert report["copula"]["quantile_knots"] == knots
    assert report["copula"]["shrinkage"] == .01
    assert report["test_used"] is False and report["split"] == "val"
    assert "test.npz" not in report["prepared_data_hashes"]
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("tamper", ["data_hash", "selection_hash"])
def test_rejects_tampered_reference_binding(experiment, forbid_test_access, tmp_path, tamper):
    prepared, checkpoint, _, _ = experiment
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if tamper == "data_hash":
        state["data_hashes"]["val.npz"] = "0" * 64
        message = "fingerprint.*val.npz"
    else:
        state["selection"]["train_indices_sha256"] = "0" * 64
        message = "exact training subset"
    invalid = tmp_path / "invalid.pt"
    torch.save(state, invalid)
    with pytest.raises(ValueError, match=message):
        evaluation.evaluate(prepared, invalid)


def test_optional_saved_model_replays_report_samples(experiment, forbid_test_access, tmp_path):
    prepared, checkpoint, arrays, _ = experiment
    model_output = tmp_path / "copula.npz"
    report = evaluation.evaluate(prepared, checkpoint, samples_per_condition=3,
                                 model_output=model_output)
    loaded = ConditionalGaussianCopula.load(model_output)
    assert loaded.quantile_knots is None and loaded.shrinkage == .01
    labels = arrays["val"]["y"][selected_validation(arrays, 2026, 3)]
    replay = loaded.generate(labels, seed=2026)
    assert report["frozen_samples"]["generated_profiles_sha256"] == digest(replay, "<f8")


def test_cli_defaults_exact_margins_and_refuses_report_overwrite(
        experiment, forbid_test_access, tmp_path):
    prepared, checkpoint, _, _ = experiment
    output, model_output = tmp_path / "report.json", tmp_path / "fitted.npz"
    args = ["--prepared", str(prepared), "--reference-checkpoint", str(checkpoint),
            "--output", str(output), "--model-output", str(model_output),
            "--samples-per-condition", "3"]
    evaluation.main(args)
    original = output.read_bytes()
    report = json.loads(original)
    assert report["copula"]["quantile_knots"] is None
    assert report["seed"] == 2026
    assert report["validation_selection"]["condition_counts"] == {"0": 3, "4": 3}
    assert model_output.is_file()
    with pytest.raises(FileExistsError):
        evaluation.main(args)
    assert output.read_bytes() == original
    with pytest.raises(SystemExit):
        evaluation.main(args + ["--split", "test"])
