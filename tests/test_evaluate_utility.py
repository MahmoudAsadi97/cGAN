"""Tiny synthetic-only integration checks for the validation utility command."""
import builtins
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from fluvius_cgan.copula import fit_conditional_copula
from fluvius_cgan.evaluation import resample_baseline
from fluvius_cgan.models import Generator
from fluvius_cgan.training import CHECKPOINT_FORMAT, _config, generate
from fluvius_cgan.utility import RidgeProfileForecaster

SPEC = importlib.util.spec_from_file_location(
    "evaluate_utility", Path(__file__).resolve().parents[1] / "scripts" / "evaluate_utility.py")
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


def digest(values, dtype):
    return hashlib.sha256(np.asarray(values, dtype=dtype).tobytes(order="C")).hexdigest()


@pytest.fixture(scope="module")
def experiment(tmp_path_factory):
    root = tmp_path_factory.mktemp("utility-evaluation")
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
            "profiles": 16, "meters": 4, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "meter_ids": sorted(set(arrays[split]["meter_id"].tolist())),
        }
    (prepared / "test.npz").write_bytes(b"sealed synthetic test sentinel")
    scaler = {"minimum": float(arrays["train"]["x"].min()), "maximum": float(arrays["train"]["x"].max())}
    metadata = {"resolution": 24, "target": "net", "units": "kWh/interval",
                "source_sha256": "a" * 64, "split_seed": 42, "splits": manifest}
    (prepared / "scaler.json").write_text(json.dumps(scaler))
    (prepared / "metadata.json").write_text(json.dumps(metadata))
    selected = np.sort(np.random.default_rng(7).choice(16, 12, replace=False))
    config = _config({"seed": 7, "max_train_samples": 12, "latent_dim": 4, "hidden_dim": 4})
    with torch.random.fork_rng():
        torch.manual_seed(91)
        generator = Generator(latent_dim=4, hidden_dim=4, output_dim=24)
    state = {
        "format_version": CHECKPOINT_FORMAT, "epoch": 1, "config": config,
        "metadata": metadata, "scaler": scaler, "output_dim": 24, "condition_dim": 64,
        "generator_state_dict": generator.state_dict(),
        "data_hashes": {name: hashlib.sha256((prepared / name).read_bytes()).hexdigest()
                        for name in ("train.npz", "val.npz", "scaler.json", "metadata.json")},
        "selection": {"train_profiles_available": 16, "train_profiles_used": 12,
                      "train_indices_sha256": digest(selected, "<i8")},
        "supported_labels": np.unique(arrays["train"]["y"][selected]).tolist(),
    }
    checkpoint = root / "checkpoint.pt"
    torch.save(state, checkpoint)
    return prepared, checkpoint, arrays, selected


@pytest.fixture
def forbid_test_access(monkeypatch):
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


def validation_indices(arrays, seed, cap):
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.choice(np.flatnonzero(arrays["val"]["y"] == label), cap, replace=False)
                           for label in np.unique(arrays["val"]["y"])])


@pytest.mark.parametrize("cap", [None, 8])
def test_identical_labels_exact_source_subset_and_manual_all_method_scores(
        experiment, forbid_test_access, cap):
    prepared, checkpoint, arrays, actual = experiment
    seed, validation_cap = 23, 3
    report = evaluation.evaluate(prepared, checkpoint, samples_per_condition=validation_cap,
                                 seed=seed, max_train_profiles=cap, copula=True)
    selected = actual if cap is None else np.sort(np.random.default_rng(seed + 100).choice(actual, cap, replace=False))
    indices = validation_indices(arrays, seed, validation_cap)
    source_x, source_y = (arrays["train"][key][actual] for key in ("x", "y"))
    train_x, train_y = (arrays["train"][key][selected] for key in ("x", "y"))
    vx, vy = (arrays["val"][key][indices] for key in ("x", "y"))
    sources = {
        "training_real": train_x,
        "training_cgan": generate(checkpoint, train_y, seed=seed),
        "training_resampling": resample_baseline(source_x, source_y, train_y, seed=seed),
        "training_copula": fit_conditional_copula(source_x, source_y, quantile_knots=None,
                                                 shrinkage=.01).generate(train_y, seed=seed),
    }
    for name, values in sources.items():
        manual = RidgeProfileForecaster().fit(values, train_y).evaluate(vx, vy)
        persistence = manual.pop("persistence")
        assert report["methods"][name] == manual
        assert report["methods"]["persistence"] == persistence
        assert report["training"][name]["profiles_sha256"] == digest(values, "<f8")
        assert report["training"][name]["labels_sha256"] == digest(train_y, "<i8")
        assert report["training"][name]["n_profiles"] == len(selected)
    assert report["source_training_pool"]["actual_gan_profiles"] == 12
    assert report["gan_preprocessing"]["fitting_profiles"] == 16
    assert report["gan_preprocessing"]["fitted_on"] == "full_prepared_train_before_training_subset_selection"
    assert report["gan_preprocessing"]["uses_validation_values"] is False
    assert report["source_training_pool"]["actual_indices_sha256"] == digest(actual, "<i8")
    assert report["copula"]["source_indices_sha256"] == digest(actual, "<i8")
    assert report["copula"]["quantile_knots"] is None
    assert report["utility_training_selection"]["selected_indices_sha256"] == digest(selected, "<i8")
    assert report["utility_training_selection"]["cap_applied"] is (cap is not None)
    assert report["validation_selection"]["selected_indices_sha256"] == digest(indices, "<i8")
    assert report["validation_selection"]["profiles_sha256"] == digest(vx, "<f8")
    assert report["invariants"]["training_condition_counts_match"] is True
    assert report["invariants"]["training_label_order_matches"] is True
    assert report["invariants"]["fit_uses_validation_values"] is False
    assert report["split"] == "val" and report["test_used"] is False
    assert "test.npz" not in report["prepared_data_hashes"]
    json.dumps(report, allow_nan=False)


def test_default_omits_copula_and_replays_coefficients(experiment, forbid_test_access, monkeypatch):
    prepared, checkpoint, _, _ = experiment

    def forbid_copula(*args, **kwargs):
        raise AssertionError("copula must be opt-in")

    monkeypatch.setattr(evaluation, "fit_conditional_copula", forbid_copula)
    first = evaluation.evaluate(prepared, checkpoint, samples_per_condition=2)
    second = evaluation.evaluate(prepared, checkpoint, samples_per_condition=2)
    assert first["methods"] == second["methods"]
    assert first["training"] == second["training"]
    assert first["copula"] == {"enabled": False}
    assert set(first["methods"]) == {"training_real", "training_cgan", "training_resampling", "persistence"}


def test_cli_refuses_overwrite_and_test_split(experiment, forbid_test_access, tmp_path):
    prepared, checkpoint, _, _ = experiment
    output = tmp_path / "utility.json"
    args = ["--prepared", str(prepared), "--checkpoint", str(checkpoint),
            "--output", str(output), "--samples-per-condition", "2", "--copula"]
    evaluation.main(args)
    original = output.read_bytes()
    report = json.loads(original)
    assert report["config"]["seed"] == 2026
    assert report["config"]["max_train_profiles"] is None
    with pytest.raises(FileExistsError):
        evaluation.main(args)
    assert output.read_bytes() == original
    with pytest.raises(SystemExit):
        evaluation.main(args + ["--split", "test"])


@pytest.mark.parametrize("field", ["data_hash", "indices_hash"])
def test_corrupted_checkpoint_binding_rejected(experiment, forbid_test_access, tmp_path, field):
    prepared, checkpoint, _, _ = experiment
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if field == "data_hash":
        state["data_hashes"]["val.npz"] = "0" * 64
    else:
        state["selection"]["train_indices_sha256"] = "0" * 64
    invalid = tmp_path / "invalid.pt"
    torch.save(state, invalid)
    with pytest.raises(ValueError, match="fingerprint|exact training subset"):
        evaluation.evaluate(prepared, invalid)


def test_cap_that_loses_condition_support_fails_without_narrowing_validation(experiment, forbid_test_access):
    prepared, checkpoint, _, _ = experiment
    with pytest.raises(ValueError, match="cap omitted validation conditions"):
        evaluation.evaluate(prepared, checkpoint, max_train_profiles=1)


@pytest.mark.parametrize("options", [{"samples_per_condition": 0}, {"samples_per_condition": True},
                                      {"seed": -1}, {"seed": 2 ** 32}, {"seed": True},
                                      {"max_train_profiles": 0}, {"max_train_profiles": True},
                                      {"copula": "yes"}])
def test_invalid_settings_rejected_without_data_access(options, monkeypatch):
    def forbid_load(*args, **kwargs):
        raise AssertionError("invalid settings must fail before data access")

    monkeypatch.setattr(evaluation, "_load_prepared", forbid_load)
    with pytest.raises(ValueError):
        evaluation.evaluate("unused", "unused", **options)
