"""Integration checks for validation-only, provenance-bound paired comparisons."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from fluvius_cgan.evaluation import mean_baseline, resample_baseline
from fluvius_cgan.training import generate, train
from fluvius_cgan.uncertainty import paired_condition_macro_meter_bootstrap


SPEC = importlib.util.spec_from_file_location(
    "compare_checkpoints", Path(__file__).resolve().parents[1] / "scripts" / "compare_checkpoints.py")
comparison = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparison)


@pytest.fixture(scope="module")
def experiment(tmp_path_factory):
    root = tmp_path_factory.mktemp("paired-checkpoints")
    prepared = root / "prepared"
    prepared.mkdir()
    rng = np.random.default_rng(6)
    manifest = {}
    arrays = {}
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
            "meter_ids": sorted(set(arrays[split]["meter_id"])),
        }
    # A synthetic unreadable sentinel; access guards below also forbid hashing it.
    (prepared / "test.npz").write_bytes(b"sealed synthetic test sentinel")
    (prepared / "scaler.json").write_text(json.dumps({
        "minimum": float(arrays["train"]["x"].min()),
        "maximum": float(arrays["train"]["x"].max()),
    }))
    (prepared / "metadata.json").write_text(json.dumps({
        "resolution": 24, "target": "net", "units": "kWh/interval",
        "source_sha256": "a" * 64, "split_seed": 42, "splits": manifest,
    }))
    options = {"epochs": 1, "hidden_dim": 4, "latent_dim": 4,
               "batch_size": 4, "validation_samples": 8, "num_threads": 1}
    candidate, comparator = root / "candidate", root / "comparator"
    train(prepared, candidate, {**options, "seed": 7, "max_train_samples": 12})
    train(prepared, comparator, {**options, "seed": 8, "normalization": "standard"})
    return prepared, candidate / "best.pt", comparator / "best.pt", arrays


@pytest.fixture
def forbid_test_access(monkeypatch):
    original = Path.open

    def guarded(path, *args, **kwargs):
        assert path.name != "test.npz", "test split must not be opened or hashed"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)


@pytest.mark.parametrize("baseline", ["conditional_resampling", "conditional_mean"])
def test_baselines_use_exact_candidate_subset_and_match_manual_comparison(experiment, forbid_test_access, baseline):
    prepared, checkpoint, _, arrays = experiment
    seed, cap, bootstrap = 23, 100, 5
    report = comparison.compare(prepared, checkpoint, baseline=baseline,
                                seed=seed, samples_per_condition=cap, bootstrap=bootstrap)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    selected_train = np.sort(np.random.default_rng(7).choice(16, 12, replace=False))
    rng = np.random.default_rng(seed)
    selected_val = np.concatenate([
        rng.choice(np.flatnonzero(arrays["val"]["y"] == label), 8, replace=False)
        for label in (0, 4)
    ])
    vx, vy, ids = (arrays["val"][key][selected_val] for key in ("x", "y", "meter_id"))
    tx, ty = (arrays["train"][key][selected_train] for key in ("x", "y"))
    generated = generate(checkpoint, vy, seed=seed)
    reference = (resample_baseline(tx, ty, vy, seed=seed) if baseline == "conditional_resampling"
                 else mean_baseline(tx, ty, vy))
    manual = paired_condition_macro_meter_bootstrap(
        vx, vy, generated, vy, reference, vy, ids,
        interval_hours=1, n_bootstrap=bootstrap, seed=seed,
        real_meter_strata=vy // 8,
    )
    assert report["paired_comparison"] == manual
    assert report["comparator"]["training_selection"]["train_profiles_used"] == 12
    assert report["candidate"]["training_selection"]["train_indices_sha256"] == state["selection"]["train_indices_sha256"]
    assert report["validation_selection"]["selected_indices_sha256"] == hashlib.sha256(selected_val.astype("<i8").tobytes()).hexdigest()
    assert report["training_subsets_match"] is True
    assert report["test_used"] is False and report["split"] == "val"
    assert "test.npz" not in report["prepared_data_hashes"]
    json.dumps(report, allow_nan=False)


def test_checkpoint_comparator_verifies_both_and_discloses_different_training_subsets(experiment, forbid_test_access):
    prepared, candidate, comparator, _ = experiment
    report = comparison.compare(prepared, candidate, comparator=comparator, bootstrap=3)
    assert report["training_subsets_match"] is False
    assert "training-data differences" in report["training_subset_interpretation"]
    assert report["candidate"]["training_selection"]["train_profiles_used"] == 12
    assert report["comparator"]["training_selection"]["train_profiles_used"] == 16
    assert report["candidate"]["data_hashes"] == report["comparator"]["data_hashes"]
    assert report["paired_comparison"]["interval_status"] == "complete"
    same = comparison.compare(prepared, candidate, comparator=candidate, bootstrap=3)
    for metric in same["paired_comparison"]["intervals"].values():
        assert metric == {"estimate": 0., "lower": 0., "upper": 0.}


@pytest.mark.parametrize("which", ["candidate", "comparator"])
def test_rejects_changed_checkpoint_data_binding(experiment, forbid_test_access, tmp_path, which):
    prepared, candidate, comparator, _ = experiment
    source = candidate if which == "candidate" else comparator
    state = torch.load(source, map_location="cpu", weights_only=True)
    state["data_hashes"]["val.npz"] = "0" * 64
    invalid = tmp_path / "invalid.pt"
    torch.save(state, invalid)
    with pytest.raises(ValueError, match="fingerprint.*val.npz"):
        comparison.compare(prepared, invalid if which == "candidate" else candidate,
                           comparator=invalid if which == "comparator" else comparator, bootstrap=2)


def test_rejects_incorrect_training_subset_hash(experiment, forbid_test_access, tmp_path):
    prepared, candidate, _, _ = experiment
    state = torch.load(candidate, map_location="cpu", weights_only=True)
    state["selection"]["train_indices_sha256"] = "0" * 64
    invalid = tmp_path / "invalid.pt"
    torch.save(state, invalid)
    with pytest.raises(ValueError, match="exact training subset"):
        comparison.compare(prepared, invalid, bootstrap=2)


def test_cli_defaults_and_refuses_overwrite(experiment, forbid_test_access, tmp_path):
    prepared, candidate, _, _ = experiment
    output = tmp_path / "comparison.json"
    args = ["--prepared", str(prepared), "--candidate", str(candidate),
            "--output", str(output), "--samples-per-condition", "3", "--bootstrap", "2"]
    comparison.main(args)
    original = output.read_bytes()
    report = json.loads(original)
    assert report["comparator"]["kind"] == "conditional_resampling"
    assert report["seed"] == 2026
    assert report["validation_selection"]["condition_counts"] == {"0": 3, "4": 3}
    with pytest.raises(FileExistsError):
        comparison.main(args)
    assert output.read_bytes() == original
    with pytest.raises(SystemExit):
        comparison.main(args + ["--split", "test"])


def test_rejects_both_comparators_and_invalid_options_before_data_reads():
    with pytest.raises(ValueError, match="either"):
        comparison.compare("missing", "missing", comparator="missing", baseline="conditional_mean")
    with pytest.raises(ValueError, match="bootstrap"):
        comparison.compare("missing", "missing", bootstrap=1)
    with pytest.raises(ValueError, match="seed"):
        comparison.compare("missing", "missing", seed=2 ** 32)
