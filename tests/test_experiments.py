"""Seed aggregation and orchestration without expensive GAN training."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import run_experiments


def write_report(directory, seed, value, **changes):
    directory.mkdir(parents=True, exist_ok=True)
    report = {
        "seed": seed,
        "split": "val",
        "max_per_condition": 200,
        "n_evaluated": 400,
        "dataset_sha256": "dataset-hash",
        "checkpoint_sha256": f"checkpoint-{seed}",
        "classes": ["A", "B"],
        "cgan": {"global": {
            "per_slot_wasserstein": value,
            "autocorrelation_mae_lag_1": None,
            "metadata": {"profile_value_unit": "kWh", "interval_hours": 0.25},
            "warnings": [],
        }},
        "conditional_bootstrap": {"global": {
            "per_slot_wasserstein": value / 2,
            "metadata": {"profile_value_unit": "kWh", "interval_hours": 0.25},
            "warnings": [],
        }},
    }
    report.update(changes)
    path = directory / "report.json"
    path.write_text(json.dumps(report))
    return path


def test_summary_has_known_mean_and_sample_standard_deviation(tmp_path):
    paths = [write_report(tmp_path / str(seed), seed, value)
             for seed, value in [(42, 1.0), (43, 3.0), (44, 5.0)]]
    summary = run_experiments.build_summary(paths, expected_seeds=[42, 43, 44])
    metric = summary["cgan"]["global"]["per_slot_wasserstein"]
    assert metric["mean"] == pytest.approx(3.0)
    assert metric["sample_std"] == pytest.approx(2.0)
    assert metric["n_available"] == 3
    assert summary["conditional_bootstrap"]["global"]["per_slot_wasserstein"]["mean"] == 1.5
    assert summary["n_completed"] == 3
    assert "household" in summary["uncertainty_note"].lower()
    assert summary["cgan"]["global"]["autocorrelation_mae_lag_1"]["mean"] is None
    assert "metadata" not in summary["cgan"]["global"]


def test_summary_rejects_missing_seed_instead_of_aggregating_subset(tmp_path):
    report = write_report(tmp_path / "42", 42, 1.0)
    with pytest.raises(ValueError, match="missing"):
        run_experiments.build_summary([report], expected_seeds=[42, 43])


def test_summary_rejects_nonexistent_report(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_experiments.build_summary([tmp_path / "missing.json"], expected_seeds=[42])


def test_summary_rejects_duplicate_expected_and_completed_seeds(tmp_path):
    report = write_report(tmp_path / "42", 42, 1.0)
    with pytest.raises(ValueError, match="unique"):
        run_experiments.build_summary([report], expected_seeds=[42, 42])
    with pytest.raises(ValueError, match="duplicate"):
        run_experiments.build_summary([report, report], expected_seeds=[42])


def test_summary_rejects_mismatched_data_or_split(tmp_path):
    first = write_report(tmp_path / "42", 42, 1.0)
    second = write_report(tmp_path / "43", 43, 3.0, split="test")
    with pytest.raises(ValueError, match="split"):
        run_experiments.build_summary([first, second], expected_seeds=[42, 43])
    second = write_report(tmp_path / "43", 43, 3.0, dataset_sha256="other")
    with pytest.raises(ValueError, match="dataset_sha256"):
        run_experiments.build_summary([first, second], expected_seeds=[42, 43])


def test_single_seed_standard_deviation_is_undefined(tmp_path):
    report = write_report(tmp_path / "42", 42, 1.0)
    summary = run_experiments.build_summary([report], expected_seeds=[42])
    assert summary["cgan"]["global"]["per_slot_wasserstein"]["sample_std"] is None


def test_unavailable_metric_does_not_silently_drop_a_seed(tmp_path):
    first = write_report(tmp_path / "42", 42, 1.0)
    second = write_report(tmp_path / "43", 43, 3.0)
    data = json.loads(second.read_text())
    data["cgan"]["global"]["per_slot_wasserstein"] = None
    second.write_text(json.dumps(data))
    summary = run_experiments.build_summary([first, second], expected_seeds=[42, 43])
    metric = summary["cgan"]["global"]["per_slot_wasserstein"]
    assert metric["n_available"] == 1
    assert metric["mean"] is None
    assert metric["sample_std"] is None


def test_missing_metric_is_a_schema_error(tmp_path):
    first = write_report(tmp_path / "42", 42, 1.0)
    second = write_report(tmp_path / "43", 43, 3.0)
    data = json.loads(second.read_text())
    del data["cgan"]["global"]["per_slot_wasserstein"]
    second.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="metric"):
        run_experiments.build_summary([first, second], expected_seeds=[42, 43])


def test_nonfinite_metric_is_rejected(tmp_path):
    report = write_report(tmp_path / "42", 42, float("nan"))
    with pytest.raises(ValueError, match="finite"):
        run_experiments.build_summary([report], expected_seeds=[42])


def test_defaults_are_five_seeds_and_validation():
    args = run_experiments.make_parser().parse_args(["--data", "data", "--output", "runs"])
    assert args.seeds == [42, 43, 44, 45, 46]
    assert args.split == "val"
    assert args.max_per_condition == 200


def test_runner_preserves_existing_directory(tmp_path, monkeypatch):
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("unchanged")
    monkeypatch.setattr(run_experiments.subprocess, "run", lambda *a, **k: pytest.fail("must not launch"))
    with pytest.raises(FileExistsError):
        run_experiments.main(["--data", str(tmp_path), "--output", str(output)])
    assert sentinel.read_text() == "unchanged"


def test_runner_matches_training_and_evaluation_seeds(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"epochs": 1, "seed": 99}))
    output = tmp_path / "runs"
    calls = []

    def fake_run(command, check):
        assert check is True
        assert command[:3] == [sys.executable, "-m", "energy_cgan"]
        calls.append(command)
        seed = int(command[command.index("--seed") + 1])
        destination = Path(command[command.index("--output") + 1])
        if command[3] == "train":
            destination.mkdir()
            (destination / "best.pt").write_bytes(b"test-checkpoint")
        else:
            assert command[3] == "evaluate"
            assert command[command.index("--split") + 1] == "val"
            write_report(destination.parent, seed, float(seed - 40))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(run_experiments.subprocess, "run", fake_run)
    run_experiments.main([
        "--data", str(data), "--output", str(output), "--config", str(config),
        "--seeds", "42", "43",
    ])
    assert [call[3] for call in calls] == ["train", "evaluate", "train", "evaluate"]
    assert [call[call.index("--seed") + 1] for call in calls] == ["42", "42", "43", "43"]
    summary = json.loads((output / "summary.json").read_text())
    assert summary["n_completed"] == 2
    assert json.loads(config.read_text())["seed"] == 99


def test_runner_failure_never_creates_a_partial_summary(tmp_path, monkeypatch):
    config = tmp_path / "config.json"
    config.write_text("{}")
    output = tmp_path / "runs"

    def fail(command, check):
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(run_experiments.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        run_experiments.main([
            "--data", str(tmp_path), "--output", str(output), "--config", str(config),
        ])
    assert not (output / "summary.json").exists()
