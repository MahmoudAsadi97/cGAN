"""Offline command contracts, including physical output and reporting provenance."""
import csv
import hashlib
import json
import subprocess
import sys

import numpy as np
import pytest


@pytest.fixture
def prepared(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    rng = np.random.default_rng(71)
    metadata = {"resolution": 24, "interval_hours": 1.0, "condition_dim": 64, "splits": {}}
    for split, count in [("train", 8), ("val", 4), ("test", 4)]:
        x = rng.uniform(-2, 5, (count, 24)).astype(np.float32)
        path = data / f"{split}.npz"
        np.savez(path, x=x, y=np.arange(count, dtype=np.int64) % 2,
                 meter_id=np.array([f"{split}_{i}" for i in range(count)]),
                 date=np.full(count, "2024-01-01"))
        metadata["splits"][split] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        if split == "train":
            scaler = {"minimum": float(x.min()), "maximum": float(x.max()), "fitted_on": "train"}
    (data / "metadata.json").write_text(json.dumps(metadata))
    (data / "scaler.json").write_text(json.dumps(scaler))
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"epochs": 1, "batch_size": 8, "num_threads": 1,
                                  "validation_per_condition": 2, "validation_every": 1}))
    return data, config


def test_subprocess_help_and_phase2_excluded():
    result = subprocess.run([sys.executable, "-m", "fluvius_cgan", "--help"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    commands = {"download", "prepare", "curate", "hourly", "train", "generate", "evaluate", "freeze", "study", "summarize"}
    assert all(command in result.stdout for command in commands)
    assert all(word not in result.stdout.lower() for word in ["federated", "calibrate", "phase2"])
    bad = subprocess.run([sys.executable, "-m", "fluvius_cgan", "phase2"], capture_output=True, text=True)
    assert bad.returncode == 2


def test_offline_train_generate_evaluate_freeze_and_summary(prepared, tmp_path):
    from fluvius_cgan.cli import main
    from fluvius_cgan.training import generate
    data, config = prepared
    run, output, report = tmp_path / "run", tmp_path / "samples.csv", tmp_path / "report.json"
    assert main(["train", "--data", str(data), "--run", str(run), "--config", str(config)]) == 0
    arguments = ["generate", "--checkpoint", str(run / "best.pt"), "--condition", "1", "--count", "3", "--seed", "31001", "--output", str(output)]
    assert main(arguments) == 0
    with output.open() as handle:
        reader = csv.reader(handle)
        assert next(reader) == [f"hour_{hour:02d}" for hour in range(24)]
        values = np.array(list(reader), dtype=float)
    np.testing.assert_allclose(values, generate(run / "best.pt", np.ones(3, dtype=np.int64), 31001))
    provenance = json.loads(output.with_suffix(".json").read_text())
    assert provenance["condition"] == 1 and provenance["seed"] == 31001
    assert provenance["profile_units"] == "kWh per hourly interval"
    assert provenance["checkpoint_sha256"] == hashlib.sha256((run / "best.pt").read_bytes()).hexdigest()
    before = output.read_bytes(), output.with_suffix(".json").read_bytes()
    assert main(arguments) == 2
    assert before == (output.read_bytes(), output.with_suffix(".json").read_bytes())
    assert main(["evaluate", "--data", str(data), "--method", str(run / "best.pt"), "--output", str(report)]) == 0
    saved = json.loads(report.read_text())
    assert saved["split"] == "val" and saved["generation_seeds"] == [31001, 31002, 31003]
    assert not saved["test_performance_access"]
    assert "NaN" not in report.read_text()
    summary = tmp_path / "summary.json"
    assert main(["summarize", "--reports", str(report), "--output", str(summary)]) == 0
    combined = json.loads(summary.read_text())
    assert combined["reports"][0]["training_seed"] == 11
    assert combined["reports"][0]["monte_carlo"]["generation_seeds"] == saved["generation_seeds"]
    assert combined["reports"][0]["monte_carlo"]["macro_summary"] == saved["macro_summary"]
    assert combined["training_seed_summary"][0]["training_seeds"] == [11]
    assert main(["evaluate", "--data", str(data), "--method", "mean", "--split", "test", "--output", str(tmp_path / "blocked.json")]) == 2
    assert not (tmp_path / "blocked.json").exists()
    protocol = tmp_path / "protocol.json"
    assert main(["freeze", "--data", str(data), "--config", str(config), "--output", str(protocol)]) == 0
    assert json.loads(protocol.read_text())["frozen"] is True
    assert main(["freeze", "--data", str(data), "--config", str(config), "--output", str(protocol)]) == 2
    assert main(["evaluate", "--data", str(data), "--method", "mean", "--split", "test", "--protocol", str(protocol), "--output", str(tmp_path / "test-report.json")]) == 0


@pytest.mark.parametrize("kind", ["data", "config", "malformed"])
def test_actionable_input_failures_are_nonzero(prepared, tmp_path, capsys, kind):
    from fluvius_cgan.cli import main
    data, config = prepared
    if kind == "data":
        data = tmp_path / "missing-data"
    elif kind == "config":
        config = tmp_path / "missing-config.json"
    else:
        config.write_text("{broken")
    assert main(["train", "--data", str(data), "--run", str(tmp_path / "run"), "--config", str(config)]) == 2
    assert "error:" in capsys.readouterr().err


def test_prepare_and_other_data_commands_use_source_resolution(monkeypatch, tmp_path):
    from fluvius_cgan.cli import main
    import fluvius_cgan.data
    import fluvius_cgan.curation
    import fluvius_cgan.prepared
    import fluvius_cgan.release
    calls = []
    monkeypatch.setattr(fluvius_cgan.data, "prepare_archive", lambda *a, **k: calls.append((a, k)))
    monkeypatch.setattr(fluvius_cgan.curation, "curate_prepared", lambda *a: calls.append((a, {})))
    monkeypatch.setattr(fluvius_cgan.prepared, "aggregate_hourly", lambda *a: calls.append((a, {})))
    monkeypatch.setattr(fluvius_cgan.release, "download_release", lambda *a: calls.append((a, {})))
    assert main(["prepare", "--archive", "raw.zip", "--output", "source", "--chunksize", "123"]) == 0
    assert calls[-1][1] == {"resolution": 96, "seed": 42, "chunksize": 123}
    for command in ["curate", "hourly"]:
        assert main([command, "--data", "source", "--output", "destination"]) == 0
    assert main(["download", "--output", "raw.zip"]) == 0
    assert len(calls) == 4


def test_study_trains_both_arms_sequentially_without_test_access(prepared, tmp_path, monkeypatch):
    from fluvius_cgan.cli import main
    import fluvius_cgan.training
    data, config = prepared
    calls = []
    def fake_train(data_dir, run_dir, configuration, resume=False):
        calls.append((configuration["seed"], configuration["unconditional"]))
        return {"status": "completed", "config": configuration}
    monkeypatch.setattr(fluvius_cgan.training, "train", fake_train)
    assert main(["study", "--data", str(data), "--config", str(config), "--runs", str(tmp_path / "runs")]) == 0
    assert calls == [(seed, arm) for seed in [11, 22, 33, 44, 55] for arm in [False, True]]
    manifest = json.loads((tmp_path / "runs/study_manifest.json").read_text())
    assert len(manifest["runs"]) == 10
    assert not manifest["test_performance_access"]
    assert not list(tmp_path.rglob("test_access.jsonl"))


def test_programming_errors_keep_tracebacks(monkeypatch):
    from fluvius_cgan.cli import main
    import fluvius_cgan.release
    def bug(*args):
        raise TypeError("programming bug")
    monkeypatch.setattr(fluvius_cgan.release, "download_release", bug)
    with pytest.raises(TypeError, match="programming bug"):
        main(["download", "--output", "unused.zip"])


def test_summary_separates_training_variation_from_sampling_noise(tmp_path):
    from fluvius_cgan.cli import main
    paths = []
    for seed, mean in [(11, 10.0), (22, 14.0)]:
        path = tmp_path / f"report-{seed}.json"
        path.write_text(json.dumps({"method": "conditional", "split": "val",
                                   "model": {"config": {"seed": seed, "epochs": 1}},
                                   "generation_seeds": [31001, 31002, 31003],
                                   "macro_summary": {"profile_sw1_kwh": {"mean": mean, "sampling_sd": 100.0}}}))
        paths.append(str(path))
    output = tmp_path / "summary.json"
    assert main(["summarize", "--reports", *paths, "--output", str(output)]) == 0
    metric = json.loads(output.read_text())["training_seed_summary"][0]["metrics"]["profile_sw1_kwh"]
    assert metric == {"mean": 12.0, "training_seed_sd": pytest.approx(8**0.5), "valid_training_seeds": 2}
    before = output.read_bytes()
    assert main(["summarize", "--reports", *paths, "--output", str(output)]) == 2
    assert output.read_bytes() == before


def test_study_preserves_failures_and_does_not_replace_seeds(prepared, tmp_path, monkeypatch):
    from fluvius_cgan.cli import main
    import fluvius_cgan.training
    data, config = prepared
    calls = []
    def failed_train(data_dir, run_dir, configuration, resume=False):
        calls.append((configuration["seed"], configuration["unconditional"]))
        if configuration["seed"] == 11:
            raise FloatingPointError("Nonfinite generator loss")
        return {"status": "completed"}
    monkeypatch.setattr(fluvius_cgan.training, "train", failed_train)
    args = ["study", "--data", str(data), "--config", str(config), "--runs", str(tmp_path / "runs"), "--seeds", "11", "22"]
    assert main(args) == 2
    assert calls == [(11, False), (11, True), (22, False), (22, True)]
    saved = json.loads((tmp_path / "runs/study_manifest.json").read_text())
    assert [run["status"] for run in saved["runs"]] == ["failed", "failed", "completed", "completed"]
    assert all("Nonfinite" in run["error"] for run in saved["runs"][:2])
    assert main(args) == 2
    assert len(calls) == 4


def test_existing_sidecar_is_preserved_without_creating_csv(tmp_path):
    from fluvius_cgan.cli import main
    sidecar = tmp_path / "samples.json"
    sidecar.write_text("original provenance")
    assert main(["generate", "--checkpoint", "missing.pt", "--condition", "0", "--count", "1", "--seed", "1", "--output", str(tmp_path / "samples.csv")]) == 2
    assert sidecar.read_text() == "original provenance"
    assert not (tmp_path / "samples.csv").exists()


def test_malformed_report_is_an_actionable_failure(tmp_path, capsys):
    from fluvius_cgan.cli import main
    report = tmp_path / "bad.json"
    report.write_text(json.dumps({"method": "conditional", "split": "val", "generation_seeds": [1],
                                  "macro_summary": {}, "model": "invalid"}))
    assert main(["summarize", "--reports", str(report), "--output", str(tmp_path / "summary.json")]) == 2
    assert "model" in capsys.readouterr().err
