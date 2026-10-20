"""The parallel experiment runner is tested without starting real training."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import threading
import time

import pytest


@pytest.fixture
def runner():
    path = Path(__file__).resolve().parents[1] / "scripts/run_phase1.py"
    spec = importlib.util.spec_from_file_location("phase1_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def inputs(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    hashes = {}
    for name in ["train.npz", "val.npz", "test.npz", "scaler.json", "metadata.json"]:
        path = data / name
        path.write_bytes(name.encode())
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({"phase": 1, "frozen": True, "training_config": {"epochs": 2},
                                    "training_seeds": [11, 22, 33],
                                    "methods": ["conditional", "unconditional", "mean"], "data_hashes": hashes}))
    return data, protocol, tmp_path / "runs"


def fake_git(command):
    return subprocess.CompletedProcess(command, 0, stdout="a" * 40 + "\n" if "rev-parse" in command else " M src/fluvius_cgan/training.py\n", stderr="")


def test_bounded_concurrent_jobs_configs_provenance_and_failure_preservation(runner, inputs, monkeypatch):
    data, protocol, runs = inputs
    calls, state = [], {"active": 0, "maximum": 0}
    lock = threading.Lock()
    def fake_run(command, **kwargs):
        if command[0] == "git":
            return fake_git(command)
        config_path = Path(command[command.index("--config") + 1])
        config = json.loads(config_path.read_text())
        with lock:
            calls.append((command, kwargs, config))
            state["active"] += 1
            state["maximum"] = max(state["maximum"], state["active"])
        time.sleep(.03)
        kwargs["stdout"].write("mock training log\n")
        with lock:
            state["active"] -= 1
        return subprocess.CompletedProcess(command, 7 if config["seed"] == 11 and not config["unconditional"] else 0)
    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    assert runner.main(["--data", str(data), "--protocol", str(protocol), "--runs", str(runs), "--python", "/custom/python"]) == 1
    assert len(calls) == 6
    assert 1 < state["maximum"] <= 3
    for command, kwargs, config in calls:
        assert command[:4] == ["/custom/python", "-m", "fluvius_cgan", "train"]
        assert config["epochs"] == 2
        assert all(kwargs["env"][key] == "1" for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"])
        assert "--resume" not in command
    manifest = json.loads((runs / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["workers"] == 3
    assert manifest["protocol_sha256"] == hashlib.sha256(protocol.read_bytes()).hexdigest()
    assert manifest["git"]["head"] == "a" * 40
    assert manifest["git"]["dirty_files"]
    assert "src/fluvius_cgan/training.py" in manifest["source_py_sha256"]
    assert len(manifest["jobs"]) == 6
    assert [job["exit_code"] for job in manifest["jobs"]].count(7) == 1
    for job in manifest["jobs"]:
        assert job["started_at_utc"] and job["ended_at_utc"]
        assert Path(job["log"]).read_text() == "mock training log\n"
        assert json.loads(Path(job["config_path"]).read_text()) == job["config"]
    before = (runs / "manifest.json").read_bytes()
    assert runner.main(["--data", str(data), "--protocol", str(protocol), "--runs", str(runs)]) == 2
    assert (runs / "manifest.json").read_bytes() == before
    assert len(calls) == 6


@pytest.mark.parametrize("problem", ["hash", "missing_arm", "workers_zero", "workers_high", "seeds"])
def test_preflight_rejects_invalid_inputs_before_launch(runner, inputs, monkeypatch, problem):
    data, protocol, runs = inputs
    value = json.loads(protocol.read_text())
    arguments = ["--data", str(data), "--protocol", str(protocol), "--runs", str(runs)]
    if problem == "hash":
        (data / "train.npz").write_bytes(b"changed")
    elif problem == "missing_arm":
        value["methods"] = ["conditional"]
    elif problem == "seeds":
        value["training_seeds"] = [11, 11]
    else:
        arguments += ["--workers", "0" if problem == "workers_zero" else "5"]
    protocol.write_text(json.dumps(value))
    def unexpected(*args, **kwargs):
        raise AssertionError("No process may start before preflight succeeds")
    monkeypatch.setattr(runner.subprocess, "run", unexpected)
    assert runner.main(arguments) == 2
    assert not runs.exists()


def test_all_success_and_process_start_failures(runner, inputs, monkeypatch):
    data, protocol, runs = inputs
    launches = []
    def fake_run(command, **kwargs):
        if command[0] == "git":
            return fake_git(command)
        launches.append(command)
        if len(launches) == 1:
            raise FileNotFoundError("interpreter missing")
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    arguments = ["--data", str(data), "--protocol", str(protocol), "--runs", str(runs), "--workers", "1"]
    assert runner.main(arguments) == 1
    saved = json.loads((runs / "manifest.json").read_text())
    assert len(launches) == 6 and sum(job["status"] == "failed" for job in saved["jobs"]) == 1
    assert "interpreter missing" in saved["jobs"][0]["error"]
    assert runner.main([*arguments[:5], str(runs.parent / "success"), "--workers", "1"]) == 0
