"""Tiny verified artifacts exercise bundling without touching active experiments."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import zipfile

import pytest


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


@pytest.fixture
def bundler(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/archive_phase1.py"
    spec = importlib.util.spec_from_file_location("phase1_archive", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout="b" * 40 + "\n"))
    return module


@pytest.fixture
def artifacts(tmp_path):
    runs, evaluations, figures = [tmp_path / name for name in ("runs", "evaluations", "figures")]
    for root in (runs, evaluations, figures):
        root.mkdir()
    protocol = {"phase": 1, "frozen": True, "training_seeds": [11, 22, 33, 44, 55],
                "data_hashes": {"train.npz": "not-read"}}
    write_json(evaluations / "protocol.snapshot.json", protocol)
    protocol_hash = digest(evaluations / "protocol.snapshot.json")
    jobs, checkpoints, manifests = [], {}, {}
    names = []
    for arm in ("conditional", "unconditional"):
        for seed in protocol["training_seeds"]:
            name = f"{arm}_seed_{seed}"
            names.append(name)
            directory = runs / arm / f"seed_{seed}"
            directory.mkdir(parents=True)
            config = {"epochs": 1, "seed": seed, "unconditional": arm == "unconditional"}
            for filename in ("best.pt", "last.pt", "history.json"):
                (directory / filename).write_bytes(filename.encode())
            manifest = {"status": "completed", "completed_epochs": 1, "config": config,
                        "artifact_hashes": {filename: digest(directory / filename) for filename in ("best.pt", "last.pt", "history.json")}}
            write_json(directory / "manifest.json", manifest)
            config_path = runs / "configs" / f"{name}.json"
            write_json(config_path, config)
            log = runs / "logs" / f"{name}.log"
            log.parent.mkdir(exist_ok=True)
            log.write_text("training completed\n")
            jobs.append({"name": name, "method": arm, "training_seed": seed, "status": "completed", "exit_code": 0,
                         "config": config, "config_path": str(config_path), "log": str(log)})
            checkpoints[name] = {"path": str(directory / "best.pt"), "sha256": digest(directory / "best.pt")}
            manifests[name] = {"path": str(directory / "manifest.json"), "sha256": digest(directory / "manifest.json")}
    write_json(runs / "manifest.json", {"status": "completed", "jobs": jobs, "git": {"head": "a" * 40}, "protocol_sha256": protocol_hash})
    write_json(evaluations / "execution_plan.json", {"provenance": {"protocol": {"sha256": protocol_hash}, "checkpoints": checkpoints, "run_manifests": manifests}})
    records = []
    names += ["mean", "resample", "copula"]
    for name in names:
        report = evaluations / "reports" / f"{name}.json"
        write_json(report, {"method": name.split("_seed_")[0], "split": "test", "protocol_sha256": protocol_hash,
                            "checkpoint_sha256": checkpoints.get(name, {}).get("sha256")})
        records.append({"name": name, "returncode": 0, "report_sha256": digest(report)})
    write_json(evaluations / "evaluation_execution.json", {"jobs": records})
    for name in [*names, "summarize", "inference"]:
        for stream in ("stdout", "stderr"):
            path = evaluations / "logs" / f"{name}.{stream}.log"
            path.parent.mkdir(exist_ok=True)
            path.write_text("mock log\n")
    for name in ("summary", "inference"):
        write_json(evaluations / f"{name}.json", {"result": name})
        write_json(evaluations / f"{name}_execution.json", {"returncode": 0})
    write_json(evaluations / "completed.json", {"protocol_sha256": protocol_hash,
                "summary_sha256": digest(evaluations / "summary.json"), "inference_sha256": digest(evaluations / "inference.json")})
    (evaluations / "test_access.jsonl").write_text("{}\n")
    (evaluations / "reports/test_access.jsonl").write_text("{}\n")
    for name in ("primary.png", "primary.pdf", "metrics.csv"):
        (figures / name).write_bytes(name.encode())
    write_json(figures / "render_manifest.json", {"schema_version": 1,
               "source_json_sha256": {str(evaluations / "inference.json"): digest(evaluations / "inference.json")},
               "output_sha256": {name: digest(figures / name) for name in ("primary.png", "primary.pdf", "metrics.csv")}})
    return runs, evaluations, figures, tmp_path / "bundle.zip"


def arguments(artifacts):
    runs, evaluations, figures, output = artifacts
    return ["--runs", str(runs), "--evaluations", str(evaluations), "--figures", str(figures), "--output", str(output)]


def test_verified_inventory_crc_and_explicit_data_exclusion(bundler, artifacts):
    runs, evaluations, figures, output = artifacts
    (runs / "raw.zip").write_bytes(b"forbidden")
    (evaluations / "train.npz").write_bytes(b"forbidden")
    (figures / "unrelated.json").write_text("do not include")
    assert bundler.main(arguments(artifacts)) == 0
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        inventory = json.loads(archive.read("bundle_manifest.json"))
        assert inventory["git_head"] == "b" * 40
        assert inventory["training_source_commit"] == "a" * 40
        assert set(archive.namelist()) == set(inventory["members"]) | {"bundle_manifest.json"}
        assert not any(name.endswith((".npz", ".zip")) or "unrelated" in name for name in archive.namelist())
        assert sum(name.endswith("best.pt") for name in archive.namelist()) == 10
        assert sum(name.endswith("last.pt") for name in archive.namelist()) == 10
        for name, metadata in inventory["members"].items():
            content = archive.read(name)
            assert metadata == {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
    before = output.read_bytes()
    assert bundler.main(arguments(artifacts)) == 2
    assert output.read_bytes() == before


@pytest.mark.parametrize("problem", ["launch", "run", "evaluation", "summary", "inference", "report", "checkpoint", "figure", "external_figure_source"])
def test_incomplete_or_mutated_artifacts_are_rejected(bundler, artifacts, problem, tmp_path):
    runs, evaluations, figures, output = artifacts
    if problem in {"launch", "run"}:
        path = runs / ("manifest.json" if problem == "launch" else "conditional/seed_11/manifest.json")
        value = json.loads(path.read_text())
        value["status"] = "running"
        write_json(path, value)
    elif problem == "evaluation":
        (evaluations / "completed.json").unlink()
    elif problem == "external_figure_source":
        manifest = json.loads((figures / "render_manifest.json").read_text())
        manifest["source_json_sha256"] = {str(tmp_path / "outside.json"): "not-read"}
        write_json(figures / "render_manifest.json", manifest)
    else:
        path = {"summary": evaluations / "summary.json", "inference": evaluations / "inference.json",
                "report": evaluations / "reports/mean.json", "checkpoint": runs / "conditional/seed_11/best.pt",
                "figure": figures / "primary.png"}[problem]
        path.write_bytes(b"changed")
    assert bundler.main(arguments(artifacts)) == 2
    assert not output.exists()
    assert not list(output.parent.glob(".bundle.zip-*.tmp"))


@pytest.mark.parametrize("kind", ["file", "directory", "root"])
def test_source_symlinks_are_rejected(bundler, artifacts, tmp_path, kind):
    runs, evaluations, figures, output = artifacts
    if kind == "file":
        path = runs / "conditional/seed_11/best.pt"
        outside = tmp_path / "outside.pt"
        outside.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(outside)
    elif kind == "directory":
        directory = runs / "conditional"
        outside = tmp_path / "outside"
        directory.rename(outside)
        directory.symlink_to(outside, target_is_directory=True)
    else:
        link = tmp_path / "linked-runs"
        link.symlink_to(runs, target_is_directory=True)
        artifacts = link, evaluations, figures, output
    assert bundler.main(arguments(artifacts)) == 2
    assert not output.exists()


def test_failure_during_writing_cleans_temporary_archive(bundler, artifacts, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("simulated write failure")
    monkeypatch.setattr(bundler.zipfile.ZipFile, "open", fail)
    assert bundler.main(arguments(artifacts)) == 2
    assert not artifacts[-1].exists()
    assert not list(artifacts[-1].parent.glob(".bundle.zip-*.tmp"))


def test_missing_training_commit_is_rejected(bundler, artifacts):
    path = artifacts[0] / "manifest.json"
    launch = json.loads(path.read_text())
    launch.pop("git")
    write_json(path, launch)
    assert bundler.main(arguments(artifacts)) == 2
    assert not artifacts[-1].exists()


def test_change_after_inventory_is_rejected_and_cleaned(bundler, artifacts, monkeypatch):
    def change_after_inventory(*args, **kwargs):
        (artifacts[0] / "logs/conditional_seed_11.log").write_text("changed after inventory")
        return subprocess.CompletedProcess(args, 0, stdout="b" * 40 + "\n")
    monkeypatch.setattr(bundler.subprocess, "run", change_after_inventory)
    assert bundler.main(arguments(artifacts)) == 2
    assert not artifacts[-1].exists()
    assert not list(artifacts[-1].parent.glob(".bundle.zip-*.tmp"))
