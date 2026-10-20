"""JSON extraction/binding contracts; no figure styling or model/data execution."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "plot_training_histories", Path(__file__).parents[1] / "scripts/plot_training_histories.py")
plotter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plotter)


def fixture():
    report = {"schema_version": 1, "kind": "completed_reference_validation_study", "status": "complete",
              "split": "val", "test_used": False, "candidate_kind": "swd10",
              "training_seeds": list(plotter.SEEDS), "runs": {}, "artifacts": {}}
    documents = {}
    for arm in plotter.ARMS:
        report["runs"][arm] = {}
        for seed in plotter.SEEDS:
            width = 24 if arm == "bce24" else 96
            normalization = "standard" if arm == "candidate96" else "minmax"
            history = [{"epoch": epoch, "validation_score": abs(epoch - 8) / 100 + .01}
                       for epoch in range(1, 27)]
            selection = {"validation_selection": "conditional_w1", "validation_samples_requested": 4096}
            provenance = {"git_commit": "a" * 40}
            summary = {"test_used": False, "stopped_early": False, "validation_metric": plotter.METRIC,
                       "resolution": width, "target": "net", "units": "kWh/interval", "epochs_completed": 26,
                       "generator_updates": 30784, "discriminator_updates": 30784,
                       "scaler": {"kind": normalization}, "selection": selection, "provenance": provenance,
                       "best_epoch": 8, "best_validation_score": .01, "last_epoch_metrics": history[-1]}
            artifacts = {}
            for name, content in (("history.json", history), ("summary.json", summary)):
                path = f"{arm}/{seed}/{name}"
                raw = json.dumps(content).encode()
                record = {"path": path, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
                artifacts[name] = report["artifacts"][path] = record
                documents[path] = content
            checkpoint_path = f"{arm}/{seed}/best.pt"
            artifacts["best.pt"] = report["artifacts"][checkpoint_path] = {
                "path": checkpoint_path, "sha256": "b" * 64, "bytes": 100}
            report["runs"][arm][str(seed)] = {
                "arm": arm, "training_seed": seed, "width": width, "epochs_completed": 26,
                "generator_updates": 30784, "discriminator_updates": 30784, "best_epoch": 8,
                "checkpoint_sha256": "b" * 64, "training_selection": selection, "provenance": provenance,
                "config": {"seed": seed, "normalization": normalization, "validation_selection": "conditional_w1",
                           "validation_samples": 4096, "epochs": 26}, "artifacts": artifacts}
    return report, documents


def extract(report, documents):
    return plotter.extract(report, lambda record, name: documents[record["path"]])


def test_extract_retains_every_seed_and_epoch():
    report, documents = fixture()
    panels = extract(report, documents)
    assert tuple(panels) == plotter.ARMS
    for panel in panels.values():
        assert [trace["seed"] for trace in panel["traces"]] == list(plotter.SEEDS)
        assert all(trace["epochs"] == list(range(1, 27)) and trace["best_epoch"] == 8
                   for trace in panel["traces"])


@pytest.mark.parametrize("omission", ["seed", "arm"])
def test_missing_coverage_rejected_before_read(omission):
    report, _ = fixture()
    if omission == "seed":
        del report["runs"]["bce24"]["55"]
    else:
        del report["runs"]["bce24"]
    with pytest.raises(ValueError):
        plotter.extract(report, lambda *args: pytest.fail("must reject coverage before reading artifacts"))


@pytest.mark.parametrize("mutation", ["gap", "late_tie", "summary", "test", "metric", "normalization", "binding", "nonfinite"])
def test_invalid_history_contract_rejected(mutation):
    report, documents = fixture()
    run = report["runs"]["candidate96"]["11"]
    history = documents[run["artifacts"]["history.json"]["path"]]
    summary = documents[run["artifacts"]["summary.json"]["path"]]
    if mutation == "gap":
        history[1]["epoch"] = 1
    elif mutation == "late_tie":
        history[6]["validation_score"] = .01
    elif mutation == "summary":
        summary["best_validation_score"] = .02
    elif mutation == "test":
        summary["test_used"] = True
    elif mutation == "metric":
        summary["validation_metric"] = "physical_w1"
    elif mutation == "normalization":
        summary["scaler"]["kind"] = "minmax"
    elif mutation == "nonfinite":
        history[0]["validation_score"] = float("inf")
    else:
        run["artifacts"]["history.json"] = deepcopy(run["artifacts"]["history.json"])
        run["artifacts"]["history.json"]["sha256"] = "c" * 64
    with pytest.raises(ValueError):
        extract(report, documents)


def write_fixture(tmp_path):
    report, documents = fixture()
    for path, content in documents.items():
        destination = tmp_path / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(content))
    report_path = tmp_path / "collector.json"
    report_path.write_text(json.dumps(report))
    return report_path


def test_file_hash_binding_rejects_changed_history(tmp_path):
    report_path = write_fixture(tmp_path)
    plotter.load_histories(report_path)
    path = tmp_path / "candidate96/11/history.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="bound JSON content changed"):
        plotter.load_histories(report_path)


def test_profile_symlink_rejected_before_content_read(tmp_path, monkeypatch):
    report_path = write_fixture(tmp_path)
    forbidden = tmp_path / "test.npz"
    forbidden.write_bytes(b"reserved synthetic content")
    path = tmp_path / "candidate96/11/history.json"
    path.unlink()
    path.symlink_to(forbidden)
    original = Path.read_bytes

    def guarded(self):
        if self.resolve() == forbidden:
            pytest.fail("reserved profile must not be opened")
        return original(self)

    monkeypatch.setattr(Path, "read_bytes", guarded)
    with pytest.raises(ValueError, match="non-JSON"):
        plotter.load_histories(report_path)
