"""Controller guards and real OS lock lifetimes, with no model/data execution."""
import copy
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location("decoder_controller_test", Path(__file__).resolve().parents[1] / "scripts/run_quantile_decoder_pilot.py")
controller = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(controller)


def gate_value():
    return {"schema_version": 1, "test_used": False,
        "status": "completed_current_reference_and_capacity_training_phases",
        "completed_runs": [{"arm": arm, "seed": seed, "epochs": 26, "updates_per_adversary": 30784,
            "artifacts_sha256": {name: "a" * 64 for name in
                ("summary.json", "history.json", "best.pt", "last.pt", "validation_report.json")}}
            for arm in sorted(controller.REFERENCE_ARMS) for seed in (11, 22, 33, 44, 55)]}


def test_exact_copy_retains_declaration_bytes_and_refuses_overwrite(tmp_path):
    original, saved = tmp_path / "original.json", tmp_path / "saved.json"
    original.write_bytes(b'{"z":1,  "a":2}\n\n')
    controller.copy_exact_new(original, saved, controller.sha(original))
    assert saved.read_bytes() == original.read_bytes()
    with pytest.raises(FileExistsError):
        controller.copy_exact_new(original, saved, controller.sha(original))
    with pytest.raises(ValueError, match="changed"):
        controller.copy_exact_new(original, tmp_path / "bad.json", "f" * 64)
    assert not (tmp_path / "bad.json").exists()


@pytest.mark.parametrize("change", ["missing", "duplicate", "budget", "hash"])
def test_resource_receipt_requires_complete_reference_inventory(tmp_path, change):
    value = gate_value()
    if change == "missing":
        value["completed_runs"].pop()
    elif change == "duplicate":
        value["completed_runs"].append(copy.deepcopy(value["completed_runs"][0]))
    elif change == "budget":
        value["completed_runs"][0]["epochs"] = 25
    else:
        value["completed_runs"][0]["artifacts_sha256"]["best.pt"] = "not-a-digest"
    path = tmp_path / "gate.json"
    controller.dump_new(path, value)
    with pytest.raises(ValueError):
        controller.verify_resource_gate(path, controller.sha(path))


def test_all_five_inputs_are_bound_and_reserved_alias_rejected(tmp_path):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    for name in controller.DATA_FILES:
        (prepared / name).write_bytes(name.encode())
    declaration = {"checkpoint_data_hashes": {name: controller.sha(prepared / name) for name in controller.DATA_FILES}}
    assert controller.verify_data(prepared, declaration) == declaration["checkpoint_data_hashes"]
    (prepared / "curation_manifest.json").write_text("changed")
    with pytest.raises(ValueError, match="manifest changed"):
        controller.verify_data(prepared, declaration)
    sentinel = prepared / "test.npz"
    sentinel.write_bytes(b"synthetic reserved data")
    (prepared / "train.npz").unlink()
    os.link(sentinel, prepared / "train.npz")
    with pytest.raises(ValueError, match="TEST alias"):
        controller.verify_data(prepared, declaration)


def test_resource_decision_keeps_declared_formula_and_eight_gib_ceiling():
    # 3.25GiB each plus1.5GiB exactly fills8GiB.
    assert controller.remaining_parallelism(int(3.25 * 2**20), 8 * 2**30)[0] == 2
    assert controller.remaining_parallelism(int(3.25 * 2**20) + 1, 16 * 2**30)[0] == 1
    assert controller.remaining_parallelism(int(3 * 2**20), 7 * 2**30)[0] == 1
    with pytest.raises(ValueError):
        controller.remaining_parallelism(True, 8 * 2**30)


def test_child_retains_exclusive_lock_after_controller_descriptor_closes(tmp_path):
    lock_path, ready, release = (tmp_path / name for name in ("lock", "ready", "release"))
    code = "import os,pathlib,sys,time;os.fstat(int(sys.argv[1]));pathlib.Path(sys.argv[2]).write_text('ready');\nwhile not pathlib.Path(sys.argv[3]).exists(): time.sleep(.01)"
    lock = lock_path.open("a")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
    with (tmp_path / "child.log").open("w") as log:
        process = controller.spawn_locked([sys.executable, "-c", code, str(lock.fileno()), str(ready), str(release)],
                                          lock=lock, cwd=tmp_path, env=os.environ.copy(), log=log)
        try:
            lock.close()
            # The inherited FD exists immediately after Popen returns; no timing
            # assumption or trained-model subprocess is needed for this check.
            with lock_path.open("a") as contender:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(contender.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                release.write_text("done")
                assert process.wait(timeout=5) == 0
                fcntl.flock(contender.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
            lock.close()


def test_cleanup_stops_child_before_lock_release_and_retains_exit(tmp_path):
    with (tmp_path / "lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        log = (tmp_path / "child.log").open("w")
        process = controller.spawn_locked([sys.executable, "-c", "import time;time.sleep(60)"],
                                          lock=lock, cwd=tmp_path, env=os.environ.copy(), log=log)
        active = {22: {"process": process, "log": log}}
        records = controller.stop_children(active)
        assert not active and process.poll() is not None and log.closed
        assert records == [{"seed": 22, "pid": process.pid, "returncode": process.returncode, "cleanup": True}]
        with (tmp_path / "lock").open("a") as contender:
            with pytest.raises(BlockingIOError):
                fcntl.flock(contender.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_signals_become_cleanup_exceptions_and_prior_handlers_restore():
    before = signal.getsignal(signal.SIGTERM)
    with pytest.raises(InterruptedError, match="signal"):
        with controller.interrupted_as_exception():
            signal.raise_signal(signal.SIGTERM)
    assert signal.getsignal(signal.SIGTERM) == before


def test_pinned_assessor_validator_runs_before_any_training(tmp_path):
    source, prepared = tmp_path / "source", tmp_path / "prepared"
    (source / "scripts").mkdir(parents=True)
    prepared.mkdir()
    (source / controller.SCRIPT).write_bytes(Path(controller.__file__).read_bytes())
    assessor = source / "scripts/assess_quantile_decoder_pilot.py"
    assessor.write_text("def load_declaration(path):\n    raise ValueError('pinned strict validator invoked')\n")
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=source, text=True).strip()
    git("init", "-q")
    git("add", "scripts")
    git("-c", "user.name=Synthetic", "-c", "user.email=synthetic@example.invalid", "commit", "-qm", "fixture")
    declaration = {"source_commit": git("rev-parse", "HEAD"), "source_tree": git("rev-parse", "HEAD^{tree}"),
        "source_files_sha256": {name: controller.sha(source / name) for name in
            (controller.SCRIPT, "scripts/assess_quantile_decoder_pilot.py")}}
    path = tmp_path / "declaration.json"
    controller.dump_new(path, declaration)
    args = SimpleNamespace(source=source, prepared=prepared, declaration=path,
        expected_declaration_sha256=controller.sha(path), config=tmp_path / "config.json",
        resource_gate=tmp_path / "gate.json", lock=tmp_path / "lock")
    with pytest.raises(ValueError, match="pinned strict validator invoked"):
        controller.verify_declaration(args)
    declaration["source_tree"] = "0" * 40
    with pytest.raises(ValueError, match="source tree"):
        controller.check_source(source, declaration)


def test_failed_phase_receipt_retained_and_same_output_cannot_restart(tmp_path, monkeypatch):
    args = SimpleNamespace(output=tmp_path / "attempt", declaration=tmp_path / "declaration.json",
        config=tmp_path / "config.json", resource_gate=tmp_path / "gate.json", expected_resource_gate_sha256="f" * 64)
    args.declaration.write_text('{"fixed":true}\n')
    args.config.write_text('{"seed":11}\n')
    args.resource_gate.write_text('{}')
    args.expected_declaration_sha256 = controller.sha(args.declaration)
    declaration = {"config_sha256": controller.sha(args.config)}
    monkeypatch.setattr(controller, "verify_declaration", lambda _: (declaration, {"seed": 11}))
    with pytest.raises(ValueError, match="resource receipt SHA"):
        controller.run(args)
    failure = json.loads((args.output / "queue_failed.json").read_text())
    assert failure["unstarted_seeds"] == [11, 22, 33] and not failure["automatic_retry"]
    assert failure["error_type"] == "ValueError" and not failure["test_used"]
    assert (args.output / "declaration.json").read_bytes() == args.declaration.read_bytes()
    with pytest.raises(FileExistsError):
        controller.run(args)


def test_cli_requires_immutable_resource_receipt_digest():
    with pytest.raises(SystemExit):
        controller.main([])
