"""Execute the declared decoder pilot with immutable inputs and an inherited lock.

This controller starts real fitting only after a published execution declaration
and the completed-reference resource receipt pass verification. It never reads
TEST. A failed or interrupted phase is retained and cannot be resumed in place.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

SCRIPT = "scripts/run_quantile_decoder_pilot.py"
SEEDS = (11, 22, 33)
DATA_FILES = {"train.npz", "val.npz", "metadata.json", "scaler.json", "curation_manifest.json"}
REFERENCE_ARMS = {"curated_reference_bce96", "curated_reference_unconditional96", "curated_reference_bce24"}
GIB = 2 ** 30


def check(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def dump_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def copy_exact_new(source, target, expected):
    payload = Path(source).read_bytes()
    check(hashlib.sha256(payload).hexdigest() == expected, "immutable input changed before copying")
    with Path(target).open("xb") as stream:
        stream.write(payload)


def preflight(prepared, paths):
    """Reject reserved TEST names and aliases using metadata only."""
    reserved = Path(prepared).resolve() / "test.npz"
    for value in paths:
        path = Path(value)
        check(path.name != "test.npz" and path.resolve().name != "test.npz", "TEST input forbidden")
        check(path.resolve() != reserved, "TEST input forbidden")
        if path.exists() and reserved.exists():
            check(not path.samefile(reserved), "TEST alias forbidden")


def check_source(source, declaration):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=source, text=True).strip()
    check(git("rev-parse", "HEAD") == declaration["source_commit"], "source commit differs")
    check(git("rev-parse", "HEAD^{tree}") == declaration["source_tree"], "source tree differs")
    check(not git("status", "--porcelain"), "source worktree must remain clean")
    for name, digest in declaration["source_files_sha256"].items():
        path = Path(name)
        check(not path.is_absolute() and ".." not in path.parts and path.as_posix() == name
              and (source / path).resolve().is_relative_to(source.resolve()), "invalid source path")
        check(sha(source / path) == digest, f"source fingerprint differs: {name}")


def verify_declaration(args):
    preflight(args.prepared, [args.declaration, args.config, args.resource_gate, args.lock, __file__,
                             *(args.prepared / name for name in DATA_FILES)])
    check(sha(args.declaration) == args.expected_declaration_sha256, "execution declaration SHA differs")
    declaration = json.loads(args.declaration.read_text())
    sources = declaration.get("source_files_sha256", {})
    check(isinstance(sources, dict) and SCRIPT in sources and
          "scripts/assess_quantile_decoder_pilot.py" in sources, "controller/assessor must be source-bound")
    preflight(args.prepared, [args.source / name for name in sources])
    check_source(args.source, declaration)
    check(sha(__file__) == sources[SCRIPT], "running controller differs from declared source")
    # Use the pinned report auditor's strict commit/blob/config/data/budget
    # validator. This imports only standard-library code and opens no arrays.
    path = args.source / "scripts/assess_quantile_decoder_pilot.py"
    spec = importlib.util.spec_from_file_location("pinned_decoder_declaration", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    verified = module.load_declaration(args.declaration)
    check(verified["_sha256"] == args.expected_declaration_sha256, "declaration changed during verification")
    check(sha(args.config) == declaration["config_sha256"], "configuration SHA differs")
    config = json.loads(args.config.read_text())
    for seed in SEEDS:
        expected = module.expected_config(seed, verified)
        check(all(key == "seed" or module._equal(value, expected.get(key)) for key, value in config.items()),
              "configuration differs from declared decoder package")
    return declaration, config


def verify_data(prepared, declaration):
    hashes = declaration["checkpoint_data_hashes"]
    check(set(hashes) == DATA_FILES, "exact five development-data bindings required")
    preflight(prepared, [prepared / name for name in DATA_FILES])
    actual = {name: sha(prepared / name) for name in sorted(DATA_FILES)}
    check(actual == hashes, "development data or curation manifest changed")
    return actual


def verify_resource_gate(path, expected):
    check(sha(path) == expected, "resource receipt SHA differs")
    value = json.loads(Path(path).read_text())
    check(value.get("schema_version") == 1 and value.get("test_used") is False and
          value.get("status") == "completed_current_reference_and_capacity_training_phases", "resource receipt status differs")
    rows = value.get("completed_runs", [])
    keys = [(row.get("arm"), row.get("seed")) for row in rows]
    required = {(arm, seed) for arm in REFERENCE_ARMS for seed in (11, 22, 33, 44, 55)}
    check(len(keys) == len(set(keys)) and required <= set(keys), "all 15 reference completions are required")
    for row in rows:
        check(row.get("epochs") == 26 and row.get("updates_per_adversary") == 30784,
              "resource receipt contains incomplete training")
        hashes = row.get("artifacts_sha256", {})
        check({"summary.json", "history.json", "best.pt", "last.pt", "validation_report.json"} <= set(hashes)
              and all(isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v) for v in hashes.values()),
              "resource receipt artifact inventory incomplete")
    return value


def remaining_parallelism(peak_rss_kib, memory_limit_bytes):
    check(type(peak_rss_kib) is int and peak_rss_kib > 0 and type(memory_limit_bytes) is int
          and memory_limit_bytes > 0, "invalid measured memory bounds")
    estimated = 2 * peak_rss_kib * 1024 + 1.5 * GIB
    return (2 if estimated <= min(memory_limit_bytes, 8 * GIB) else 1), estimated


def spawn_locked(command, *, lock, cwd, env, log):
    # pass_fds preserves the *same open file description*. If this controller
    # disappears, each surviving child retains the exclusive flock until exit.
    return subprocess.Popen(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT,
                            pass_fds=(lock.fileno(),), start_new_session=True)


def stop_children(active):
    records = []
    for seed, item in active.items():
        process = item["process"]
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    for seed, item in active.items():
        process = item["process"]
        try:
            code = process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            code = process.wait(timeout=10)
        item["log"].close()
        records.append({"seed": seed, "pid": process.pid, "returncode": code, "cleanup": True})
    active.clear()
    return records


@contextmanager
def interrupted_as_exception():
    def interrupted(number, frame):
        raise InterruptedError(f"controller received signal {number}")
    prior = {number: signal.signal(number, interrupted) for number in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    try:
        yield
    finally:
        for number, handler in prior.items():
            signal.signal(number, handler)


CHILD = '''
from pathlib import Path
from datetime import datetime, timezone
import hashlib,json,os,resource,sys,time
from fluvius_cgan.training import train
from fluvius_cgan.cli import evaluate_checkpoint,write_json
prepared,output,config_path,seed=Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3]),int(sys.argv[4])
lease_fd=int(sys.argv[5]);os.fstat(lease_fd)
start=time.perf_counter()
print('START',datetime.now(timezone.utc).isoformat(),seed,flush=True)
config=json.loads(config_path.read_text())
if config['seed'] != seed: raise ValueError('seed config changed')
summary=train(prepared,output,config)
report=evaluate_checkpoint(prepared,output/'best.pt',split='val',samples_per_condition=100,seed=2026)
write_json(output/'validation_report.json',report)
record={'seed':seed,'completed_utc':datetime.now(timezone.utc).isoformat(),'seconds':time.perf_counter()-start,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'epochs':summary['epochs_completed'],'best_epoch':summary['best_epoch'],'generator_updates':summary['generator_updates'],'discriminator_updates':summary['discriminator_updates'],'samples_seen':summary['samples_seen'],'raw_validation_complete':True}
write_json(output/'execution.json',record)
print('DECODER TRAINING COMPLETE',json.dumps(record),flush=True)
'''


def run(args):
    declaration, config = verify_declaration(args)
    args.output.mkdir(parents=True, exist_ok=False)
    active, completed, pending, failures = {}, [], list(SEEDS), []
    try:
        with interrupted_as_exception():
            copy_exact_new(args.declaration, args.output / "declaration.json", args.expected_declaration_sha256)
            copy_exact_new(args.config, args.output / "frozen_config.json", declaration["config_sha256"])
            (args.output / "declared_configs").mkdir()
            seed_configs = {}
            for seed in SEEDS:
                path = args.output / "declared_configs" / f"seed_{seed}.json"
                dump_new(path, {**config, "seed": seed})
                seed_configs[seed] = {"path": path.resolve(), "sha256": sha(path)}
            dump_new(args.output / "schedule.json", {
                "created_utc": now(), "declaration_sha256": args.expected_declaration_sha256,
                "queue_script_sha256": sha(__file__), "resource_gate_sha256": args.expected_resource_gate_sha256,
                "resource_policy": "Seed11 alone; then two measured child peaks plus1.5GiB must fit8GiB and cgroup bound. Whole phase holds inherited exclusive heavy lock.",
                "test_used": False, "declared_seed_config_sha256": {str(k): v["sha256"] for k, v in seed_configs.items()}})
            while not args.resource_gate.is_file():
                time.sleep(10)
            verify_resource_gate(args.resource_gate, args.expected_resource_gate_sha256)
            env = {**os.environ, "PYTHONPATH": str(args.source / "src"), "OMP_NUM_THREADS": "1",
                   "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
            with args.lock.open("a") as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                try:
                    check_source(args.source, declaration)
                    verify_resource_gate(args.resource_gate, args.expected_resource_gate_sha256)
                    data_hashes = verify_data(args.prepared, declaration)
                    dump_new(args.output / "phase_started.json", {"started_utc": now(), "input_sha256": data_hashes,
                        "resource_gate_sha256": args.expected_resource_gate_sha256, "source_commit": declaration["source_commit"],
                        "source_tree": declaration["source_tree"], "lock": str(args.lock.resolve()),
                        "memory_events": Path("/sys/fs/cgroup/memory.events").read_text()})
                    limit = 1
                    while pending or active:
                        while pending and len(active) < limit and not failures:
                            seed = pending[0]
                            check_source(args.source, declaration)
                            inputs = verify_data(args.prepared, declaration)
                            path = seed_configs[seed]["path"]
                            check(sha(path) == seed_configs[seed]["sha256"], "frozen seed config changed")
                            target = args.output / f"seed_{seed}"
                            check(not target.exists(), "seed output already exists")
                            dump_new(args.output / f"seed_{seed}_started.json", {"seed": seed, "started_utc": now(),
                                "input_sha256": inputs, "config_sha256": seed_configs[seed]["sha256"], "test_used": False})
                            log = (args.output / f"seed_{seed}.log").open("x")
                            command = [sys.executable, "-u", "-c", CHILD, str(args.prepared.resolve()),
                                       str(target.resolve()), str(path), str(seed), str(lock.fileno())]
                            try:
                                process = spawn_locked(command, lock=lock, cwd=args.source, env=env, log=log)
                            except BaseException:
                                log.close()
                                raise
                            active[seed] = {"process": process, "log": log}
                            pending.pop(0)
                            print("LAUNCHED", seed, now(), flush=True)
                        for seed, item in list(active.items()):
                            code = item["process"].poll()
                            if code is None:
                                continue
                            item["log"].close()
                            del active[seed]
                            if code:
                                failures.append({"seed": seed, "returncode": code})
                                dump_new(args.output / f"seed_{seed}_failed.json", failures[-1])
                                continue
                            check_source(args.source, declaration)
                            post = verify_data(args.prepared, declaration)
                            check(sha(seed_configs[seed]["path"]) == seed_configs[seed]["sha256"], "frozen seed config changed")
                            record = json.loads((args.output / f"seed_{seed}" / "execution.json").read_text())
                            check(record["seed"] == seed and record["epochs"] == 26 and
                                  record["generator_updates"] == record["discriminator_updates"] == 30784 and
                                  record["samples_seen"] == 15761408 and record["raw_validation_complete"] is True,
                                  "child did not complete the fixed budget")
                            completed.append(record)
                            dump_new(args.output / f"seed_{seed}_verified.json", {"execution": record, "post_input_sha256": post,
                                "artifacts_sha256": {name: sha(args.output / f"seed_{seed}" / name) for name in
                                    ("best.pt", "last.pt", "summary.json", "history.json", "config.json", "validation_report.json", "execution.json")}})
                            if seed == 11:
                                memory_limit = int(Path("/sys/fs/cgroup/memory.max").read_text())
                                limit, estimate = remaining_parallelism(record["peak_rss_kib"], memory_limit)
                                dump_new(args.output / "remaining_concurrency.json", {"first_seed_peak_rss_kib": record["peak_rss_kib"],
                                    "rss_bytes_per_kib": 1024, "gib_bytes": GIB, "cgroup_memory_limit_bytes": memory_limit,
                                    "protocol_memory_limit_bytes": 8 * GIB, "two_child_peak_plus_reserve_bytes": estimate,
                                    "remaining_parallelism": limit, "basis": "resource measurement only; no endpoint selection"})
                        if failures:
                            raise RuntimeError("Training failure retained; no automatic seed replacement or restart")
                        if pending or active:
                            time.sleep(5)
                    check_source(args.source, declaration)
                    post = verify_data(args.prepared, declaration)
                    dump_new(args.output / "queue_complete.json", {"completed_utc": now(), "seeds": completed,
                        "post_input_sha256": post, "memory_events": Path("/sys/fs/cgroup/memory.events").read_text(), "test_used": False})
                except BaseException:
                    # Do not release the parent lock while children still run.
                    failures.extend(stop_children(active))
                    raise
    except BaseException as error:
        failures.extend(stop_children(active))
        dump_new(args.output / "queue_failed.json", {"failed_utc": now(), "error_type": type(error).__name__,
            "error": str(error), "failures": failures, "unstarted_seeds": pending, "completed": completed,
            "test_used": False, "automatic_retry": False})
        raise
    print("QUANTILE DECODER TRAINING COMPLETE", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "declaration", "prepared", "config", "output", "resource-gate", "lock"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--expected-declaration-sha256", required=True)
    parser.add_argument("--expected-resource-gate-sha256", required=True)
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
