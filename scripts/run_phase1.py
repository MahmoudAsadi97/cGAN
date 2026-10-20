"""Launch the frozen Phase 1 training arms with bounded CPU concurrency.

Example: python scripts/run_phase1.py --data DATA --protocol FROZEN.json --runs NEW
This launcher never evaluates, freezes, resumes, retries, or replaces a run.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading


ROOT = Path(__file__).resolve().parents[1]
DATA_FILES = {"train.npz", "val.npz", "test.npz", "scaler.json", "metadata.json"}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _launch(args):
    if not 1 <= args.workers <= 4:
        raise ValueError("workers must be between 1 and 4")
    data, protocol_path, runs = args.data.resolve(), args.protocol.resolve(), args.runs.resolve()
    if runs.exists():
        raise FileExistsError(f"Runs output must be a new directory: {runs}")
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    if not isinstance(protocol, dict) or protocol.get("frozen") is not True or protocol.get("phase") != 1:
        raise ValueError("A frozen Phase 1 protocol is required")
    config, seeds, methods = (protocol.get(key) for key in ("training_config", "training_seeds", "methods"))
    if not isinstance(config, dict):
        raise ValueError("Protocol training_config must be an object")
    if (not isinstance(seeds, list) or not seeds
            or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("Protocol training_seeds must be distinct integers in [0, 2**32)")
    if not isinstance(methods, list) or not {"conditional", "unconditional"}.issubset(methods):
        raise ValueError("Protocol must declare both conditional and unconditional arms")
    hashes = protocol.get("data_hashes")
    if not isinstance(hashes, dict) or set(hashes) != DATA_FILES:
        raise ValueError("Protocol must bind all five prepared data files")
    for name, expected in hashes.items():
        if _sha256(data / name) != expected:
            raise ValueError(f"Frozen protocol data hash mismatch: {name}")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain=v1", "--untracked-files=all"], cwd=ROOT,
                           capture_output=True, text=True, check=True).stdout.splitlines()
    source_files = sorted([*(ROOT / "src").rglob("*.py"), *(ROOT / "scripts").rglob("*.py")])
    manifest = {"schema_version": 1, "phase": 1, "status": "running", "launched_at_utc": _now(),
                "ended_at_utc": None, "workers": args.workers, "python": args.python,
                "protocol": str(protocol_path), "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
                "data": str(data), "data_hashes": hashes, "git": {"head": head, "dirty_files": dirty},
                "source_py_sha256": {str(path.relative_to(ROOT)): _sha256(path) for path in source_files},
                "thread_environment": {key: "1" for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
                "jobs": []}
    # Exclusive directory creation reserves every output before any subprocess.
    runs.mkdir(parents=True, exist_ok=False)
    (runs / "configs").mkdir()
    (runs / "logs").mkdir()
    for seed in seeds:
        for arm in ("conditional", "unconditional"):
            name = f"{arm}_seed_{seed}"
            job_config = {**config, "seed": seed, "unconditional": arm == "unconditional"}
            config_path = runs / "configs" / f"{name}.json"
            with config_path.open("x") as handle:
                handle.write(json.dumps(job_config, indent=2, allow_nan=False) + "\n")
            command = [args.python, "-m", "fluvius_cgan", "train", "--data", str(data),
                       "--run", str(runs / arm / f"seed_{seed}"), "--config", str(config_path)]
            manifest["jobs"].append({"name": name, "method": arm, "training_seed": seed,
                                     "config": job_config, "config_path": str(config_path),
                                     "run": str(runs / arm / f"seed_{seed}"),
                                     "command": command, "log": str(runs / "logs" / f"{name}.log"),
                                     "status": "queued", "started_at_utc": None, "ended_at_utc": None,
                                     "exit_code": None})
    lock = threading.Lock()
    def save_manifest():
        temporary = runs / "manifest.json.tmp"
        temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        os.replace(temporary, runs / "manifest.json")
    save_manifest()

    def run_job(job):
        with lock:
            job.update(status="running", started_at_utc=_now())
            save_manifest()
        result = {"status": "failed", "exit_code": None}
        try:
            with Path(job["log"]).open("x") as log:
                completed = subprocess.run(job["command"], stdout=log, stderr=subprocess.STDOUT,
                                           env={**os.environ, **manifest["thread_environment"]}, check=False)
            result.update(exit_code=completed.returncode, status="completed" if completed.returncode == 0 else "failed")
        except OSError as error:
            result["error"] = f"{type(error).__name__}: {error}"
        with lock:
            job.update(**result, ended_at_utc=_now())
            save_manifest()
        print(f"{job['name']}: {job['status']} (exit {job['exit_code']})", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_job, job) for job in manifest["jobs"]]
        for future in as_completed(futures):
            future.result()
    failed = any(job["status"] != "completed" for job in manifest["jobs"])
    manifest.update(status="failed" if failed else "completed", ended_at_utc=_now())
    save_manifest()
    return 1 if failed else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--workers", default=3, type=int)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args(argv)
    try:
        return _launch(args)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"run_phase1: error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
