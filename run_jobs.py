"""Run a list of training jobs, a few at a time, resuming interrupted ones.

    python run_jobs.py jobs/final.txt --parallel 3

Each non-empty line of the job file is ``<config> <seed> [name] [key=value ...]``
(``#`` starts a comment), e.g.::

    faithful 0
    improved 0 improved_big hidden_size=512 depth=4 input_size=64

Every job is started as ``train.py --resume``: a finished run is skipped, an
interrupted run continues from its last checkpoint (``resume.pt``) and a new run
starts from scratch.  Running the same job file again after an interruption
(power loss, closed session, reboot) therefore continues where the jobs left
off.  Output of each job goes to ``runs/logs/<name>_seed<k>.log`` (appended).
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

from train import keep_awake

ROOT = Path(__file__).resolve().parent


def parse_jobs(path: Path) -> list[dict]:
    jobs = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        config, seed, rest = parts[0], int(parts[1]), parts[2:]
        name = rest.pop(0) if rest and "=" not in rest[0] else config
        jobs.append({"config": config, "seed": seed, "name": name, "set": rest})
    return jobs


def command(job: dict, runs: str) -> list[str]:
    cmd = [sys.executable, str(ROOT / "train.py"), "--config", job["config"], "--seed", str(job["seed"]),
           "--name", job["name"], "--out", runs, "--resume"]
    if job["set"]:
        cmd += ["--set", *job["set"]]
    return cmd


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jobs", type=Path)
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--runs", default="runs")
    args = ap.parse_args(argv)
    keep_awake()
    logs = Path(args.runs, "logs")
    logs.mkdir(parents=True, exist_ok=True)
    pending = parse_jobs(args.jobs)
    running = []
    failed = []
    while pending or running:
        for job, proc in list(running):
            if proc.poll() is not None:
                running.remove((job, proc))
                status = "done" if proc.returncode == 0 else f"FAILED ({proc.returncode})"
                if proc.returncode:
                    failed.append(job)
                print(f"{time.strftime('%H:%M:%S')} {status}: {job['name']} seed {job['seed']}", flush=True)
        while pending and len(running) < args.parallel:
            job = pending.pop(0)
            if Path(args.runs, job["name"], f"seed{job['seed']}", "train_summary.json").exists():
                print(f"skip (finished): {job['name']} seed {job['seed']}", flush=True)
                continue
            log = open(logs / f"{job['name']}_seed{job['seed']}.log", "a")
            proc = subprocess.Popen(command(job, args.runs), cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            running.append((job, proc))
            print(f"{time.strftime('%H:%M:%S')} start: {job['name']} seed {job['seed']}", flush=True)
            time.sleep(5)
        time.sleep(10)
    print(f"{time.strftime('%H:%M:%S')} all jobs ended; {len(failed)} failed", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
