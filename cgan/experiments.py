"""Run the full experiment matrix (resumable).

    python -m cgan.experiments --plan configs/experiments.yaml
    python -m cgan.experiments --plan configs/experiments.yaml --only models --jobs 3

Training runs are executed as separate processes (``--jobs`` at a time, one
GPU); finished runs are evaluated in this process as soon as they complete.
Anything whose output already exists is skipped, so the command can simply be
re-run after an interruption.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

from .baselines import fit_baseline
from .sample import resolve_ckpt
from .evaluate import BaselineSource, GeneratorSource, RealSource, evaluate, get_splits, summarise
from .utils import load_config, write_json

ROOT = Path(__file__).resolve().parents[1]
EVAL_SEED = 2025


def _log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def _train_job(config: str, seed: int, out: Path) -> tuple[str, int, int]:
    cfg = load_config(ROOT / config)
    name = cfg["name"]
    run_dir = out / name / f"seed{seed}"
    if (run_dir / "last.pt").exists():
        return name, seed, 0
    log_dir = out / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / f"{name}_seed{seed}.log", "w") as fh:
        rc = subprocess.run(
            [sys.executable, "-m", "cgan.train", "--config", str(config), "--seed", str(seed), "--out", str(out)],
            cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT,
        ).returncode
    return name, seed, rc


def _write_eval(df, out_dir: Path, tag: str) -> None:
    df.to_csv(out_dir / f"{tag}.csv", index=False)
    write_json(out_dir / f"{tag}.json", summarise(df))


def _eval_run(run_dir: Path, ckpts, split: str, seed: int, data_cfg) -> None:
    for ckpt in ckpts:
        tag = f"eval_{split}_{ckpt}"
        if (run_dir / f"{tag}.csv").exists() or not (run_dir / f"{resolve_ckpt(ckpt)[0]}.pt").exists():
            continue
        src = GeneratorSource(run_dir, ckpt)
        df = evaluate(src, get_splits(data_cfg), data_cfg, split, EVAL_SEED + seed, log=_log,
                      save_samples=run_dir / f"samples_{split}_{ckpt}.npz")
        _write_eval(df, run_dir, tag)
        _log(f"evaluated {run_dir} [{ckpt}]")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", default="configs/experiments.yaml")
    ap.add_argument("--only", nargs="*", default=None,
                    help="subset of: models ablations baselines real_reference")
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--no-eval", action="store_true")
    args = ap.parse_args(argv)

    plan = yaml.safe_load((ROOT / args.plan).read_text())
    out = ROOT / plan.get("out", "runs")
    split = plan.get("eval_split", "test")
    jobs = args.jobs or plan.get("jobs", 1)
    sections = args.only or ["models", "ablations", "baselines", "real_reference"]
    base_cfg = load_config(ROOT / "configs/base.yaml")
    data_cfg = base_cfg["data"]

    train_items = []
    for sec in ("models", "ablations"):
        if sec in sections:
            for item in plan.get(sec, []):
                for seed in item["seeds"]:
                    train_items.append((item["config"], seed, item.get("ckpts", ["best"])))

    pool = ThreadPoolExecutor(max_workers=jobs)
    futures = {pool.submit(_train_job, c, s, out): (c, s, ck) for c, s, ck in train_items}
    _log(f"{len(futures)} training runs scheduled ({jobs} concurrent)")

    if not args.no_eval:
        # Baselines and the real-data reference do not need the GPU queue.
        if "baselines" in sections:
            b = plan["baselines"]
            for kind in b["kinds"]:
                for seed in b["seeds"]:
                    run_dir = out / f"baseline_{kind}" / f"seed{seed}"
                    try:
                        if not (run_dir / "model.npz").exists():
                            fit_baseline(kind, base_cfg, out, seed, get_splits(data_cfg))
                            _log(f"fitted baseline {kind} seed {seed}")
                        tag = f"eval_{split}"
                        if not (run_dir / f"{tag}.csv").exists():
                            df = evaluate(BaselineSource(run_dir), get_splits(data_cfg), data_cfg, split,
                                          EVAL_SEED + seed, log=_log, save_samples=run_dir / f"samples_{split}.npz")
                            _write_eval(df, run_dir, tag)
                            _log(f"evaluated baseline {kind} seed {seed}")
                    except Exception:
                        _log(f"baseline {kind} seed {seed} failed:\n{traceback.format_exc()}")
        if "real_reference" in sections:
            r = plan["real_reference"]
            for seed in r["seeds"]:
                run_dir = out / f"real_{r['split']}" / f"seed{seed}"
                tag = f"eval_{split}"
                if not (run_dir / f"{tag}.csv").exists():
                    run_dir.mkdir(parents=True, exist_ok=True)
                    splits = get_splits(data_cfg)
                    df = evaluate(RealSource(splits[r["split"]], {"data": data_cfg}, r["split"], seed), splits, data_cfg, split,
                                  EVAL_SEED + seed, log=_log, save_samples=run_dir / f"samples_{split}.npz")
                    _write_eval(df, run_dir, tag)
                    _log(f"evaluated real reference ({r['split']}) seed {seed}")

    for fut in as_completed(futures):
        config, seed, ckpts = futures[fut]
        try:
            name, seed, rc = fut.result()
        except Exception:
            _log(f"training {config} seed {seed} crashed:\n{traceback.format_exc()}")
            continue
        if rc != 0:
            _log(f"training {name} seed {seed} FAILED (exit {rc}); see {out / 'logs'}")
            continue
        _log(f"training {name} seed {seed} done")
        if not args.no_eval:
            try:
                _eval_run(out / name / f"seed{seed}", ckpts, split, seed, data_cfg)
            except Exception:
                _log(f"evaluation of {name} seed {seed} failed:\n{traceback.format_exc()}")
    pool.shutdown()
    _log("all done")


if __name__ == "__main__":
    main()
