#!/usr/bin/env python3
"""Run matched cGAN/baseline evaluations across seeds, then summarize all runs.

Install the project first (``python -m pip install -e '.[dev]'``). Example:
``python scripts/run_experiments.py --data data/prepared --output runs/five_seeds``.
Validation is the default evaluation split; use ``--split test`` only after
freezing the model and experiment choices. Existing output directories are
refused, including partially completed experiments; use a new directory.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev
import subprocess
import sys
from typing import Any, Sequence


DEFAULT_SEEDS = [42, 43, 44, 45, 46]
_METHODS = ("cgan", "conditional_bootstrap")


def _validated_seeds(seeds: Sequence[int]) -> list[int]:
    values = list(seeds)
    if not values or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
                         for seed in values):
        raise ValueError("seeds must be a nonempty list of nonnegative integers")
    if len(set(values)) != len(values):
        raise ValueError("seeds must be unique")
    return values


def _metric_values(report: dict[str, Any], method: str) -> dict[str, float | None]:
    try:
        metrics = report[method]["global"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"report missing {method}.global metrics") from exc
    if not isinstance(metrics, dict):
        raise ValueError(f"{method}.global metrics must be an object")
    result = {}
    for name, value in metrics.items():
        if value is None:
            result[name] = None
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            if not math.isfinite(value):
                raise ValueError(f"{method}.{name} must be finite or null")
            result[name] = float(value)
    if not result:
        raise ValueError(f"{method}.global has no numeric metrics")
    return result


def build_summary(
    report_paths: Sequence[str | Path], expected_seeds: Sequence[int]
) -> dict[str, Any]:
    """Aggregate exactly the requested seeds; never silently summarize a subset.

    Only scalar global metrics are aggregated. Nested diagnostic objects and
    counts in metadata remain in each source report. If any seed has an
    undefined metric, its aggregate mean/std stay null and availability is
    explicit. A one-seed sample standard deviation is likewise undefined.
    """
    expected = _validated_seeds(expected_seeds)
    by_seed: dict[int, tuple[Path, dict[str, Any]]] = {}
    for raw_path in report_paths:
        path = Path(raw_path)
        report = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(report, dict):
            raise ValueError(f"report must be a JSON object: {path}")
        seed = report.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError(f"report requires a nonnegative integer seed: {path}")
        if seed in by_seed:
            raise ValueError(f"duplicate completed seed: {seed}")
        by_seed[seed] = (path, report)
    missing = sorted(set(expected) - set(by_seed))
    unexpected = sorted(set(by_seed) - set(expected))
    if missing or unexpected:
        raise ValueError(f"seed mismatch: missing {missing}; unexpected {unexpected}")
    reports = [by_seed[seed][1] for seed in expected]
    first = reports[0]
    shared_fields = ("split", "dataset_sha256", "max_per_condition", "classes", "n_evaluated")
    for field in shared_fields:
        if field not in first:
            raise ValueError(f"report missing required field: {field}")
        if any(report.get(field) != first[field] for report in reports[1:]):
            raise ValueError(f"inconsistent {field} across seed reports")
    if first["split"] not in ("val", "test"):
        raise ValueError("report split must be val or test")
    summary: dict[str, Any] = {
        "expected_seeds": expected,
        "completed_seeds": expected,
        "n_completed": len(reports),
        **{field: first[field] for field in shared_fields},
        "source_reports": {str(seed): str(by_seed[seed][0]) for seed in expected},
        "uncertainty_note": (
            "Mean and sample standard deviation (ddof=1) across training/generation "
            "seeds, including seeded evaluation subsampling and baseline resampling. "
            "These are not household-level confidence intervals or independent "
            "dataset replications. cGAN and conditional bootstrap use matched "
            "evaluation profiles within each run."
        ),
        "aggregation_note": (
            "Scalar global metrics only; inspect source reports for condition "
            "support, warnings, and nearest-neighbor diagnostics. Mean and std "
            "are null if any expected seed has an undefined metric. Sample std "
            "is null for a single seed."
        ),
    }
    for method in _METHODS:
        metrics_by_seed = [_metric_values(report, method) for report in reports]
        names = set(metrics_by_seed[0])
        if any(set(metrics) != names for metrics in metrics_by_seed[1:]):
            raise ValueError(f"inconsistent {method}.global metric schema across seed reports")
        first_metadata = first[method]["global"].get("metadata", {})
        for field in ("profile_value_unit", "interval_hours", "n_time_slots"):
            reference = first_metadata.get(field)
            if any(report[method]["global"].get("metadata", {}).get(field) != reference
                   for report in reports[1:]):
                raise ValueError(f"inconsistent {method} metric metadata: {field}")
        aggregated = {}
        for name in sorted(names):
            values = [metrics[name] for metrics in metrics_by_seed]
            available = [value for value in values if value is not None]
            complete = len(available) == len(expected)
            aggregated[name] = {
                "mean": mean(available) if complete else None,
                "sample_std": stdev(available) if complete and len(available) > 1 else None,
                "n_available": len(available),
                "n_expected": len(expected),
                "values_by_seed": {str(seed): value for seed, value in zip(expected, values)},
            }
        summary[method] = {"global": aggregated, "units": {
            field: first_metadata.get(field)
            for field in ("profile_value_unit", "interval_hours", "n_time_slots")
        }}
    return summary


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Prepared dataset directory")
    parser.add_argument("--output", required=True, help="New experiment directory; never overwritten")
    parser.add_argument("--config", default="configs/upstream_15min.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--max-per-condition", type=int, default=200)
    return parser


def _write_new_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def main(argv: list[str] | None = None) -> None:
    args = make_parser().parse_args(argv)
    seeds = _validated_seeds(args.seeds)
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing experiment directory: {output}")
    if args.max_per_condition < 1:
        raise ValueError("max_per_condition must be positive")
    data = Path(args.data).resolve()
    if not data.is_dir():
        raise NotADirectoryError(f"Prepared dataset directory does not exist: {data}")
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("training config must be a JSON object")
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "training_config.json"
    _write_new_json(snapshot, config)
    _write_new_json(output / "experiment.json", {
        "expected_seeds": seeds,
        "data": str(data),
        "source_config": str(config_path),
        "training_config_snapshot": str(snapshot),
        "seed_override": "Each seed overrides the training config and is reused for evaluation.",
        "split": args.split,
        "max_per_condition": args.max_per_condition,
    })
    reports = []
    for seed in seeds:
        run_dir = output / f"seed_{seed}"
        subprocess.run([
            sys.executable, "-m", "energy_cgan", "train",
            "--data", str(data), "--output", str(run_dir),
            "--config", str(snapshot), "--seed", str(seed),
        ], check=True)
        report = run_dir / "report.json"
        subprocess.run([
            sys.executable, "-m", "energy_cgan", "evaluate",
            "--data", str(data), "--checkpoint", str(run_dir / "best.pt"),
            "--split", args.split, "--output", str(report), "--seed", str(seed),
            "--max-per-condition", str(args.max_per_condition),
        ], check=True)
        reports.append(report)
    summary = build_summary(reports, expected_seeds=seeds)
    _write_new_json(output / "summary.json", summary)
    print(output / "summary.json")


if __name__ == "__main__":
    main()
