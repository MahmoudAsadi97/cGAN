"""Explicit Phase 1 commands; data, checkpoints and reports stay user-owned."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys
import zipfile


EXPECTED_ERRORS = (OSError, ValueError, FloatingPointError, zipfile.BadZipFile)


def _read_json(path):
    def reject_constant(value):
        raise ValueError(f"Nonfinite JSON constant in {path}: {value}")
    return json.loads(Path(path).read_text(), parse_constant=reject_constant)


def _config(path):
    value = _read_json(path)
    if not isinstance(value, dict):
        raise ValueError("Training config must be a JSON object")
    return value


def _seed_list(seeds):
    if not seeds or len(set(seeds)) != len(seeds) or any(not 0 <= seed < 2**32 for seed in seeds):
        raise ValueError("Seeds must be distinct integers in [0, 2**32)")
    return seeds


def _parser():
    parser = argparse.ArgumentParser(prog="fluvius-cgan", description="Phase 1 reference GAN study on hourly Fluvius energy profiles.")
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("download", help="Download and verify the pinned source archive")
    command.add_argument("--output", required=True, type=Path)
    command = commands.add_parser("prepare", help="Prepare original 96-slot meter-disjoint splits")
    command.add_argument("--archive", required=True, type=Path)
    command.add_argument("--output", required=True, type=Path)
    command.add_argument("--seed", default=42, type=int)
    command.add_argument("--chunksize", default=250_000, type=int)
    for name, description in [("curate", "Remove complete-trace aliases at source resolution"),
                              ("hourly", "Sum curated quarter-hours into hourly kWh")]:
        command = commands.add_parser(name, help=description)
        command.add_argument("--data", required=True, type=Path)
        command.add_argument("--output", required=True, type=Path)
    command = commands.add_parser("train", help="Train one reference model using train and validation")
    command.add_argument("--data", required=True, type=Path)
    command.add_argument("--run", required=True, type=Path)
    command.add_argument("--config", required=True, type=Path)
    command.add_argument("--resume", action="store_true")
    command = commands.add_parser("generate", help="Write physical hourly-kWh CSV and JSON provenance")
    command.add_argument("--checkpoint", required=True, type=Path)
    command.add_argument("--condition", required=True, type=int, choices=range(64), metavar="INT")
    command.add_argument("--count", required=True, type=int)
    command.add_argument("--seed", required=True, type=int)
    command.add_argument("--output", required=True, type=Path)
    command = commands.add_parser("evaluate", help="Evaluate mean/resample/copula or a checkpoint; test requires a frozen protocol")
    command.add_argument("--data", required=True, type=Path)
    command.add_argument("--method", required=True)
    command.add_argument("--split", choices=("val", "test"), default="val")
    command.add_argument("--seeds", type=int, nargs="+", default=[31001, 31002, 31003])
    command.add_argument("--output", required=True, type=Path)
    command.add_argument("--protocol", type=Path)
    command = commands.add_parser("freeze", help="Freeze a data-bound Phase 1 protocol explicitly")
    command.add_argument("--data", required=True, type=Path)
    command.add_argument("--config", required=True, type=Path)
    command.add_argument("--output", required=True, type=Path)
    command = commands.add_parser("study", help="Sequentially train conditional and unconditional arms; records all failures")
    command.add_argument("--data", required=True, type=Path)
    command.add_argument("--config", required=True, type=Path)
    command.add_argument("--runs", required=True, type=Path)
    command.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33, 44, 55])
    command = commands.add_parser("summarize", help="Summarize saved reports with separate generation and training seed uncertainty")
    command.add_argument("--reports", required=True, type=Path, nargs="+")
    command.add_argument("--output", required=True, type=Path)
    return parser


def _generate(args):
    import numpy as np
    from .study import sha256
    from .training import generate
    output, sidecar = args.output, args.output.with_suffix(".json")
    if output.suffix.lower() != ".csv":
        raise ValueError("Generation output must have a .csv suffix; provenance uses the matching .json path")
    if args.count < 1:
        raise ValueError("Generation count must be a positive integer")
    for path in (output, sidecar):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite generated output: {path}")
    digest = sha256(args.checkpoint)
    values = generate(args.checkpoint, np.full(args.count, args.condition, dtype=np.int64), args.seed)
    if digest != sha256(args.checkpoint):
        raise ValueError("Checkpoint changed during generation")
    metadata = {"schema_version": 1, "phase": 1, "checkpoint": str(args.checkpoint.resolve()),
                "checkpoint_sha256": digest, "condition": args.condition, "count": args.count,
                "seed": args.seed, "profile_units": "kWh per hourly interval", "interval_hours": 1.0,
                "columns": [f"hour_{hour:02d}" for hour in range(24)]}
    serialized = json.dumps(metadata, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    created = []
    try:
        with output.open("x", newline="") as handle:
            created.append(output)
            with sidecar.open("x") as provenance:
                created.append(sidecar)
                writer = csv.writer(handle)
                writer.writerow(metadata["columns"])
                writer.writerows(values)
                provenance.write(serialized)
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        raise


def _study(args):
    from .study import write_json
    from .training import train, _configuration
    config = _configuration(_config(args.config))
    seeds = _seed_list(args.seeds)
    if args.runs.exists() and (not args.runs.is_dir() or any(args.runs.iterdir())):
        raise FileExistsError(f"Study requires a new or empty runs directory: {args.runs}")
    records = []
    for seed in seeds:
        for unconditional in (False, True):
            arm = "unconditional" if unconditional else "conditional"
            run = args.runs / arm / f"seed_{seed}"
            record = {"method": arm, "training_seed": seed, "run": str(run)}
            try:
                result = train(args.data, run, {**config, "seed": seed, "unconditional": unconditional})
                record["status"] = result["status"]
            except EXPECTED_ERRORS as error:
                record.update(status="failed", error=f"{type(error).__name__}: {error}")
                print(f"{arm} seed {seed}: {record['error']}", file=sys.stderr)
            records.append(record)
    write_json(args.runs / "study_manifest.json", {
        "schema_version": 1, "phase": 1, "training_config": config,
        "training_seeds": seeds, "runs": records, "test_performance_access": False,
    })
    if any(record["status"] != "completed" for record in records):
        raise ValueError(f"Study contains failed runs; see {args.runs / 'study_manifest.json'}")


def _summarize(args):
    from .study import sha256, write_json
    if args.output.exists():
        raise FileExistsError(args.output)
    reports, groups = [], {}
    for path in args.reports:
        report = _read_json(path)
        if (not isinstance(report, dict) or not isinstance(report.get("macro_summary"), dict)
                or not isinstance(report.get("generation_seeds"), list)
                or report.get("split") not in {"val", "test"}
                or not isinstance(report.get("method"), str)):
            raise ValueError(f"Not a saved Phase 1 evaluation report: {path}")
        model = report.get("model") or {}
        if not isinstance(model, dict):
            raise ValueError(f"Invalid model metadata in report: {path}")
        config = model.get("config", {})
        if not isinstance(config, dict):
            raise ValueError(f"Invalid model config in report: {path}")
        seed = config.get("seed")
        reports.append({"report": str(path.resolve()), "report_sha256": sha256(path),
                        "method": report["method"], "split": report["split"],
                        "model": model, "configuration": config, "training_seed": seed,
                        "monte_carlo": {"generation_seeds": report["generation_seeds"],
                                        "macro_summary": report["macro_summary"]}})
        if seed is None:
            continue
        configuration = {key: value for key, value in config.items() if key != "seed"}
        grouping = {"method": report["method"], "split": report["split"], "configuration": configuration,
                    "data_hashes": report.get("data_hashes"), "protocol_sha256": report.get("protocol_sha256"),
                    "generation_seeds": report["generation_seeds"]}
        key = json.dumps(grouping, sort_keys=True, allow_nan=False)
        group = groups.setdefault(key, {**grouping, "training_seeds": [], "values": {}})
        if seed in group["training_seeds"]:
            raise ValueError(f"Duplicate training seed {seed} in comparable saved reports")
        group["training_seeds"].append(seed)
        for metric, summary in report["macro_summary"].items():
            if not isinstance(summary, dict) or "mean" not in summary:
                raise ValueError(f"Invalid macro summary for {metric}: {path}")
            value = summary["mean"]
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
                raise ValueError(f"Invalid metric mean for {metric}: {path}")
            if value is not None:
                group["values"].setdefault(metric, []).append(value)
    summaries = []
    for group in groups.values():
        values = group.pop("values")
        group["metrics"] = {
            metric: {"mean": statistics.mean(samples), "training_seed_sd": statistics.stdev(samples) if len(samples) > 1 else None,
                     "valid_training_seeds": len(samples)} for metric, samples in values.items()
        }
        summaries.append(group)
    write_json(args.output, {"schema_version": 1, "reports": reports, "training_seed_summary": summaries,
                             "uncertainty_note": "macro_summary sampling_sd describes Monte Carlo generation replicates within one model. training_seed_sd describes variation of model replicate means across distinct training seeds; one seed cannot estimate it."})


def _dispatch(args):
    if args.command == "download":
        from .release import download_release
        download_release(args.output)
    elif args.command == "prepare":
        from .data import prepare_archive
        prepare_archive(args.archive, args.output, resolution=96, seed=args.seed, chunksize=args.chunksize)
    elif args.command == "curate":
        from .curation import curate_prepared
        curate_prepared(args.data, args.output)
    elif args.command == "hourly":
        from .prepared import aggregate_hourly
        aggregate_hourly(args.data, args.output)
    elif args.command == "train":
        from .training import train
        train(args.data, args.run, _config(args.config), resume=args.resume)
    elif args.command == "generate":
        _generate(args)
    elif args.command == "evaluate":
        from .study import evaluate_run
        evaluate_run(args.data, args.method, args.split, args.seeds, args.output, protocol=args.protocol)
    elif args.command == "freeze":
        from .study import freeze_protocol
        from .training import _configuration
        freeze_protocol(args.data, args.output, _configuration(_config(args.config)))
    elif args.command == "study":
        _study(args)
    elif args.command == "summarize":
        _summarize(args)


def main(argv=None):
    """Return an exit status; expected input failures are concise stderr errors."""
    args = _parser().parse_args(argv)
    try:
        _dispatch(args)
    except EXPECTED_ERRORS as error:
        print(f"fluvius-cgan: error: {error}", file=sys.stderr)
        return 2
    return 0
