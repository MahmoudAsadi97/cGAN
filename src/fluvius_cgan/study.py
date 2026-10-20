"""Shared, provenance-bound evaluation for the frozen Phase 1 study."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import wasserstein_distance

from .baselines import fit_baseline
from .data import load_split
from .evaluation import conditional_metrics, projection_directions


GENERATION_SEEDS = [31001, 31002, 31003]
TRAINING_SEEDS = [11, 22, 33, 44, 55]
DATA_FILES = ("train.npz", "val.npz", "test.npz", "scaler.json", "metadata.json")


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    """Write a new report; never silently replace an earlier experiment."""
    path = Path(path)
    serialized = json.dumps(value, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        handle.write(serialized)


def _seeds(seeds):
    seeds = list(seeds)
    if (not seeds or any(type(s) is not int or s < 0 or s >= 2**32 for s in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("generation seeds must be distinct nonnegative integers")
    return seeds


def freeze_protocol(data_dir, output, training_config, generation_seeds=None):
    """Bind the intended final experiment before any test performance access."""
    if Path(output).exists():
        raise FileExistsError(output)
    root = Path(data_dir)
    value = {
        "schema_version": 1, "phase": 1, "frozen": True,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_hashes": {name: sha256(root / name) for name in DATA_FILES},
        "training_config": training_config,
        "training_seeds": TRAINING_SEEDS,
        "generation_seeds": _seeds(GENERATION_SEEDS if generation_seeds is None else generation_seeds),
        "methods": ["conditional", "unconditional", "mean", "resample", "copula"],
        "primary_endpoint": "condition-macro normalized profile sliced Wasserstein-1",
        "projection_count": 128, "projection_seed": 2026,
        "prior_access": (
            "V2 documents validation exploration and deterministic test identity access. "
            "Its inspected records state final test performance was unseen; this study "
            "cannot independently establish every prior user's access. V3 preserves the "
            "original retained test membership and logs its own performance accesses."
        ),
    }
    write_json(output, value)
    return value


def _test_protocol(data_dir, protocol, seeds):
    if protocol is None:
        raise ValueError("test evaluation requires a frozen protocol")
    value = json.loads(Path(protocol).read_text())
    if value.get("frozen") is not True or value.get("phase") != 1:
        raise ValueError("test evaluation requires a frozen protocol for Phase 1")
    if value.get("generation_seeds") != seeds:
        raise ValueError("generation seeds differ from the frozen protocol")
    for name in DATA_FILES:
        if value.get("data_hashes", {}).get(name) != sha256(Path(data_dir) / name):
            raise ValueError(f"frozen protocol data hash mismatch: {name}")
    return value


def _verified_data(data_dir, split):
    root = Path(data_dir)
    metadata = json.loads((root / "metadata.json").read_text())
    scaler = json.loads((root / "scaler.json").read_text())
    if metadata.get("resolution") != 24 or metadata.get("interval_hours") != 1.0:
        raise ValueError("Phase 1 evaluation requires hourly physical energy profiles")
    names = ("train.npz", f"{split}.npz", "metadata.json", "scaler.json")
    hashes = {name: sha256(root / name) for name in names}
    for name in ("train", split):
        if metadata.get("splits", {}).get(name, {}).get("sha256") != hashes[f"{name}.npz"]:
            raise ValueError(f"prepared split hash mismatch: {name}")
    train, real = load_split(root / "train.npz"), load_split(root / f"{split}.npz")
    if set(train["meter_id"]) & set(real["meter_id"]):
        raise ValueError("training and evaluation meter identifiers overlap")
    if (scaler.get("fitted_on") != "train"
            or scaler.get("minimum") != float(train["x"].min())
            or scaler.get("maximum") != float(train["x"].max())):
        raise ValueError("evaluation scaler must match training bounds only")
    if hashes != {name: sha256(root / name) for name in names}:
        raise ValueError("prepared files changed while loading")
    return train, real, metadata, scaler, hashes


def _profile_curves(real, fake, labels):
    probabilities = [.01, .05, .5, .95, .99]
    curves = {}
    for label in np.unique(labels):
        a, b = real[labels == label], fake[labels == label]
        curves[str(int(label))] = {
            "quantile_probabilities": probabilities,
            "real_mean_kwh": a.mean(axis=0, dtype=np.float64).tolist(),
            "fake_mean_kwh": b.mean(axis=0, dtype=np.float64).tolist(),
            "real_quantiles_kwh": np.quantile(a, probabilities, axis=0).tolist(),
            "fake_quantiles_kwh": np.quantile(b, probabilities, axis=0).tolist(),
            "slot_w1_kwh": [float(wasserstein_distance(a[:, j], b[:, j]))
                            for j in range(a.shape[1])],
        }
    return curves


def evaluate_run(data_dir, checkpoint_or_baseline, split, seeds, output, *, protocol=None):
    """Evaluate matched samples; default callers should explicitly use ``val``.

    Final test access requires a frozen, data-bound protocol. This is an audit
    guard and cannot prevent a researcher from reading files outside the tool.
    """
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if split not in {"val", "test"}:
        raise ValueError("evaluation split must be val or test")
    seeds = _seeds(seeds)
    frozen = _test_protocol(data_dir, protocol, seeds) if split == "test" else None
    access_log = output.parent / "test_access.jsonl"
    protocol_hash = sha256(protocol) if protocol is not None else None

    def event(name, **extra):
        if split == "test":
            access_log.parent.mkdir(parents=True, exist_ok=True)
            with access_log.open("a") as handle:
                handle.write(json.dumps({
                    "event": name, "time_utc": datetime.now(timezone.utc).isoformat(),
                    "output": str(output), "protocol_sha256": protocol_hash, **extra,
                }, allow_nan=False) + "\n")

    event("test_evaluation_started")
    try:
        train, real, metadata, scaler, hashes = _verified_data(data_dir, split)
        method = str(checkpoint_or_baseline)
        checkpoint_hash = None
        model_metadata = None
        if method in {"mean", "resample", "copula"}:
            model = fit_baseline(method, train["x"], train["y"])
            sample = lambda seed: model.sample(real["y"], seed)
        else:
            import torch
            from .training import generate
            checkpoint = Path(checkpoint_or_baseline)
            checkpoint_hash = sha256(checkpoint)
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            for name in ("train.npz", "scaler.json", "metadata.json"):
                if payload["data_hashes"][name] != sha256(Path(data_dir) / name):
                    raise ValueError(f"checkpoint dataset hash mismatch: {name}")
            config = payload["config"]
            method = "unconditional" if config["unconditional"] else "conditional"
            model_metadata = {"config": config, "epoch": payload["epoch"],
                              "selection": payload["selection"]}
            if frozen is not None:
                if config["seed"] not in frozen["training_seeds"]:
                    raise ValueError("checkpoint seed is not in frozen protocol")
                for key, value in frozen["training_config"].items():
                    if key not in {"seed", "unconditional"} and config.get(key) != value:
                        raise ValueError(f"checkpoint config differs from frozen protocol: {key}")
            sample = lambda seed: generate(checkpoint, real["y"], seed)
        if frozen is not None and method not in frozen["methods"]:
            raise ValueError("method is not in frozen protocol")
        directions = projection_directions(24, count=128, seed=2026)
        denominator = scaler["maximum"] - scaler["minimum"] or 1.0
        replicates = []
        for seed in seeds:
            fake = sample(seed)
            metrics = conditional_metrics(real["x"], fake, real["y"], real["y"],
                                          interval_hours=1.0, directions=directions)
            for grouping in ("macro", "weighted"):
                metrics[grouping]["normalized_profile_sw1"] = (
                    metrics[grouping]["profile_sw1_kwh"] / denominator)
            replicates.append({"generation_seed": seed, "metrics": metrics,
                               "curves": _profile_curves(real["x"], fake, real["y"])})
        summary = {}
        for metric in replicates[0]["metrics"]["macro"]:
            values = [r["metrics"]["macro"][metric] for r in replicates]
            finite = [v for v in values if v is not None]
            summary[metric] = {"mean": float(np.mean(finite)) if finite else None,
                               "sampling_sd": float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.,
                               "valid_replicates": len(finite)}
        result = {
            "schema_version": 1, "method": method, "split": split,
            "test_performance_access": split == "test",
            "protocol_sha256": protocol_hash, "data_hashes": hashes,
            "source_sha256": metadata.get("source_sha256"),
            "checkpoint_sha256": checkpoint_hash, "model": model_metadata,
            "generation_seeds": seeds, "profile_units": "kWh/hour",
            "sample_policy": "all eligible real rows; equal generated count within every condition",
            "directions_sha256": hashlib.sha256(directions.astype("<f8").tobytes()).hexdigest(),
            "normalization": {"minimum": scaler["minimum"], "maximum": scaler["maximum"],
                              "fitted_on": "train"},
            "replicates": replicates, "macro_summary": summary,
        }
        write_json(output, result)
        event("test_evaluation_completed", report_sha256=sha256(output))
        return result
    except Exception as error:
        event("test_evaluation_failed", error_type=type(error).__name__, message=str(error))
        raise
