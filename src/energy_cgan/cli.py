"""Explicit, file-based commands for reproducible experiments."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

import numpy as np

RELEASE_URL = "https://github.com/MahmoudAsadi97/cGAN/releases/download/data/raw.zip"
RELEASE_SHA256 = "934cc871e9fe18dbe30530472816d711ece878ec7cd8725d9d8417bbdb389bd8"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_data(output: str | Path) -> Path:
    """Fetch only the pinned release; never trust a mutable asset name alone."""
    output = Path(output)
    if output.exists():
        if sha256(output) != RELEASE_SHA256:
            raise ValueError("Existing dataset checksum does not match pinned release; move it before downloading.")
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    if temporary.exists():
        raise FileExistsError(f"Partial download exists: {temporary}; remove it before retrying")
    try:
        with urllib.request.urlopen(RELEASE_URL, timeout=120) as response, temporary.open("xb") as dest:
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                dest.write(chunk)
        if sha256(temporary) != RELEASE_SHA256:
            raise ValueError("Downloaded dataset checksum does not match pinned release")
        temporary.replace(output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return output


def evaluation_indices(labels: np.ndarray, max_per_condition: int, seed: int) -> np.ndarray:
    if max_per_condition < 1:
        raise ValueError("max_per_condition must be positive")
    rng = np.random.default_rng(seed)
    selected = []
    for label in np.unique(labels):
        eligible = np.flatnonzero(labels == label)
        selected.extend(rng.choice(eligible, min(max_per_condition, len(eligible)), replace=False))
    return np.array(sorted(selected), dtype=np.int64)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch-data", help="Download and checksum the pinned release")
    fetch.add_argument("--output", default="data/raw.zip")
    prepare = commands.add_parser("prepare", help="Build daily profiles and household splits")
    prepare.add_argument("--archive", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--config", default="configs/data_15min.json")
    training = commands.add_parser("train", help="Train with validation-only checkpoint selection")
    training.add_argument("--data", required=True)
    training.add_argument("--output", required=True)
    training.add_argument("--config", default="configs/upstream_15min.json")
    training.add_argument("--seed", type=int)
    sampling = commands.add_parser("generate", help="Generate profiles in original units")
    sampling.add_argument("--checkpoint", required=True)
    sampling.add_argument("--condition", type=int, required=True)
    sampling.add_argument("--num-samples", type=int, default=100)
    sampling.add_argument("--seed", type=int, default=42)
    sampling.add_argument("--output", required=True)
    evaluation = commands.add_parser("evaluate", help="Compare cGAN and conditional resampling")
    evaluation.add_argument("--data", required=True)
    evaluation.add_argument("--checkpoint", required=True)
    evaluation.add_argument("--split", choices=["val", "test"], default="val")
    evaluation.add_argument("--seed", type=int, default=42)
    evaluation.add_argument("--max-per-condition", type=int, default=200)
    evaluation.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = make_parser().parse_args(argv)
    if args.command == "fetch-data":
        print(fetch_data(args.output))
        return
    if args.command in ("prepare", "train"):
        config = json.loads(Path(args.config).read_text())
        if args.command == "prepare":
            from .data import prepare
            result = prepare(args.archive, args.output, config)
        else:
            from .training import train
            if args.seed is not None:
                config["seed"] = args.seed
            result = train(args.data, args.output, config)
        print(json.dumps(result, indent=2, allow_nan=False))
        return
    from .training import generate, validate_checkpoint_data
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.command == "generate":
        if args.num_samples < 1:
            raise ValueError("num_samples must be positive")
        labels = np.full(args.num_samples, args.condition, dtype=np.int64)
        x = generate(args.checkpoint, labels, args.seed)
        import torch
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        metadata = checkpoint["metadata"]
        provenance = {
            "checkpoint_sha256": sha256(Path(args.checkpoint)), "seed": args.seed,
            "classes": metadata["classes"], "unit": metadata["unit"],
            "interval_hours": metadata["interval_hours"], "target": metadata.get("target"),
            "training_dataset_sha256": checkpoint["manifest"]["dataset_sha256"],
            "training_metadata_sha256": checkpoint["manifest"]["metadata_sha256"],
            "output_activation": checkpoint["config"]["output_activation"],
            "num_samples": args.num_samples,
        }
        with output.open("xb") as dest:
            np.savez_compressed(dest, x=x, y=labels,
                                provenance=np.array(json.dumps(provenance, allow_nan=False)))
        print(output)
        return
    from .evaluation import conditional_bootstrap, evaluate_conditional
    data_dir = Path(args.data)
    binding = validate_checkpoint_data(args.checkpoint, data_dir)
    metadata = json.loads((data_dir / "metadata.json").read_text())
    with np.load(data_dir / "dataset.npz", allow_pickle=False) as bundle:
        train_mask = bundle["split"] == "train"
        selected = np.flatnonzero(bundle["split"] == args.split)
        if not len(selected):
            raise ValueError(f"No profiles for split {args.split}")
        selected = selected[evaluation_indices(bundle["y"][selected], args.max_per_condition, args.seed)]
        real, labels = bundle["x"][selected], bundle["y"][selected]
        train_x, train_y = bundle["x"][train_mask], bundle["y"][train_mask]
    fake = generate(args.checkpoint, labels, args.seed)
    empirical = conditional_bootstrap(train_x, train_y, labels, args.seed)
    kwargs = dict(interval_hours=metadata["interval_hours"], unit=metadata["unit"],
                  reference_train=train_x, train_labels=train_y)
    result = {
        "split": args.split, "seed": args.seed,
        "max_per_condition": args.max_per_condition,
        "n_evaluated": len(real), "dataset_sha256": sha256(data_dir / "dataset.npz"),
        "checkpoint_sha256": sha256(Path(args.checkpoint)),
        "checkpoint_data_binding": binding,
        "classes": metadata["classes"],
        "sampling_note": "Fixed seeded cap per condition; inspect counts; global mixture reflects this sample.",
        "cgan": evaluate_conditional(real, fake, labels, **kwargs),
        "conditional_bootstrap": evaluate_conditional(real, empirical, labels, **kwargs),
    }
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(output)
