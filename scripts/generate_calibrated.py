"""Generate physical-unit profiles from a frozen cGAN and its bound fixed map."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fluvius_cgan.calibration import ConditionalMarginalCalibrator
from fluvius_cgan.release import sha256_file
from fluvius_cgan.training import generate


def generate_profiles(checkpoint, calibrator, labels, *, seed=42, batch_size=1024):
    checkpoint_hash = sha256_file(checkpoint)
    map_hash = sha256_file(calibrator)
    model = ConditionalMarginalCalibrator.load(calibrator)
    if model.provenance.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("calibrator does not belong to this checkpoint")
    raw = generate(checkpoint, np.asarray(labels), seed=seed, batch_size=batch_size)
    result = model.transform(raw, labels)
    if sha256_file(checkpoint) != checkpoint_hash or sha256_file(calibrator) != map_hash:
        raise ValueError("checkpoint or calibrator changed during generation")
    metadata = {
        "checkpoint_sha256": checkpoint_hash, "calibrator_sha256": map_hash,
        "seed": seed, "batch_size": batch_size, "units": "kWh/interval",
        "resolution": model.n_features, "calibration": model.provenance,
        "post_generation_clipping": False,
        "support": "Fixed empirical-quantile calibration is bounded by training margins.",
        "limitations": "Synthetic profiles; no extrapolation, privacy, causal or publication guarantee.",
    }
    return result, metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--calibrator", required=True)
    parser.add_argument("--labels", nargs="+", type=int, required=True)
    parser.add_argument("--samples-per-label", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.samples_per_label < 1 or args.batch_size < 1:
        parser.error("sample count and batch size must be positive")
    if args.output.suffix.lower() != ".npz" or args.output.exists():
        parser.error("output must be a new .npz file")
    labels = np.repeat(np.asarray(args.labels, dtype=np.int64), args.samples_per_label)
    profiles, metadata = generate_profiles(args.checkpoint, args.calibrator, labels,
                                          seed=args.seed, batch_size=args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as handle:
        np.savez_compressed(handle, x=profiles, y=labels,
                            metadata=np.asarray(json.dumps(metadata, allow_nan=False)))
    print(json.dumps({"output": str(args.output), "profiles": len(labels),
                      "resolution": profiles.shape[1]}, allow_nan=False))


if __name__ == "__main__":
    main()
