"""Prepare a five-seed sealed evaluation bundle without opening/hashing TEST."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fluvius_cgan.frozen_test import freeze_test


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--run", action="append", nargs=2, required=True, metavar=("CHECKPOINT", "CALIBRATOR"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--decision-rationale", required=True)
    parser.add_argument("--protocol", action="append", required=True, type=Path)
    parser.add_argument("--evidence", action="append", default=[], type=Path)
    parser.add_argument("--no-utility", action="store_true", help="Explicitly omit utility before sealing (default: utility on)")
    parser.add_argument("--bootstrap-replicates", type=int, default=500, help="0 explicitly omits uncertainty")
    args = parser.parse_args(argv)
    result = freeze_test(args.prepared, args.run, args.output, decision_rationale=args.decision_rationale,
        protocol_files=args.protocol, evidence_files=args.evidence, utility=not args.no_utility,
        bootstrap_replicates=args.bootstrap_replicates)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
