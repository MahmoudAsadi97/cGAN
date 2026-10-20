#!/usr/bin/env python3
"""Curate exact complete-trace identities with TRAIN > VAL > TEST priority.

The original test split is loaded solely for identity fingerprinting and writing
retained rows. This command never performs model evaluation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Also permit execution directly from an uninstalled source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fluvius_cgan.curation import curate_prepared


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True,
                        help="Source prepared-data directory containing train, val, and test splits")
    parser.add_argument("--output", type=Path, required=True,
                        help="Destination for the curated prepared-data copy")
    args = parser.parse_args(argv)
    try:
        metadata = curate_prepared(args.prepared, args.output)
        curation = metadata["curation"]
        summary = {
            "output": str(args.output.resolve()),
            "before": curation["before"],
            "after": curation["after"],
            "duplicates_removed": curation["duplicates_removed"],
            "test_access_scope": curation["test_access_scope"],
        }
        print(json.dumps(summary, allow_nan=False, sort_keys=True, separators=(",", ":")))
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f"Prepared-data curation failed: {error}\n")


if __name__ == "__main__":
    main()
