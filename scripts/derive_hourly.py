"""Derive hourly TRAIN/VAL from canonical curated96; never access TEST contents."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fluvius_cgan.derived import derive_hourly


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True, type=Path, help="Direct pinned curated-v2 96-slot preparation")
    parser.add_argument("--output", required=True, type=Path, help="New hourly TRAIN/VAL-only output directory")
    parser.add_argument("--chunk-rows", type=int, default=4096, help="Rows per bounded-memory transformation chunk")
    args = parser.parse_args(argv)
    try:
        metadata = derive_hourly(args.parent, args.output, chunk_rows=args.chunk_rows)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f"error: {error}\n")
    print(json.dumps({"output": str(args.output), "resolution": metadata["resolution"],
                      "splits": {key: {"profiles": value["profiles"], "meters": value["meters"]}
                                 for key, value in metadata["splits"].items()},
                      "test24_materialized": False, "qualification": metadata["derived"]["qualification"]}, indent=2))


if __name__ == "__main__":
    main()
