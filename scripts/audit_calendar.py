#!/usr/bin/env python3
"""Audit the supplied Fluvius 2024 local-clock calendar without extracting CSVs."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def calendar_template():
    """Return nominal local slots, preserving the archive's sorted autumn folds."""
    dates = pd.date_range("2024-01-01", "2024-12-31 23:45", freq="15min")
    lookup = {t.strftime("%Y-%m-%dT%H:%M:%S.000Z"): i for i, t in enumerate(dates)}
    expected = []
    for stamp, index in lookup.items():
        if stamp.startswith("2024-03-31T02:"):
            continue
        expected.append(index)
        if stamp.startswith("2024-10-27T02:"):
            expected.append(index)
    return lookup, np.asarray(expected, dtype=np.int64)


def audit(archive, chunk_size):
    lookup, expected = calendar_template()
    reports = []
    with zipfile.ZipFile(archive) as source:
        members = [n for n in source.namelist() if n.lower().endswith(".csv")]
        if not members:
            raise ValueError("Archive contains no CSV files")
        for member in members:
            counts = Counter()
            mismatches = 0
            unknown = 0
            with source.open(member) as stream:
                for frame in pd.read_csv(
                    stream, usecols=["EAN_ID", "Datum_Startuur"], chunksize=chunk_size
                ):
                    if frame.empty:
                        continue
                    ids = pd.to_numeric(frame.EAN_ID, errors="raise").to_numpy()
                    if not np.isfinite(ids).all() or not np.equal(ids, np.floor(ids)).all():
                        raise ValueError(f"{member}: EAN_ID must be a nonmissing integer")
                    offsets = frame.EAN_ID.map(lambda value: counts[int(value)]).to_numpy()
                    positions = frame.groupby("EAN_ID", sort=False).cumcount().to_numpy() + offsets
                    actual = frame.Datum_Startuur.map(lookup)
                    unknown += int(actual.isna().sum())
                    mismatches += int((actual.to_numpy() != expected[positions % len(expected)]).sum())
                    counts.update({int(k): int(v) for k, v in frame.groupby("EAN_ID").size().items()})
            if not counts:
                raise ValueError(f"{member}: CSV contains no observations")
            reports.append({
                "file": member,
                "rows": sum(counts.values()),
                "meters": len(counts),
                "unknown_timestamp_rows": unknown,
                "template_slot_mismatches": mismatches,
                "meters_with_unexpected_row_count": sum(n != len(expected) for n in counts.values()),
            })
            print(f"Checked calendar: {Path(member).name}", flush=True)
    return {
        "schema_version": 1,
        "archive_basename": archive.name,
        "source_sha256": sha256(archive),
        "year": 2024,
        "interpretation": "CET/CEST source-local wall-clock labels despite literal Z suffix",
        "expected_rows_per_meter": len(expected),
        "expected_normal_days_per_meter": 364,
        "dst_days": {
            "2024-03-31": {"rows_per_meter": 92, "absent_slots": ["02:00", "02:15", "02:30", "02:45"]},
            "2024-10-27": {"rows_per_meter": 100, "repeated_slots": ["02:00", "02:15", "02:30", "02:45"]},
        },
        "files": reports,
        "total_rows_checked": sum(r["rows"] for r in reports),
        "total_template_slot_mismatches": sum(r["template_slot_mismatches"] for r in reports),
        "total_unknown_timestamp_rows": sum(r["unknown_timestamp_rows"] for r in reports),
        "total_meters_with_unexpected_row_count": sum(r["meters_with_unexpected_row_count"] for r in reports),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True, help="Source ZIP archive")
    parser.add_argument("--output", type=Path, required=True, help="Compact JSON output path")
    parser.add_argument("--chunk-size", type=int, default=500000, help="CSV rows per chunk (default: 500000)")
    args = parser.parse_args()
    if args.chunk_size < 1:
        parser.error("--chunk-size must be positive")
    if args.output.resolve() == args.archive.resolve():
        parser.error("--output must not overwrite the source archive")
    try:
        result = audit(args.archive, args.chunk_size)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    except (ValueError, OSError, zipfile.BadZipFile, KeyError) as error:
        parser.exit(2, f"Audit failed: {error}\n")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
