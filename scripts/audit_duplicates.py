#!/usr/bin/env python3
"""Audit exact prepared-profile duplicates using only explicit train/validation NPZ inputs."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path

# The audit does not need threaded linear algebra.
for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_variable] = "1"

import numpy as np


KINDS = ("constant_zero", "other_constant", "nonconstant")
EXPECTED_DATES = np.arange("2024-01-01", "2025-01-01", dtype="datetime64[D]").astype("U10")
EXPECTED_DATES = EXPECTED_DATES[~np.isin(EXPECTED_DATES, ["2024-03-31", "2024-10-27"])]


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_input(path):
    # Never discover or load an additional split, even for metadata.
    if path.name.lower() == "test.npz" or path.resolve().name.lower() == "test.npz":
        raise ValueError("This audit must not open test.npz")
    with np.load(path, allow_pickle=False) as source:
        if set(source.files) != {"x", "y", "meter_id", "date"}:
            raise ValueError(f"{path.name}: expected x, y, meter_id, date")
        arrays = {name: source[name] for name in source.files}
    x = arrays["x"]
    if x.dtype != np.dtype("<f4") or x.ndim != 2 or x.shape[1] != 96 or not len(x):
        raise ValueError(f"{path.name}: expected nonempty little-endian float32 [N,96] x")
    if not np.isfinite(x).all():
        raise ValueError(f"{path.name}: x contains nonfinite values")
    for name in ("y", "meter_id", "date"):
        if arrays[name].shape != (len(x),):
            raise ValueError(f"{path.name}: {name} is not aligned with x")
    if arrays["y"].dtype != np.int64 or np.any((arrays["y"] < 0) | (arrays["y"] >= 64)):
        raise ValueError(f"{path.name}: expected int64 condition labels in [0,63]")
    if arrays["meter_id"].dtype.kind != "U" or arrays["date"].dtype.kind != "U":
        raise ValueError(f"{path.name}: meter_id and date must be Unicode vectors")
    arrays["x"] = np.ascontiguousarray(x, dtype="<f4")
    return arrays


def profile_kind(values):
    if np.all(values == 0):
        return "constant_zero"
    return "other_constant" if np.all(values == values.flat[0]) else "nonconstant"


def trace_bytes(arrays, rows):
    """Rows must already be sorted by date. No ID or condition enters this key."""
    date_bytes = arrays["date"][rows].astype("S10").tobytes(order="C")
    value_bytes = np.ascontiguousarray(arrays["x"][rows], dtype="<f4").tobytes(order="C")
    return date_bytes + value_bytes


def choose_two(count):
    return count * (count - 1) // 2


def audit_traces(splits):
    groups = defaultdict(list)
    kinds = {name: Counter() for name in splits}
    calendar_counts = Counter()
    for name, arrays in splits.items():
        order = np.lexsort((arrays["date"], arrays["meter_id"]))
        sorted_ids = arrays["meter_id"][order]
        starts = np.r_[0, np.flatnonzero(sorted_ids[1:] != sorted_ids[:-1]) + 1]
        stops = np.r_[starts[1:], len(order)]
        for start, stop in zip(starts, stops):
            rows = order[start:stop]
            meter = str(sorted_ids[start])
            dates = arrays["date"][rows]
            if len(np.unique(dates)) != len(dates):
                raise ValueError(f"{name} meter {meter}: repeated source dates")
            if not np.array_equal(dates, EXPECTED_DATES):
                raise ValueError(f"{name} meter {meter}: not the expected 364-date retained 2024 calendar")
            calendar_counts[name] += 1
            packed = trace_bytes(arrays, rows)
            digest = hashlib.sha256(packed).hexdigest()
            kind = profile_kind(arrays["x"][rows])
            kinds[name][kind] += 1
            if groups[digest]:
                previous = groups[digest][0]
                if packed != trace_bytes(splits[previous["split"]], previous["rows"]):
                    raise ValueError("SHA256 collision: date/value bytes differ; audit aborted")
            groups[digest].append({
                "split": name, "meter_id": meter, "rows": rows, "kind": kind,
                "technology_codes": np.unique(arrays["y"][rows] // 8).tolist(),
            })
    duplicate_groups = []
    summary = {kind: {
        "train_meters": kinds["train"][kind], "validation_meters": kinds["validation"][kind],
        "within_train_duplicate_groups": 0, "within_train_duplicate_pairs": 0,
        "within_validation_duplicate_groups": 0, "within_validation_duplicate_pairs": 0,
        "cross_split_duplicate_groups": 0, "cross_split_duplicate_pairs": 0,
        "train_meters_matching_validation": 0, "validation_meters_matching_train": 0,
    } for kind in KINDS}
    for digest, members in sorted(groups.items()):
        if len(members) < 2:
            continue
        train = [m for m in members if m["split"] == "train"]
        validation = [m for m in members if m["split"] == "validation"]
        stats = summary[members[0]["kind"]]
        for name, selected in (("train", train), ("validation", validation)):
            stats[f"within_{name}_duplicate_groups"] += int(len(selected) >= 2)
            stats[f"within_{name}_duplicate_pairs"] += choose_two(len(selected))
        stats["cross_split_duplicate_groups"] += int(bool(train and validation))
        stats["cross_split_duplicate_pairs"] += len(train) * len(validation)
        if train and validation:
            stats["train_meters_matching_validation"] += len(train)
            stats["validation_meters_matching_train"] += len(validation)
        first = members[0]
        reference_conditions = splits[first["split"]]["y"][first["rows"]]
        codes = sorted({code for member in members for code in member["technology_codes"]})
        duplicate_groups.append({
            "sha256": digest, "kind": first["kind"], "retained_days": 364, "slots_per_day": 96,
            "train_meters": sorted(m["meter_id"] for m in train),
            "validation_meters": sorted(m["meter_id"] for m in validation),
            "technology_consistent": len(codes) == 1,
            "technology_codes": codes,
            "technology_flags": [{"code": code, "HP": code // 4, "EV": (code // 2) % 2, "PV": code % 2} for code in codes],
            "condition_sequences_identical": all(np.array_equal(reference_conditions, splits[m["split"]]["y"][m["rows"]]) for m in members),
        })
    return {
        "expected_retained_dates": 364,
        "excluded_dst_dates": ["2024-03-31", "2024-10-27"],
        "meters_with_exact_expected_date_set": dict(calendar_counts),
        "distinct_date_aligned_traces": len(groups),
        "duplicate_groups": len(duplicate_groups),
        "by_kind": summary,
        "groups": duplicate_groups,
    }


def duplicate_metrics():
    return {
        "duplicate_groups": 0, "rows_in_duplicate_groups": 0, "redundant_rows": 0,
        "duplicate_pairs": 0, "same_meter_pairs": 0, "different_meter_pairs": 0,
        "groups_spanning_distinct_meters": 0, "rows_in_groups_spanning_distinct_meters": 0,
    }


def audit_days(splits):
    n_train = len(splits["train"]["x"])
    total = n_train + len(splits["validation"]["x"])
    hashes = np.empty(total, dtype="V32")
    kinds = np.empty(total, dtype=np.uint8)
    offset = 0
    for arrays in splits.values():
        x = arrays["x"]
        for start in range(0, len(x), 8192):
            stop = min(start + 8192, len(x))
            block = x[start:stop]
            zero = np.all(block == 0, axis=1)
            constant = np.all(block == block[:, :1], axis=1)
            kinds[offset + start:offset + stop] = np.where(zero, 0, np.where(constant, 1, 2))
        raw = memoryview(x).cast("B")
        row_bytes = x.shape[1] * x.dtype.itemsize
        for row in range(len(x)):
            hashes[offset + row] = np.void(hashlib.sha256(raw[row * row_bytes:(row + 1) * row_bytes]).digest())
        offset += len(x)
    unique_hashes, inverse, counts = np.unique(hashes, return_inverse=True, return_counts=True)
    group_kinds = np.empty(len(counts), dtype=np.uint8)
    group_kinds[inverse] = kinds
    train_counts = np.bincount(inverse[:n_train], minlength=len(counts))
    validation_counts = np.bincount(inverse[n_train:], minlength=len(counts))
    report = {}
    for code, kind in enumerate(KINDS):
        mask = group_kinds == code
        report[kind] = {
            "profile_rows": {"train": int((kinds[:n_train] == code).sum()), "validation": int((kinds[n_train:] == code).sum())},
            "distinct_profiles": {"union": int(mask.sum()), "train": int((mask & (train_counts > 0)).sum()), "validation": int((mask & (validation_counts > 0)).sum())},
            "within_train": duplicate_metrics(), "within_validation": duplicate_metrics(),
            "cross_split": {
                "duplicate_groups": 0, "train_rows_matching_validation": 0,
                "validation_rows_matching_train": 0, "duplicate_pairs": 0,
                "same_date_pairs": 0, "same_condition_pairs": 0, "same_technology_pairs": 0,
            },
        }
    order = np.argsort(inverse, kind="stable")
    starts = np.r_[0, np.cumsum(counts)]
    verified_rows = 0
    for group in np.flatnonzero(counts > 1):
        rows = order[starts[group]:starts[group + 1]]
        train_rows = rows[rows < n_train]
        validation_rows = rows[rows >= n_train] - n_train
        reference = splits["train"]["x"][train_rows[0]] if len(train_rows) else splits["validation"]["x"][validation_rows[0]]
        stats = report[KINDS[int(group_kinds[group])]]
        for name, selected in (("train", train_rows), ("validation", validation_rows)):
            arrays = splits[name]
            # Compare actual uint32 words, including signed-zero bits, after hashing.
            for start in range(0, len(selected), 8192):
                block = arrays["x"][selected[start:start + 8192]].view(np.uint32)
                if not np.all(block == reference.view(np.uint32)):
                    raise ValueError("SHA256 collision: daily float32 bytes differ; audit aborted")
            verified_rows += len(selected)
            if len(selected) >= 2:
                meter_counts = np.unique(arrays["meter_id"][selected], return_counts=True)[1]
                same_meter_pairs = sum(choose_two(int(count)) for count in meter_counts)
                metrics = stats[f"within_{name}"]
                metrics["duplicate_groups"] += 1
                metrics["rows_in_duplicate_groups"] += len(selected)
                metrics["redundant_rows"] += len(selected) - 1
                metrics["duplicate_pairs"] += choose_two(len(selected))
                metrics["same_meter_pairs"] += same_meter_pairs
                metrics["different_meter_pairs"] += choose_two(len(selected)) - same_meter_pairs
                metrics["groups_spanning_distinct_meters"] += int(len(meter_counts) > 1)
                metrics["rows_in_groups_spanning_distinct_meters"] += len(selected) if len(meter_counts) > 1 else 0
        if len(train_rows) and len(validation_rows):
            metrics = stats["cross_split"]
            metrics["duplicate_groups"] += 1
            metrics["train_rows_matching_validation"] += len(train_rows)
            metrics["validation_rows_matching_train"] += len(validation_rows)
            metrics["duplicate_pairs"] += len(train_rows) * len(validation_rows)
            for field, metric in (("date", "same_date_pairs"), ("y", "same_condition_pairs")):
                left = Counter(splits["train"][field][train_rows])
                right = Counter(splits["validation"][field][validation_rows])
                metrics[metric] += sum(count * right[value] for value, count in left.items())
            left = np.bincount(splits["train"]["y"][train_rows] // 8, minlength=8)
            right = np.bincount(splits["validation"]["y"][validation_rows] // 8, minlength=8)
            metrics["same_technology_pairs"] += int(left @ right)
    return {
        "total_profiles": total, "unique_exact_profiles_in_union": len(unique_hashes),
        "rows_in_repeated_hash_groups_confirmed_by_byte_comparison": verified_rows,
        "by_kind": report,
    }


def audit(train_path, validation_path):
    if train_path.resolve() == validation_path.resolve():
        raise ValueError("Train and validation inputs must be different files")
    splits = {"train": load_input(train_path), "validation": load_input(validation_path)}
    overlapping = np.intersect1d(np.unique(splits["train"]["meter_id"]), np.unique(splits["validation"]["meter_id"]))
    traces = audit_traces(splits)
    print("Completed date-aligned trace fingerprints", flush=True)
    days = audit_days(splits)
    inputs = {}
    for name, path in (("train", train_path), ("validation", validation_path)):
        arrays = splits[name]
        inputs[name] = {
            "basename": path.name, "sha256": file_sha256(path), "shape": list(arrays["x"].shape),
            "dtype": str(arrays["x"].dtype), "meters": len(np.unique(arrays["meter_id"])),
            "date_min": str(min(arrays["date"])), "date_max": str(max(arrays["date"])),
        }
    return {
        "schema_version": 1, "scope": "Only the explicit training and validation NPZ inputs were opened; no test data were opened",
        "inputs": inputs, "overlapping_ean_ids": overlapping.tolist(),
        "fingerprint_algorithm": {
            "daily": "SHA256 of one 96-slot row's exact C-order little-endian float32 bytes; date, ID and condition excluded",
            "yearly": "Sort each meter by ISOdate; SHA256(dates.astype('S10').tobytes(order='C') + ascontiguousarray(x,dtype='<f4').tobytes(order='C')); no header, separators, IDs or conditions",
            "verification": "Every repeated daily hash is verified against actual uint32 words; every repeated yearly hash is verified against full date+value bytes; collisions fail the audit",
            "classification": "Constant zero: all samples numerically zero. Other constant: every sample equals the first, excluding zero. Nonconstant: otherwise. Exact identity preserves signed-zero bits",
            "technology": "condition // 8 = HP*4 + EV*2 + PV",
            "pair_count_definition": "Within-split unordered row/meter pairs use n*(n-1)//2; cross-split pairs use n_train*n_validation",
        },
        "yearly_traces": traces, "daily_profiles": days,
        "interpretation": [
            "Identical daily profiles can occur naturally, especially zero or other constant profiles; duplicate days alone do not establish leakage.",
            "Duplicated nonconstant date-aligned whole retained-year traces across train and validation are a serious independence failure despite disjoint EAN IDs.",
            "These checks establish identity of prepared float32 net-energy inputs, not separate raw import/export channel identity or the reason for duplication.",
            "Prior validation results using trace-duplicated splits should be treated as exploratory; this report does not inspect or characterize the held-out test set.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True, help="Explicit training NPZ path")
    parser.add_argument("--validation", type=Path, required=True, help="Explicit validation NPZ path")
    parser.add_argument("--output", type=Path, required=True, help="JSON report path")
    args = parser.parse_args()
    if args.output.resolve() in (args.train.resolve(), args.validation.resolve()):
        parser.error("--output must not overwrite an input NPZ")
    try:
        result = audit(args.train, args.validation)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f"Duplicate audit failed: {error}\n")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
