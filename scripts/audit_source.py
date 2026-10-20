#!/usr/bin/env python3
"""Stream a Fluvius 2024 ZIP and write compact, reproducible source-data checks."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

from audit_calendar import calendar_template, sha256


COLUMNS = [
    "EAN_ID", "Datum", "Datum_Startuur", "Volume_Afname_KWh", "Volume_Injectie_KWh",
    "Warmtepomp_Indicator", "Elektrisch_Voertuig_Indicator", "PV_Installatie_Indicator",
    "Contract_Categorie",
]
VOLUMES = COLUMNS[3:5]
METADATA = COLUMNS[5:]


def audit_file(source, member, chunk_size):
    lookup, _ = calendar_template()
    date_lookup = {t.strftime("%Y-%m-%d"): i for i, t in enumerate(pd.date_range("2024-01-01", "2024-12-31"))}
    report = {
        "file": member, "rows": 0, "null_cells": 0, "unknown_timestamp_rows": 0,
        "date_timestamp_mismatch_rows": 0, "adjacent_duplicate_timestamp_rows": 0,
        "out_of_order_adjacent_rows": 0, "invalid_indicator_cells": 0,
        "non_residential_rows": 0, "nonpv_positive_export_rows": 0,
        "nonpv_export_kwh": 0.0,
    }
    meter_counts, day_counts = Counter(), Counter()
    metadata = defaultdict(set)
    nonpv_export_meters = set()
    values = {
        column: {"min": None, "max": None, "sum": 0.0, "negative": 0, "zero": 0, "nonfinite": 0, "nonnumeric": 0}
        for column in VOLUMES
    }
    last = None
    with source.open(member) as stream:
        for frame in pd.read_csv(stream, chunksize=chunk_size):
            if list(frame.columns) != COLUMNS:
                raise ValueError(f"{member}: unexpected CSV schema {list(frame.columns)}")
            if frame.empty:
                continue
            ids = pd.to_numeric(frame.EAN_ID, errors="raise").to_numpy()
            if not np.isfinite(ids).all() or not np.equal(ids, np.floor(ids)).all():
                raise ValueError(f"{member}: EAN_ID must be a nonmissing integer")
            ids = ids.astype(np.int64)
            report["rows"] += len(frame)
            report["null_cells"] += int(frame.isna().sum().sum())
            meter_counts.update({int(k): int(v) for k, v in frame.groupby("EAN_ID").size().items()})
            day_counts.update({(int(k[0]), str(k[1])): int(v) for k, v in frame.groupby(["EAN_ID", "Datum"], dropna=False).size().items()})
            times = frame.Datum_Startuur.map(lookup)
            dates = frame.Datum.map(date_lookup)
            report["unknown_timestamp_rows"] += int(times.isna().sum())
            report["date_timestamp_mismatch_rows"] += int((dates != times // 96).sum())
            slots = times.fillna(-1).to_numpy(dtype=np.int64)
            previous_ids, current_ids = ids[:-1], ids[1:]
            previous_slots, current_slots = slots[:-1], slots[1:]
            valid_pairs = (previous_slots >= 0) & (current_slots >= 0)
            report["adjacent_duplicate_timestamp_rows"] += int((valid_pairs & (previous_ids == current_ids) & (previous_slots == current_slots)).sum())
            report["out_of_order_adjacent_rows"] += int((valid_pairs & ((current_ids < previous_ids) | ((current_ids == previous_ids) & (current_slots < previous_slots)))).sum())
            first = (int(ids[0]), int(slots[0]))
            if last is not None and last[1] >= 0 and first[1] >= 0:
                report["adjacent_duplicate_timestamp_rows"] += int(first == last)
                report["out_of_order_adjacent_rows"] += int(first < last)
            last = (int(ids[-1]), int(slots[-1]))
            report["invalid_indicator_cells"] += int((~frame[METADATA[:3]].isin([0, 1])).sum().sum())
            report["non_residential_rows"] += int((frame.Contract_Categorie != "Residentieel").sum())
            for row in frame[["EAN_ID"] + METADATA].drop_duplicates().itertuples(index=False, name=None):
                metadata[int(row[0])].add(tuple(None if pd.isna(value) else value for value in row[1:]))
            for column in VOLUMES:
                numeric = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
                finite = numeric[np.isfinite(numeric)]
                stats = values[column]
                stats["nonnumeric"] += int((pd.isna(numeric) & frame[column].notna().to_numpy()).sum())
                stats["nonfinite"] += int((~np.isfinite(numeric)).sum())
                stats["negative"] += int((numeric < 0).sum())
                stats["zero"] += int((numeric == 0).sum())
                stats["sum"] += float(finite.sum())
                if len(finite):
                    stats["min"] = float(finite.min()) if stats["min"] is None else min(stats["min"], float(finite.min()))
                    stats["max"] = float(finite.max()) if stats["max"] is None else max(stats["max"], float(finite.max()))
                if column == "Volume_Injectie_KWh":
                    nonpv = (frame.PV_Installatie_Indicator.to_numpy() == 0) & (numeric > 0) & np.isfinite(numeric)
                    report["nonpv_positive_export_rows"] += int(nonpv.sum())
                    report["nonpv_export_kwh"] += float(numeric[nonpv].sum())
                    nonpv_export_meters.update(int(value) for value in ids[nonpv])
    if not meter_counts:
        raise ValueError(f"{member}: CSV contains no observations")
    profile_counts = Counter(profile for profiles in metadata.values() for profile in profiles)
    unusual_dates = defaultdict(Counter)
    for (_, date), count in day_counts.items():
        if count != 96:
            unusual_dates[date][count] += 1
    report.update({
        "meters": len(meter_counts), "meter_id_min": min(meter_counts), "meter_id_max": max(meter_counts),
        "rows_per_meter_distribution": dict(sorted(Counter(meter_counts.values()).items())),
        "days_per_meter_distribution": dict(sorted(Counter(Counter(k[0] for k in day_counts).values()).items())),
        "rows_per_day_distribution": dict(sorted(Counter(day_counts.values()).items())),
        "nonstandard_dates": {date: dict(sorted(counts.items())) for date, counts in sorted(unusual_dates.items())},
        "metadata_conflict_meters": sum(len(profiles) != 1 for profiles in metadata.values()),
        "metadata_profiles": [{**dict(zip(METADATA, profile)), "meters": count} for profile, count in profile_counts.items()],
        "nonpv_positive_export_meters": len(nonpv_export_meters), "volumes": values,
    })
    return report, set(meter_counts)


def audit(archive, chunk_size, calendar_report=None):
    digest = sha256(archive)
    calendar = None
    if calendar_report is not None:
        calendar = json.loads(calendar_report.read_text())
        if calendar.get("source_sha256") != digest:
            raise ValueError("Calendar report belongs to a different source SHA256")
    reports, all_ids = [], Counter()
    with zipfile.ZipFile(archive) as source:
        members = [info for info in source.infolist() if info.filename.lower().endswith(".csv")]
        if not members:
            raise ValueError("Archive contains no CSV files")
        csv_bytes = sum(info.file_size for info in members)
        for member in members:
            report, ids = audit_file(source, member.filename, chunk_size)
            report["uncompressed_bytes"] = member.file_size
            reports.append(report)
            all_ids.update(ids)
            print(f"Audited {Path(member.filename).name}: {report['rows']:,} rows", flush=True)
    day_counts = Counter()
    for report in reports:
        day_counts.update(report["rows_per_day_distribution"])
    result = {
        "schema_version": 1, "archive_basename": archive.name, "source_sha256": digest,
        "year": 2024, "columns": COLUMNS,
        "units": "kWh during the quarter-hour following source-local Datum_Startuur",
        "timestamp_interpretation": "CET/CEST local wall time despite literal Z suffix",
        "csv_files": len(reports), "csv_uncompressed_bytes": csv_bytes,
        "total_rows": sum(r["rows"] for r in reports), "unique_meters": len(all_ids),
        "meter_id_min": min(all_ids), "meter_id_max": max(all_ids),
        "meter_ids_contiguous": len(all_ids) == max(all_ids) - min(all_ids) + 1,
        "overlapping_meter_ids_count": sum(count > 1 for count in all_ids.values()),
        "rows_per_day_distribution": dict(sorted(day_counts.items())),
        "retained_96_slot_meter_days": day_counts[96],
        "retained_96_slot_rows": day_counts[96] * 96,
        "full_year_import_kwh": sum(r["volumes"][VOLUMES[0]]["sum"] for r in reports),
        "full_year_export_kwh": sum(r["volumes"][VOLUMES[1]]["sum"] for r in reports),
        "nonpv_positive_export_rows": sum(r["nonpv_positive_export_rows"] for r in reports),
        "nonpv_export_kwh": sum(r["nonpv_export_kwh"] for r in reports),
        "files": reports,
    }
    if calendar is not None:
        source_counts = {r["file"]: (r["rows"], r["meters"]) for r in reports}
        calendar_counts = {r["file"]: (r["rows"], r["meters"]) for r in calendar["files"]}
        if source_counts != calendar_counts:
            raise ValueError("Calendar report file, row or meter counts do not match the source scan")
        result["calendar_validation"] = calendar
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True, help="Source ZIP archive")
    parser.add_argument("--output", type=Path, required=True, help="Compact JSON output path")
    parser.add_argument("--calendar-report", type=Path, help="Attach independently generated audit_calendar.py JSON, verified by SHA256 and counts")
    parser.add_argument("--chunk-size", type=int, default=400000, help="CSV rows per chunk (default: 400000)")
    args = parser.parse_args()
    if args.chunk_size < 1:
        parser.error("--chunk-size must be positive")
    if args.output.resolve() == args.archive.resolve():
        parser.error("--output must not overwrite the source archive")
    try:
        result = audit(args.archive, args.chunk_size, args.calendar_report)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    except (ValueError, OSError, zipfile.BadZipFile, KeyError) as error:
        parser.exit(2, f"Audit failed: {error}\n")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
