"""Streaming, audited preparation of the Fluvius local-wall-clock release.

CSV rows must be contiguous by meter and ordered by local timestamp, except for
the repeated hour on the autumn DST transition. The archive is never extracted.
Prepared arrays contain physical energy, not normalized model inputs.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
import hashlib
import json
import math
from numbers import Real
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Iterator
from zipfile import ZipFile
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


COLUMNS = (
    "EAN_ID", "Datum", "Datum_Startuur", "Volume_Afname_KWh",
    "Volume_Injectie_KWh", "Warmtepomp_Indicator",
    "Elektrisch_Voertuig_Indicator", "PV_Installatie_Indicator",
    "Contract_Categorie",
)
INDICATORS = COLUMNS[5:8]
RELEASE_SHA256 = "934cc871e9fe18dbe30530472816d711ece878ec7cd8725d9d8417bbdb389bd8"
# Early stopping is safe only for these exact, hash-verified release bytes.
_RELEASE_GROUPS = {
    "P6269_Open_Data_geen_ZP.csv": 0,
    "P6269_Open_Data_enkel_ZP.csv": 1,
    "P6269_Open_Data_EV_geen_ZP.csv": 2,
    "P6269_Open_Data_EV_met_ZP.csv": 3,
    "P6269_Open_Data_WP_geen_ZP.csv": 4,
    "P6269_Open_Data_WP_met_ZP.csv": 5,
    "P6269_Open_Data_WP_EV_geen_ZP.csv": 6,
    "P6269_Open_Data_WP_EV_met_ZP.csv": 7,
}
TIMEZONE_POLICY = (
    "Europe/Brussels local CET/CEST wall-clock, as specified by the source legend; "
    "the literal trailing Z is removed, never interpreted as UTC. Preserve source "
    "date and quarter-hour slot. Drop all local DST transition days, and all "
    "incomplete, duplicate-slot, off-grid or invalid-energy days; no imputation."
)
EXCLUSION_REASONS = (
    "dst_transition_day", "incomplete_day", "duplicate_slots", "off_grid",
    "nonfinite_energy", "negative_source_energy",
)


@dataclass
class _MeterProfiles:
    meter_id: str
    technology: int
    x: np.ndarray
    y: np.ndarray
    dates: np.ndarray


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _iter_meters(source, chunksize: int, counts: Counter) -> Iterator[pd.DataFrame]:
    """Carry the complete trailing meter, even when it spans many chunks."""
    carry = None
    with pd.read_csv(source, dtype=str, chunksize=chunksize) as chunks:
        for chunk in chunks:
            if tuple(chunk.columns) != COLUMNS:
                raise ValueError(f"Unexpected CSV schema: {list(chunk.columns)!r}")
            counts["rows_read"] += len(chunk)
            if chunk.empty:
                continue
            ids = chunk["EAN_ID"]
            if ids.isna().any() or ids.str.strip().eq("").any():
                raise ValueError("Missing meter identifier")
            if not ids.eq(ids.str.strip()).all():
                raise ValueError("Whitespace in meter identifier")
            if carry is not None:
                chunk = pd.concat([carry, chunk], ignore_index=True)
            ids = chunk["EAN_ID"].to_numpy()
            starts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1]
            for start, stop in zip(starts[:-1], starts[1:]):
                yield chunk.iloc[start:stop]
            carry = chunk.iloc[starts[-1]:].copy()
        if carry is not None:
            yield carry


def _meter_metadata(frame: pd.DataFrame, meter_id: str) -> tuple[int, bool]:
    values = []
    for column in INDICATORS:
        data = frame[column]
        if not data.isin(("0", "1")).all():
            raise ValueError(f"Meter {meter_id}: {column} must be strict non-null binary 0/1")
        if data.nunique(dropna=False) != 1:
            raise ValueError(f"Meter {meter_id}: {column} must be constant")
        values.append(int(data.iloc[0]))
    contracts = frame["Contract_Categorie"]
    if contracts.isna().any() or contracts.str.strip().eq("").any():
        raise ValueError(f"Meter {meter_id}: missing contract metadata")
    if contracts.nunique(dropna=False) != 1:
        raise ValueError(f"Meter {meter_id}: contract metadata must be constant")
    return values[0] * 4 + values[1] * 2 + values[2], contracts.iloc[0] == "Residentieel"


@lru_cache(maxsize=4096)
def _calendar(date: str) -> tuple[bool, bool, int, int]:
    """Return transition, autumn transition, weekend, season for a local date."""
    local = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=ZoneInfo("Europe/Brussels"))
    before, after = local.utcoffset(), (local + timedelta(days=1)).utcoffset()
    season = (local.month % 12) // 3
    return before != after, after < before, int(local.weekday() >= 5), season


def _convert_meter(frame: pd.DataFrame, meter_id: str, technology: int,
                   resolution: int, target: str, counts: Counter,
                   exclusions: Counter) -> _MeterProfiles | None:
    dates = frame["Datum"]
    timestamps = frame["Datum_Startuur"]
    if not dates.str.fullmatch(r"\d{4}-\d{2}-\d{2}", na=False).all():
        raise ValueError(f"Meter {meter_id}: invalid source date format")
    if not timestamps.str.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?Z", na=False
    ).all():
        raise ValueError(f"Meter {meter_id}: unexpected timestamp format/offset; expected local literal Z")
    if not timestamps.str.slice(0, 10).eq(dates).all():
        raise ValueError(f"Meter {meter_id}: source date does not agree with timestamp date")
    try:
        local = pd.DatetimeIndex(pd.to_datetime(timestamps.str[:-1], format="ISO8601", errors="raise"))
    except (ValueError, TypeError) as error:
        raise ValueError(f"Meter {meter_id}: invalid local timestamp") from error
    day_values = dates.to_numpy()
    starts = np.r_[0, np.flatnonzero(day_values[1:] != day_values[:-1]) + 1]
    stops = np.r_[starts[1:], len(frame)]
    unique_days = day_values[starts]
    if np.any(unique_days[1:] <= unique_days[:-1]):
        raise ValueError(f"Meter {meter_id}: rows must be sorted by date and timestamp")
    times_ns = local.to_numpy(dtype="datetime64[ns]").view(np.int64)
    backward = np.flatnonzero(times_ns[1:] < times_ns[:-1])
    for row in backward:
        if day_values[row] != day_values[row + 1] or not _calendar(day_values[row])[1]:
            raise ValueError(f"Meter {meter_id}: rows must be sorted by timestamp")
    imports = pd.to_numeric(frame["Volume_Afname_KWh"], errors="coerce").to_numpy(dtype=np.float64)
    injection = pd.to_numeric(frame["Volume_Injectie_KWh"], errors="coerce").to_numpy(dtype=np.float64)
    invalid = ~(np.isfinite(imports) & np.isfinite(injection))
    negative = (imports < 0) | (injection < 0)
    slots = (local.hour * 4 + local.minute // 15).to_numpy()
    grid = ((local.minute % 15 == 0) & (local.second == 0)
            & (local.microsecond == 0) & (local.nanosecond == 0))
    energy = imports - injection if target == "net" else imports if target == "import" else injection
    profiles, labels, kept_dates = [], [], []
    for start, stop, date in zip(starts, stops, unique_days):
        counts["days_seen"] += 1
        transition, _, weekend, season = _calendar(date)
        day_slots = slots[start:stop]
        unique_slots = np.unique(day_slots)
        reasons = []
        if transition:
            reasons.append("dst_transition_day")
        if stop - start != 96 or len(unique_slots) != 96:
            reasons.append("incomplete_day")
        if len(unique_slots) != stop - start:
            reasons.append("duplicate_slots")
        if not grid[start:stop].all():
            reasons.append("off_grid")
        if invalid[start:stop].any():
            reasons.append("nonfinite_energy")
        if negative[start:stop].any():
            reasons.append("negative_source_energy")
        if reasons:
            counts["days_excluded"] += 1
            exclusions.update(reasons)
            continue
        profile = energy[start:stop]
        if resolution == 24:
            profile = profile.reshape(24, 4).sum(axis=1)
        if np.abs(profile).max() > np.finfo(np.float32).max:
            raise ValueError(f"Meter {meter_id}: energy exceeds float32 storage range")
        profiles.append(profile)
        labels.append(((technology * 2) + weekend) * 4 + season)
        kept_dates.append(date)
    if not profiles:
        counts["meters_without_valid_days"] += 1
        return None
    counts["profiles_kept"] += len(profiles)
    counts["meters_kept"] += 1
    return _MeterProfiles(meter_id, technology, np.asarray(profiles, dtype=np.float32),
                          np.asarray(labels, dtype=np.int64), np.asarray(kept_dates, dtype="U10"))


def _split_meters(meters: list[_MeterProfiles], seed: int,
                  train_fraction: float, val_fraction: float) -> dict[str, list[_MeterProfiles]]:
    rng = np.random.default_rng(seed)
    splits: dict[str, list[_MeterProfiles]] = {name: [] for name in ("train", "val", "test")}
    for technology in sorted({meter.technology for meter in meters}):
        group = sorted((m for m in meters if m.technology == technology), key=lambda m: m.meter_id)
        n = len(group)
        if n < 3:
            raise ValueError(f"Technology group {technology} requires at least 3 eligible meters; got {n}")
        n_train = min(n - 2, max(1, int(np.floor(n * train_fraction))))
        n_val = min(n - n_train - 1, max(1, int(np.floor(n * val_fraction))))
        order = rng.permutation(n)
        for split, indices in zip(splits, (order[:n_train], order[n_train:n_train + n_val],
                                          order[n_train + n_val:])):
            splits[split].extend(group[int(index)] for index in indices)
    if any(not records for records in splits.values()):
        raise ValueError("All three splits require eligible residential meters")
    return splits


def _pack(meters: list[_MeterProfiles]) -> dict[str, np.ndarray]:
    return {
        "x": np.concatenate([m.x for m in meters]),
        "y": np.concatenate([m.y for m in meters]),
        "meter_id": np.concatenate([np.full(len(m.x), m.meter_id, dtype=f"U{len(m.meter_id)}")
                                     for m in meters]),
        "date": np.concatenate([m.dates for m in meters]),
    }


def prepare_archive(archive_path, output_dir, *, resolution: int = 96,
                    target: str = "net", seed: int = 42, train_fraction: float = .7,
                    val_fraction: float = .15, chunksize: int = 250_000,
                    max_meters_per_group: int | None = None) -> dict:
    """Prepare a ZIP in bounded raw-row memory, with meter-disjoint splits.

    ``max_meters_per_group`` deliberately chooses the earliest eligible meters
    encountered per technology and is only for smoke runs. Exact release bytes
    permit early stopping of homogeneous CSVs; other archives are scanned fully.
    Exclusion reasons are nonexclusive; ``days_excluded`` counts unique days.
    Only the train split fits a scalar [0,1] min/max transform. The returned
    summary is also written as metadata.json; x arrays retain original kWh.
    """
    if type(resolution) is not int or resolution not in (24, 96):
        raise ValueError("resolution must be 24 or 96")
    if target not in ("net", "import", "injection"):
        raise ValueError("target must be net, import or injection")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v)
           for v in (train_fraction, val_fraction)):
        raise ValueError("Split fractions must be finite numbers")
    train_fraction, val_fraction = float(train_fraction), float(val_fraction)
    if not (0 < train_fraction < 1 and 0 < val_fraction < 1 and train_fraction + val_fraction < 1):
        raise ValueError("Split fractions must be positive and leave a positive test fraction")
    if type(chunksize) is not int or chunksize < 1:
        raise ValueError("chunksize must be a positive integer")
    if max_meters_per_group is not None and (type(max_meters_per_group) is not int
                                            or max_meters_per_group < 3):
        raise ValueError("max_meters_per_group must be at least 3")
    archive_path, output_dir = Path(archive_path), Path(output_dir)
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f"Refusing to overwrite nonempty prepared directory: {output_dir}")
    digest = _sha256(archive_path)
    counts = Counter({name: 0 for name in (
        "rows_read", "rows_processed", "meters_seen", "meters_kept", "days_seen",
        "days_excluded", "profiles_kept", "meters_without_valid_days",
        "nonresidential_meters", "sampled_out_meters",
    )})
    exclusions = Counter({reason: 0 for reason in EXCLUSION_REASONS})
    meters, seen, selected, early_stopped = [], set(), Counter(), []
    observed_groups = set()
    with ZipFile(archive_path) as archive:
        infos = archive.infolist()
        member_names = [info.filename for info in infos]
        if len(member_names) != len(set(member_names)):
            raise ValueError("Duplicate archive member names")
        for info in infos:
            path = PurePosixPath(info.filename)
            if path.is_absolute() or ".." in path.parts or "\\" in info.filename:
                raise ValueError(f"Unsafe archive member: {info.filename}")
        csvs = sorted(info.filename for info in infos if info.filename.lower().endswith(".csv"))
        if not csvs:
            raise ValueError("Archive contains no CSV files")
        for name in csvs:
            known_group = _RELEASE_GROUPS.get(PurePosixPath(name).name) if digest == RELEASE_SHA256 else None
            with archive.open(name) as source:
                frames = _iter_meters(source, chunksize, counts)
                try:
                    for frame in frames:
                        meter_id = str(frame["EAN_ID"].iloc[0])
                        if meter_id in seen:
                            raise ValueError(f"Duplicate meter {meter_id}: IDs must be contiguous and unique across files")
                        seen.add(meter_id)
                        counts["meters_seen"] += 1
                        technology, residential = _meter_metadata(frame, meter_id)
                        if known_group is not None and technology != known_group:
                            raise ValueError(f"Unexpected technology metadata in pinned release member {name}")
                        if not residential:
                            counts["nonresidential_meters"] += 1
                            continue
                        observed_groups.add(technology)
                        if max_meters_per_group is not None and selected[technology] >= max_meters_per_group:
                            counts["sampled_out_meters"] += 1
                            continue
                        counts["rows_processed"] += len(frame)
                        record = _convert_meter(frame, meter_id, technology, resolution, target, counts, exclusions)
                        if record is not None:
                            meters.append(record)
                            selected[technology] += 1
                        if (known_group is not None and max_meters_per_group is not None
                                and selected[known_group] >= max_meters_per_group):
                            early_stopped.append(name)
                            break
                finally:
                    frames.close()
    # A represented residential stratum with no usable meters must not disappear.
    if not meters or any(selected[group] < 3 for group in observed_groups):
        raise ValueError("Every observed residential technology group requires at least 3 eligible meters")
    splits = _split_meters(meters, seed, train_fraction, val_fraction)
    minimum = min(float(m.x.min()) for m in splits["train"])
    maximum = max(float(m.x.max()) for m in splits["train"])
    scaler = {"minimum": minimum, "maximum": maximum, "fitted_on": "train",
              "range": [0, 1], "constant_range": minimum == maximum,
              "transform": "(x - minimum) / (maximum - minimum); use denominator 1 if constant"}
    summary = {
        "schema_version": 1, "source_sha256": digest,
        "source_is_pinned_release": digest == RELEASE_SHA256,
        "source_files": csvs, "resolution": resolution, "target": target,
        "target_definition": {"net": "Volume_Afname_KWh - Volume_Injectie_KWh",
                              "import": "Volume_Afname_KWh", "injection": "Volume_Injectie_KWh"}[target],
        "units": "kWh/interval", "interval_hours": 24 / resolution,
        "timezone_policy": TIMEZONE_POLICY, "split_seed": seed,
        "split_fractions_requested": {"train": train_fraction, "val": val_fraction,
                                      "test": 1 - train_fraction - val_fraction},
        "split_method": "Seeded permutation of sorted meter IDs within technology strata; floor train/val sizes, minimum one meter per split",
        "label_formula": "((HP*4 + EV*2 + PV)*2 + weekend)*4 + season",
        "condition_dim": 64, "season_codes": {"winter": 0, "spring": 1, "summer": 2, "autumn": 3},
        "counts": dict(counts), "exclusions": dict(exclusions),
        "exclusion_count_policy": "Reasons overlap; counts.days_excluded counts each excluded day once",
        "sampling": {"mode": "smoke" if max_meters_per_group is not None else "full",
                     "max_meters_per_group": max_meters_per_group,
                     "selection": "Earliest eligible meters in sorted archive-member/source order; selection bias, not a representative research sample"
                                  if max_meters_per_group is not None else "All eligible residential meters",
                     "early_stopped_files": early_stopped,
                     "complete_archive_validation": max_meters_per_group is None,
                     "validation_scope": "Selected meters only; unselected/unread rows are not fully audited and duplicate IDs in unread tails cannot be ruled out"
                                         if max_meters_per_group is not None else "All meter IDs and metadata; all residential timestamp/energy rows; nonresidential rows excluded after metadata validation"},
        "splits": {},
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}-", dir=output_dir.parent) as staging:
        stage = Path(staging)
        for name, records in splits.items():
            arrays = _pack(records)
            np.savez_compressed(stage / f"{name}.npz", **arrays)
            summary["splits"][name] = {
                "profiles": len(arrays["x"]), "meters": len(records),
                "sha256": _sha256(stage / f"{name}.npz"),
                "meter_ids": [record.meter_id for record in records],
                "technology_meter_counts": dict(sorted(Counter(str(m.technology) for m in records).items())),
                "condition_counts": dict(sorted(Counter(str(int(y)) for y in arrays["y"]).items(), key=lambda p: int(p[0]))),
                "date_min": min(str(record.dates[0]) for record in records),
                "date_max": max(str(record.dates[-1]) for record in records),
            }
        (stage / "scaler.json").write_text(json.dumps(scaler, indent=2) + "\n")
        (stage / "metadata.json").write_text(json.dumps(summary, indent=2) + "\n")
        os.replace(stage, output_dir)
    return summary


def load_split(path) -> dict[str, np.ndarray]:
    """Load a prepared split without pickle; fail on malformed data."""
    with np.load(path, allow_pickle=False) as archive:
        if len(archive.files) != 4 or set(archive.files) != {"x", "y", "meter_id", "date"}:
            raise ValueError("Prepared split must contain x, y, meter_id and date")
        result = {name: archive[name] for name in archive.files}
    x, y = result["x"], result["y"]
    if x.dtype != np.float32 or x.ndim != 2 or x.shape[1] not in (24, 96) or not len(x) or not np.isfinite(x).all():
        raise ValueError("Prepared x must be nonempty finite [N,24] or [N,96]")
    if x.dtype != np.float32 or y.dtype != np.int64 or y.shape != (len(x),) or np.any((y < 0) | (y >= 64)):
        raise ValueError("Prepared x/y must have float32/int64 dtype and valid condition labels")
    for key in ("meter_id", "date"):
        if result[key].shape != (len(x),) or result[key].dtype.kind != "U":
            raise ValueError(f"Prepared {key} must be a unicode vector aligned with x")
    ids, dates = result["meter_id"], result["date"]
    if np.any(ids == "") or np.any(np.char.strip(ids) != ids):
        raise ValueError("Prepared meter IDs must be nonempty without surrounding whitespace")
    try:
        parsed = dates.astype("datetime64[D]")
    except ValueError as error:
        raise ValueError("Prepared dates must be exact YYYY-MM-DD calendar dates") from error
    if np.isnat(parsed).any() or np.any(np.datetime_as_string(parsed, unit="D") != dates):
        raise ValueError("Prepared dates must be exact YYYY-MM-DD calendar dates")
    identity = np.rec.fromarrays([ids, dates], names=["meter_id", "date"])
    if len(np.unique(identity)) != len(x):
        raise ValueError("Prepared split contains duplicate meter/date rows")
    return result
