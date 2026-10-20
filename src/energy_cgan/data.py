"""Stream the Fluvius 2024 ZIP into auditable, meter-disjoint daily profiles.

The release dictionary says CET/CEST wall time although timestamp strings end in
``Z``. The default deliberately follows that dictionary without shifting clock
times. Brussels clock-change days are excluded instead of interpolated.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date, datetime, time, timedelta
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
from typing import Any
from zoneinfo import ZoneInfo
import zipfile

import numpy as np
import pandas as pd


_COLUMNS = [
    "EAN_ID", "Datum", "Datum_Startuur", "Volume_Afname_KWh",
    "Volume_Injectie_KWh", "Warmtepomp_Indicator",
    "Elektrisch_Voertuig_Indicator", "PV_Installatie_Indicator", "Contract_Categorie",
]
_EQUIPMENT = ["PV_Installatie_Indicator", "Elektrisch_Voertuig_Indicator", "Warmtepomp_Indicator"]
_DAY_NS = 86_400_000_000_000
_QUARTER_NS = 900_000_000_000


def _config(config: dict[str, Any] | None) -> dict[str, Any]:
    settings = {
        "target": "import", "resolution_minutes": 15,
        "conditioning": "equipment_calendar", "seed": 42,
        "split_fractions": [0.7, 0.15, 0.15], "chunksize": 100_000,
        "max_meters_per_file": None, "timestamp_policy": "source_wall_clock",
        "timezone": "Europe/Brussels",
    }
    if config is not None and not isinstance(config, dict):
        raise ValueError("Preparation config must be a mapping")
    unknown = set(config or {}) - set(settings)
    if unknown:
        raise ValueError(f"Unknown preparation config keys: {sorted(unknown)}")
    settings.update(config or {})
    for field in ("resolution_minutes", "chunksize", "seed", "max_meters_per_file"):
        value = settings[field]
        if field == "max_meters_per_file" and value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{field} must be an integer")
        if value < (0 if field == "seed" else 1):
            raise ValueError(f"{field} must be {'nonnegative' if field == 'seed' else 'positive'}")
        settings[field] = int(value)
    if settings["target"] not in {"import", "export", "net"}:
        raise ValueError("target must be import, export, or net")
    if settings["resolution_minutes"] not in {15, 60}:
        raise ValueError("resolution_minutes must be 15 or 60")
    if settings["conditioning"] not in {"calendar", "pv_ev_calendar", "equipment_calendar"}:
        raise ValueError("Unknown conditioning; use calendar, pv_ev_calendar, or equipment_calendar")
    if settings["timestamp_policy"] not in {"source_wall_clock", "utc_to_local"}:
        raise ValueError("timestamp_policy must be source_wall_clock or utc_to_local")
    ZoneInfo(settings["timezone"])
    fractions = np.asarray(settings["split_fractions"], dtype=float)
    if fractions.shape != (3,) or not np.isfinite(fractions).all() or (fractions <= 0).any() or not np.isclose(fractions.sum(), 1):
        raise ValueError("split_fractions must contain three positive fractions summing to one")
    settings["split_fractions"] = fractions.tolist()
    settings["seed"] = int(settings["seed"])
    return settings


def _source_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _label(day: str, equipment: tuple[int, int, int], conditioning: str) -> str:
    calendar_day = date.fromisoformat(day)
    season = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
              6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}[calendar_day.month]
    parts = []
    if conditioning in {"pv_ev_calendar", "equipment_calendar"}:
        parts.extend([f"PV={equipment[0]}", f"EV={equipment[1]}"])
    if conditioning == "equipment_calendar":
        parts.append(f"HP={equipment[2]}")
    parts.extend([f"season={season}", f"weekend={int(calendar_day.weekday() >= 5)}"])
    return "|".join(parts)


def _split_meters(
    equipment_by_meter: dict[str, tuple[int, int, int]],
    fractions: list[float], seed: int,
) -> dict[str, str]:
    strata: dict[tuple[int, int, int], list[str]] = defaultdict(list)
    for meter, equipment in equipment_by_meter.items():
        strata[equipment].append(meter)
    rng = np.random.default_rng(seed)
    result = {}
    for equipment in sorted(strata):
        meters = sorted(strata[equipment])
        n = len(meters)
        if n < 3:
            raise ValueError(
                f"Equipment stratum PV/EV/HP={equipment} has {n} usable meters; "
                "at least 3 meters are required for disjoint train/val/test splits. "
                "Increase max_meters_per_file or use the full release."
            )
        # Largest-remainder allocation with a guaranteed meter in every split.
        desired = np.asarray(fractions) * n
        counts = np.maximum(np.floor(desired).astype(int), 1)
        while counts.sum() > n:
            candidates = np.where(counts > 1)[0]
            index = candidates[np.argmax((counts - desired)[candidates])]
            counts[index] -= 1
        while counts.sum() < n:
            counts[np.argmax(desired - counts)] += 1
        shuffled = rng.permutation(meters)
        boundaries = np.cumsum(counts)
        for name, values in zip(("train", "val", "test"), np.split(shuffled, boundaries[:-1])):
            result.update({str(meter): name for meter in values})
    return result


def _parse_chunk(frame: pd.DataFrame, settings: dict[str, Any]) -> pd.DataFrame:
    """Parse once per chunk, leaving day validation to the streaming consumer."""
    frame = frame.copy()
    frame["EAN_ID"] = frame["EAN_ID"].str.strip()
    if frame["EAN_ID"].eq("").any():
        raise ValueError("Missing EAN_ID; meter identities cannot be inferred")
    raw_dates = pd.to_datetime(frame["Datum"], format="%Y-%m-%d", errors="coerce")
    if raw_dates.isna().any():
        raise ValueError("Invalid Datum; cannot assign the affected row to a daily profile")
    if settings["timestamp_policy"] == "source_wall_clock":
        # Z contradicts the release's CET/CEST legend. Remove it, do not shift.
        stamp = pd.to_datetime(frame["Datum_Startuur"].str.replace(r"Z$", "", regex=True),
                               format="ISO8601", errors="coerce")
        if isinstance(stamp.dtype, pd.DatetimeTZDtype) or not pd.api.types.is_datetime64_dtype(stamp.dtype):
            raise ValueError("source_wall_clock expects naive clock strings with optional trailing Z")
        raw_mismatch = stamp.dt.normalize().ne(raw_dates) & stamp.notna()
        day = raw_dates
    else:
        stamp_utc = pd.to_datetime(frame["Datum_Startuur"], format="ISO8601", errors="coerce", utc=True)
        # Datum refers to the source string's date, before any requested shift.
        raw_mismatch = frame["Datum_Startuur"].str.slice(0, 10).ne(frame["Datum"])
        stamp = stamp_utc.dt.tz_convert(settings["timezone"]).dt.tz_localize(None)
        day = stamp.dt.normalize().fillna(raw_dates)
    frame["_day"] = day.dt.strftime("%Y-%m-%d")
    frame["_timestamp"] = stamp.dt.as_unit("ns").astype("int64")
    frame["_bad_timestamp"] = stamp.isna() | raw_mismatch
    for name in [*_EQUIPMENT, "Volume_Afname_KWh", "Volume_Injectie_KWh"]:
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    return frame


def prepare(archive_path: str | Path, output_dir: str | Path, config: dict | None = None) -> dict:
    """Prepare daily kWh profiles without materializing the release's raw rows.

    CSV members must contain contiguous meter blocks and chronological readings
    within each meter. Numeric and lexicographic meter ordering are both allowed.
    Duplicate meter/date profiles across members fail, including rejected days.
    All output arrays use non-object dtypes and can be loaded with allow_pickle=False.
    """
    settings = _config(config)
    source = Path(archive_path)
    output = Path(output_dir)
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}; choose a new prepared-data directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    source_hash = _source_sha256(source)
    audit = {
        "rows_loaded": 0, "rows_processed": 0, "profiles_seen": 0, "profiles_kept": 0,
        "profiles_dropped": 0, "dropped_days_by_reason": {}, "rows_by_issue": {},
        "subset": settings["max_meters_per_file"] is not None, "csv_members": [],
        "meters_per_file": {},
    }
    dropped = Counter()
    row_issues = Counter()
    equipment_audit = {}
    equipment_by_meter: dict[str, tuple[int, int, int]] = {}
    usable_meters: set[str] = set()
    intervals = 96 if settings["resolution_minutes"] == 15 else 24
    dst_cache: dict[str, bool] = {}
    # Source-wall-clock timestamps always follow the dictionary's Brussels zone.
    day_zone = ZoneInfo("Europe/Brussels" if settings["timestamp_policy"] == "source_wall_clock" else settings["timezone"])

    with tempfile.TemporaryDirectory(prefix="energy-cgan-", dir=output.parent) as scratch:
        temporary = Path(scratch)
        database = sqlite3.connect(temporary / "profile_keys.sqlite")
        database.execute("PRAGMA journal_mode=OFF")
        database.execute("PRAGMA synchronous=OFF")
        database.execute("CREATE TABLE days (meter TEXT, day TEXT, row_index INTEGER, label TEXT, PRIMARY KEY (meter,day))")
        values_file = (temporary / "profiles.f32").open("wb")

        def consume_day(group: pd.DataFrame) -> None:
            meter, day = str(group["EAN_ID"].iloc[0]), str(group["_day"].iloc[0])
            audit["rows_processed"] += len(group)
            audit["profiles_seen"] += 1
            try:
                database.execute("INSERT INTO days VALUES (?,?,NULL,NULL)", (meter, day))
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"Duplicate daily profile for EAN_ID={meter}, date={day}; IDs are global across CSV members") from exc
            if not group["Contract_Categorie"].str.strip().eq("Residentieel").all():
                raise ValueError(f"Expected residential Contract_Categorie='Residentieel' for EAN_ID={meter}, date={day}")
            equipment_rows = group[_EQUIPMENT].to_numpy(dtype=float)
            if not np.isin(equipment_rows, (0, 1)).all():
                raise ValueError(f"Missing or invalid equipment indicators for EAN_ID={meter}, date={day}; cannot infer conditions")
            equipment = tuple(int(v) for v in equipment_rows[0])
            if not (equipment_rows == equipment_rows[0]).all() or (meter in equipment_by_meter and equipment_by_meter[meter] != equipment):
                raise ValueError(f"Inconsistent equipment indicators for EAN_ID={meter}; metadata must be stable across all files")
            equipment_by_meter[meter] = equipment
            equipment_key = "|".join(f"{name}={value}" for name, value in zip(("PV", "EV", "HP"), equipment))
            subgroup = equipment_audit.setdefault(equipment_key, {
                "profiles_seen": 0, "profiles_kept": 0, "profiles_dropped": 0,
                "dropped_days_by_reason": Counter(),
            })
            subgroup["profiles_seen"] += 1
            reasons = set()
            stamps = group["_timestamp"].to_numpy(dtype=np.int64)
            invalid_stamp = group["_bad_timestamp"].to_numpy(dtype=bool)
            if invalid_stamp.any():
                reasons.add("invalid_timestamp")
                row_issues["invalid_timestamp"] += int(invalid_stamp.sum())
            valid_stamps = stamps[~invalid_stamp]
            duplicate_count = len(valid_stamps) - len(np.unique(valid_stamps))
            if duplicate_count:
                reasons.add("duplicate_interval")
                row_issues["duplicate_interval"] += duplicate_count
            if day not in dst_cache:
                current = date.fromisoformat(day)
                start = datetime.combine(current, time(), day_zone)
                end = datetime.combine(current + timedelta(days=1), time(), day_zone)
                dst_cache[day] = start.utcoffset() != end.utcoffset()
            if dst_cache[day]:
                reasons.add("dst_transition_day")
            elif len(valid_stamps) > 1 and (np.diff(valid_stamps) < 0).any():
                raise ValueError(f"Input must be sorted chronologically within each meter: EAN_ID={meter}, date={day}")
            slots = np.remainder(valid_stamps, _DAY_NS)
            if (slots % _QUARTER_NS != 0).any():
                reasons.add("off_grid_interval")
                row_issues["off_grid_interval"] += int((slots % _QUARTER_NS != 0).sum())
            # Duplicate days have their own specific reason, rather than being
            # counted twice merely because their row count differs from 96.
            if not duplicate_count and (len(group) != 96 or len(np.unique(slots)) != 96):
                reasons.add("incomplete_day")
            incoming = group["Volume_Afname_KWh"].to_numpy(dtype=float)
            outgoing = group["Volume_Injectie_KWh"].to_numpy(dtype=float)
            invalid_energy = ~np.isfinite(incoming) | ~np.isfinite(outgoing)
            if invalid_energy.any():
                reasons.add("invalid_energy")
                row_issues["invalid_energy"] += int(invalid_energy.sum())
            negative_energy = (incoming < 0) | (outgoing < 0)
            if negative_energy.any():
                reasons.add("negative_energy")
                row_issues["negative_energy"] += int(negative_energy.sum())
            if reasons:
                audit["profiles_dropped"] += 1
                dropped.update(reasons)
                subgroup["profiles_dropped"] += 1
                subgroup["dropped_days_by_reason"].update(reasons)
                return
            values = {"import": incoming, "export": outgoing, "net": incoming - outgoing}[settings["target"]]
            if intervals == 24:
                values = values.reshape(24, 4).sum(axis=1)
            if not np.isfinite(values.astype(np.float32)).all():
                audit["profiles_dropped"] += 1
                dropped["float32_overflow"] += 1
                subgroup["profiles_dropped"] += 1
                subgroup["dropped_days_by_reason"]["float32_overflow"] += 1
                return
            values.astype(np.float32).tofile(values_file)
            label = _label(day, equipment, settings["conditioning"])
            database.execute("UPDATE days SET row_index=?,label=? WHERE meter=? AND day=?",
                             (audit["profiles_kept"], label, meter, day))
            audit["profiles_kept"] += 1
            subgroup["profiles_kept"] += 1
            usable_meters.add(meter)

        try:
            with zipfile.ZipFile(source) as archive:
                members = sorted(name for name in archive.namelist() if name.lower().endswith(".csv") and not name.endswith("/"))
                if not members:
                    raise ValueError("Archive contains no CSV members")
                for member in members:
                    audit["csv_members"].append(member)
                    carry = None
                    current_meter = None
                    last_day = None
                    closed_meters = set()
                    meters_seen = 0
                    stop = False
                    with archive.open(member) as handle:
                        reader = pd.read_csv(handle, dtype=str, keep_default_na=False, chunksize=settings["chunksize"])
                        for raw in reader:
                            missing = set(_COLUMNS) - set(raw.columns)
                            if missing:
                                raise ValueError(f"Missing required columns in {member}: {sorted(missing)}")
                            audit["rows_loaded"] += len(raw)
                            parsed = _parse_chunk(raw[_COLUMNS], settings)
                            frame = parsed if carry is None else pd.concat([carry, parsed], ignore_index=True)
                            meter_array = frame["EAN_ID"].to_numpy()
                            day_array = frame["_day"].to_numpy()
                            boundaries = np.flatnonzero((meter_array[1:] != meter_array[:-1]) | (day_array[1:] != day_array[:-1])) + 1
                            starts = np.r_[0, boundaries]
                            ends = np.r_[boundaries, len(frame)]
                            carry = None
                            for start, end in zip(starts, ends):
                                meter, day = str(meter_array[start]), str(day_array[start])
                                if meter != current_meter:
                                    maximum = settings["max_meters_per_file"]
                                    if maximum is not None and meters_seen >= maximum:
                                        stop = True
                                        break
                                    if meter in closed_meters:
                                        raise ValueError(f"Input must be sorted into contiguous meter blocks in {member}; EAN_ID={meter} repeats")
                                    if current_meter is not None:
                                        closed_meters.add(current_meter)
                                    current_meter, last_day = meter, None
                                    meters_seen += 1
                                if last_day is not None and day < last_day:
                                    raise ValueError(f"Input must be sorted by date within each meter in {member}: EAN_ID={meter}")
                                last_day = day
                                group = frame.iloc[start:end]
                                if end == len(frame):
                                    carry = group.copy()
                                else:
                                    consume_day(group)
                            if stop:
                                break
                        if carry is not None and not stop:
                            consume_day(carry)
                    audit["meters_per_file"][member] = meters_seen
            values_file.close()
            database.commit()
            count = audit["profiles_kept"]
            if count == 0:
                raise ValueError(f"No complete daily profiles survived validation; dropped-day reasons: {dict(dropped)}")
            meter_splits = _split_meters({meter: equipment_by_meter[meter] for meter in usable_meters},
                                         settings["split_fractions"], settings["seed"])
            classes = sorted({label for meter, label in database.execute("SELECT meter,label FROM days WHERE row_index IS NOT NULL") if meter_splits[meter] == "train"})
            vocabulary = {label: index for index, label in enumerate(classes)}
            x = np.memmap(temporary / "profiles.f32", dtype=np.float32, mode="r", shape=(count, intervals))
            meter_width = max(map(len, usable_meters))
            meter_ids = np.lib.format.open_memmap(temporary / "meter_id.npy", mode="w+", dtype=f"U{meter_width}", shape=(count,))
            dates = np.lib.format.open_memmap(temporary / "date.npy", mode="w+", dtype="U10", shape=(count,))
            splits = np.lib.format.open_memmap(temporary / "split.npy", mode="w+", dtype="U5", shape=(count,))
            labels = np.lib.format.open_memmap(temporary / "y.npy", mode="w+", dtype=np.int64, shape=(count,))
            profile_counts = Counter()
            condition_profiles = Counter()
            condition_meters = defaultdict(set)
            for meter, day, index, label in database.execute("SELECT meter,day,row_index,label FROM days WHERE row_index IS NOT NULL ORDER BY row_index"):
                if label not in vocabulary:
                    raise ValueError(f"Held-out label {label!r} is absent from the training vocabulary (meter={meter}, date={day}); use a larger training sample")
                meter_ids[index], dates[index], splits[index], labels[index] = meter, day, meter_splits[meter], vocabulary[label]
                profile_counts[meter_splits[meter]] += 1
                condition_profiles[(label, meter_splits[meter])] += 1
                condition_meters[(label, meter_splits[meter])].add(meter)
            audit["dropped_days_by_reason"] = dict(sorted(dropped.items()))
            audit["rows_by_issue"] = dict(sorted(row_issues.items()))
            audit["meters_seen"] = len(equipment_by_meter)
            audit["meters_kept"] = len(usable_meters)
            audit["split_profile_counts"] = dict(sorted(profile_counts.items()))
            audit["split_meter_counts"] = dict(sorted(Counter(meter_splits.values()).items()))
            audit["equipment_groups"] = {
                key: {**subgroup, "dropped_days_by_reason": dict(sorted(subgroup["dropped_days_by_reason"].items()))}
                for key, subgroup in sorted(equipment_audit.items())
            }
            condition_split_counts = {
                label: {
                    split: {"profiles": condition_profiles[(label, split)],
                            "unique_meters": len(condition_meters[(label, split)])}
                    for split in ("train", "val", "test")
                }
                for label in classes
            }
            metadata = {
                "schema_version": 1, "dataset_file": "dataset.npz", "source_sha256": source_hash,
                "source_archive": source.name, "classes": classes, "unit": "kWh",
                "interval_hours": settings["resolution_minutes"] / 60,
                "resolution_minutes": settings["resolution_minutes"], "n_intervals": intervals,
                "target": settings["target"], "conditioning": settings["conditioning"],
                "timestamp_policy": settings["timestamp_policy"], "timezone": str(day_zone),
                "timestamp_note": (
                    "Release legend specifies CET/CEST despite trailing Z; source clock times are retained without conversion. Brussels DST transition dates are excluded."
                    if settings["timestamp_policy"] == "source_wall_clock" else
                    "Explicit alternative: timestamp suffixes are interpreted as UTC and converted to the requested timezone. Clock-change dates are excluded."
                ),
                "calendar_policy": "Meteorological seasons DJF/MAM/JJA/SON; weekend is Saturday or Sunday; public holidays are not included.",
                "equipment_note": "Indicators are source classifications. EV=0 means not detected, not proven absence; the HP dictionary wording has an apparent copy/paste ambiguity.",
                "split_policy": "Deterministic meter-level split stratified by PV/EV/HP indicators, with at least one meter per split in every stratum.",
                "label_vocabulary_fit": "train_only", "condition_split_counts": condition_split_counts,
                "config": settings, "audit": audit,
            }
            # No final files are created until every input and vocabulary check succeeds.
            np.savez_compressed(temporary / "dataset.npz", x=x, y=labels, split=splits, meter_id=meter_ids, date=dates)
            (temporary / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            output.mkdir(parents=True, exist_ok=True)
            (temporary / "dataset.npz").replace(output / "dataset.npz")
            (temporary / "metadata.json").replace(output / "metadata.json")
            return metadata
        finally:
            values_file.close()
            database.close()
