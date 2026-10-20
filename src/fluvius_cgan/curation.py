"""Remove complete-trace aliases without transferring meters between splits.

TEST energy is used only in deterministic identity keys and retained-row copying.
No model, performance metric, energy-distribution summary, or tuning occurs here.
The default identity calendar is the audited 364 retained dates of 2024. An
explicit complete calendar supports other preparations and small synthetic tests.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path
import tempfile

import numpy as np

from .data import load_split


SPLIT_PRIORITY = ("train", "val", "test")
INPUT_FILES = ("metadata.json", "scaler.json", "train.npz", "val.npz", "test.npz")
TEST_ACCESS_SCOPE = (
    "Original TEST profiles are read only for deterministic complete-trace identity "
    "fingerprinting, structural/condition validation, and copying retained original-TEST "
    "rows. No performance or energy-distribution metrics are computed."
)
FINGERPRINT_ALGORITHM = (
    "Sort each complete meter trace by ISO date; copy profiles as contiguous '<f4'; "
    "replace numerical zeros with +0.0 in the identity copy only; SHA256 of "
    "dates.astype('S10').tobytes(order='C') + profiles.tobytes(order='C'). "
    "No headers, separators, meter IDs or conditions enter the key. Repeated hashes "
    "are confirmed by direct comparison of the complete canonicalized bytes."
)


@dataclass
class _Trace:
    split: str
    meter_id: str
    rows: np.ndarray
    technology: int


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity_digest(value):
    return hashlib.sha256(value).hexdigest()


def _identity_bytes(arrays, rows):
    # Preserve original arrays, including signed-zero bits, in every output row.
    values = np.array(arrays["x"][rows], dtype="<f4", order="C", copy=True)
    values[values == 0] = np.float32(0.0)
    return arrays["date"][rows].astype("S10").tobytes(order="C") + values.tobytes(order="C")


def _calendar(expected_dates):
    if expected_dates is None:
        days = np.arange("2024-01-01", "2025-01-01", dtype="datetime64[D]").astype("U10")
        expected_dates = days[~np.isin(days, ["2024-03-31", "2024-10-27"])]
    dates = np.asarray(expected_dates)
    if dates.ndim != 1 or len(dates) < 2 or dates.dtype.kind != "U":
        raise ValueError("expected_dates must be a complete calendar of at least two ISO dates")
    try:
        parsed = dates.astype("datetime64[D]")
    except ValueError as error:
        raise ValueError("expected_dates contains invalid dates") from error
    if np.isnat(parsed).any() or np.any(np.datetime_as_string(parsed, unit="D") != dates):
        raise ValueError("expected_dates must contain exact YYYY-MM-DD dates")
    dates = np.sort(dates)
    if len(np.unique(dates)) != len(dates):
        raise ValueError("expected_dates must not repeat dates")
    return dates


def _load_inputs(directory):
    hashes = {name: _sha256(directory / name) for name in INPUT_FILES}
    metadata = json.loads((directory / "metadata.json").read_text())
    scaler = json.loads((directory / "scaler.json").read_text())
    if not isinstance(metadata, dict) or not isinstance(scaler, dict):
        raise ValueError("metadata and scaler must be JSON objects")
    if not isinstance(metadata.get("splits"), dict):
        raise ValueError("metadata splits must be an object")
    source_hash = metadata.get("source_sha256")
    if not isinstance(source_hash, str) or len(source_hash) != 64 or any(character not in "0123456789abcdef" for character in source_hash):
        raise ValueError("metadata source_sha256 must be a SHA256 hex digest")
    seed = metadata.get("split_seed")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("metadata split_seed must be a nonnegative integer")
    arrays = {name: load_split(directory / f"{name}.npz") for name in SPLIT_PRIORITY}
    if hashes != {name: _sha256(directory / name) for name in INPUT_FILES}:
        raise ValueError("original prepared files changed while loading")
    if metadata.get("resolution") not in (24, 96) or isinstance(metadata.get("resolution"), bool):
        raise ValueError("metadata resolution must be 24 or 96")
    if metadata.get("units") != "kWh/interval" or metadata.get("target") not in ("net", "import", "injection"):
        raise ValueError("metadata must declare supported physical units and target")
    minimum, maximum = scaler.get("minimum"), scaler.get("maximum")
    if any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) for v in (minimum, maximum)):
        raise ValueError("original scaler bounds must be finite numbers")
    if minimum != float(arrays["train"]["x"].min()) or maximum != float(arrays["train"]["x"].max()):
        raise ValueError("original scaler does not match original TRAIN values")
    known_ids = set()
    for name, data in arrays.items():
        if data["x"].shape[1] != metadata["resolution"]:
            raise ValueError(f"{name}: profile width disagrees with metadata")
        if np.any(data["meter_id"] == "") or np.any(data["date"] == ""):
            raise ValueError(f"{name}: empty meter IDs or dates")
        ids = set(data["meter_id"].tolist())
        if known_ids & ids:
            raise ValueError("original meter IDs overlap across splits")
        known_ids.update(ids)
        recorded = metadata.get("splits", {}).get(name)
        if not isinstance(recorded, dict) or recorded.get("profiles") != len(data["x"]) or recorded.get("meters") != len(ids):
            raise ValueError(f"{name}: original counts disagree with metadata")
        if "sha256" in recorded and recorded["sha256"] != hashes[f"{name}.npz"]:
            raise ValueError(f"{name}: original file hash disagrees with metadata")
        if "meter_ids" in recorded and (len(recorded["meter_ids"]) != len(ids) or set(recorded["meter_ids"]) != ids):
            raise ValueError(f"{name}: original meter IDs disagree with metadata")
    return arrays, metadata, scaler, hashes


def _collect_traces(arrays, expected_dates):
    groups = defaultdict(list)
    records = {name: [] for name in SPLIT_PRIORITY}
    days = expected_dates.astype("datetime64[D]")
    months = days.astype("datetime64[M]").astype(np.int64) % 12 + 1
    date_conditions = ((days.astype(np.int64) + 3) % 7 >= 5).astype(np.int64) * 4 + (months % 12) // 3
    for name, data in arrays.items():
        order = np.lexsort((data["date"], data["meter_id"]))
        sorted_ids = data["meter_id"][order]
        starts = np.r_[0, np.flatnonzero(sorted_ids[1:] != sorted_ids[:-1]) + 1]
        for start, stop in zip(starts, np.r_[starts[1:], len(order)]):
            rows = order[start:stop]
            meter = str(sorted_ids[start])
            if not np.array_equal(data["date"][rows], expected_dates):
                raise ValueError(f"{name} meter {meter}: not the complete expected date calendar")
            labels = data["y"][rows]
            technologies = np.unique(labels // 8)
            if len(technologies) != 1 or not np.array_equal(labels % 8, date_conditions):
                raise ValueError(f"{name} meter {meter}: inconsistent date-aligned conditions/technology")
            trace = _Trace(name, meter, rows, int(technologies[0]))
            packed = _identity_bytes(data, rows)
            digest = _identity_digest(packed)
            if groups[digest]:
                reference = groups[digest][0]
                reference_data = arrays[reference.split]
                if packed != _identity_bytes(reference_data, reference.rows):
                    raise ValueError("SHA256 collision: canonicalized trace bytes differ")
                if not np.array_equal(labels, reference_data["y"][reference.rows]):
                    raise ValueError("duplicate trace has conflicting date-aligned conditions/technology")
            groups[digest].append(trace)
            records[name].append(trace)
    return records, groups


def _split_counts(records, data):
    rows = np.sort(np.concatenate([record.rows for record in records])) if records else np.array([], dtype=np.int64)
    return {
        "profiles": int(len(rows)), "meters": len(records),
        "meter_ids": sorted(record.meter_id for record in records),
        "technology_meter_counts": dict(sorted(Counter(str(record.technology) for record in records).items())),
        "condition_counts": dict(sorted(Counter(str(int(label)) for label in data["y"][rows]).items(), key=lambda item: int(item[0]))),
        "date_min": str(min(data["date"][rows])) if len(rows) else None,
        "date_max": str(max(data["date"][rows])) if len(rows) else None,
    }


def _member(trace):
    return {"split": trace.split, "meter_id": trace.meter_id,
            "profiles": len(trace.rows), "technology": trace.technology}


def _write_json(path, value):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def curate_prepared(prepared_dir, output_dir, *, expected_dates=None):
    """Write a new prepared copy containing one representative per exact trace.

    Default calendar: all 2024 dates except the two local DST transition dates.
    ``expected_dates`` must describe the complete intended trace, never a selected
    daily window. Equal days within otherwise different traces do not remove a
    meter. Canonical IDs use lexicographic ordering within the earliest split.
    The original input directory is immutable; output must not already exist or
    be inside it. A sibling staging directory is atomically published on success.
    """
    source, output = Path(prepared_dir).resolve(), Path(output_dir).resolve()
    if output == source or source in output.parents:
        raise ValueError("output must be outside the original prepared directory")
    if output.exists():
        raise FileExistsError("curated output must be a new directory")
    calendar = _calendar(expected_dates)
    arrays, original_metadata, original_scaler, input_hashes = _load_inputs(source)
    records, groups = _collect_traces(arrays, calendar)
    before = {name: _split_counts(records[name], arrays[name]) for name in SPLIT_PRIORITY}
    for name in SPLIT_PRIORITY:
        recorded = original_metadata["splits"][name]
        for key in ("technology_meter_counts", "condition_counts", "date_min", "date_max"):
            if key in recorded and recorded[key] != before[name][key]:
                raise ValueError(f"{name}: original {key} disagrees with metadata")
    retained = {name: [] for name in SPLIT_PRIORITY}
    audit_groups, aliases = [], []
    for digest, members in sorted(groups.items()):
        winner = min(members, key=lambda trace: (SPLIT_PRIORITY.index(trace.split), trace.meter_id))
        retained[winner.split].append(winner)
        dropped = [trace for trace in members if trace is not winner]
        audit_groups.append({"fingerprint_sha256": digest, "canonical": _member(winner),
                             "members": [_member(trace) for trace in members]})
        aliases.extend({**_member(trace), "canonical_split": winner.split,
                        "canonical_meter_id": winner.meter_id, "fingerprint_sha256": digest,
                        "reason": "exact_complete_trace_alias"} for trace in dropped)
    if any(not retained[name] for name in SPLIT_PRIORITY):
        raise ValueError("curation would leave an empty split; no output was published")
    after = {name: _split_counts(retained[name], arrays[name]) for name in SPLIT_PRIORITY}
    metadata = json.loads(json.dumps(original_metadata))
    curation = {
        "schema_version": 1, "fingerprint_version": 2, "fingerprint_algorithm": FINGERPRINT_ALGORITHM,
        "signed_zero_canonicalized_for_identity_only": True,
        "nonzero_rounding_or_tolerance": False, "byte_comparison_verified": True,
        "split_priority": list(SPLIT_PRIORITY), "canonical_id_order": "lexicographic within the earliest original split",
        "expected_dates": calendar.tolist(), "input_files_sha256": input_hashes,
        "implementation_files_sha256": {name: _sha256(Path(__file__).parent / name) for name in ("curation.py", "data.py")},
        "numpy_version": str(np.__version__), "before": before, "after": after,
        "identity_groups": audit_groups, "aliases_excluded": aliases,
        "duplicates_removed": {"meters": len(aliases), "profiles": sum(alias["profiles"] for alias in aliases)},
        "duplicate_groups": sum(len(members) > 1 for members in groups.values()),
        "original_preparation_counts": original_metadata.get("counts", {}),
        "test_access_scope": TEST_ACCESS_SCOPE,
        "original_split_assignments_preserved": True, "is_new_random_split": False,
        "development_traces_transferred_to_test": False,
        "curation_computed_performance_metrics": False, "curation_computed_energy_distribution_metrics": False,
        "train_statistics_scope": "Only original/retained TRAIN min/max bounds for scaler verification and refitting",
        "validation_status": "exploratory; previous validation results used the original preparation",
        "requires_new_training_artifacts": True, "pre_curation_checkpoints_are_fresh_models": False,
        "holdout": {
            "description": "Identity-curated subset of the original TEST split, not a new random test split",
            "retained_original_test_meter_ids": after["test"]["meter_ids"],
            "excluded_original_test_meter_ids": sorted(alias["meter_id"] for alias in aliases if alias["split"] == "test"),
            "performance_evaluated_by_curation": False,
            "prior_performance_access_history": "not inferred by this operation; retain the original experiment access history",
        },
        "identity_claim": "No exact full-trace overlap under the declared fingerprint; this does not prove independent households or exclude near-duplicates",
    }
    metadata["curation"] = curation
    metadata["counts"] = {**metadata.get("counts", {}),
                          "meters_kept": sum(value["meters"] for value in after.values()),
                          "profiles_kept": sum(value["profiles"] for value in after.values())}
    metadata["counts_scope"] = (
        "Original source-preprocessing counters are preserved except meters_kept and "
        "profiles_kept, which describe the curated copy. Source exclusions remain "
        "unchanged; removed alias meters/profiles are recorded separately in "
        "curation.duplicates_removed. Full original counters are retained in "
        "curation.original_preparation_counts."
    )
    metadata["splits"] = {}
    train_rows = np.sort(np.concatenate([trace.rows for trace in retained["train"]]))
    minimum, maximum = float(arrays["train"]["x"][train_rows].min()), float(arrays["train"]["x"][train_rows].max())
    scaler = {**original_scaler, "minimum": minimum, "maximum": maximum,
              "fitted_on": "train", "range": [0, 1], "constant_range": minimum == maximum,
              "transform": "(x - minimum) / (maximum - minimum); use denominator 1 if constant"}
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.parent / f".{output.name}.curation.lock"
    try:
        lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise FileExistsError("another curation already reserved this output") from error
    os.close(lock_fd)
    try:
        if output.exists():
            raise FileExistsError("curated output must remain a new directory")
        with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as staging:
            stage = Path(staging)
            for name in SPLIT_PRIORITY:
                rows = np.sort(np.concatenate([trace.rows for trace in retained[name]]))
                with (stage / f"{name}.npz").open("wb") as handle:
                    np.savez_compressed(handle, **{key: value[rows] for key, value in arrays[name].items()})
                    handle.flush()
                    os.fsync(handle.fileno())
                metadata["splits"][name] = {**after[name], "sha256": _sha256(stage / f"{name}.npz")}
            _write_json(stage / "scaler.json", scaler)
            _write_json(stage / "metadata.json", metadata)
            _write_json(stage / "curation_manifest.json", {
                "schema_version": 1, "fingerprint_version": 2,
                "input_files_sha256": input_hashes,
                "output_files_sha256": {name: _sha256(stage / name) for name in INPUT_FILES},
                "split_priority": list(SPLIT_PRIORITY), "canonical_id_order": curation["canonical_id_order"],
                "test_access_scope": TEST_ACCESS_SCOPE, "holdout": curation["holdout"],
                "identity_claim": curation["identity_claim"], "requires_new_training_artifacts": True,
            })
            if input_hashes != {name: _sha256(source / name) for name in INPUT_FILES}:
                raise ValueError("original prepared files changed during curation")
            if output.exists():
                raise FileExistsError("curated output appeared during curation")
            os.rename(stage, output)
    finally:
        lock.unlink(missing_ok=True)
    return metadata
