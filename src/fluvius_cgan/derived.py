"""Bounded-memory, development-only hourly derivation of curated 96-slot data.

The parent TEST path is inspected only for aliases; its contents are never opened,
hashed or copied. Quarter-hour canonical representatives are retained without a
new identity audit. Hourly exact-trace uniqueness is not asserted.
"""
from __future__ import annotations

from collections import Counter
from contextlib import ExitStack, contextmanager
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import re
import tempfile
import zipfile

import numpy as np

from .release import RELEASE_SHA256, sha256_file

DEVELOPMENT_FILES = ("metadata.json", "scaler.json", "train.npz", "val.npz")
MANIFEST = "derivation_manifest.json"
ARRAYS = ("x", "y", "meter_id", "date")
MAX_CHUNK_BYTES = 8 * 1024 * 1024
PARENT_PATH_POLICY = "relative_to_child_prepared_directory"
OPERATION = {
    "name": "consecutive_quarter_hour_energy_sum",
    "parent_resolution": 96, "child_resolution": 24, "group_size": 4,
    "input_dtype": "<f4", "accumulator_dtype": "<f8", "output_dtype": "<f4",
    "formula": "x96.astype('<f8').reshape(N, 24, 4).sum(axis=2).astype('<f4')",
    "row_order_preserved": True, "row_metadata_bytes_preserved": True,
    "canonical_membership_preserved": True, "new_identity_audit": False,
    "hourly_trace_uniqueness_verified": False,
}
QUALIFICATION = (
    "Verified hourly sum of the original curated-v2 TRAIN/VAL rows; quarter-hour "
    "canonical representatives are preserved. No new identity audit was performed; "
    "hourly trace uniqueness and independent households are not established. "
    "Validation remains exploratory; TEST24 is unmaterialized."
)


def _json(path):
    from .training import _json_snapshot
    return _json_snapshot(path)


def _write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _same(left, right):
    """Compare JSON including scalar types (True, 1 and 1.0 are distinct)."""
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def _chunk_rows(value):
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 65536:
        raise ValueError("chunk_rows must be an integer in [1, 65536]")
    return value


def _preflight(parent, child=None):
    """Reject direct, symlink and hard-link TEST aliases before content reads."""
    forbidden = [parent / "test.npz"]
    if child is not None:
        forbidden.append(child / "test.npz")
        if os.path.lexists(child / "test.npz"):
            raise ValueError("derived TEST24 must remain unmaterialized")
    paths = [parent / name for name in (*DEVELOPMENT_FILES, "curation_manifest.json")]
    if child is not None:
        paths += [child / name for name in (*DEVELOPMENT_FILES, MANIFEST)]
    for path in paths:
        for test_path in forbidden:
            if path.resolve() == test_path.resolve() or (
                path.exists() and test_path.exists() and path.samefile(test_path)
            ):
                raise ValueError(f"development input aliases reserved TEST: {path.name}")


def _parent(parent):
    from .training import _training_curation_guard
    _preflight(parent)
    metadata, metadata_hash = _json(parent / "metadata.json")
    if ("derived" in metadata or "curation" not in metadata or
            type(metadata.get("resolution")) is not int or metadata["resolution"] != 96 or
            metadata.get("source_sha256") != RELEASE_SHA256 or
            metadata.get("target") != "net" or metadata.get("units") != "kWh/interval"):
        raise ValueError("hourly derivation requires a direct pinned curated-v2 96-slot net-energy parent")
    status = _training_curation_guard(parent, {"allow_uncurated_data": False})
    if status["status"] != "verified_curation_v2" or status["metadata_sha256"] != metadata_hash:
        raise ValueError("parent curation binding changed or is not verified v2")
    scaler, scaler_hash = _json(parent / "scaler.json")
    if scaler_hash != status["verified_file_hashes"]["scaler.json"]:
        raise ValueError("parent scaler changed during verification")
    manifest, manifest_hash = _json(parent / "curation_manifest.json")
    if manifest_hash != status["manifest_sha256"]:
        raise ValueError("parent curation manifest changed during verification")
    test_digest = manifest["output_files_sha256"].get("test.npz")
    if not isinstance(test_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", test_digest):
        raise ValueError("parent reserved TEST96 metadata requires a valid recorded digest")
    curation = metadata["curation"]
    dates = curation.get("expected_dates")
    if not isinstance(dates, list) or not dates or any(not isinstance(x, str) for x in dates):
        raise ValueError("parent curation must specify its complete calendar")
    try:
        days = np.asarray(dates, dtype="U10").astype("datetime64[D]")
        exact = np.datetime_as_string(days, unit="D").tolist()
    except (ValueError, OverflowError) as error:
        raise ValueError("invalid parent curation calendar") from error
    if exact != dates or dates != sorted(set(dates)) or np.isnat(days).any():
        raise ValueError("parent calendar must be unique ordered exact ISO dates")
    months = days.astype("datetime64[M]").astype(np.int64) % 12 + 1
    calendar_labels = ((days.astype(np.int64) + 3) % 7 >= 5) * 4 + (months % 12) // 3
    calendar = {day: (1 << i, int(label)) for i, (day, label) in enumerate(zip(dates, calendar_labels))}
    canonical = {split: {} for split in ("train", "val", "test")}
    groups = curation.get("identity_groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("parent curation requires canonical identity groups")
    for group in groups:
        record = group.get("canonical", {}) if isinstance(group, dict) else {}
        split, meter, technology = record.get("split"), record.get("meter_id"), record.get("technology")
        if (split not in canonical or not isinstance(meter, str) or not meter or
                type(technology) is not int or not 0 <= technology < 8 or
                type(record.get("profiles")) is not int or record["profiles"] != len(dates) or
                meter in canonical[split]):
            raise ValueError("invalid parent canonical identity record")
        canonical[split][meter] = technology
    seen_ids = set()
    for split, meters in canonical.items():
        if not meters or seen_ids.intersection(meters):
            raise ValueError("parent canonical membership must be nonempty and split-disjoint")
        seen_ids.update(meters)
        counts = Counter(str(tech * 8 + int(label)) for tech in meters.values() for label in calendar_labels)
        expected = {
            "profiles": len(meters) * len(dates), "meters": len(meters), "meter_ids": sorted(meters),
            "technology_meter_counts": dict(Counter(str(tech) for tech in meters.values())),
            "condition_counts": dict(counts), "date_min": dates[0], "date_max": dates[-1],
        }
        recorded = metadata.get("splits", {}).get(split, {})
        if (any(recorded.get(key) != value for key, value in expected.items()) or
                curation.get("after", {}).get(split) != expected or
                recorded.get("sha256") != manifest["output_files_sha256"].get(f"{split}.npz")):
            raise ValueError(f"parent {split} canonical membership/count/hash binding disagrees")
    holdout = curation.get("holdout", {})
    if (holdout.get("retained_original_test_meter_ids") != sorted(canonical["test"]) or
            manifest.get("holdout") != holdout or holdout.get("performance_evaluated_by_curation") is not False):
        raise ValueError("parent reserved TEST metadata disagrees with curation")
    return metadata, scaler, status, calendar, canonical


class _Rows:
    def __init__(self, archive, stack, path, resolution):
        if sorted(archive.namelist()) != sorted(f"{key}.npy" for key in ARRAYS):
            raise ValueError(f"{path.name}: expected exactly x/y/meter_id/date NPZ arrays")
        self.streams, self.headers = {}, {}
        for key in ARRAYS:
            stream = stack.enter_context(archive.open(f"{key}.npy"))
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
            elif version == (2, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
            else:
                raise ValueError("unsupported NPZ NPY header version")
            if fortran or dtype.hasobject or dtype.fields is not None:
                raise ValueError("derived inputs require ordinary C-order nonobject arrays")
            if not shape or any(type(v) is not int or v < 1 for v in shape):
                raise ValueError("derived input arrays must be nonempty")
            if archive.getinfo(f"{key}.npy").file_size != stream.tell() + math.prod(shape) * dtype.itemsize:
                raise ValueError("NPY member length does not match its declared shape")
            self.streams[key] = stream
            self.headers[key] = (shape, dtype)
        self.n = self.headers["x"][0][0]
        if self.headers["x"] != ((self.n, resolution), np.dtype("<f4")):
            raise ValueError(f"{path.name}: profiles must have shape (N, {resolution}) and dtype <f4")
        for key in ARRAYS[1:]:
            shape, dtype = self.headers[key]
            if shape != (self.n,) or dtype.kind not in ("iu" if key == "y" else "U"):
                raise ValueError(f"{path.name}: invalid {key} vector")
        self.row_bytes = sum(math.prod(shape[1:]) * dtype.itemsize for shape, dtype in self.headers.values())
        if self.row_bytes > MAX_CHUNK_BYTES:
            raise ValueError("NPZ row exceeds the bounded streaming memory allowance")

    def read(self, count):
        result = {}
        for key, (shape, dtype) in self.headers.items():
            row_shape = shape[1:]
            size = count * int(np.prod(row_shape, dtype=np.int64)) * dtype.itemsize
            raw = self.streams[key].read(size)
            if len(raw) != size:
                raise ValueError("truncated NPZ array")
            result[key] = np.frombuffer(raw, dtype=dtype).reshape((count, *row_shape))
        return result

    def finish(self):
        if any(stream.read(1) for stream in self.streams.values()):
            raise ValueError("unexpected trailing NPZ array bytes")


@contextmanager
def _rows(path, resolution):
    with ExitStack() as stack:
        handle = stack.enter_context(Path(path).open("rb"))
        archive = stack.enter_context(zipfile.ZipFile(handle))
        yield _Rows(archive, stack, Path(path), resolution)


def _sum4(values):
    return values.astype("<f8").reshape(-1, 24, 4).sum(axis=2).astype("<f4")


def _scan(parent, split, calendar, canonical, chunk_rows, *, child=None, writer=None):
    """Validate complete parent rows and optionally compare or write child values."""
    observed = {meter: 0 for meter in canonical[split]}
    full_calendar = (1 << len(calendar)) - 1
    minimum, maximum = float("inf"), float("-inf")
    parent_min, parent_max = float("inf"), float("-inf")
    with ExitStack() as stack:
        source = stack.enter_context(_rows(parent / f"{split}.npz", 96))
        target = stack.enter_context(_rows(child / f"{split}.npz", 24)) if child is not None else None
        if source.n != len(observed) * len(calendar):
            raise ValueError(f"parent {split} row count disagrees with complete canonical traces")
        if target is not None and (target.n != source.n or any(
                source.headers[key] != target.headers[key] for key in ARRAYS[1:])):
            raise ValueError(f"derived {split} row metadata shape/dtype changed")
        bounded_rows = min(chunk_rows, MAX_CHUNK_BYTES // max(source.row_bytes, target.row_bytes if target else 1))
        output = stack.enter_context(writer.open("x.npy", "w", force_zip64=True)) if writer is not None else None
        if output is not None:
            np.lib.format.write_array_header_1_0(output, {
                "descr": "<f4", "fortran_order": False, "shape": (source.n, 24)})
        for offset in range(0, source.n, bounded_rows):
            count = min(bounded_rows, source.n - offset)
            original = source.read(count)
            if not np.isfinite(original["x"]).all():
                raise ValueError(f"parent {split} contains nonfinite profile values")
            for meter, date, label in zip(original["meter_id"], original["date"], original["y"]):
                if meter not in observed or date not in calendar:
                    raise ValueError(f"parent {split} row membership/date disagrees with canonical metadata")
                bit, calendar_label = calendar[date]
                if label != canonical[split][meter] * 8 + calendar_label or observed[meter] & bit:
                    raise ValueError(f"parent {split} row label or unique meter/date binding disagrees")
                observed[meter] |= bit
            expected = _sum4(original["x"])
            if not np.isfinite(expected).all():
                raise ValueError("hourly float32 sum must remain finite")
            parent_min, parent_max = min(parent_min, float(original["x"].min())), max(parent_max, float(original["x"].max()))
            minimum, maximum = min(minimum, float(expected.min())), max(maximum, float(expected.max()))
            if target is not None:
                actual = target.read(count)
                if any(original[key].tobytes() != actual[key].tobytes() for key in ARRAYS[1:]):
                    raise ValueError(f"derived {split} row metadata/order differs from parent")
                if expected.tobytes() != actual["x"].tobytes():
                    raise ValueError(f"derived {split} profiles differ from the exact float64 sum4 to float32 transformation")
            if output is not None:
                output.write(expected.tobytes())
        source.finish()
        if target is not None:
            target.finish()
    if any(mask != full_calendar for mask in observed.values()):
        raise ValueError(f"parent {split} does not retain complete canonical traces")
    return minimum, maximum, parent_min, parent_max


def resolve_parent_directory(child24, derived=None):
    """Resolve portable lineage against the child directory, never the process cwd.

    Pass an already-read ``metadata['derived']`` to avoid another metadata read.
    The path policy is explicit and only canonical relative POSIX paths qualify.
    This helper resolves paths without reading any parent or TEST contents.
    """
    child = Path(child24).resolve()
    if derived is None:
        metadata, _ = _json(child / "metadata.json")
        derived = metadata.get("derived")
    parent = derived.get("parent") if isinstance(derived, dict) else None
    reference = parent.get("prepared_dir") if isinstance(parent, dict) else None
    if (not isinstance(parent, dict) or parent.get("path_policy") != PARENT_PATH_POLICY or
            not isinstance(reference, str) or not reference or "\\" in reference or "\x00" in reference or
            Path(reference).is_absolute() or ":" in reference.split("/")[0] or
            not Path(reference).parts or Path(reference).as_posix() != reference):
        raise ValueError("derived parent requires a canonical relative-to-child prepared directory path")
    resolved = (child / reference).resolve()
    if Path(os.path.relpath(resolved, child)).as_posix() != reference:
        raise ValueError("derived parent path must be canonical relative to the child directory")
    return resolved


def _parent_binding(parent, child, status):
    return {"prepared_dir": Path(os.path.relpath(parent, child)).as_posix(),
            "path_policy": PARENT_PATH_POLICY, "development_files_sha256": status["verified_file_hashes"],
            "curation_manifest_sha256": status["manifest_sha256"]}


def _reserved_test(metadata):
    return {"resolution": 96, "split": deepcopy(metadata["splits"]["test"]),
            "access": "metadata_only_not_content_verified", "materialized_in_child": False}


def _scaler(minimum, maximum):
    return {"minimum": minimum, "maximum": maximum, "fitted_on": "train", "range": [0, 1],
            "constant_range": minimum == maximum,
            "transform": "(x - minimum) / (maximum - minimum); use denominator 1 if constant"}


def _metadata(original, parent, child, status, splits):
    metadata = deepcopy(original)
    metadata.pop("curation")
    metadata.update({
        "resolution": 24, "interval_hours": 1, "splits": splits,
        "reserved_parent_test96": _reserved_test(original),
        "counts": {"meters_kept": sum(v["meters"] for v in splits.values()),
                   "profiles_kept": sum(v["profiles"] for v in splits.values())},
        "counts_scope": "Materialized hourly TRAIN/VAL only; reserved parent TEST96 is metadata-only.",
        "derived": {"schema_version": 1, "operation": deepcopy(OPERATION),
                    "parent": _parent_binding(parent, child, status), "test24_materialized": False,
                    "qualification": QUALIFICATION},
    })
    return metadata


def _unchanged(parent, status, child=None, child_hashes=None, manifest_hash=None):
    _preflight(parent, child)
    if any(sha256_file(parent / name) != digest for name, digest in status["verified_file_hashes"].items()):
        raise ValueError("parent development files changed during hourly verification")
    if sha256_file(parent / "curation_manifest.json") != status["manifest_sha256"]:
        raise ValueError("parent curation manifest changed during hourly verification")
    if child is not None:
        if any(sha256_file(child / name) != digest for name, digest in child_hashes.items()):
            raise ValueError("derived development files changed during hourly verification")
        if sha256_file(child / MANIFEST) != manifest_hash:
            raise ValueError("derivation manifest changed during verification")


def verify_hourly_development(parent96, child24, *, chunk_rows=4096):
    """Verify direct parent provenance and every derived TRAIN/VAL row, no TEST reads.

    Only row chunks and per-meter calendar bitsets are retained. The returned
    provenance contains hashes and qualification, never profile arrays.
    """
    chunk_rows = _chunk_rows(chunk_rows)
    parent, child = Path(parent96).resolve(), Path(child24).resolve()
    if parent == child:
        raise ValueError("derived and parent directories must differ")
    _preflight(parent, child)
    metadata, metadata_hash = _json(child / "metadata.json")
    if resolve_parent_directory(child, metadata.get("derived")) != parent:
        raise ValueError("explicit parent differs from the recorded relative hourly lineage")
    original, parent_scaler, status, calendar, canonical = _parent(parent)
    scaler, scaler_hash = _json(child / "scaler.json")
    try:
        manifest, manifest_hash = _json(child / MANIFEST)
    except OSError as error:
        raise ValueError("claimed hourly derivation requires derivation_manifest.json") from error
    derived = metadata.get("derived")
    expected_derived = {"schema_version": 1, "operation": OPERATION, "parent": _parent_binding(parent, child, status),
                        "test24_materialized": False, "qualification": QUALIFICATION}
    if not _same(derived, expected_derived) or "curation" in metadata or (child / "curation_manifest.json").exists():
        raise ValueError("malformed hourly derived schema/parent binding or copied curation claim")
    if (metadata.get("resolution") != 24 or metadata.get("interval_hours") != 1 or
            metadata.get("source_sha256") != RELEASE_SHA256 or metadata.get("target") != original["target"] or
            metadata.get("units") != "kWh/interval" or metadata.get("split_seed") != original.get("split_seed") or
            metadata.get("reserved_parent_test96") != _reserved_test(original) or
            set(metadata.get("splits", {})) != {"train", "val"}):
        raise ValueError("derived physical target/split/reserved TEST metadata disagrees")
    expected_manifest = {"schema_version": 1, "kind": "curated96_to_hourly24_development",
                         "source_sha256": RELEASE_SHA256, "derived": expected_derived}
    if (set(manifest) != set(expected_manifest) | {"output_files_sha256"} or
            any(not _same(manifest.get(key), value) for key, value in expected_manifest.items())):
        raise ValueError("derivation manifest disagrees with derived provenance")
    hashes = {"metadata.json": metadata_hash, "scaler.json": scaler_hash,
              **{f"{split}.npz": sha256_file(child / f"{split}.npz") for split in ("train", "val")}}
    if manifest.get("output_files_sha256") != hashes:
        raise ValueError("derivation manifest does not bind the exact child development files")
    expected_splits = {split: {**original["splits"][split], "sha256": hashes[f"{split}.npz"]}
                       for split in ("train", "val")}
    if not _same(metadata, _metadata(original, parent, child, status, expected_splits)):
        raise ValueError("derived metadata does not preserve the declared parent fields and development counts")
    for split in ("train", "val"):
        expected_split = {**original["splits"][split], "sha256": hashes[f"{split}.npz"]}
        if metadata["splits"][split] != expected_split:
            raise ValueError(f"derived {split} metadata differs from parent canonical membership")
        minimum, maximum, parent_min, parent_max = _scan(parent, split, calendar, canonical, chunk_rows, child=child)
        if split == "train":
            if parent_scaler.get("minimum") != parent_min or parent_scaler.get("maximum") != parent_max:
                raise ValueError("parent scaler does not match its full TRAIN min/max")
            if not _same(scaler, _scaler(minimum, maximum)):
                raise ValueError("derived scaler is not the exact hourly full-TRAIN min/max scaler")
    _unchanged(parent, status, child, hashes, manifest_hash)
    return {"status": "verified_hourly_derivation_v1", "pinned_release": True,
            "metadata_sha256": metadata_hash, "verified_file_hashes": hashes,
            "derived_manifest_sha256": manifest_hash, "parent": expected_derived["parent"],
            "qualification": QUALIFICATION}


def derive_hourly(parent96, output24, *, chunk_rows=4096):
    """Atomically publish a new hourly TRAIN/VAL-only preparation; never overwrite."""
    chunk_rows = _chunk_rows(chunk_rows)
    parent, output = Path(parent96).resolve(), Path(output24).resolve()
    if parent == output or parent in output.parents or output in parent.parents:
        raise ValueError("hourly output must be separate from the parent prepared directory")
    if os.path.lexists(output):
        raise FileExistsError("hourly output must be a new directory")
    original, parent_scaler, status, calendar, canonical = _parent(parent)
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.parent / f".{output.name}.derivation.lock"
    lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(lock_fd)
    try:
        with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
            stage = Path(temporary)
            splits = {}
            for split in ("train", "val"):
                with (stage / f"{split}.npz").open("xb") as handle, zipfile.ZipFile(
                        handle, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as writer:
                    minimum, maximum, parent_min, parent_max = _scan(
                        parent, split, calendar, canonical, chunk_rows, writer=writer)
                    with (parent / f"{split}.npz").open("rb") as raw, zipfile.ZipFile(raw) as archive:
                        for key in ARRAYS[1:]:
                            with archive.open(f"{key}.npy") as source, writer.open(f"{key}.npy", "w", force_zip64=True) as target:
                                for block in iter(lambda: source.read(1024 * 1024), b""):
                                    target.write(block)
                if split == "train":
                    if parent_scaler.get("minimum") != parent_min or parent_scaler.get("maximum") != parent_max:
                        raise ValueError("parent scaler does not match its full TRAIN min/max")
                    scaler = _scaler(minimum, maximum)
                splits[split] = {**original["splits"][split], "sha256": sha256_file(stage / f"{split}.npz")}
            metadata = _metadata(original, parent, output, status, splits)
            _write_json(stage / "scaler.json", scaler)
            _write_json(stage / "metadata.json", metadata)
            _write_json(stage / MANIFEST, {
                "schema_version": 1, "kind": "curated96_to_hourly24_development", "source_sha256": RELEASE_SHA256,
                "derived": metadata["derived"],
                "output_files_sha256": {name: sha256_file(stage / name) for name in DEVELOPMENT_FILES},
            })
            # Verify the written bytes, transformation and parent stability before publication.
            verify_hourly_development(parent, stage, chunk_rows=chunk_rows)
            if os.path.lexists(output):
                raise FileExistsError("hourly output appeared during derivation")
            os.rename(stage, output)
        return metadata
    finally:
        lock.unlink(missing_ok=True)
