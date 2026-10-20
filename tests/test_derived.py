"""Synthetic-only hourly derivation checks; never open real prepared data."""

import builtins
from collections import Counter
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
import pytest

from fluvius_cgan.curation import curate_prepared
from fluvius_cgan.derived import derive_hourly, resolve_parent_directory, verify_hourly_development
from fluvius_cgan.release import RELEASE_SHA256
from fluvius_cgan import training


DATES = np.array(["2024-01-01", "2024-01-02", "2024-01-03"], dtype="U10")
DEVELOPMENT_FILES = {"metadata.json", "scaler.json", "train.npz", "val.npz"}


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


@pytest.fixture
def curated96(tmp_path):
    """Create and curate toy TEST before installing any no-TEST access guard."""
    source = tmp_path / "original96"
    source.mkdir()
    base = np.arange(288, dtype=np.float32).reshape(3, 96) / 4 - 20
    # Float32 summation loses a unit here; declared float64 accumulation gives 2.
    base[:, :4] = np.array([2**24, 1, -(2**24), 1], dtype=np.float32)
    hourly_alias = base.copy()
    hourly_alias[:, 4] += 1
    hourly_alias[:, 5] -= 1
    members = {
        "train": [("train_z", base, 0), ("train_a", base, 0),
                  ("train_hourly_alias", hourly_alias, 1)],
        "val": [("val_a", np.full((3, 96), 1_000_000, dtype=np.float32), 0),
                ("val_b", np.full((3, 96), 2_000_000, dtype=np.float32), 1)],
        "test": [("test_reserved", np.full((3, 96), 3_000_000, dtype=np.float32), 0)],
    }
    metadata = {
        "schema_version": 1, "resolution": 96, "target": "net",
        "target_definition": "Volume_Afname_KWh - Volume_Injectie_KWh",
        "units": "kWh/interval", "interval_hours": 0.25,
        "source_sha256": RELEASE_SHA256, "source_is_pinned_release": True,
        "split_seed": 42, "timezone_policy": "CET/CEST source local calendar",
        "split_method": "Synthetic original assignments, preserved during curation",
        "label_formula": "((HP*4 + EV*2 + PV)*2 + weekend)*4 + season",
        "condition_dim": 64, "season_codes": {"winter": 0, "spring": 1, "summer": 2, "autumn": 3},
        "counts": {"meters_kept": 6, "profiles_kept": 18}, "splits": {},
    }
    for name, records in members.items():
        x = np.concatenate([values for _, values, _ in records])
        order = np.roll(np.arange(len(x))[::-1], 1)
        arrays = {
            "x": x[order],
            "y": np.repeat([technology * 8 for _, _, technology in records], len(DATES)).astype(np.int64)[order],
            "meter_id": np.repeat([meter for meter, _, _ in records], len(DATES))[order],
            "date": np.tile(DATES, len(records))[order],
        }
        np.savez_compressed(source / f"{name}.npz", **arrays)
        metadata["splits"][name] = {
            "profiles": len(x), "meters": len(records),
            "meter_ids": sorted(meter for meter, _, _ in records),
            "sha256": _digest(source / f"{name}.npz"),
        }
    train = _read_arrays(source / "train.npz")["x"]
    _write_json(source / "scaler.json", {
        "minimum": float(train.min()), "maximum": float(train.max()), "fitted_on": "train",
        "range": [0, 1], "constant_range": False,
        "transform": "(x - minimum) / (maximum - minimum); use denominator 1 if constant",
    })
    _write_json(source / "metadata.json", metadata)
    parent = tmp_path / "curated96"
    curate_prepared(source, parent, expected_dates=DATES)
    return source, parent


def _forbid_test_access(monkeypatch):
    """Trap pathlib, NumPy and ZIP/builtin file access to either toy TEST file."""
    original_path_open = Path.open
    original_read_bytes = Path.read_bytes
    original_numpy_load = np.load
    original_builtin_open = builtins.open
    original_io_open = io.open

    def check(value):
        if not isinstance(value, (str, bytes, Path)):
            value = getattr(value, "name", None)
        if isinstance(value, bytes):
            value = value.decode()
        if isinstance(value, (str, Path)):
            path = Path(value)
            test_path = path.parent / "test.npz"
            if (path.name == "test.npz" or path.resolve().name == "test.npz" or
                    (path.exists() and test_path.exists() and path.samefile(test_path))):
                pytest.fail("Hourly development must not open, hash or copy TEST")

    def path_open(path, *args, **kwargs):
        check(path)
        return original_path_open(path, *args, **kwargs)

    def read_bytes(path, *args, **kwargs):
        check(path)
        return original_read_bytes(path, *args, **kwargs)

    def numpy_load(file, *args, **kwargs):
        check(file)
        return original_numpy_load(file, *args, **kwargs)

    def builtin_open(file, *args, **kwargs):
        check(file)
        return original_builtin_open(file, *args, **kwargs)

    def io_open(file, *args, **kwargs):
        check(file)
        return original_io_open(file, *args, **kwargs)

    monkeypatch.setattr(Path, "open", path_open)
    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    monkeypatch.setattr(np, "load", numpy_load)
    monkeypatch.setattr(builtins, "open", builtin_open)
    monkeypatch.setattr(io, "open", io_open)


def _rebind_child(child, changed_split=None):
    """Rebind local file hashes so tamper checks cannot rely on a stale digest."""
    metadata = json.loads((child / "metadata.json").read_text())
    if changed_split is not None:
        arrays = _read_arrays(child / f"{changed_split}.npz")
        summary = metadata["splits"][changed_split]
        summary["sha256"] = _digest(child / f"{changed_split}.npz")
        summary["condition_counts"] = dict(sorted(Counter(str(int(y)) for y in arrays["y"]).items()))
        technology = {str(meter): int(y) // 8 for meter, y in zip(arrays["meter_id"], arrays["y"])}
        summary["technology_meter_counts"] = dict(sorted(Counter(str(value) for value in technology.values()).items()))
        _write_json(child / "metadata.json", metadata)
    manifest = json.loads((child / "derivation_manifest.json").read_text())
    assert set(manifest["output_files_sha256"]) == DEVELOPMENT_FILES
    manifest["output_files_sha256"] = {name: _digest(child / name) for name in sorted(DEVELOPMENT_FILES)}
    _write_json(child / "derivation_manifest.json", manifest)


@pytest.mark.parametrize("chunk_rows", [1, 3, 5])
def test_hourly_sum_preserves_rows_dtypes_membership_and_train_only_scaler(curated96, tmp_path, monkeypatch, chunk_rows):
    _, parent = curated96
    before = {name: _digest(parent / name) for name in DEVELOPMENT_FILES | {"curation_manifest.json"}}
    _forbid_test_access(monkeypatch)
    child = tmp_path / f"hourly_{chunk_rows}"
    metadata = derive_hourly(parent, child, chunk_rows=chunk_rows)
    assert metadata == json.loads((child / "metadata.json").read_text())
    assert metadata["source_sha256"] == RELEASE_SHA256
    assert metadata["resolution"] == 24 and metadata["interval_hours"] == 1
    assert metadata["units"] == "kWh/interval" and metadata["target"] == "net"
    assert set(metadata["splits"]) == {"train", "val"}
    assert "curation" not in metadata
    assert metadata["derived"]["test24_materialized"] is False
    assert metadata["reserved_parent_test96"]["materialized_in_child"] is False
    assert metadata["reserved_parent_test96"]["resolution"] == 96
    assert metadata["reserved_parent_test96"]["access"] == "metadata_only_not_content_verified"
    parent_metadata = json.loads((parent / "metadata.json").read_text())
    assert metadata["reserved_parent_test96"]["split"] == parent_metadata["splits"]["test"]
    assert not (child / "test.npz").exists()
    assert not (child / "curation_manifest.json").exists()

    child_arrays = {}
    for split in ("train", "val"):
        original = _read_arrays(parent / f"{split}.npz")
        actual = child_arrays[split] = _read_arrays(child / f"{split}.npz")
        expected64 = original["x"].astype(np.float64).reshape(-1, 24, 4).sum(axis=2)
        expected = expected64.astype(np.float32)
        assert actual["x"].shape == (len(original["x"]), 24)
        assert actual["x"].dtype == np.dtype("float32")
        np.testing.assert_array_equal(actual["x"].view(np.uint32), expected.view(np.uint32))
        for key in ("y", "meter_id", "date"):
            assert actual[key].dtype == original[key].dtype
            assert actual[key].tobytes() == original[key].tobytes()
        # The only permitted daily-energy discrepancy is the stored hourly rounding.
        rounding = np.abs(expected.astype(np.float64) - expected64).sum(axis=1)
        discrepancy = np.abs(actual["x"].astype(np.float64).sum(axis=1) - original["x"].astype(np.float64).sum(axis=1))
        assert np.all(discrepancy <= rounding + 1e-10)

    train = child_arrays["train"]
    assert np.all(train["x"][:, 0] == 2), "Four-quarter energy sums must use float64 accumulation"
    assert set(train["meter_id"]) == {"train_a", "train_hourly_alias"}
    for date in DATES:
        first = train["x"][(train["meter_id"] == "train_a") & (train["date"] == date)]
        second = train["x"][(train["meter_id"] == "train_hourly_alias") & (train["date"] == date)]
        np.testing.assert_array_equal(first, second)
    scaler = json.loads((child / "scaler.json").read_text())
    assert scaler["minimum"] == float(train["x"].min())
    assert scaler["maximum"] == float(train["x"].max())
    assert scaler["fitted_on"] == "train"
    assert child_arrays["val"]["x"].min() > scaler["maximum"]
    status = verify_hourly_development(parent, child, chunk_rows=chunk_rows)
    assert status["status"] == "verified_hourly_derivation_v1"
    assert set(status["verified_file_hashes"]) == DEVELOPMENT_FILES
    assert status["derived_manifest_sha256"] == _digest(child / "derivation_manifest.json")
    assert before == {name: _digest(parent / name) for name in before}


def test_derive_verify_guard_and_training_loader_work_without_parent_test(curated96, tmp_path, monkeypatch):
    _, parent = curated96
    (parent / "test.npz").unlink()
    _forbid_test_access(monkeypatch)
    child = tmp_path / "hourly"
    derive_hourly(parent, child, chunk_rows=3)
    direct = verify_hourly_development(parent, child, chunk_rows=3)
    guarded = training._training_curation_guard(child, {"allow_uncurated_data": False})
    assert direct["status"] == guarded["status"] == "verified_hourly_derivation_v1"
    train, val, metadata, scaler, hashes = training._load_prepared(child)
    assert train["x"].shape == val["x"].shape == (6, 24)
    assert metadata["resolution"] == 24
    assert scaler["maximum"] == float(train["x"].max())
    assert "test.npz" not in hashes


@pytest.mark.parametrize("tamper", ["x", "y", "row_order", "scaler"])
def test_rebound_child_tampering_still_fails_parent_relation(curated96, tmp_path, monkeypatch, tamper):
    _, parent = curated96
    _forbid_test_access(monkeypatch)
    child = tmp_path / "hourly"
    derive_hourly(parent, child, chunk_rows=3)
    if tamper == "scaler":
        scaler = json.loads((child / "scaler.json").read_text())
        scaler["minimum"] -= 1
        _write_json(child / "scaler.json", scaler)
        _rebind_child(child)
    else:
        arrays = _read_arrays(child / "train.npz")
        if tamper == "x":
            arrays["x"][0, 7] += np.float32(0.25)
        elif tamper == "y":
            arrays["y"] += 8
        else:
            order = np.arange(len(arrays["x"]))
            order[[0, 1]] = order[[1, 0]]
            arrays = {name: value[order] for name, value in arrays.items()}
        np.savez_compressed(child / "train.npz", **arrays)
        _rebind_child(child, "train")
    with pytest.raises(ValueError):
        verify_hourly_development(parent, child, chunk_rows=3)
    with pytest.raises(ValueError):
        training._training_curation_guard(child, {"allow_uncurated_data": True})
    with pytest.raises(ValueError):
        training._load_prepared(child)


@pytest.mark.parametrize("filename", ["train.npz", "val.npz", "metadata.json", "scaler.json", "curation_manifest.json"])
def test_parent_development_tampering_is_rejected(curated96, tmp_path, monkeypatch, filename):
    _, parent = curated96
    _forbid_test_access(monkeypatch)
    child = tmp_path / "hourly"
    derive_hourly(parent, child, chunk_rows=3)
    with (parent / filename).open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError):
        verify_hourly_development(parent, child, chunk_rows=3)
    with pytest.raises(ValueError):
        training._training_curation_guard(child, {"allow_uncurated_data": True})


def test_uncurated_and_nested_parents_are_rejected_without_override_escape(curated96, tmp_path, monkeypatch):
    original, parent = curated96
    _forbid_test_access(monkeypatch)
    with pytest.raises(ValueError):
        derive_hourly(original, tmp_path / "uncurated_hourly", chunk_rows=3)
    child = tmp_path / "hourly"
    derive_hourly(parent, child, chunk_rows=3)
    with pytest.raises(ValueError):
        derive_hourly(child, tmp_path / "nested_hourly", chunk_rows=3)
    metadata = json.loads((child / "metadata.json").read_text())
    metadata["derived"]["parent"]["prepared_dir"] = str(original)
    _write_json(child / "metadata.json", metadata)
    _rebind_child(child)
    with pytest.raises(ValueError):
        training._training_curation_guard(child, {"allow_uncurated_data": True})


def test_missing_derivation_manifest_cannot_use_uncurated_override(curated96, tmp_path, monkeypatch):
    _, parent = curated96
    _forbid_test_access(monkeypatch)
    child = tmp_path / "hourly"
    derive_hourly(parent, child, chunk_rows=3)
    (child / "derivation_manifest.json").unlink()
    with pytest.raises(ValueError):
        training._training_curation_guard(child, {"allow_uncurated_data": True})


def test_derivation_does_not_overwrite_parent_or_existing_output(curated96, tmp_path, monkeypatch):
    _, parent = curated96
    _forbid_test_access(monkeypatch)
    before = {name: _digest(parent / name) for name in DEVELOPMENT_FILES | {"curation_manifest.json"}}
    for output in (parent, parent / "nested"):
        with pytest.raises((ValueError, FileExistsError)):
            derive_hourly(parent, output, chunk_rows=3)
    output = tmp_path / "already_exists"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("existing output")
    with pytest.raises((ValueError, FileExistsError)):
        derive_hourly(parent, output, chunk_rows=3)
    assert marker.read_text() == "existing output"
    assert before == {name: _digest(parent / name) for name in before}


def test_parent_profiles_are_streamed_without_numpy_loading_parent_npz(curated96, tmp_path, monkeypatch):
    _, parent = curated96
    _forbid_test_access(monkeypatch)
    guarded_load = np.load

    def no_parent_npz(file, *args, **kwargs):
        path = Path(file) if isinstance(file, (str, Path)) else None
        if path is not None and path.parent == parent and path.name in {"train.npz", "val.npz"}:
            pytest.fail("Compressed parent NPZ must be streamed; NumPy mmap does not bound its memory")
        return guarded_load(file, *args, **kwargs)

    monkeypatch.setattr(np, "load", no_parent_npz)
    child = tmp_path / "hourly"
    derive_hourly(parent, child, chunk_rows=3)
    status = verify_hourly_development(parent, child, chunk_rows=3)
    assert status["status"] == "verified_hourly_derivation_v1"


def test_tiny_training_records_derived_provenance_without_test_access(curated96, tmp_path, monkeypatch):
    import torch
    from fluvius_cgan.cli import evaluate_checkpoint

    _, parent = curated96
    _forbid_test_access(monkeypatch)
    child = tmp_path / "hourly"
    derive_hourly(parent, child, chunk_rows=3)
    run = tmp_path / "run"
    summary = training.train(child, run, {
        "epochs": 1, "batch_size": 3, "latent_dim": 4, "hidden_dim": 4,
        "validation_samples": 4, "num_threads": 1, "seed": 11,
    })
    checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    assert summary["data_curation"]["status"] == "verified_hourly_derivation_v1"
    assert checkpoint["data_curation"] == summary["data_curation"]
    assert checkpoint["data_hashes"]["derivation_manifest.json"] == _digest(child / "derivation_manifest.json")
    assert "test.npz" not in checkpoint["data_hashes"]
    assert summary["test_used"] is False
    report = evaluate_checkpoint(child, run / "best.pt", split="val", samples_per_condition=2)
    assert report["split"] == "val" and report["units"] == "kWh/interval"
    assert report["profile_count"] == 4 and report["meter_count"] == 2
    assert report["training_profiles"] == {"prepared": 6, "model": 6, "baselines": 6}
    for method in report["methods"].values():
        assert method["overall"]["interval_hours"] == 1
        assert method["overall"]["n_intervals"] == 24
        assert method["overall"]["n_real"] == method["overall"]["n_generated"] == 4
    assert report["data_curation_status"] == "verified_hourly_derivation_v1"
    assert report["derivation_manifest_sha256"] == _digest(child / "derivation_manifest.json")
    assert report["derivation_metadata"] == checkpoint["metadata"]["derived"]
    assert report["prepared_data_hashes"] == {name: checkpoint["data_hashes"][name] for name in DEVELOPMENT_FILES}
    assert report["test_used"] is False
    checkpoint["data_hashes"]["derivation_manifest.json"] = "0" * 64
    unbound_checkpoint = run / "unbound.pt"
    torch.save(checkpoint, unbound_checkpoint)
    with pytest.raises(ValueError, match="derivation provenance"):
        evaluate_checkpoint(child, unbound_checkpoint, split="val", samples_per_condition=2)


def test_hourly_test_evaluation_rejects_before_checkpoint_or_split_reads(curated96, tmp_path, monkeypatch):
    import torch
    from fluvius_cgan.cli import evaluate_checkpoint

    _, parent = curated96
    child = tmp_path / "hourly"
    derive_hourly(parent, child, chunk_rows=3)
    # An unexpected child TEST file must not turn development-only output into a
    # supported holdout. Create the toy sentinel before installing read traps.
    (child / "test.npz").write_bytes(b"reserved TEST must not be read")
    _forbid_test_access(monkeypatch)

    def forbidden_load(*args, **kwargs):
        pytest.fail("Derived TEST rejection must precede checkpoint or split loading")

    monkeypatch.setattr(torch, "load", forbidden_load)
    monkeypatch.setattr(training, "_read_split", forbidden_load)
    with pytest.raises(ValueError, match="TEST.*unmaterialized"):
        evaluate_checkpoint(child, tmp_path / "nonexistent.pt", split="test", samples_per_condition=2)


@pytest.mark.parametrize("filename,kind", [
    ("train.npz", "symlink"), ("train.npz", "hardlink"), ("metadata.json", "symlink"),
])
def test_parent_development_alias_to_test_is_rejected_before_content_access(curated96, tmp_path, monkeypatch, filename, kind):
    _, parent = curated96
    _forbid_test_access(monkeypatch)
    child = tmp_path / "hourly"
    derive_hourly(parent, child, chunk_rows=3)
    alias = parent / filename
    alias.unlink()
    if kind == "symlink":
        alias.symlink_to(parent / "test.npz")
    else:
        os.link(parent / "test.npz", alias)
    with pytest.raises(ValueError):
        derive_hourly(parent, tmp_path / "aliased_hourly", chunk_rows=3)
    with pytest.raises(ValueError):
        verify_hourly_development(parent, child, chunk_rows=3)
    with pytest.raises(ValueError):
        training._training_curation_guard(child, {"allow_uncurated_data": True})


@pytest.mark.parametrize("malformed", ["none", "boolean_version", "missing", "changed_source", "stripped_derivation"])
def test_malformed_derived_claim_never_uses_uncurated_override(curated96, tmp_path, monkeypatch, malformed):
    _, parent = curated96
    _forbid_test_access(monkeypatch)
    child = tmp_path / "hourly"
    derive_hourly(parent, child, chunk_rows=3)
    metadata = json.loads((child / "metadata.json").read_text())
    if malformed == "none":
        metadata["derived"] = None
    elif malformed == "boolean_version":
        metadata["derived"]["schema_version"] = True
    elif malformed in {"missing", "stripped_derivation"}:
        del metadata["derived"]
        if malformed == "stripped_derivation":
            (child / "derivation_manifest.json").unlink()
    else:
        metadata["source_sha256"] = "a" * 64
    _write_json(child / "metadata.json", metadata)
    if malformed != "stripped_derivation":
        _rebind_child(child)
    with pytest.raises(ValueError):
        training._training_curation_guard(child, {"allow_uncurated_data": True})
    with pytest.raises(ValueError):
        training._load_prepared(child)


def test_parent_and_child_can_relocate_together_without_changing_artifact_bytes(curated96, tmp_path, monkeypatch):
    original, parent = curated96
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    original.rename(bundle / "original96")
    parent = parent.rename(bundle / "curated96")
    child = bundle / "hourly"
    _forbid_test_access(monkeypatch)
    metadata = derive_hourly(parent, child, chunk_rows=3)
    binding = metadata["derived"]["parent"]
    assert binding["prepared_dir"] == "../curated96"
    assert binding["path_policy"] == "relative_to_child_prepared_directory"
    assert resolve_parent_directory(child) == parent.resolve()
    assert resolve_parent_directory(child, metadata["derived"]) == parent.resolve()
    files = {
        "curated96": DEVELOPMENT_FILES | {"curation_manifest.json"},
        "hourly": DEVELOPMENT_FILES | {"derivation_manifest.json"},
    }
    before = {
        (directory, name): (bundle / directory / name).read_bytes()
        for directory, names in files.items() for name in names
    }
    digests = {key: hashlib.sha256(value).hexdigest() for key, value in before.items()}
    relocated = bundle.rename(tmp_path / "relocated_bundle")
    moved_parent, moved_child = relocated / "curated96", relocated / "hourly"
    # Resolution must be anchored to the child artifact, never the process cwd.
    unrelated = tmp_path / "unrelated_working_directory"
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)
    assert resolve_parent_directory(moved_child) == moved_parent.resolve()
    status = verify_hourly_development(moved_parent, moved_child, chunk_rows=3)
    assert status["status"] == "verified_hourly_derivation_v1"
    train, val, loaded_metadata, _, _ = training._load_prepared(moved_child)
    assert train["x"].shape == val["x"].shape == (6, 24)
    assert loaded_metadata == metadata
    for (directory, name), content in before.items():
        path = relocated / directory / name
        assert path.read_bytes() == content
        assert _digest(path) == digests[(directory, name)]


def test_wrong_explicit_parent_is_rejected_even_when_development_bytes_match(curated96, tmp_path, monkeypatch):
    _, parent = curated96
    _forbid_test_access(monkeypatch)
    child = tmp_path / "hourly"
    derive_hourly(parent, child, chunk_rows=3)
    wrong_parent = tmp_path / "other_parent"
    wrong_parent.mkdir()
    for name in DEVELOPMENT_FILES | {"curation_manifest.json"}:
        # Copy development provenance only. TEST contents are never copied.
        (wrong_parent / name).write_bytes((parent / name).read_bytes())
    assert resolve_parent_directory(child) == parent.resolve()
    with pytest.raises(ValueError):
        verify_hourly_development(wrong_parent, child, chunk_rows=3)


def test_absolute_parent_binding_is_rejected_after_consistent_manifest_rebinding(curated96, tmp_path, monkeypatch):
    _, parent = curated96
    _forbid_test_access(monkeypatch)
    child = tmp_path / "hourly"
    derive_hourly(parent, child, chunk_rows=3)
    metadata = json.loads((child / "metadata.json").read_text())
    metadata["derived"]["parent"]["prepared_dir"] = str(parent.resolve())
    _write_json(child / "metadata.json", metadata)
    _rebind_child(child)
    manifest = json.loads((child / "derivation_manifest.json").read_text())
    manifest["derived"] = metadata["derived"]
    _write_json(child / "derivation_manifest.json", manifest)
    with pytest.raises(ValueError):
        resolve_parent_directory(child)
    with pytest.raises(ValueError):
        verify_hourly_development(parent, child, chunk_rows=3)
    with pytest.raises(ValueError):
        training._training_curation_guard(child, {"allow_uncurated_data": True})
    with pytest.raises(ValueError):
        training._load_prepared(child)
