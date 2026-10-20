"""Synthetic-only identity curation; no real prepared or model files are opened."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from fluvius_cgan.curation import curate_prepared
from fluvius_cgan.data import load_split


DATES = np.array(["2024-01-01", "2024-01-02", "2024-01-03"])


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_source(tmp_path, *, members=None, signed_zero=False):
    source = tmp_path / "original"
    source.mkdir()
    profiles = {name: (np.arange(72, dtype=np.float32).reshape(3, 24) / 10 + offset)
                for name, offset in zip("ABCDEF", [0, 100, 200, 300, 400, 1_000_000])}
    profiles["B"][0] = profiles["A"][0]  # Equal day, different complete trace.
    if members is None:
        members = {
            "train": [("train_z", "A"), ("train_a", "A"), ("train_unique", "B")],
            "val": [("val_train_alias", "A"), ("val_z", "C"), ("val_a", "C"), ("val_unique", "D")],
            "test": [("aaa_test_train_alias", "A"), ("aaa_test_val_alias", "C"),
                     ("test_z", "E"), ("test_a", "E"), ("test_unique", "F")],
        }
    metadata = {"resolution": 24, "target": "net", "units": "kWh/interval",
                "source_sha256": "a" * 64, "split_seed": 42,
                "timezone_policy": "source local calendar", "split_method": "original seeded stratification",
                "counts": {"meters_kept": sum(map(len, members.values())), "profiles_kept": sum(map(len, members.values())) * 3},
                "exclusions": {"dst_transition_day": 2}, "splits": {}}
    original_arrays = {}
    for split, records in members.items():
        x = np.concatenate([profiles[name] for _, name in records]).copy()
        meter_ids = np.repeat([meter for meter, _ in records], 3)
        dates = np.tile(DATES, len(records))
        if signed_zero and split == "train":
            x[np.flatnonzero(meter_ids == "train_a")[0], 0] = -0.0
        # Deliberately interleave and reverse row order; identities align by date.
        order = np.arange(len(x))[::-1]
        arrays = {"x": x[order], "y": np.zeros(len(x), dtype=np.int64),
                  "meter_id": meter_ids[order], "date": dates[order]}
        original_arrays[split] = arrays
        np.savez_compressed(source / f"{split}.npz", **arrays)
        metadata["splits"][split] = {"profiles": len(x), "meters": len(records),
                                    "meter_ids": sorted(meter for meter, _ in records),
                                    "sha256": digest(source / f"{split}.npz")}
    train = original_arrays["train"]["x"]
    (source / "scaler.json").write_text(json.dumps({"minimum": float(train.min()), "maximum": float(train.max())}))
    (source / "metadata.json").write_text(json.dumps(metadata))
    return source, original_arrays


def refresh_metadata_hash(source, split):
    metadata = json.loads((source / "metadata.json").read_text())
    metadata["splits"][split]["sha256"] = digest(source / f"{split}.npz")
    (source / "metadata.json").write_text(json.dumps(metadata))


def test_curation_preserves_priority_original_rows_and_full_manifest(tmp_path):
    source, original = write_source(tmp_path)
    before_hashes = {path.name: digest(path) for path in source.iterdir()}
    output = tmp_path / "curated"
    result = curate_prepared(source, output, expected_dates=DATES)
    expected = {"train": {"train_a", "train_unique"}, "val": {"val_a", "val_unique"},
                "test": {"test_a", "test_unique"}}
    for split, ids in expected.items():
        retained = load_split(output / f"{split}.npz")
        assert set(retained["meter_id"]) == ids
        mask = np.isin(original[split]["meter_id"], sorted(ids))
        for key, values in retained.items():
            np.testing.assert_array_equal(values, original[split][key][mask])
        assert result["splits"][split]["sha256"] == digest(output / f"{split}.npz")
        assert result["splits"][split]["technology_meter_counts"] == {"0": 2}
    assert {path.name: digest(path) for path in source.iterdir()} == before_hashes
    assert result["source_sha256"] == "a" * 64 and result["split_seed"] == 42
    assert result["split_method"] == "original seeded stratification"
    assert result["exclusions"] == {"dst_transition_day": 2}
    curation = result["curation"]
    assert curation["duplicates_removed"] == {"meters": 6, "profiles": 18}
    assert curation["duplicate_groups"] == 3
    assert curation["before"]["test"]["meters"] == 5
    assert curation["after"]["test"]["meters"] == 2
    assert len(curation["identity_groups"]) == 6
    assert sum(len(group["members"]) for group in curation["identity_groups"]) == 12
    assert curation["input_files_sha256"] == before_hashes
    assert curation["requires_new_training_artifacts"]
    assert not curation["development_traces_transferred_to_test"]
    assert not curation["is_new_random_split"]
    assert not curation["curation_computed_performance_metrics"]
    assert not curation["curation_computed_energy_distribution_metrics"]
    assert "exploratory" in curation["validation_status"]
    assert "does not prove independent households" in curation["identity_claim"]
    assert "No performance or energy-distribution metrics" in curation["test_access_scope"]
    assert result["counts"]["meters_kept"] == 6
    assert result["counts"]["profiles_kept"] == 18
    assert curation["original_preparation_counts"]["meters_kept"] == 12
    scaler = json.loads((output / "scaler.json").read_text())
    assert scaler["minimum"] == float(original["train"]["x"].min())
    assert scaler["maximum"] == float(original["train"]["x"].max())
    manifest = json.loads((output / "curation_manifest.json").read_text())
    for name, value in manifest["output_files_sha256"].items():
        assert value == digest(output / name)
    assert manifest["input_files_sha256"] == before_hashes
    assert manifest["holdout"]["retained_original_test_meter_ids"] == ["test_a", "test_unique"]


def test_signed_zero_canonicalization_is_identity_only_and_replay_is_deterministic(tmp_path):
    source, original = write_source(tmp_path, signed_zero=True)
    first, second = tmp_path / "first", tmp_path / "second"
    result = curate_prepared(source, first, expected_dates=DATES)
    curate_prepared(source, second, expected_dates=DATES)
    train = load_split(first / "train.npz")
    zero_row = (train["meter_id"] == "train_a") & (train["date"] == DATES[0])
    assert np.signbit(train["x"][zero_row, 0]).all()
    assert result["curation"]["fingerprint_version"] == 2
    assert result["curation"]["signed_zero_canonicalized_for_identity_only"]
    assert result["curation"]["duplicates_removed"]["meters"] == 6
    assert {path.name: digest(path) for path in first.iterdir()} == {path.name: digest(path) for path in second.iterdir()}
    # Validate the declared key directly against an independently built v2 key.
    raw = original["train"]
    rows = np.flatnonzero(raw["meter_id"] == "train_a")
    rows = rows[np.argsort(raw["date"][rows])]
    values = np.ascontiguousarray(raw["x"][rows], dtype="<f4")
    values[values == 0] = 0.0
    expected = hashlib.sha256(raw["date"][rows].astype("S10").tobytes() + values.tobytes()).hexdigest()
    group = next(group for group in result["curation"]["identity_groups"] if group["canonical"]["meter_id"] == "train_a")
    assert group["fingerprint_sha256"] == expected


@pytest.mark.parametrize("change", ["technology", "date_condition", "date_missing", "nonfinite", "wrong_dtype"])
def test_invalid_identity_inputs_fail_without_output(tmp_path, change):
    source, original = write_source(tmp_path)
    values = {key: value.copy() for key, value in original["val"].items()}
    alias = values["meter_id"] == "val_train_alias"
    if change == "technology":
        values["y"][alias] += 8
    elif change == "date_condition":
        values["y"][alias] += 4
    elif change == "date_missing":
        values["date"][np.flatnonzero(alias)[0]] = "2024-01-04"
    elif change == "nonfinite":
        values["x"][0, 0] = np.nan
    else:
        values["x"] = values["x"].astype(np.float64)
    np.savez_compressed(source / "val.npz", **values)
    refresh_metadata_hash(source, "val")
    with pytest.raises(ValueError):
        curate_prepared(source, tmp_path / "curated", expected_dates=DATES)
    assert not (tmp_path / "curated").exists()


def test_hash_collision_is_confirmed_by_bytes_and_rejected(tmp_path, monkeypatch):
    source, _ = write_source(tmp_path)
    monkeypatch.setattr("fluvius_cgan.curation._identity_digest", lambda _: "0" * 64)
    with pytest.raises(ValueError, match="collision"):
        curate_prepared(source, tmp_path / "curated", expected_dates=DATES)
    assert not (tmp_path / "curated").exists()


@pytest.mark.parametrize("change", ["split_hash", "counts", "scaler", "ids", "source_hash", "split_seed"])
def test_original_manifest_and_scaler_must_match(tmp_path, change):
    source, _ = write_source(tmp_path)
    metadata = json.loads((source / "metadata.json").read_text())
    if change == "split_hash":
        metadata["splits"]["val"]["sha256"] = "0" * 64
    elif change == "counts":
        metadata["splits"]["val"]["profiles"] += 1
    elif change == "ids":
        metadata["splits"]["test"]["meter_ids"][0] = "invented"
    elif change == "source_hash":
        metadata["source_sha256"] = "bad"
    elif change == "split_seed":
        metadata.pop("split_seed")
    else:
        (source / "scaler.json").write_text('{"minimum": -100, "maximum": 100}')
    (source / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        curate_prepared(source, tmp_path / "curated", expected_dates=DATES)
    assert not (tmp_path / "curated").exists()


def test_no_partial_output_when_writing_fails(tmp_path, monkeypatch):
    source, _ = write_source(tmp_path)
    before = {path.name: digest(path) for path in source.iterdir()}
    original_save = np.savez_compressed
    count = 0

    def fail_second(*args, **kwargs):
        nonlocal count
        count += 1
        if count == 2:
            raise OSError("synthetic write failure")
        return original_save(*args, **kwargs)

    monkeypatch.setattr("fluvius_cgan.curation.np.savez_compressed", fail_second)
    with pytest.raises(OSError, match="synthetic write failure"):
        curate_prepared(source, tmp_path / "curated", expected_dates=DATES)
    assert not (tmp_path / "curated").exists()
    assert not list(tmp_path.glob(".curated*"))
    assert {path.name: digest(path) for path in source.iterdir()} == before


def test_existing_or_nested_output_is_rejected_and_default_calendar_is_full_year(tmp_path):
    source, _ = write_source(tmp_path)
    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "keep.txt").write_text("keep")
    with pytest.raises(FileExistsError):
        curate_prepared(source, existing, expected_dates=DATES)
    with pytest.raises(ValueError, match="outside"):
        curate_prepared(source, source / "nested", expected_dates=DATES)
    with pytest.raises(ValueError, match="complete expected date"):
        curate_prepared(source, tmp_path / "curated")
    assert (existing / "keep.txt").read_text() == "keep"
    assert not (source / "nested").exists()


def test_aliases_cannot_empty_a_heldout_split_silently(tmp_path):
    source, _ = write_source(tmp_path, members={
        "train": [("train", "A")], "val": [("val", "B")], "test": [("test", "A")]})
    with pytest.raises(ValueError, match="empty split"):
        curate_prepared(source, tmp_path / "curated", expected_dates=DATES)
    assert not (tmp_path / "curated").exists()


def test_cli_default_calendar_on_synthetic_full_year(tmp_path):
    dates = np.arange("2024-01-01", "2025-01-01", dtype="datetime64[D]")
    dates = dates[~np.isin(dates.astype("U10"), ["2024-03-31", "2024-10-27"])]
    months = dates.astype("datetime64[M]").astype(np.int64) % 12 + 1
    labels = ((dates.astype(np.int64) + 3) % 7 >= 5).astype(np.int64) * 4 + (months % 12) // 3
    source, output = tmp_path / "original", tmp_path / "curated"
    source.mkdir()
    metadata = {"resolution": 24, "units": "kWh/interval", "target": "net",
                "source_sha256": "a" * 64, "split_seed": 42, "splits": {}}
    for number, name in enumerate(("train", "val", "test")):
        values = np.arange(len(dates) * 24, dtype=np.float32).reshape(len(dates), 24) + number * 100_000
        np.savez_compressed(source / f"{name}.npz", x=values, y=labels,
                            date=dates.astype("U10"), meter_id=np.full(len(dates), name))
        metadata["splits"][name] = {"profiles": len(dates), "meters": 1,
                                    "sha256": digest(source / f"{name}.npz")}
    (source / "metadata.json").write_text(json.dumps(metadata))
    (source / "scaler.json").write_text(json.dumps({"minimum": 0., "maximum": float(len(dates) * 24 - 1)}))
    script = Path(__file__).resolve().parents[1] / "scripts" / "curate_prepared.py"
    completed = subprocess.run([sys.executable, str(script), "--prepared", str(source), "--output", str(output)],
                               check=True, capture_output=True, text=True)
    report = json.loads(completed.stdout)
    assert report["duplicates_removed"] == {"meters": 0, "profiles": 0}
    assert report["after"]["test"]["profiles"] == 364
    assert len(json.loads((output / "metadata.json").read_text())["curation"]["expected_dates"]) == 364
