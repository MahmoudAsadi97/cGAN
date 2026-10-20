"""Synthetic release archives exercise preparation without downloading the release."""

import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
import pytest

from fluvius_cgan.data import prepare_archive, load_split


COLUMNS = ["EAN_ID", "Datum", "Datum_Startuur", "Volume_Afname_KWh",
           "Volume_Injectie_KWh", "Warmtepomp_Indicator",
           "Elektrisch_Voertuig_Indicator", "PV_Installatie_Indicator",
           "Contract_Categorie"]


def day(meter, date="2024-01-01", group=0, amount=None):
    times = pd.date_range(date, periods=96, freq="15min")
    return pd.DataFrame({
        "EAN_ID": str(meter), "Datum": date,
        "Datum_Startuur": times.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "Volume_Afname_KWh": float(meter) if amount is None else amount,
        "Volume_Injectie_KWh": 0.25,
        "Warmtepomp_Indicator": group // 4,
        "Elektrisch_Voertuig_Indicator": (group // 2) % 2,
        "PV_Installatie_Indicator": group % 2,
        "Contract_Categorie": "Residentieel",
    }, columns=COLUMNS)


def archive(tmp_path, frames=None):
    frames = frames if frames is not None else {
        f"group_{g}.csv": pd.concat([day(100 * g + m, group=g)
                                    for m in range(1, 7)], ignore_index=True)
        for g in range(8)
    }
    path = tmp_path / "fixture.zip"
    with ZipFile(path, "w", ZIP_DEFLATED) as zipped:
        for name, frame in frames.items():
            zipped.writestr(name, frame.to_csv(index=False))
    return path


def read_json(path):
    return json.loads(Path(path).read_text())


def all_arrays(path):
    return {name: load_split(path / f"{name}.npz")
            for name in ("train", "val", "test")}


def test_streaming_chunk_boundaries_splits_scaler_and_original_units(tmp_path):
    source = archive(tmp_path)
    small, large = tmp_path / "small", tmp_path / "large"
    summary = prepare_archive(source, small, chunksize=37, seed=42)
    prepare_archive(source, large, chunksize=100_000, seed=42)
    arrays, other = all_arrays(small), all_arrays(large)
    meter_sets = []
    for split, data in arrays.items():
        for key in data:
            np.testing.assert_array_equal(data[key], other[split][key])
        assert data["x"].dtype == np.float32
        assert data["y"].dtype == np.int64
        assert data["meter_id"].dtype.kind == "U"
        assert data["date"].dtype.kind == "U"
        assert data["x"].shape[1] == 96
        np.testing.assert_array_equal(data["x"][:, 0], data["meter_id"].astype(float) - .25)
        assert len(set(data["y"] // 8)) == 8
        meter_sets.append(set(data["meter_id"]))
    assert not meter_sets[0] & meter_sets[1]
    assert not meter_sets[0] & meter_sets[2]
    assert not meter_sets[1] & meter_sets[2]
    assert sum(map(len, meter_sets)) == 48
    scaler = read_json(small / "scaler.json")
    assert scaler["minimum"] == float(arrays["train"]["x"].min())
    assert scaler["maximum"] == float(arrays["train"]["x"].max())
    assert scaler["fitted_on"] == "train"
    assert summary["counts"]["profiles_kept"] == 48
    assert summary["units"] == "kWh/interval"
    assert len(summary["source_sha256"]) == 64
    assert summary["counts"] == read_json(large / "metadata.json")["counts"]


def test_dst_missing_duplicate_null_and_negative_days_are_audited(tmp_path):
    frames = [day(m) for m in range(1, 4)]
    # Keep both transition days invalid even if synthetically complete.
    spring = day(1, "2024-03-31").drop(index=range(8, 12))
    autumn = day(1, "2024-10-27")
    autumn = pd.concat([autumn.iloc[:12], autumn.iloc[8:12], autumn.iloc[12:]])
    missing = day(1, "2024-01-02").iloc[:-1]
    duplicate = day(1, "2024-01-03")
    duplicate.loc[10, "Datum_Startuur"] = duplicate.loc[9, "Datum_Startuur"]
    null = day(1, "2024-01-04")
    null.loc[5, "Volume_Afname_KWh"] = np.nan
    negative = day(1, "2024-01-05")
    negative.loc[6, "Volume_Injectie_KWh"] = -1
    full_dst = day(2, "2024-03-31")
    frames[0] = pd.concat([frames[0], missing, duplicate, null, negative, spring, autumn])
    frames[1] = pd.concat([frames[1], full_dst])
    source = archive(tmp_path, {"one.csv": pd.concat(frames)})
    result = prepare_archive(source, tmp_path / "out", chunksize=51)
    assert result["counts"]["profiles_kept"] == 3
    assert result["counts"]["days_excluded"] == 7
    assert result["exclusions"]["dst_transition_day"] == 3
    assert result["exclusions"]["incomplete_day"] >= 1
    assert result["exclusions"]["duplicate_slots"] >= 1
    assert result["exclusions"]["nonfinite_energy"] == 1
    assert result["exclusions"]["negative_source_energy"] == 1


@pytest.mark.parametrize("target,value", [("net", -1.), ("import", 1.), ("injection", 2.)])
def test_targets_hourly_sums_and_negative_net(tmp_path, target, value):
    frames = [day(m, amount=1.) for m in range(1, 4)]
    for frame in frames:
        frame["Volume_Injectie_KWh"] = 2.
    source = archive(tmp_path, {"one.csv": pd.concat(frames)})
    result = prepare_archive(source, tmp_path / "out", resolution=24, target=target)
    for split in all_arrays(tmp_path / "out").values():
        assert split["x"].shape == (1, 24)
        np.testing.assert_allclose(split["x"], value * 4)
        np.testing.assert_allclose(split["x"].sum(axis=1), value * 96)
    assert result["resolution"] == 24
    assert result["interval_hours"] == 1.


@pytest.mark.parametrize("column,value,match", [
    ("PV_Installatie_Indicator", 2, "binary"),
    ("PV_Installatie_Indicator", np.nan, "binary"),
    ("PV_Installatie_Indicator", 1, "constant"),
    ("Datum", "2024-01-02", "date"),
    ("Datum_Startuur", "2024-01-01T00:00:00+00:00", "timestamp"),
    ("EAN_ID", np.nan, "meter"),
])
def test_invalid_metadata_and_date_formats_fail(tmp_path, column, value, match):
    frame = pd.concat([day(m) for m in range(1, 4)], ignore_index=True)
    frame.loc[0, column] = value
    source = archive(tmp_path, {"one.csv": frame})
    with pytest.raises(ValueError, match=match):
        prepare_archive(source, tmp_path / "out", chunksize=37)


def test_duplicate_meter_across_files_and_reopened_meter_fail(tmp_path):
    source = archive(tmp_path, {"a.csv": pd.concat([day(m) for m in range(1, 4)]),
                                "b.csv": day(1)})
    with pytest.raises(ValueError, match="[Dd]uplicate meter"):
        prepare_archive(source, tmp_path / "cross")
    source = archive(tmp_path, {"a.csv": pd.concat([day(1), day(2), day(1), day(3)])})
    with pytest.raises(ValueError, match="[Dd]uplicate meter"):
        prepare_archive(source, tmp_path / "reopen", chunksize=37)


def test_unsorted_timestamp_fails(tmp_path):
    frame = pd.concat([day(m) for m in range(1, 4)], ignore_index=True)
    frame.iloc[[2, 3]] = frame.iloc[[3, 2]].to_numpy()
    source = archive(tmp_path, {"one.csv": frame})
    with pytest.raises(ValueError, match="sorted"):
        prepare_archive(source, tmp_path / "out")


def test_labels_are_calendar_and_technology_correct_and_meter_days_stay_together(tmp_path):
    frames = []
    dates = ["2024-01-01", "2024-03-02", "2024-06-03", "2024-09-01"]
    expected = [56, 61, 58, 63]  # HP=EV=PV=1; weekday/weekend and four seasons.
    for meter in range(1, 4):
        frames.extend(day(meter, date, group=7) for date in dates)
    source = archive(tmp_path, {"one.csv": pd.concat(frames)})
    prepare_archive(source, tmp_path / "out")
    for data in all_arrays(tmp_path / "out").values():
        assert len(set(data["meter_id"])) == 1
        np.testing.assert_array_equal(data["date"], dates)
        np.testing.assert_array_equal(data["y"], expected)


def test_too_few_eligible_meters_and_overwrite_fail(tmp_path):
    source = archive(tmp_path, {"one.csv": pd.concat([day(1), day(2)])})
    with pytest.raises(ValueError, match="three|3"):
        prepare_archive(source, tmp_path / "out")
    source = archive(tmp_path)
    destination = tmp_path / "existing"
    destination.mkdir()
    (destination / "keep.txt").write_text("keep")
    with pytest.raises(FileExistsError):
        prepare_archive(source, destination)
    assert (destination / "keep.txt").read_text() == "keep"


def test_smoke_selection_is_explicit_and_limit_per_group(tmp_path):
    source = archive(tmp_path)
    result = prepare_archive(source, tmp_path / "out", max_meters_per_group=3, chunksize=97)
    assert result["counts"]["meters_kept"] == 24
    assert result["sampling"]["mode"] == "smoke"
    assert "bias" in result["sampling"]["selection"].lower()
    assert result["sampling"]["max_meters_per_group"] == 3


def test_schema_and_unsafe_archive_member_rejected(tmp_path):
    frame = day(1).rename(columns={"Datum": "date"})
    source = archive(tmp_path, {"one.csv": frame})
    with pytest.raises(ValueError, match="schema"):
        prepare_archive(source, tmp_path / "bad_schema")
    source = archive(tmp_path, {"../one.csv": day(1)})
    with pytest.raises(ValueError, match="archive"):
        prepare_archive(source, tmp_path / "unsafe")


def test_heldout_extreme_cannot_change_train_scaler(tmp_path):
    frames = {"one.csv": pd.concat([day(m) for m in range(1, 4)], ignore_index=True)}
    source = archive(tmp_path, frames)
    first = prepare_archive(source, tmp_path / "first")
    heldout_id = first["splits"]["test"]["meter_ids"][0]
    frames["one.csv"].loc[frames["one.csv"]["EAN_ID"] == heldout_id, "Volume_Afname_KWh"] = 1_000_000.
    source = archive(tmp_path, frames)
    second = prepare_archive(source, tmp_path / "second")
    assert first["splits"]["train"]["meter_ids"] == second["splits"]["train"]["meter_ids"]
    assert read_json(tmp_path / "first/scaler.json") == read_json(tmp_path / "second/scaler.json")
    assert load_split(tmp_path / "second/test.npz")["x"].max() > 999_999.


def test_zero_eligible_stratum_fails_even_when_other_group_is_valid(tmp_path):
    invalid = day(4, group=1).iloc[:-1]
    source = archive(tmp_path, {"one.csv": pd.concat([day(m) for m in range(1, 4)] + [invalid])})
    with pytest.raises(ValueError, match="3 eligible meters"):
        prepare_archive(source, tmp_path / "out")


def test_nonresidential_excluded_and_off_grid_day_counted(tmp_path):
    off_grid = day(1, "2024-01-02")
    off_grid.loc[0, "Datum_Startuur"] = "2024-01-02T00:00:01.000Z"
    nonresidential = day(4)
    nonresidential["Contract_Categorie"] = "Professioneel"
    source = archive(tmp_path, {"one.csv": pd.concat([day(1), off_grid, day(2), day(3), nonresidential])})
    result = prepare_archive(source, tmp_path / "out", chunksize=67)
    assert result["counts"]["nonresidential_meters"] == 1
    assert result["counts"]["profiles_kept"] == 3
    assert result["exclusions"]["off_grid"] == 1
