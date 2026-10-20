import csv
import io
import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from energy_cgan.data import _config, _parse_chunk, prepare


COLUMNS = [
    "EAN_ID", "Datum", "Datum_Startuur", "Volume_Afname_KWh",
    "Volume_Injectie_KWh", "Warmtepomp_Indicator",
    "Elektrisch_Voertuig_Indicator", "PV_Installatie_Indicator", "Contract_Categorie",
]


def day(meter, date="2024-01-01", *, pv=0, ev=0, hp=0, value=0.25):
    return [
        [str(meter), ts.strftime("%Y-%m-%d"), ts.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
         value, 0, hp, ev, pv, "Residentieel"]
        for ts in pd.date_range(date, periods=96, freq="15min", tz="UTC")
    ]


def archive(tmp_path, members):
    path = tmp_path / "raw.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, rows in members.items():
            out = io.StringIO()
            writer = csv.writer(out)
            writer.writerow(COLUMNS)
            writer.writerows(rows)
            zf.writestr(name, out.getvalue())
    return path


def fixture_rows(meters=(1, 2, 10)):
    return [row for meter in meters for date in ("2024-01-01", "2024-01-06") for row in day(meter, date)]


def test_prepare_preserves_units_and_split_leakage(tmp_path):
    source = archive(tmp_path, {"profiles.csv": fixture_rows()})
    metadata = prepare(source, tmp_path / "prepared", {"chunksize": 31})
    with np.load(tmp_path / "prepared" / "dataset.npz") as data:
        assert data["x"].shape == (6, 96)
        assert data["x"].dtype == np.float32
        np.testing.assert_allclose(data["x"].sum(axis=1), 24)
        assert set(data["meter_id"]) == {"1", "2", "10"}
        assert set(data["split"]) == {"train", "val", "test"}
        for meter in np.unique(data["meter_id"]):
            assert len(np.unique(data["split"][data["meter_id"] == meter])) == 1
        assert set(data["y"][data["split"] == "train"]) == {0, 1}
    assert metadata["unit"] == "kWh"
    assert metadata["interval_hours"] == 0.25
    assert metadata["audit"]["profiles_kept"] == 6
    assert len(metadata["source_sha256"]) == 64
    assert json.loads((tmp_path / "prepared" / "metadata.json").read_text()) == metadata


def test_hourly_aggregation_conserves_energy(tmp_path):
    source = archive(tmp_path, {"profiles.csv": fixture_rows()})
    metadata = prepare(source, tmp_path / "prepared", {"resolution_minutes": 60, "chunksize": 101})
    with np.load(tmp_path / "prepared" / "dataset.npz") as data:
        assert data["x"].shape == (6, 24)
        np.testing.assert_allclose(data["x"], 1)
        np.testing.assert_allclose(data["x"].sum(axis=1), 24)
    assert metadata["interval_hours"] == 1


def test_bad_days_dropped_with_reasons(tmp_path):
    rows = []
    for meter in (1, 2, 3):
        rows += day(meter)
        rows += day(meter, "2024-01-02")[:-1]
        duplicate = day(meter, "2024-01-03")
        duplicate.insert(1, duplicate[0].copy())
        rows += duplicate
        invalid = day(meter, "2024-01-04")
        invalid[0][3] = "NaN"
        rows += invalid
        negative = day(meter, "2024-01-05")
        negative[0][3] = -1
        rows += negative
    source = archive(tmp_path, {"profiles.csv": rows})
    metadata = prepare(source, tmp_path / "prepared", {"chunksize": 29})
    assert metadata["audit"]["profiles_kept"] == 3
    dropped = metadata["audit"]["dropped_days_by_reason"]
    assert dropped["incomplete_day"] == 3
    assert dropped["duplicate_interval"] == 3
    assert dropped["invalid_energy"] == 3
    assert dropped["negative_energy"] == 3


def test_duplicate_profile_across_files_fails(tmp_path):
    source = archive(tmp_path, {"a.csv": fixture_rows(), "b.csv": day(1)})
    with pytest.raises(ValueError, match="Duplicate.*profile|duplicate.*profile"):
        prepare(source, tmp_path / "prepared", {})


def test_same_identifier_across_files_keeps_same_split(tmp_path):
    source = archive(tmp_path, {
        "a.csv": [r for m in (1, 2, 3) for r in day(m)],
        "b.csv": [r for m in (1, 2, 3) for r in day(m, "2024-01-06")],
    })
    prepare(source, tmp_path / "prepared", {})
    with np.load(tmp_path / "prepared" / "dataset.npz") as data:
        for meter in ("1", "2", "3"):
            assert len(set(data["split"][data["meter_id"] == meter])) == 1


def test_missing_and_inconsistent_equipment_fails(tmp_path):
    rows = fixture_rows()
    rows[0][5] = ""
    source = archive(tmp_path, {"profiles.csv": rows})
    with pytest.raises(ValueError, match="equipment"):
        prepare(source, tmp_path / "prepared", {})
    rows = fixture_rows()
    rows[100][5] = 1
    source = archive(tmp_path, {"profiles.csv": rows})
    with pytest.raises(ValueError, match="equipment"):
        prepare(source, tmp_path / "prepared", {})


def test_small_stratum_fails(tmp_path):
    source = archive(tmp_path, {"profiles.csv": fixture_rows((1, 2))})
    with pytest.raises(ValueError, match="at least 3 meters"):
        prepare(source, tmp_path / "prepared", {})


def test_subset_reads_full_meters(tmp_path):
    source = archive(tmp_path, {"profiles.csv": fixture_rows((1, 2, 3, 4, 5))})
    metadata = prepare(source, tmp_path / "prepared", {"max_meters_per_file": 3, "chunksize": 103})
    with np.load(tmp_path / "prepared" / "dataset.npz") as data:
        assert data["x"].shape == (6, 96)
        assert set(data["meter_id"]) == {"1", "2", "3"}
    assert metadata["audit"]["subset"] is True


def test_net_can_be_negative_and_existing_output_is_not_overwritten(tmp_path):
    rows = fixture_rows()
    for row in rows:
        row[4] = 0.5
    source = archive(tmp_path, {"profiles.csv": rows})
    prepare(source, tmp_path / "prepared", {"target": "net"})
    with np.load(tmp_path / "prepared" / "dataset.npz") as data:
        np.testing.assert_allclose(data["x"], -0.25)
    with pytest.raises(FileExistsError):
        prepare(source, tmp_path / "prepared", {"target": "net"})


def test_out_of_order_input_fails(tmp_path):
    rows = day(1, "2024-01-02") + day(1) + day(2) + day(3)
    source = archive(tmp_path, {"profiles.csv": rows})
    with pytest.raises(ValueError, match="sorted|order"):
        prepare(source, tmp_path / "prepared", {})


def test_unseen_calendar_label_in_held_out_data_fails(tmp_path):
    # Every meter has its own season: whichever meter is training cannot cover the others.
    rows = day(1, "2024-01-01") + day(2, "2024-04-01") + day(3, "2024-07-01")
    source = archive(tmp_path, {"profiles.csv": rows})
    with pytest.raises(ValueError, match="training.*vocabulary|training.*label"):
        prepare(source, tmp_path / "prepared", {})


def test_source_wall_clock_follows_legend_and_drops_nominal_96_dst_days(tmp_path):
    rows = [r for m in (1, 2, 3) for date in ("2024-01-01", "2024-03-31", "2024-10-27") for r in day(m, date)]
    source = archive(tmp_path, {"profiles.csv": rows})
    metadata = prepare(source, tmp_path / "prepared", {"chunksize": 47})
    assert metadata["timestamp_policy"] == "source_wall_clock"
    assert metadata["timezone"] == "Europe/Brussels"
    assert metadata["audit"]["dropped_days_by_reason"]["dst_transition_day"] == 6
    with np.load(tmp_path / "prepared" / "dataset.npz") as data:
        assert data["x"].shape == (3, 96)
        assert set(data["date"]) == {"2024-01-01"}


def test_utc_to_local_is_explicit_and_drops_dst_without_sort_failure(tmp_path):
    rows = []
    for meter in (1, 2, 3):
        # A complete local pre-change day plus the 25-hour fall transition day.
        for ts in pd.date_range("2024-10-25T22:00:00Z", "2024-10-27T22:45:00Z", freq="15min"):
            rows.append([str(meter), ts.strftime("%Y-%m-%d"), ts.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                         0.25, 0, 0, 0, 0, "Residentieel"])
    source = archive(tmp_path, {"profiles.csv": rows})
    metadata = prepare(source, tmp_path / "prepared", {"timestamp_policy": "utc_to_local", "timezone": "Europe/Brussels", "chunksize": 31})
    assert metadata["audit"]["dropped_days_by_reason"]["dst_transition_day"] == 3
    with np.load(tmp_path / "prepared" / "dataset.npz") as data:
        assert data["x"].shape == (3, 96)
        assert set(data["date"]) == {"2024-10-26"}


def test_rejected_profile_still_cannot_repeat_in_another_file(tmp_path):
    incomplete = day(1)[:-1]
    source = archive(tmp_path, {"a.csv": incomplete, "b.csv": fixture_rows()})
    with pytest.raises(ValueError, match="Duplicate daily profile"):
        prepare(source, tmp_path / "prepared", {})


def test_chunk_size_does_not_change_values_or_splits_and_preserves_leading_zero_ids(tmp_path):
    source = archive(tmp_path, {"profiles.csv": fixture_rows(("001", "002", "010"))})
    prepare(source, tmp_path / "small", {"chunksize": 17})
    prepare(source, tmp_path / "large", {"chunksize": 100_000})
    with np.load(tmp_path / "small" / "dataset.npz") as small, np.load(tmp_path / "large" / "dataset.npz") as large:
        for name in small.files:
            np.testing.assert_array_equal(small[name], large[name])
        assert set(small["meter_id"]) == {"001", "002", "010"}


@pytest.mark.parametrize("config", [
    {"chunkszie": 100}, {"chunksize": 1.5}, {"chunksize": True},
    {"seed": 1.5}, {"seed": -1}, {"max_meters_per_file": 3.5},
    {"resolution_minutes": 15.0},
])
def test_invalid_configuration_is_rejected(config):
    with pytest.raises(ValueError):
        _config(config)


def test_nonresidential_input_is_rejected(tmp_path):
    rows = fixture_rows()
    rows[0][-1] = "Niet-residentieel"
    source = archive(tmp_path, {"profiles.csv": rows})
    with pytest.raises(ValueError, match="Residentieel|residential"):
        prepare(source, tmp_path / "prepared", {})


def test_timestamp_storage_is_nanoseconds_even_if_parser_returns_microseconds(monkeypatch):
    original = pd.to_datetime

    def microsecond_datetime(*args, **kwargs):
        result = original(*args, **kwargs)
        return result.dt.as_unit("us") if isinstance(result, pd.Series) else result.as_unit("us")

    monkeypatch.setattr(pd, "to_datetime", microsecond_datetime)
    frame = pd.DataFrame(day("001"), columns=COLUMNS).astype(str)
    parsed = _parse_chunk(frame, _config({}))
    assert parsed["_timestamp"].iloc[0] == pd.Timestamp("2024-01-01").value
    assert parsed["_timestamp"].iloc[1] - parsed["_timestamp"].iloc[0] == 900_000_000_000


def test_equipment_audit_and_condition_split_counts_reconcile(tmp_path):
    rows = []
    for pv in (0, 1):
        for meter in range(1 + 3 * pv, 4 + 3 * pv):
            rows += day(meter, "2024-01-01", pv=pv)
            rows += day(meter, "2024-01-02", pv=pv)
            rows += day(meter, "2024-01-03", pv=pv)[:-1]
    source = archive(tmp_path, {"profiles.csv": rows})
    metadata = prepare(source, tmp_path / "prepared", {"chunksize": 43})
    audit = metadata["audit"]
    groups = audit["equipment_groups"]
    assert set(groups) == {"PV=0|EV=0|HP=0", "PV=1|EV=0|HP=0"}
    for metric in ("profiles_seen", "profiles_kept", "profiles_dropped"):
        assert sum(group[metric] for group in groups.values()) == audit[metric]
    for group in groups.values():
        assert group["profiles_seen"] == 9
        assert group["profiles_kept"] == 6
        assert group["profiles_dropped"] == 3
        assert group["dropped_days_by_reason"] == {"incomplete_day": 3}
    counts = metadata["condition_split_counts"]
    assert set(counts) == set(metadata["classes"])
    for condition in counts.values():
        assert condition == {split: {"profiles": 2, "unique_meters": 1} for split in ("train", "val", "test")}
    assert sum(cell["profiles"] for condition in counts.values() for cell in condition.values()) == audit["profiles_kept"]
