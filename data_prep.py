"""Prepare the Fluvius 2024 smart-meter release for the cGAN.

    python data_prep.py --raw data/raw/raw.zip --out data/processed/fluvius_hourly.npz

``raw.zip`` (release ``data`` of github.com/MahmoudAsadi97/cGAN) holds one CSV
per technology group with quarter-hourly import (``Volume_Afname_KWh``) and
export (``Volume_Injectie_KWh``) of 300 residential meters each, plus the
per-meter heat pump, EV and PV indicators.  Timestamps are local wall-clock
time (CET/CEST, see the legend in the archive), so 31 March has 92 and
27 October 100 quarter hours.

The script aggregates the quarter hours to hourly energy per meter and day and
stores, for every meter, the 366 x 24 import / export arrays, the number of
quarter hours behind each hourly value and the indicators.  ``load_profiles``
turns the arrays into labelled daily profiles, following the data processing
of Li, Dong & Qiu (2025, Sec. 2.1-2.2):

* day type: weekday / weekend, calendar (Belgian public) holidays counted as weekend;
* season: spring MAM, summer JJA, fall SON, winter DJF;
* technology group from the indicators (bit 0 PV, bit 1 heat pump, bit 2 EV);
* label = daytype * 32 + group * 4 + season  (64 labels, as in the reference);
* profiles with missing or incorrect recordings are removed: days whose hours
  do not each consist of exactly four quarter hours (the two DST days) and
  meter-days without any energy flow in all 24 hours.

The release has a yes/no EV indicator but no charger levels, so the three EV
charger classes of the reference (Level 1, Level 2, both) are replaced by the
heat-pump indicator, which keeps 8 groups and 64 labels.

Households are split into train / validation / test (70 / 15 / 15 %,
stratified by group); no household contributes days to two splits.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

YEAR = 2024
N_DAYS = 366
N_HOURS = 24
N_METERS = 2400

# Order of the groups follows the reference (no technology, PV, then the EV classes):
# group = pv + 2 * hp + 4 * ev.
GROUP_NAMES = ["No PV HP EV", "PV", "HP", "PV+HP", "EV", "PV+EV", "HP+EV", "PV+HP+EV"]
SEASON_NAMES = ["Spring", "Summer", "Fall", "Winter"]
DAYTYPE_NAMES = ["Weekday", "Weekend"]
N_GROUPS, N_SEASONS, N_DAYTYPES = len(GROUP_NAMES), len(SEASON_NAMES), len(DAYTYPE_NAMES)
N_LABELS = N_DAYTYPES * N_GROUPS * N_SEASONS

# Belgian public holidays 2024.
HOLIDAYS = ["2024-01-01", "2024-04-01", "2024-05-01", "2024-05-09", "2024-05-20",
            "2024-07-21", "2024-08-15", "2024-11-01", "2024-11-11", "2024-12-25"]


# ---------------------------------------------------------------------------
# raw.zip -> hourly arrays
# ---------------------------------------------------------------------------
def _read_csv(f):
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.csv as pv

    tb = pv.read_csv(f, convert_options=pv.ConvertOptions(column_types={
        "EAN_ID": pa.int64(), "Datum": pa.string(), "Datum_Startuur": pa.string(),
        "Volume_Afname_KWh": pa.float64(), "Volume_Injectie_KWh": pa.float64(),
        "Warmtepomp_Indicator": pa.int64(), "Elektrisch_Voertuig_Indicator": pa.int64(),
        "PV_Installatie_Indicator": pa.int64(), "Contract_Categorie": pa.string()}))
    ts = tb["Datum_Startuur"]
    if not pc.all(pc.equal(pc.utf8_slice_codeunits(ts, 0, 10), tb["Datum"])).as_py():
        raise ValueError("Datum and Datum_Startuur disagree")
    epoch = (dt.date(YEAR, 1, 1) - dt.date(1970, 1, 1)).days
    day = pc.cast(pc.cast(tb["Datum"], pa.date32()), pa.int32()).to_numpy() - epoch
    hour = pc.cast(pc.utf8_slice_codeunits(ts, 11, 13), pa.int64()).to_numpy()
    cols = {
        "ean": tb["EAN_ID"].to_numpy(), "day": day.astype(np.int64), "hour": hour,
        "imp": tb["Volume_Afname_KWh"].to_numpy(), "exp": tb["Volume_Injectie_KWh"].to_numpy(),
        "hp": tb["Warmtepomp_Indicator"].to_numpy(), "ev": tb["Elektrisch_Voertuig_Indicator"].to_numpy(),
        "pv": tb["PV_Installatie_Indicator"].to_numpy(),
        "contract": pc.unique(tb["Contract_Categorie"]).to_pylist(),
    }
    return cols


def convert(raw: Path, out: Path) -> dict:
    """Aggregate every CSV in ``raw`` to hourly arrays and save them to ``out`` (npz)."""
    shape = (N_METERS, N_DAYS, N_HOURS)
    imp = np.zeros(shape)
    exp_ = np.zeros(shape)
    count = np.zeros(shape, np.int64)
    flags = np.full((N_METERS, 3), -1, np.int64)   # pv, hp, ev
    source = np.full(N_METERS, "", object)
    log = []
    with zipfile.ZipFile(raw) as z:
        for name in sorted(n for n in z.namelist() if n.endswith(".csv")):
            t0 = time.time()
            with z.open(name) as f:
                c = _read_csv(f)
            if (c["ean"] < 1).any() or (c["ean"] > N_METERS).any():
                raise ValueError(f"{name}: EAN_ID outside 1..{N_METERS}")
            if (c["day"] < 0).any() or (c["day"] >= N_DAYS).any() or (c["hour"] < 0).any() or (c["hour"] >= 24).any():
                raise ValueError(f"{name}: timestamp outside {YEAR}")
            m = c["ean"] - 1
            if (source[np.unique(m)] != "").any():
                raise ValueError(f"{name}: EAN_ID already seen in another file")
            idx = np.ravel_multi_index((m, c["day"], c["hour"]), shape)
            imp += np.bincount(idx, c["imp"], imp.size).reshape(shape)
            exp_ += np.bincount(idx, c["exp"], exp_.size).reshape(shape)
            count += np.bincount(idx, None, count.size).reshape(shape).astype(np.int64)
            for j, k in enumerate(("pv", "hp", "ev")):
                lo = np.full(N_METERS, 2, np.int64)
                hi = np.full(N_METERS, -1, np.int64)
                np.minimum.at(lo, m, c[k])
                np.maximum.at(hi, m, c[k])
                um = np.unique(m)
                if (lo[um] != hi[um]).any():
                    raise ValueError(f"{name}: indicator {k} varies within a meter")
                flags[um, j] = lo[um]
            source[np.unique(m)] = Path(name).stem
            entry = {"file": Path(name).name, "rows": int(len(m)), "meters": int(len(np.unique(m))),
                     "contracts": c["contract"], "negative_values": int((c["imp"] < 0).sum() + (c["exp"] < 0).sum()),
                     "seconds": round(time.time() - t0, 1)}
            log.append(entry)
            print(entry, flush=True)
    if (source == "").any():
        raise ValueError(f"{int((source == '').sum())} meters missing from the archive")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, import_kwh=imp.astype(np.float32), export_kwh=exp_.astype(np.float32),
                        count=count.astype(np.int8), pv=flags[:, 0], hp=flags[:, 1], ev=flags[:, 2],
                        ean=np.arange(1, N_METERS + 1), source=source.astype(str))
    return {"files": log}


# ---------------------------------------------------------------------------
# hourly arrays -> labelled daily profiles
# ---------------------------------------------------------------------------
def calendar():
    """Day type, season and date of every day of 2024."""
    dates = [dt.date(YEAR, 1, 1) + dt.timedelta(days=i) for i in range(N_DAYS)]
    hol = {dt.date.fromisoformat(h) for h in HOLIDAYS}
    daytype = np.array([int(d.weekday() >= 5 or d in hol) for d in dates])
    month = np.array([d.month for d in dates])
    season = np.select([np.isin(month, [3, 4, 5]), np.isin(month, [6, 7, 8]), np.isin(month, [9, 10, 11])],
                       [0, 1, 2], 3)
    return daytype, season, dates


def label_of(daytype, group, season):
    return np.asarray(daytype) * (N_GROUPS * N_SEASONS) + np.asarray(group) * N_SEASONS + np.asarray(season)


def decode_label(label):
    label = np.asarray(label)
    return label // (N_GROUPS * N_SEASONS), (label // N_SEASONS) % N_GROUPS, label % N_SEASONS


def label_names() -> list[str]:
    """``'Weekday - PV+EV Summer'``-style names in label order (as ``cats_`` of the reference utils)."""
    names = []
    for lab in range(N_LABELS):
        d, g, s = (int(v) for v in decode_label(lab))
        names.append(f"{DAYTYPE_NAMES[d]} - {GROUP_NAMES[g]} {SEASON_NAMES[s]}")
    return names


def duplicate_meters(imp: np.ndarray, exp_: np.ndarray) -> np.ndarray:
    """True for meters whose full-year import and export repeat an earlier meter exactly."""
    seen, dup = set(), np.zeros(len(imp), bool)
    for m in range(len(imp)):
        key = hashlib.sha1(imp[m].tobytes() + exp_[m].tobytes()).digest()
        dup[m] = key in seen
        seen.add(key)
    return dup


def split_meters(group: np.ndarray, fractions=(0.7, 0.15, 0.15), seed: int = 2024) -> np.ndarray:
    """0 train / 1 validation / 2 test for every meter, stratified by group."""
    rng = np.random.default_rng(seed)
    out = np.empty(len(group), np.int64)
    for g in np.unique(group):
        idx = rng.permutation(np.flatnonzero(group == g))
        n_tr, n_va = round(fractions[0] * len(idx)), round(fractions[1] * len(idx))
        out[idx[:n_tr]], out[idx[n_tr:n_tr + n_va]], out[idx[n_tr + n_va:]] = 0, 1, 2
    return out


@dataclass
class Split:
    X: np.ndarray                 # (n, 24) kWh per hour
    label: np.ndarray             # (n,) 0..63
    meter: np.ndarray             # (n,) meter index (EAN_ID - 1)
    day: np.ndarray               # (n,) day of year, 0..365
    info: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.X)


def load_profiles(path: str | Path = "data/processed/fluvius_hourly.npz", target: str = "net",
                  split_seed: int = 2024, fractions=(0.7, 0.15, 0.15)) -> dict[str, Split]:
    """Labelled daily profiles of the train / val / test households.

    ``target``: ``net`` = import - export (the load the grid sees; negative while
    PV exports), ``import`` or ``export``.
    """
    z = np.load(path, allow_pickle=False)
    imp, exp_ = z["import_kwh"].astype(np.float64), z["export_kwh"].astype(np.float64)
    X = {"net": imp - exp_, "import": imp, "export": exp_}[target]
    group = z["pv"] + 2 * z["hp"] + 4 * z["ev"]

    daytype, season, _ = calendar()
    complete = (z["count"] == 4).all(-1)                                 # (M, 366); False on the DST days
    no_flow = ((z["import_kwh"] == 0) & (z["export_kwh"] == 0)).all(-1)
    dup = duplicate_meters(z["import_kwh"], z["export_kwh"])
    valid = complete & ~no_flow & ~dup[:, None]
    valid &= (np.abs(X) <= 100).all(-1)                                  # the reference drops loads > 100 kWh/h

    assign = np.full(len(group), -1)
    assign[~dup] = split_meters(group[~dup], fractions, split_seed)
    m_idx, d_idx = np.nonzero(valid)
    lab = label_of(daytype[d_idx], group[m_idx], season[d_idx])
    out = {}
    for k, name in enumerate(("train", "val", "test")):
        s = assign[m_idx] == k
        out[name] = Split(X[m_idx[s], d_idx[s]].astype(np.float32), lab[s], m_idx[s], d_idx[s])
    info = {"duplicate_meters": int(dup.sum()), "dst_or_incomplete_meter_days": int((~complete).sum()),
            "no_flow_meter_days": int((no_flow & complete & ~dup[:, None]).sum()),
            "meters": {n: int(len(np.unique(s.meter))) for n, s in out.items()},
            "days": {n: len(s) for n, s in out.items()}}
    for s in out.values():
        s.info = info
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="data/raw/raw.zip")
    ap.add_argument("--out", default="data/processed/fluvius_hourly.npz")
    args = ap.parse_args(argv)
    summary = convert(Path(args.raw), Path(args.out))
    splits = load_profiles(args.out)
    summary["profiles"] = splits["train"].info
    Path(args.out).with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["profiles"], indent=2))


if __name__ == "__main__":
    main()
