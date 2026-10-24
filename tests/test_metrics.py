import numpy as np
import pytest
from scipy.stats import ks_2samp

from data_prep import N_LABELS, calendar, decode_label, label_names, label_of, split_meters
from metrics import _ks, accuracy, frechet, key_parameters, reference_key_parameters


def test_label_round_trip():
    lab = np.arange(N_LABELS)
    d, g, s = decode_label(lab)
    assert (label_of(d, g, s) == lab).all()
    assert label_names()[0] == "Weekday - No PV HP EV Spring"
    assert label_names()[63] == "Weekend - PV+HP+EV Winter"


def test_calendar():
    daytype, season, dates = calendar()
    assert len(dates) == 366
    assert daytype[0] == 1                      # 1 Jan 2024: holiday (a Monday)
    assert daytype[5] == 1 and daytype[1] == 0  # Saturday 6 Jan, Tuesday 2 Jan
    assert season[0] == 3 and season[59] == 3 and season[60] == 0   # 29 Feb winter, 1 Mar spring


def test_split_is_partition():
    group = np.repeat(np.arange(8), 300)
    a = split_meters(group)
    assert set(np.unique(a)) == {0, 1, 2}
    for g in range(8):
        assert np.bincount(a[group == g]).tolist() == [210, 45, 45]


@pytest.mark.parametrize("n,m", [(500, 700), (40, 2000)])
def test_ks_matches_scipy(n, m):
    rng = np.random.default_rng(n)
    a, b = rng.normal(size=n), np.round(rng.normal(0.2, 1.3, size=m), 1)
    assert np.isclose(_ks(a, b), ks_2samp(a, b).statistic)


def test_key_parameters_simple_day():
    x = np.full(24, 0.2)
    x[18:21] = 2.0                               # three high-load hours, 18-20 h
    x[16:18] = [0.8, 1.4]
    kp = key_parameters(x[None])
    assert kp["peak_duration"][0] == 4           # hours above (peak + base) / 2 = 1.1: 17, 18, 19, 20
    assert kp["rise_time"][0] == 2               # last near-base hour 15 -> first high-load hour 17
    assert kp["fall_time"][0] == 1               # last high-load hour 20 -> 21
    assert np.isclose(kp["daily_energy"][0], x.sum())
    ref = reference_key_parameters(x[None])
    assert ref["Peak_load"][0] == 2.0 and ref["Base_load"][0] == 0.2


def test_accuracy_bounds():
    rng = np.random.default_rng(0)
    real = rng.gamma(2, 0.3, size=(3000, 24))
    same = rng.gamma(2, 0.3, size=(3000, 24))
    mean_only = np.repeat(real.mean(0, keepdims=True), 3000, 0)
    a_same, a_mean = accuracy(real, same), accuracy(real, mean_only)
    assert a_same["accuracy"] > 0.95
    assert a_mean["mean_profile"] > 0.99 and a_mean["std_profile"] < 1e-9
    assert a_mean["accuracy"] < 0.5


def test_frechet_identity():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(1000, 5))
    c = np.cov(x, rowvar=False)
    assert abs(frechet(x.mean(0), c, x.mean(0), c)) < 1e-8


def test_key_parameters_pv_day():
    # PV day: export at midday sets the near-base load; the evening peak is the high-load period.
    x = np.array([0.3] * 7 + [0.0, -0.5, -1.5, -2.0, -2.2, -2.2, -2.0, -1.5, -0.5, 0.2] + [0.8, 1.6, 1.8, 1.2, 0.6, 0.4, 0.3])
    kp = key_parameters(x[None])
    mid = (kp["near_peak"][0] + kp["near_base"][0]) / 2
    assert kp["peak_duration"][0] == (x > mid).sum()
    # midpoint -0.26 kWh: the run around the maximum (19 h) spans 16-23 h
    assert kp["rise_time"][0] == 3               # last near-base hour 13 -> start of the run, 16 h
    assert kp["fall_time"][0] == 0               # the run lasts until the end of the day


def test_all_modules_compile():
    import py_compile
    from pathlib import Path

    for f in Path(__file__).resolve().parents[1].glob("*.py"):
        py_compile.compile(str(f), doraise=True)
