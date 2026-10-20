from pathlib import Path

import numpy as np
import pytest
import torch

from cgan.baselines import PerLabelBaseline
from cgan.data.prepare import GROUP_NAMES, group_index
from cgan.data.profiles import (N_LABELS, Split, _aggregate, calendar, decode_label, label_names, label_of,
                                split_meters)
from cgan.data.scaling import Scaler
from cgan.eval.conditional import decode_factors
from cgan.models.nets import FactorCond, build_nets, decode_label_t
from cgan.utils import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_label_layout_matches_reference():
    names = label_names()
    assert len(set(names)) == N_LABELS == 64
    assert names[0] == "Weekday - none Spring"
    assert names[5] == "Weekday - PV Summer"
    assert names[32] == "Weekend - none Spring"
    assert names[63] == "Weekend - HP+EV+PV Winter"
    d, g, s = decode_label(np.arange(64))
    np.testing.assert_array_equal(label_of(d, g, s), np.arange(64))


def test_group_bits_consistent():
    for pv in (0, 1):
        for hp in (0, 1):
            for ev in (0, 1):
                g = group_index(pv, hp, ev)
                name = GROUP_NAMES[g]
                assert ("PV" in name) == bool(pv) and ("HP" in name) == bool(hp) and ("EV" in name) == bool(ev)
    lab = torch.arange(64)
    dt, grp, se = decode_label_t(lab)
    f = decode_factors(lab.numpy())
    np.testing.assert_array_equal(grp.numpy(), f["group"])
    np.testing.assert_array_equal(dt.numpy(), f["daytype"])
    np.testing.assert_array_equal(se.numpy(), f["season"])
    assert FactorCond()(lab).shape == (64, FactorCond().dim)


def test_calendar_2024():
    daytype, season, month, dates = calendar(holidays_as_weekend=True)
    assert len(dates) == 366
    assert daytype[0] == 1          # 1 Jan 2024 (Monday) is a public holiday
    assert daytype[1] == 0          # 2 Jan Tuesday
    assert daytype[5] == 1          # 6 Jan Saturday
    assert season[0] == 3 and season[59] == 3 and season[60] == 0  # Feb 29 winter, Mar 1 spring
    d2, *_ = calendar(holidays_as_weekend=False)
    assert d2[0] == 0


def test_split_disjoint_and_stratified():
    group = np.repeat(np.arange(8), 300)
    a = split_meters(group, (0.7, 0.15, 0.15), 2024)
    for g in range(8):
        np.testing.assert_array_equal(np.bincount(a[group == g], minlength=3), [210, 45, 45])
    np.testing.assert_array_equal(a, split_meters(group, (0.7, 0.15, 0.15), 2024))


def test_aggregate_hourly_two_channels():
    x = np.arange(2 * 3 * 192, dtype=float).reshape(2, 3, 192)  # 2 channels x 96 quarters
    y = _aggregate(x, 4, channels=2)
    assert y.shape == (2, 3, 48)
    assert y[0, 0, 0] == 0 + 1 + 2 + 3
    assert y[0, 0, 24] == sum(range(96, 100))


@pytest.mark.parametrize("kind", ["minmax", "maxabs", "standard", "asinh"])
def test_scaler_roundtrip(kind):
    x = np.random.default_rng(1).normal(0.3, 2, size=(500, 24))
    s = Scaler(kind).fit(x)
    np.testing.assert_allclose(s.inverse(s.transform(x)), x, rtol=1e-9, atol=1e-9)
    t = torch.tensor(s.transform(x))
    np.testing.assert_allclose(s.inverse_torch(t).numpy(), x, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("kind", ["gaussian", "copula", "gmm", "resample"])
def test_baseline_fit_sample_save_load(kind, tmp_path):
    rng = np.random.default_rng(2)
    lab = np.repeat([0, 5], 400)
    X = rng.gamma(2, 1, size=(800, 24)).astype(np.float32) + lab[:, None]
    sp = Split(X, lab)
    m = PerLabelBaseline(kind, gmm_components=(1, 2)).fit(sp, sp)
    out = m.sample(lab, seed=3)
    assert out.shape == X.shape and np.isfinite(out).all()
    assert abs(out[lab == 5].mean() - X[lab == 5].mean()) < 0.5
    m.save(tmp_path / "m.npz")
    m2 = PerLabelBaseline.load(tmp_path / "m.npz")
    np.testing.assert_allclose(m2.sample(lab, seed=3), out)
    assert not np.allclose(m.sample(lab, seed=4), out)


@pytest.mark.parametrize("name", ["cgan_eb", "cwgan_gp", "abl_onehot", "abl_noproj", "abl_minmax",
                                  "abl_refnet_wgan"])
def test_configs_build(name):
    cfg = load_config(ROOT / "configs" / f"{name}.yaml")
    G, D = build_nets(cfg["model"], 24)
    y = torch.randint(0, 64, (7,))
    x = G(torch.randn(7, G.noise_dim), y)
    assert x.shape == (7, 24) and D(x, y).shape == (7, 1)


def test_reference_architecture_parameter_counts():
    # cGAN_EB: G 88-64-128-64-24, D 88-64-128-256-1
    G, D = build_nets(load_config(ROOT / "configs" / "cgan_eb.yaml")["model"], 24)
    assert sum(p.numel() for p in G.parameters()) == 88 * 64 + 64 + 64 * 128 + 128 + 128 * 64 + 64 + 64 * 24 + 24
    assert sum(p.numel() for p in D.parameters()) == 88 * 64 + 64 + 64 * 128 + 128 + 128 * 256 + 256 + 256 + 1
