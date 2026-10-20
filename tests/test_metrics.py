import numpy as np
import pandas as pd
import pytest
import torch
from scipy.linalg import sqrtm

from cgan.eval.fast import frechet_distance, gaussian_stats
from cgan.eval.kpis import price_kpis, reference_kpis
from cgan.eval.metrics import (aggregate_metrics, dcr_metrics, energy_distance, jsd_kl, mmd2_unbiased, prdc, scalar_metrics,
                               temporal_metrics)

rng = np.random.default_rng(0)


def test_frechet_matches_sqrtm():
    a, b = rng.normal(size=(500, 6)), rng.normal(0.3, 1.5, size=(400, 6))
    m1, s1 = gaussian_stats(a)
    m2, s2 = gaussian_stats(b)
    ref = np.sum((m1 - m2) ** 2) + np.trace(s1 + s2 - 2 * np.real(sqrtm(s1 @ s2)))
    assert frechet_distance(m1, s1, m2, s2) == pytest.approx(ref, rel=1e-6)


def test_mmd_matches_naive_loop():
    x, y = rng.normal(size=(30, 3)), rng.normal(0.5, 1, size=(25, 3))
    sig = [0.7, 1.4]

    def k(a, b):
        return sum(np.exp(-np.sum((a - b) ** 2) / (2 * s * s)) for s in sig)

    m, n = len(x), len(y)
    kxx = sum(k(x[i], x[j]) for i in range(m) for j in range(m) if i != j) / (m * (m - 1))
    kyy = sum(k(y[i], y[j]) for i in range(n) for j in range(n) if i != j) / (n * (n - 1))
    kxy = sum(k(x[i], y[j]) for i in range(m) for j in range(n)) / (m * n)
    got = mmd2_unbiased(torch.tensor(x), torch.tensor(y), sig)
    assert got == pytest.approx(kxx + kyy - 2 * kxy, rel=1e-8)


def test_energy_distance_matches_naive():
    x, y = rng.normal(size=(20, 2)), rng.normal(1, 1, size=(15, 2))
    d = lambda a, b: np.linalg.norm(a - b)  # noqa: E731
    exy = np.mean([d(a, b) for a in x for b in y])
    exx = np.sum([d(a, b) for a in x for b in x]) / (len(x) * (len(x) - 1))
    eyy = np.sum([d(a, b) for a in y for b in y]) / (len(y) * (len(y) - 1))
    assert energy_distance(torch.tensor(x), torch.tensor(y)) == pytest.approx(2 * exy - exx - eyy, rel=1e-8)


def test_prdc_identical_distributions():
    x = torch.tensor(rng.normal(size=(2000, 4)))
    y = torch.tensor(rng.normal(size=(2000, 4)))
    r = prdc(x, y, k=5)
    # Naeem et al. (2020): E[coverage] = 1 - prod_{l=1..k} (N - l) / (M + N - l) ~ 0.969, E[density] = 1
    assert r["coverage"] == pytest.approx(0.969, abs=0.02)
    assert r["density"] == pytest.approx(1.0, abs=0.08)


def test_jsd_bounds_and_identity():
    a = rng.normal(size=5000)
    jsd_same, kl_same = jsd_kl(a, rng.normal(size=5000))
    jsd_far, _ = jsd_kl(a, rng.normal(10, 1, size=5000))
    assert 0 <= jsd_same < 0.01 and kl_same < 0.02
    assert 0.5 < jsd_far <= 1.0


def test_scalar_metrics_fd1_equals_reference_formula():
    a, b = rng.normal(2, 1, 400), rng.normal(2.5, 2, 300)
    m = scalar_metrics(a, b)
    assert m["fd1"] == pytest.approx((a.mean() - b.mean()) ** 2 + (a.std(ddof=1) - b.std(ddof=1)) ** 2)


def test_temporal_metrics_zero_for_identical():
    x = rng.gamma(2, 1, size=(300, 24))
    t = temporal_metrics(x, x.copy())
    assert t["mean_profile_rmse"] == 0 and t["corr_mad"] == 0 and t["hourly_w1"] == 0


def test_dcr_copies_vs_fresh():
    train = rng.normal(size=(3000, 5))
    hold = rng.normal(size=(1000, 5))
    copies = dcr_metrics(train[rng.choice(3000, 1000, replace=False)], train, hold)
    fresh = dcr_metrics(rng.normal(size=(1000, 5)), train, hold)
    assert copies["dcr_copies"] == 1.0 and copies["dcr_q05"] == 1.0
    assert fresh["dcr_q05"] == pytest.approx(0.05, abs=0.03) and fresh["dcr_copies"] == 0
    assert fresh["dcr_ratio_med"] == pytest.approx(1.0, abs=0.1)


def test_aggregate_metrics_identical_and_collapsed():
    real = rng.gamma(2, 0.5, size=(3000, 24))
    same = aggregate_metrics(real, rng.permutation(real), seed=1)
    collapsed = aggregate_metrics(real, np.tile(real.mean(0), (3000, 1)), seed=1)
    assert abs(same["agg_peak_p95_rel"]) < 0.05
    assert collapsed["agg_cf_mean_diff"] > 0.3  # identical profiles peak simultaneously


def _reference_pandas(df):
    """Verbatim logic of cGAN_EB utils.stats_plot (rows of a 24-column frame)."""
    out = []
    for i in range(len(df)):
        row = df.iloc[i]
        peak = row.sort_values().iloc[22]
        base = row.sort_values().iloc[1]
        mean = (peak + base) / 2
        dur = len(row[row > mean])
        rise = abs(int(abs(row - peak).idxmax()) - int(abs(row - peak).idxmin()))
        fall = abs(int(abs(row - base).idxmax()) - int(abs(row - base).idxmin()))
        out.append((peak, base, dur, rise, fall))
    return np.array(out, dtype=float)


def test_reference_kpis_match_reference_code():
    X = np.round(rng.gamma(2, 1, size=(200, 24)) - 1.0, 2)  # rounding creates ties
    df = pd.DataFrame(X, columns=[str(i) for i in range(24)])
    ref = _reference_pandas(df)
    k = reference_kpis(X)
    got = np.stack([k["ref_peak"], k["ref_base"], k["ref_duration"], k["ref_rise"], k["ref_fall"]], 1)
    np.testing.assert_allclose(got, ref)


def test_price_kpis_on_a_simple_profile():
    x = np.zeros(24)
    x[17:21] = 4.0  # evening block
    x[16] = 1.0     # shoulder
    k = price_kpis(x[None])
    assert k["near_peak"][0] == pytest.approx(4.0)
    assert k["near_base"][0] == pytest.approx(0.0)
    assert k["high_load_duration"][0] == 4
    assert k["rise_time"][0] == 2   # last close-to-base hour 15 -> first high hour 17
    assert k["fall_time"][0] == 1   # last high hour 20 -> close-to-base at 21


def test_bootstrap_weighted_terms_match_duplicated_samples():
    from cgan.bootstrap import weighted_ed_mmd

    x = torch.tensor(rng.normal(size=(40, 3)))
    y = torch.tensor(rng.normal(0.3, 1, size=(30, 3)))
    w = torch.tensor(rng.integers(0, 3, size=40), dtype=torch.float64)
    w[0] = 2  # ensure duplicates
    sig = [0.8, 1.6]
    e, k = weighted_ed_mmd(x, y, w[None, :], sig)
    xd = x.repeat_interleave(w.long(), dim=0)  # explicit duplication
    assert float(e[0]) == pytest.approx(energy_distance(xd, y), rel=1e-9)
    assert float(k[0]) == pytest.approx(mmd2_unbiased(xd, y, sig), rel=1e-9)


def test_c2st_grouped_folds_ignore_household_fingerprints():
    from cgan.eval.metrics import c2st_auc

    # Real days cluster around 30 household means; synthetic days come from the same population.
    hh = rng.normal(size=(30, 4))
    g = np.repeat(np.arange(30), 40)
    real = hh[g] + 0.2 * rng.normal(size=(1200, 4))
    syn = hh[rng.integers(0, 30, 1200)] + 0.2 * rng.normal(size=(1200, 4))
    syn_new = rng.normal(size=(1200, 4)) * np.sqrt(1.04)  # new households, same population
    grouped = c2st_auc(real, syn_new, groups_real=g)
    assert grouped < 0.62
    assert c2st_auc(real, syn) < 0.62


def test_overflow_bins_penalise_overshoot():
    real = rng.normal(size=4000)
    inside, _ = jsd_kl(real, np.clip(rng.normal(size=4000), real.min(), real.max()))
    overshoot, _ = jsd_kl(real, np.r_[rng.normal(size=3600), np.full(400, real.max() + 50)])
    assert overshoot > inside + 0.03


def test_dcr_robust_to_holdout_days_in_train():
    train = rng.normal(size=(2000, 5))
    hold = np.r_[train[:200], rng.normal(size=(800, 5))]  # 20 % of holdout days also in Train
    copies = dcr_metrics(train[rng.choice(2000, 500, replace=False)], train, hold)
    assert copies["dcr_q05"] == 1.0 and copies["dcr_holdout_in_train"] == pytest.approx(0.2)
