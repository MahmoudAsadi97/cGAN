"""Validation metrics for generated daily load profiles.

Key parameters (Li, Dong & Qiu 2025, Sec. 2.4, after Price 2010), per day:

    near_peak           97.5th percentile of the hourly load
    near_base           2.5th percentile
    peak_duration       hours with load above (near_peak + near_base) / 2
    rise_time           hours from the last near-base hour before the high-load
                        period to its first hour
    fall_time           hours from the last high-load hour to the next
                        near-base hour
    daily_energy        sum of the day (added; not one of the five)

The high-load period is the contiguous run of hours above the midpoint that
contains the daily maximum.  (Taking the first and last high hour of the
whole day would make rise and fall 0 on most PV days, whose night and evening
hours lie above the midpoint set by the midday export.)  A near-base hour has
load <= near_base + 0.1 (near_peak - near_base); if no such hour exists before
(after) the high-load period, the start (end) of the day is used.

Statistical accuracy (the 85 % target) compares, per label, generated days with
real days of held-out households.  Each term is clipped to [0, 1]:

    mean_profile   1 - sum_t |mu_g(t) - mu_r(t)| / sum_t |mu_r(t)|
    std_profile    1 - sum_t |sd_g(t) - sd_r(t)| / sum_t sd_r(t)
    kpi_mean       1 - |mean_g(k) - mean_r(k)| / mean_r(|k|),  averaged over k
    kpi_std        1 - |sd_g(k) - sd_r(k)| / sd_r(k),          averaged over k
    hourly_dist    1 - KS statistic of the hourly values, averaged over hours
    kpi_dist       1 - KS statistic of each key parameter, averaged over k

    accuracy = mean(mean_profile, std_profile, (kpi_mean + kpi_std) / 2,
                    (hourly_dist + kpi_dist) / 2)

Level, spread, key-parameter statistics and full distributions weigh equally,
so matching the average day alone cannot reach a high accuracy.  The KS
statistic is the largest gap between the two empirical CDFs (no binning).

Paper metrics: mean / std differences of the key parameters (Figs. 5-6),
KL(P_real || Q_gen) on histograms with bins at the real quantiles (Figs. 9-10) and the Frechet distance
of the five key parameters (Eq. 5; Figs. 11-12 report it per parameter).

Other metrics (lower is better unless noted):

    rfd            Frechet distance of Gaussian fits to the 24-h profiles / trace of real covariance
    corr_mae       mean |difference| of the 24 x 24 hour-to-hour correlation matrices
    c2st_auc       ROC-AUC of a classifier separating real from generated days, with
                   cross-validation folds grouped by household (and stratified), so the
                   classifier cannot win by recognising households; 0.5 ideal
    precision / recall / coverage   k-NN support metrics (Naeem et al. 2020; higher is better)
    dcr_ratio      median distance of generated days to their nearest training day divided by
                   the same for real held-out days (about 1 expected; << 1 means copying)
    neg_share      share of negative hourly values in labels without PV (0 expected)
"""
from __future__ import annotations

import numpy as np

KEYS = ["near_peak", "near_base", "peak_duration", "rise_time", "fall_time", "daily_energy"]
PAPER_KEYS = KEYS[:5]
BASE_TOL = 0.1
_EPS = 1e-12


# ---------------------------------------------------------------------------
# key parameters
# ---------------------------------------------------------------------------
def key_parameters(X: np.ndarray) -> dict[str, np.ndarray]:
    X = np.asarray(X, np.float64)
    n, T = X.shape
    peak = np.percentile(X, 97.5, axis=1)
    base = np.percentile(X, 2.5, axis=1)
    high = X > ((peak + base) / 2)[:, None]
    near_base = X <= (base + BASE_TOL * (peak - base))[:, None]
    t = np.arange(T)
    p = X.argmax(1)
    has_high = high[np.arange(n), p]
    # contiguous high run around the maximum: [first, last]
    low_before = ~high & (t[None, :] < p[:, None])
    low_after = ~high & (t[None, :] > p[:, None])
    first = np.where(low_before.any(1), T - low_before[:, ::-1].argmax(1), 0)
    last = np.where(low_after.any(1), low_after.argmax(1) - 1, T - 1)
    before = near_base & (t[None, :] < first[:, None])
    start = np.where(before.any(1), T - 1 - before[:, ::-1].argmax(1), 0)
    after = near_base & (t[None, :] > last[:, None])
    end = np.where(after.any(1), after.argmax(1), T - 1)
    return {
        "near_peak": peak,
        "near_base": base,
        "peak_duration": high.sum(1).astype(np.float64),
        "rise_time": np.where(has_high, first - start, 0).astype(np.float64),
        "fall_time": np.where(has_high, end - last, 0).astype(np.float64),
        "daily_energy": X.sum(1),
    }


def reference_key_parameters(X: np.ndarray) -> dict[str, np.ndarray]:
    """The key parameters exactly as computed in cGAN_EB ``utils.stats_plot``."""
    X = np.asarray(X, np.float64)
    s = np.sort(X, axis=1)
    peak, base = s[:, 22], s[:, 1]
    dp, db = np.abs(X - peak[:, None]), np.abs(X - base[:, None])
    return {
        "Peak_load": peak,
        "Base_load": base,
        "Peak_duration": (X > ((peak + base) / 2)[:, None]).sum(1).astype(np.float64),
        "Rise": np.abs(dp.argmax(1) - dp.argmin(1)).astype(np.float64),
        "Fall": np.abs(db.argmax(1) - db.argmin(1)).astype(np.float64),
    }


# ---------------------------------------------------------------------------
# statistical accuracy
# ---------------------------------------------------------------------------
def _acc(err, scale) -> float:
    if not np.isfinite(err) or scale <= _EPS:
        return np.nan
    return float(np.clip(1.0 - err / scale, 0.0, 1.0))


def _ks(a, b) -> float:
    """Two-sample Kolmogorov-Smirnov statistic sup_x |F_a(x) - F_b(x)| (same value as scipy's ks_2samp)."""
    a, b = np.sort(a), np.sort(b)
    x = np.concatenate([a, b])
    fa = np.searchsorted(a, x, side="right") / len(a)
    fb = np.searchsorted(b, x, side="right") / len(b)
    return float(np.max(np.abs(fa - fb)))


def accuracy(real: np.ndarray, gen: np.ndarray, kp_real=None, kp_gen=None) -> dict[str, float]:
    """Statistical accuracy terms of one label (rows = days, columns = hours)."""
    real = np.asarray(real, np.float64)
    gen = np.asarray(gen, np.float64)
    kp_real = kp_real or key_parameters(real)
    kp_gen = kp_gen or key_parameters(gen)
    mu_r, mu_g = real.mean(0), gen.mean(0)
    sd_r, sd_g = real.std(0, ddof=1), gen.std(0, ddof=1)
    out = {
        "mean_profile": _acc(np.abs(mu_g - mu_r).sum(), np.abs(mu_r).sum()),
        "std_profile": _acc(np.abs(sd_g - sd_r).sum(), sd_r.sum()),
        "hourly_dist": 1.0 - float(np.mean([_ks(real[:, t], gen[:, t]) for t in range(real.shape[1])])),
    }
    km, ks, kd = [], [], []
    for k in KEYS:
        r, g = kp_real[k], kp_gen[k]
        km.append(_acc(abs(g.mean() - r.mean()), np.abs(r).mean()))
        ks.append(_acc(abs(g.std(ddof=1) - r.std(ddof=1)), r.std(ddof=1)))
        kd.append(1.0 - _ks(r, g))
        out[f"kpi_mean.{k}"], out[f"kpi_std.{k}"], out[f"kpi_dist.{k}"] = km[-1], ks[-1], kd[-1]
    out["kpi_mean"] = float(np.nanmean(km))
    out["kpi_std"] = float(np.nanmean(ks))
    out["kpi_dist"] = float(np.mean(kd))
    out["kpi"] = 0.5 * (out["kpi_mean"] + out["kpi_std"])
    out["distribution"] = 0.5 * (out["hourly_dist"] + out["kpi_dist"])
    out["accuracy"] = float(np.mean([out["mean_profile"], out["std_profile"], out["kpi"], out["distribution"]]))
    return out


# ---------------------------------------------------------------------------
# paper metrics
# ---------------------------------------------------------------------------
def kl_divergence(real, gen, bins: int = 30, alpha: float = 0.5) -> float:
    """KL(P_real || Q_gen) on bins at the quantiles of the real sample, add-alpha smoothing.

    The outer bins are open-ended, so generated values beyond the real range fall into
    them (and are penalised) instead of stretching the bin grid.  Duplicate edges of
    integer-valued parameters are merged.
    """
    real, gen = np.asarray(real, np.float64), np.asarray(gen, np.float64)
    edges = np.unique(np.quantile(real, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        edges = np.array([edges[0], edges[0]])
    edges[0], edges[-1] = -np.inf, np.inf
    p = np.histogram(real, edges)[0] + alpha
    q = np.histogram(gen, edges)[0] + alpha
    p, q = p / p.sum(), q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def frechet(mu1, s1, mu2, s2) -> float:
    """||mu1 - mu2||^2 + tr(S1 + S2 - 2 (S1 S2)^(1/2))."""
    diff = np.atleast_1d(mu1 - mu2)
    s1, s2 = np.atleast_2d(s1), np.atleast_2d(s2)
    ev = np.linalg.eigvals(s1 @ s2).real
    return float(diff @ diff + np.trace(s1) + np.trace(s2) - 2 * np.sqrt(np.clip(ev, 0, None)).sum())


def paper_metrics(kp_real: dict, kp_gen: dict) -> dict[str, float]:
    out = {}
    for k in PAPER_KEYS:
        r, g = kp_real[k], kp_gen[k]
        out[f"mean_diff.{k}"] = float(g.mean() - r.mean())
        out[f"std_diff.{k}"] = float(g.std(ddof=1) - r.std(ddof=1))
        out[f"kl.{k}"] = kl_divergence(r, g)
        out[f"fid.{k}"] = float((r.mean() - g.mean()) ** 2 + (r.std(ddof=1) - g.std(ddof=1)) ** 2)
    R = np.stack([kp_real[k] for k in PAPER_KEYS], 1)
    G = np.stack([kp_gen[k] for k in PAPER_KEYS], 1)
    out["fid"] = frechet(R.mean(0), np.cov(R, rowvar=False), G.mean(0), np.cov(G, rowvar=False))
    return out


def reference_kld(kp_real: dict, kp_gen: dict) -> dict[str, float]:
    """The "KL divergence" exactly as computed in cGAN_EB ``utils.stats_plot`` (published Figs. 9-10).

    Both samples are divided by the real maximum of each parameter, and ``rel_entr`` is applied
    to the two sorted pandas Series.  pandas aligns the Series on their index before the
    element-wise operation, which undoes the sort: real day i is paired with generated day i
    (for i < min of the two sample sizes); infinite terms (e.g. negative values) are dropped
    and |mean| is reported.  Kept only to compare with the published numbers; it is not a
    divergence between the two distributions (``kl_divergence`` is).
    """
    import pandas as pd
    from scipy.special import rel_entr

    real = pd.DataFrame(kp_real)
    gen = pd.DataFrame(kp_gen)
    real_n = real.div(real.max(axis=0), axis=1)
    gen_n = gen.div(real.max(axis=0), axis=1)
    out = {}
    for col in real.columns:
        kld = rel_entr(real_n[col].sort_values(), gen_n[col].sort_values()).dropna()
        kld = kld.replace([np.inf, -np.inf], np.nan).dropna()
        out[col] = float(abs(kld.mean())) if len(kld) else np.nan
    return out


def paper_code_metrics(real: np.ndarray, gen: np.ndarray) -> dict[str, float]:
    """Mean / std differences, "KLD" and FID of the key parameters as the cGAN_EB code computes them."""
    kr, kg = reference_key_parameters(real), reference_key_parameters(gen)
    out = {}
    for k in kr:
        out[f"mean_diff.{k}"] = float(kg[k].mean() - kr[k].mean())
        out[f"std_diff.{k}"] = float(kg[k].std(ddof=1) - kr[k].std(ddof=1))
        out[f"fid.{k}"] = float((kr[k].mean() - kg[k].mean()) ** 2 + (kr[k].std(ddof=1) - kg[k].std(ddof=1)) ** 2)
    out.update({f"kld.{k}": v for k, v in reference_kld(kr, kg).items()})
    return out


# ---------------------------------------------------------------------------
# other metrics
# ---------------------------------------------------------------------------
def profile_metrics(real, gen, train=None, rng=None, n_max: int = 2000, real_groups=None,
                    gen_groups=None) -> dict[str, float]:
    """Multivariate metrics on standardised profiles (standardise with training statistics).

    ``real_groups`` / ``gen_groups``: household of every row (``None``: one group per row),
    used to group the folds of the classifier two-sample test.
    """
    import torch

    rng = rng or np.random.default_rng(0)
    real = np.asarray(real, np.float64)
    gen = np.asarray(gen, np.float64)
    out = {}
    cr, cg = np.cov(real, rowvar=False), np.cov(gen, rowvar=False)
    out["rfd"] = frechet(real.mean(0), cr, gen.mean(0), cg) / max(np.trace(cr), _EPS)
    with np.errstate(invalid="ignore", divide="ignore"):
        rr, rg = np.corrcoef(real, rowvar=False), np.corrcoef(gen, rowvar=False)
    iu = np.triu_indices(real.shape[1], 1)
    out["corr_mae"] = float(np.nanmean(np.abs(rr[iu] - rg[iu])))

    ir, ig = rng.permutation(len(real))[:n_max], rng.permutation(len(gen))[:n_max]
    r, g = real[ir], gen[ig]
    gr = np.arange(len(real))[ir] if real_groups is None else np.asarray(real_groups)[ir]
    gg = np.arange(len(gen))[ig] if gen_groups is None else np.asarray(gen_groups)[ig]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr, tg = torch.as_tensor(r, device=dev), torch.as_tensor(g, device=dev)
    k = 5
    d_rr, d_gg, d_rg = torch.cdist(tr, tr), torch.cdist(tg, tg), torch.cdist(tr, tg)
    rad_r = d_rr.kthvalue(k + 1, 1).values
    rad_g = d_gg.kthvalue(k + 1, 1).values
    out["precision"] = float((d_rg < rad_r[:, None]).any(0).double().mean())
    out["recall"] = float((d_rg < rad_g[None, :]).any(1).double().mean())
    out["coverage"] = float((d_rg.min(1).values < rad_r).double().mean())
    if train is not None:
        tt = torch.as_tensor(np.asarray(train, np.float64), device=dev)
        d_g = torch.cdist(tg, tt).min(1).values
        d_r = torch.cdist(tr, tt).min(1).values
        out["dcr_ratio"] = float(d_g.median() / d_r.median().clamp_min(_EPS))
    out["c2st_auc"] = c2st_auc(r, g, gr, gg)
    return out


def c2st_auc(real, gen, real_groups, gen_groups, folds: int = 5, seed: int = 0) -> float:
    """ROC-AUC of a gradient-boosting classifier for real vs generated, averaged over folds.

    Folds are grouped by household and stratified by class, so days of one household
    never sit on both sides of a split.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold

    n = min(len(real), len(gen))
    X = np.concatenate([real[:n], gen[:n]])
    y = np.r_[np.ones(n), np.zeros(n)]
    # Real and generated group ids must not collide.
    _, groups = np.unique(np.r_[[f"r{v}" for v in real_groups[:n]], [f"g{v}" for v in gen_groups[:n]]],
                          return_inverse=True)
    aucs = []
    for tr, te in StratifiedGroupKFold(folds, shuffle=True, random_state=seed).split(X, y, groups):
        clf = HistGradientBoostingClassifier(max_iter=100, random_state=0).fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
    return float(np.mean(aucs))
