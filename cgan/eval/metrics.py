"""Two-sample metrics between real (held-out) and synthetic samples.

Scalar (1-D) metrics for KPIs and per-hour marginals:
    w1        Wasserstein-1 (scipy.stats.wasserstein_distance)
    ks        two-sample Kolmogorov-Smirnov statistic D
    jsd       Jensen-Shannon divergence (base 2, in [0, 1]) on shared bins with
              additive smoothing alpha = 0.5
    kl        KL(P_real || Q_syn) on the same smoothed histogram (the quantity
              Eq. 4 of Li et al. 2025 intends; their code does not compute it)
    fd1       (mu_r - mu_s)^2 + (sd_r - sd_s)^2: the 1-D Frechet distance the
              cGAN_EB code reports as "FID" (squared units)

Multivariate metrics on standardised D-dim profiles:
    fd        Frechet distance between Gaussian fits
    mmd2      unbiased MMD^2, sum of RBF kernels with bandwidths
              {0.5, 1, 2} x median pairwise distance of the real sample
    energy    energy distance 2E|X-Y| - E|X-X'| - E|Y-Y'| (U-statistics)
    prdc      precision / recall / density / coverage, k = 5 (Naeem et al. 2020)
    c2st_auc  ROC-AUC of a gradient-boosting classifier separating real from
              synthetic, 5-fold cross-validated (0.5 = indistinguishable)
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.stats import ks_2samp, wasserstein_distance

from .fast import frechet_distance, gaussian_stats

ALPHA = 0.5


# ---------------------------------------------------------------------------
# 1-D
# ---------------------------------------------------------------------------
def shared_bins(real: np.ndarray, discrete: bool, k: int = 30) -> np.ndarray:
    """Bin edges fixed from the real sample, plus two overflow bins (-inf, lo) and (hi, inf).

    Synthetic mass beyond the real range lands in the overflow bins, where the
    real sample only has the smoothing pseudo-count, so overshoot is penalised.
    """
    real = np.asarray(real, np.float64)
    if discrete:
        u = np.unique(real)
        step = np.min(np.diff(u)) if len(u) > 1 else 1.0
        edges = np.arange(np.floor(real.min()) - step / 2, np.ceil(real.max()) + step, step)
    else:
        lo, hi = np.quantile(real, [0.0, 1.0])
        if hi <= lo:
            hi = lo + 1e-9
        edges = np.linspace(lo, hi, k + 1)
    return np.r_[-np.inf, edges, np.inf]


def _hist(x, edges, alpha=ALPHA):
    c, _ = np.histogram(x, edges)
    return (c + alpha) / (c.sum() + alpha * len(c))


def jsd_kl(real, syn, discrete: bool = False, k: int = 30):
    edges = shared_bins(real, discrete, k)
    p, q = _hist(real, edges), _hist(syn, edges)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log2(p / m))
    kl_qm = np.sum(q * np.log2(q / m))
    return float(0.5 * (kl_pm + kl_qm)), float(np.sum(p * np.log(p / q)))


def scalar_metrics(real, syn, discrete: bool = False) -> dict[str, float]:
    real = np.asarray(real, np.float64)
    syn = np.asarray(syn, np.float64)
    mr, ms = real.mean(), syn.mean()
    sr, ss = real.std(ddof=1), syn.std(ddof=1)
    w1 = wasserstein_distance(real, syn)
    jsd, kl = jsd_kl(real, syn, discrete)
    return {
        "mean_diff": float(ms - mr),
        "std_diff": float(ss - sr),
        "mean_diff_std": float((ms - mr) / sr) if sr > 0 else np.nan,
        "std_ratio": float(ss / sr) if sr > 0 else np.nan,
        "w1": float(w1),
        "w1_std": float(w1 / sr) if sr > 0 else np.nan,
        "ks": float(ks_2samp(real, syn).statistic),
        "jsd": jsd,
        "kl": kl,
        "fd1": float((mr - ms) ** 2 + (sr - ss) ** 2),
    }


# ---------------------------------------------------------------------------
# multivariate
# ---------------------------------------------------------------------------
def _t(x, device):
    return torch.as_tensor(np.asarray(x), dtype=torch.float64, device=device)


def _pdist2(a, b):
    return torch.cdist(a, b).pow(2)


def median_bandwidth(x: torch.Tensor, max_n: int = 2000, seed: int = 0) -> float:
    if x.shape[0] > max_n:
        g = torch.Generator(device="cpu").manual_seed(seed)
        x = x[torch.randperm(x.shape[0], generator=g)[:max_n].to(x.device)]
    d = torch.cdist(x, x)
    iu = torch.triu_indices(len(x), len(x), 1, device=x.device)
    return float(d[iu[0], iu[1]].median())


def mmd2_unbiased(x: torch.Tensor, y: torch.Tensor, sigmas) -> float:
    m, n = x.shape[0], y.shape[0]
    dxx, dyy, dxy = _pdist2(x, x), _pdist2(y, y), _pdist2(x, y)
    total = 0.0
    for s in sigmas:
        g = 1.0 / (2 * s * s)
        kxx, kyy, kxy = torch.exp(-g * dxx), torch.exp(-g * dyy), torch.exp(-g * dxy)
        total += float((kxx.sum() - kxx.diagonal().sum()) / (m * (m - 1))
                       + (kyy.sum() - kyy.diagonal().sum()) / (n * (n - 1))
                       - 2 * kxy.mean())
    return total


def energy_distance(x: torch.Tensor, y: torch.Tensor) -> float:
    m, n = x.shape[0], y.shape[0]
    exy = torch.cdist(x, y).mean()
    exx = torch.cdist(x, x).sum() / (m * (m - 1))
    eyy = torch.cdist(y, y).sum() / (n * (n - 1))
    return float(2 * exy - exx - eyy)


def prdc(real: torch.Tensor, syn: torch.Tensor, k: int = 5) -> dict[str, float]:
    """Precision, recall, density, coverage (Naeem et al. 2020)."""
    d_rr = torch.cdist(real, real)
    d_ss = torch.cdist(syn, syn)
    d_rs = torch.cdist(real, syn)
    r_rad = d_rr.kthvalue(k + 1, dim=1).values  # k-th neighbour excluding self
    s_rad = d_ss.kthvalue(k + 1, dim=1).values
    in_real_ball = d_rs < r_rad[:, None]          # (n_real, n_syn)
    precision = in_real_ball.any(0).double().mean()
    recall = (d_rs < s_rad[None, :]).any(1).double().mean()
    density = in_real_ball.sum(0).double().mean() / k
    coverage = (d_rs.min(1).values < r_rad).double().mean()
    return {"precision": float(precision), "recall": float(recall), "density": float(density),
            "coverage": float(coverage)}


def c2st_auc(real: np.ndarray, syn: np.ndarray, folds: int = 5, seed: int = 0,
             groups_real: np.ndarray | None = None, groups_syn: np.ndarray | None = None) -> float:
    """Classifier two-sample test with household-grouped folds.

    Real days are grouped by household so that no household appears in both
    the fitting and the held-out folds (otherwise the classifier learns
    household fingerprints).  Synthetic days are independent draws unless
    ``groups_syn`` is given (e.g. real reference days of other households).
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

    X = np.concatenate([real, syn])
    y = np.r_[np.zeros(len(real)), np.ones(len(syn))]
    gr = np.arange(len(real)) if groups_real is None else np.unique(groups_real, return_inverse=True)[1]
    gs = np.arange(len(syn)) if groups_syn is None else np.unique(groups_syn, return_inverse=True)[1]
    groups = np.r_[gr, gs + gr.max() + 1]
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, early_stopping=False, random_state=seed)
    cv = StratifiedGroupKFold(folds, shuffle=True, random_state=seed)
    p = cross_val_predict(clf, X, y, groups=groups, cv=cv, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))


def profile_metrics(real_std: np.ndarray, syn_std: np.ndarray, device: str = "cpu", c2st: bool = True,
                    k: int = 5, seed: int = 0, groups_real=None, groups_syn=None) -> dict[str, float]:
    """Metrics on standardised profiles (rows = days).

    ``rfd`` = FD / tr(S_real) and ``energy_rel`` = energy / E|X - X'|_real are
    scale-free per label, so label averages are not dominated by the labels
    with the largest variability.
    """
    xr, xs = _t(real_std, device), _t(syn_std, device)
    sig = median_bandwidth(xr, seed=seed)
    mu_r, cov_r = gaussian_stats(real_std)
    mu_s, cov_s = gaussian_stats(syn_std)
    fd = frechet_distance(mu_r, cov_r, mu_s, cov_s)
    ed = energy_distance(xr, xs)
    m = xr.shape[0]
    exx = float(torch.cdist(xr, xr).sum() / (m * (m - 1)))
    out = {
        "fd": fd,
        "rfd": fd / float(np.trace(cov_r)),
        "mmd2": mmd2_unbiased(xr, xs, [0.5 * sig, sig, 2 * sig]),
        "energy": ed,
        "energy_rel": ed / exx,
    }
    out.update(prdc(xr, xs, k))
    if c2st:
        out["c2st_auc"] = c2st_auc(real_std, syn_std, seed=seed, groups_real=groups_real, groups_syn=groups_syn)
    return out


# ---------------------------------------------------------------------------
# temporal structure (raw units)
# ---------------------------------------------------------------------------
def _acf(X: np.ndarray, max_lag: int) -> np.ndarray:
    Xc = X - X.mean(1, keepdims=True)
    den = (Xc ** 2).sum(1)
    den = np.where(den > 0, den, np.nan)
    return np.stack([(Xc[:, :-l] * Xc[:, l:]).sum(1) / den for l in range(1, max_lag + 1)], 1)


def temporal_metrics(real: np.ndarray, syn: np.ndarray, max_lag: int = 6) -> dict[str, float]:
    real = np.asarray(real, np.float64)
    syn = np.asarray(syn, np.float64)
    T = real.shape[1]
    cr = np.corrcoef(real, rowvar=False)
    cs = np.corrcoef(syn, rowvar=False)
    iu = np.triu_indices(T, 1)
    acf_r = np.nanmean(_acf(real, max_lag), 0)
    acf_s = np.nanmean(_acf(syn, max_lag), 0)
    hw1 = [wasserstein_distance(real[:, t], syn[:, t]) for t in range(T)]
    hks = [ks_2samp(real[:, t], syn[:, t]).statistic for t in range(T)]
    return {
        "mean_profile_rmse": float(np.sqrt(np.mean((real.mean(0) - syn.mean(0)) ** 2))),
        "std_profile_rmse": float(np.sqrt(np.mean((real.std(0) - syn.std(0)) ** 2))),
        "q90_profile_rmse": float(np.sqrt(np.mean((np.quantile(real, 0.9, 0) - np.quantile(syn, 0.9, 0)) ** 2))),
        "corr_mad": float(np.nanmean(np.abs(cr[iu] - cs[iu]))),
        "acf_mae": float(np.mean(np.abs(acf_r - acf_s))),
        "hourly_w1": float(np.mean(hw1)),
        "hourly_ks": float(np.mean(hks)),
        "ramp_w1": float(wasserstein_distance(np.diff(real, axis=1).ravel(), np.diff(syn, axis=1).ravel())),
    }


# ---------------------------------------------------------------------------
# memorisation
# ---------------------------------------------------------------------------
def _min_dist(a: torch.Tensor, b: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    return torch.cat([torch.cdist(a[i:i + chunk], b).min(1).values for i in range(0, len(a), chunk)])


def dcr_metrics(syn: np.ndarray, train: np.ndarray, holdout: np.ndarray, device: str = "cpu") -> dict[str, float]:
    """Distance to the closest training record, compared with held-out real days.

    d_syn(s)  = min_{x in Train} ||s - x||   for synthetic days s
    d_hold(h) = min_{x in Train} ||h - x||   for real days h of unseen (test) households

    A generator that does not reproduce training days has d_syn distributed
    like d_hold.  Reported:
      dcr_q05       share of synthetic days closer to Train than the 5th
                    percentile of d_hold (0.05 expected; 1 for a copy model)
      dcr_ratio_p05 P5(d_syn) / P5(d_hold), dcr_ratio_med  median ratio
      dcr_copies    share of synthetic days identical to a training day
    Days are clustered by household, so no i.i.d. significance test is reported.
    """
    tr = _t(train, device)
    d_syn = _min_dist(_t(syn, device), tr).cpu().numpy()
    d_hold = _min_dist(_t(holdout, device), tr).cpu().numpy()
    exact_hold = d_hold < 1e-6
    # Percentile of holdout distances over holdout days that are not themselves in Train,
    # so that ties at zero cannot make a copy model look memorisation-free.
    p5 = np.quantile(d_hold[~exact_hold], 0.05) if (~exact_hold).any() else np.nan
    return {
        "dcr_q05": float(np.mean(d_syn < p5)),
        "dcr_ratio_p05": float(np.quantile(d_syn, 0.05) / p5),
        "dcr_ratio_med": float(np.median(d_syn) / np.median(d_hold)),
        "dcr_copies": float(np.mean(d_syn < 1e-6)),
        "dcr_holdout_in_train": float(np.mean(exact_hold)),
    }


def aggregate_metrics(real: np.ndarray, syn: np.ndarray, n_homes: int = 10, reps: int = 2000, seed: int = 0,
                      groups_real: np.ndarray | None = None, groups_syn: np.ndarray | None = None) -> dict[str, float]:
    """Load of ``n_homes`` days summed, as seen by a feeder.

    A real feeder takes ``n_homes`` distinct households and one random day of
    each (``groups_*`` = household ids); synthetic days are independent draws
    unless groups are given.  Compared: the aggregate peak distribution and the
    coincidence factor CF = max_t(sum_i x_i,t) / sum_i max_t(x_i,t).  Days are
    drawn independently across households, so weather-driven coincidence is not
    represented; values are interpreted relative to the real-data reference.
    Drawing ``n_homes`` of the ~45 households of a label without replacement
    shrinks the between-household variance by the finite-population factor
    (45 - n) / 44 (0.77 for n = 10) relative to i.i.d. synthetic draws.
    """
    rng = np.random.default_rng(seed)

    def draw(X, groups):
        if groups is None:
            idx = rng.integers(0, len(X), size=(reps, n_homes))
        else:
            _, inv, counts = np.unique(groups, return_inverse=True, return_counts=True)
            order = np.argsort(inv, kind="stable")              # rows grouped by household
            starts = np.r_[0, np.cumsum(counts)[:-1]]
            k = min(n_homes, len(counts))
            chosen = np.argsort(rng.random((reps, len(counts))), axis=1)[:, :k]   # k distinct households
            offset = (rng.random((reps, k)) * counts[chosen]).astype(np.int64)     # one random day each
            idx = order[starts[chosen] + offset]
        S = X[idx]                                  # reps x n x T
        peak = S.sum(1).max(1)
        indiv = S.max(2).sum(1)
        return peak, np.where(indiv > 0, peak / np.where(indiv > 0, indiv, 1), np.nan)

    pr, cr = draw(np.asarray(real, np.float64), groups_real)
    ps, cs = draw(np.asarray(syn, np.float64), groups_syn)
    sd = pr.std(ddof=1)
    return {
        "agg_peak_w1_std": float(wasserstein_distance(pr, ps) / sd) if sd > 0 else np.nan,
        "agg_peak_p95_rel": float(np.quantile(ps, 0.95) / np.quantile(pr, 0.95) - 1),
        "agg_cf_mean_diff": float(np.nanmean(cs) - np.nanmean(cr)),
    }
