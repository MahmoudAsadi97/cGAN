"""Household-level (cluster) bootstrap of the headline multivariate metrics.

    python -m cgan.bootstrap --runs runs --out results --ours cwgan_gp \
        --models cwgan_gp "cgan_eb[best_raw]" baseline_copula baseline_gmm baseline_resample

The test set contains 45 households per group, so the per-label real
distributions are themselves noisy.  Each replicate resamples the test
households within every group with replacement; per label, the real sample
becomes the days of the drawn households (with multiplicity), while the
synthetic samples saved during evaluation are kept fixed.  Energy distance and
MMD^2 are recomputed for every model on the same replicate, so differences
between models are paired.  Duplicated real days bias the real-real term of
both statistics, but that term is shared by all models and cancels in the
differences; intervals are therefore reported for differences, and the
per-model values are given only as bootstrap means.

With count weights w over the n real days of a label (m = sum w):
    E|X - Y|  = w . d / m              d_i = mean_j |x_i - y_j|
    E|X - X'| = w' D w / (m (m - 1))    D = real-real distances (zero diagonal)
and analogously for the kernel terms of MMD^2, so a replicate costs O(n^2)
once per label (shared) and O(n) per model.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data.profiles import ProfileConfig, load_profiles
from .eval.metrics import median_bandwidth
from .utils import load_config


def _sample_files(runs: Path, spec: str) -> list[Path]:
    m = re.fullmatch(r"([\w\-]+)(?:\[(\w+)\])?", spec)
    name, ckpt = m.group(1), m.group(2) or "best"
    pattern = "samples_test.npz" if name.startswith(("baseline_", "real_")) else f"samples_test_{ckpt}.npz"
    return sorted((runs / name).glob(f"seed*/{pattern}"))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="results")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--ours", default="cwgan_gp")
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    runs, out = Path(args.runs), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    test = load_profiles(ProfileConfig.from_dict(load_config(args.config)["data"]))
    train, test = test["train"], test["test"]
    mu, sd = train.X.mean(0), train.X.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    Xr = (test.X - mu) / sd

    samples = {}
    for spec in args.models:
        files = _sample_files(runs, spec)
        if not files:
            print(f"no saved samples for {spec}; skipped")
            continue
        arrs = []
        for f in files:
            z = np.load(f)
            if not np.array_equal(z["label"], test.label):
                raise ValueError(f"{f}: labels do not match the test split")
            arrs.append((z["X"] - mu) / sd)
        samples[spec] = arrs
    models = list(samples)

    # Draw replicates: within each group, resample its test meters with replacement.
    rng = np.random.default_rng(args.seed)
    meters_by_group = {g: np.unique(test.factors["meter"][test.factors["group"] == g])
                       for g in np.unique(test.factors["group"])}
    meter_weights = []
    for _ in range(args.B):
        w = {}
        for g, ms in meters_by_group.items():
            drawn, cnt = np.unique(rng.choice(ms, size=len(ms), replace=True), return_counts=True)
            w.update(zip(drawn.tolist(), cnt.tolist()))
        meter_weights.append(w)

    labels = np.unique(test.label)
    ed = {m: np.zeros((args.B, len(labels))) for m in models}
    mmd = {m: np.zeros((args.B, len(labels))) for m in models}
    for li, lab in enumerate(labels):
        idx = np.flatnonzero(test.label == lab)
        meters = test.factors["meter"][idx]
        xr = torch.as_tensor(Xr[idx], dtype=torch.float64, device=dev)
        sig = median_bandwidth(xr)
        sigmas = [0.5 * sig, sig, 2 * sig]
        D = torch.cdist(xr, xr)
        K = sum(torch.exp(-D.pow(2) / (2 * s * s)) for s in sigmas)
        W = torch.as_tensor(np.array([[w.get(int(mt), 0) for mt in meters] for w in meter_weights]),
                            dtype=torch.float64, device=dev)              # B x n
        m_tot = W.sum(1)
        exx = ((W @ D) * W).sum(1) / (m_tot * (m_tot - 1))
        kxx = (((W @ K) * W).sum(1) - len(sigmas) * m_tot) / (m_tot * (m_tot - 1))
        for model in models:
            ed_s, mmd_s = [], []
            for arr in samples[model]:
                ys = torch.as_tensor(arr[idx], dtype=torch.float64, device=dev)
                n = ys.shape[0]
                Dxy = torch.cdist(xr, ys)
                Kxy = sum(torch.exp(-Dxy.pow(2) / (2 * s * s)) for s in sigmas)
                Dyy = torch.cdist(ys, ys)
                Kyy = sum(torch.exp(-Dyy.pow(2) / (2 * s * s)) for s in sigmas)
                eyy = Dyy.sum() / (n * (n - 1))
                kyy = (Kyy.sum() - len(sigmas) * n) / (n * (n - 1))
                exy = (W @ Dxy.mean(1)) / m_tot
                kxy = (W @ Kxy.mean(1)) / m_tot
                ed_s.append((2 * exy - exx - eyy).cpu().numpy())
                mmd_s.append((kxx + kyy - 2 * kxy).cpu().numpy())
            ed[model][:, li] = np.mean(ed_s, 0)       # seed-averaged
            mmd[model][:, li] = np.mean(mmd_s, 0)

    rows, diffs = [], []
    for name, store in (("energy", ed), ("mmd2", mmd)):
        macro = {m: store[m].mean(1) for m in models}  # B values per model
        for m in models:
            rows.append({"metric": name, "model": m, "seeds": len(samples[m]), "boot_mean": macro[m].mean()})
        if args.ours in macro:
            for m in models:
                if m == args.ours:
                    continue
                d = macro[args.ours] - macro[m]
                lo, hi = np.quantile(d, [0.025, 0.975])
                diffs.append({"metric": name, "ours": args.ours, "other": m, "diff_mean": d.mean(),
                              "ci95_lo": lo, "ci95_hi": hi, "p_ours_not_better": float(np.mean(d >= 0))})
    pd.DataFrame(rows).to_csv(out / "bootstrap_household_models.csv", index=False)
    pd.DataFrame(diffs).to_csv(out / "bootstrap_household_diffs.csv", index=False)
    print(pd.DataFrame(diffs).to_string(index=False))


if __name__ == "__main__":
    main()
