"""Paper figures.

    python -m cgan.figures --runs runs --out results/figures \
        --models cgan_eb cwgan_gp baseline_copula --seed 0

Samples are generated for the test-split label counts (seed 0 runs, best
checkpoint); per-label heatmaps and training curves are read from ``runs``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .data.prepare import GROUP_NAMES  # noqa: E402
from .data.profiles import DAYTYPE_NAMES, SEASON_NAMES, label_of  # noqa: E402
from .eval.kpis import all_kpis  # noqa: E402
from .evaluate import BaselineSource, GeneratorSource, get_splits  # noqa: E402
from .utils import load_config  # noqa: E402

REAL_COLOR = "#222222"
PALETTE = ["#d1495b", "#00798c", "#edae49", "#66a182", "#2e4057", "#8d96a3"]
LABELS = {"cgan_eb": "cGAN-EB (reproduction)", "cwgan_gp": "cWGAN-GP (ours)", "baseline_copula": "Gaussian copula",
          "baseline_gaussian": "Gaussian", "baseline_gmm": "GMM", "baseline_resample": "Resample train"}


def _source(runs: Path, model: str, seed: int):
    d = runs / model / f"seed{seed}"
    return BaselineSource(d) if model.startswith("baseline_") else GeneratorSource(d, "best")


def _hours(T):
    return np.arange(T) * 24.0 / T


def mean_profiles(real, syn: dict, out: Path):
    T = real.X.shape[1]
    h = _hours(T)
    for d, dname in enumerate(DAYTYPE_NAMES):
        fig, ax = plt.subplots(len(GROUP_NAMES), 4, figsize=(14, 2.1 * len(GROUP_NAMES)), sharex=True)
        for g, gname in enumerate(GROUP_NAMES):
            for s, sname in enumerate(SEASON_NAMES):
                lab = int(label_of(d, g, s))
                m = real.label == lab
                a = ax[g, s]
                x = real.X[m]
                a.fill_between(h, np.quantile(x, 0.1, 0), np.quantile(x, 0.9, 0), color="#bbbbbb", alpha=0.5, lw=0)
                a.plot(h, x.mean(0), color=REAL_COLOR, lw=2, label="Real (test households)")
                for i, (name, X) in enumerate(syn.items()):
                    a.plot(h, X[m].mean(0), color=PALETTE[i % len(PALETTE)], lw=1.4, label=LABELS.get(name, name))
                a.axhline(0, color="#999999", lw=0.5)
                if g == 0:
                    a.set_title(sname)
                if s == 0:
                    a.set_ylabel(f"{gname}\nkWh/h")
        for a in ax[-1]:
            a.set_xlabel("Hour of day")
            a.set_xticks([0, 6, 12, 18, 24])
        handles, labels = ax[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)
        fig.suptitle(f"{dname}: mean net load (band: real 10-90th percentile)")
        fig.tight_layout(rect=(0, 0.03, 1, 0.98))
        fig.savefig(out / f"mean_profiles_{dname.lower()}.png", dpi=150)
        plt.close(fig)


def kpi_distributions(real, syn: dict, out: Path, kpis=("near_peak", "near_base", "daily_energy", "peak_hour")):
    T = real.X.shape[1]
    kr = all_kpis(real.X, 24.0 / T)
    ks = {n: all_kpis(X, 24.0 / T) for n, X in syn.items()}
    group = real.factors["group"]
    fig, ax = plt.subplots(len(kpis), len(GROUP_NAMES), figsize=(2.4 * len(GROUP_NAMES), 2.2 * len(kpis)))
    for r, k in enumerate(kpis):
        for g, gname in enumerate(GROUP_NAMES):
            m = group == g
            a = ax[r, g]
            lo, hi = np.quantile(kr[k][m], [0.005, 0.995])
            bins = np.arange(-0.5, 24.5, 1) if k == "peak_hour" else np.linspace(lo, hi, 40)
            a.hist(kr[k][m], bins=bins, density=True, color="#bbbbbb", label="Real")
            for i, (name, d) in enumerate(ks.items()):
                a.hist(d[k][m], bins=bins, density=True, histtype="step", lw=1.3,
                       color=PALETTE[i % len(PALETTE)], label=LABELS.get(name, name))
            if r == 0:
                a.set_title(gname)
            if g == 0:
                a.set_ylabel(k.replace("_", " "))
            a.set_yticks([])
    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out / "kpi_distributions.png", dpi=150)
    plt.close(fig)


def condition_effects(real, syn: dict, out: Path):
    """Hourly mean difference of each single technology relative to the 'none' group."""
    T = real.X.shape[1]
    h = _hours(T)
    techs = [("PV", 1), ("HP", 2), ("EV", 4)]
    fig, ax = plt.subplots(len(techs), 2, figsize=(10, 7), sharex=True)
    for c, season in enumerate([3, 1]):  # winter, summer
        for r, (tname, g) in enumerate(techs):
            a = ax[r, c]
            ref = int(label_of(0, 0, season))
            lab = int(label_of(0, g, season))
            eff = real.X[real.label == lab].mean(0) - real.X[real.label == ref].mean(0)
            a.plot(h, eff, color=REAL_COLOR, lw=2, label="Real")
            for i, (name, X) in enumerate(syn.items()):
                e = X[real.label == lab].mean(0) - X[real.label == ref].mean(0)
                a.plot(h, e, color=PALETTE[i % len(PALETTE)], lw=1.3, label=LABELS.get(name, name))
            a.axhline(0, color="#999999", lw=0.5)
            a.set_title(f"{tname} - none, weekday {SEASON_NAMES[season].lower()}")
            a.set_ylabel("kWh/h")
    for a in ax[-1]:
        a.set_xlabel("Hour of day")
    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out / "condition_effects.png", dpi=150)
    plt.close(fig)


def correlation_matrices(real, syn: dict, out: Path, label=None):
    label = int(label_of(0, 7, 3)) if label is None else label  # Weekday, HP+EV+PV, Winter
    m = real.label == label
    mats = {"Real": np.corrcoef(real.X[m], rowvar=False)}
    mats.update({LABELS.get(n, n): np.corrcoef(X[m], rowvar=False) for n, X in syn.items()})
    fig, ax = plt.subplots(1, len(mats), figsize=(3.2 * len(mats), 3.2))
    for a, (name, C) in zip(np.atleast_1d(ax), mats.items()):
        im = a.imshow(C, vmin=-1, vmax=1, cmap="RdBu_r")
        a.set_title(name, fontsize=9)
        a.set_xticks([0, 6, 12, 18, 23])
        a.set_yticks([0, 6, 12, 18, 23])
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(out / "correlation_matrices.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def example_profiles(real, syn: dict, out: Path, label=None, n=6, seed=0):
    label = int(label_of(0, 5, 1)) if label is None else label  # Weekday, EV+PV, Summer
    rng = np.random.default_rng(seed)
    m = np.flatnonzero(real.label == label)
    rows = {"Real": real.X[rng.choice(m, n, replace=False)]}
    rows.update({LABELS.get(k, k): X[rng.choice(m, n, replace=False)] for k, X in syn.items()})
    h = _hours(real.X.shape[1])
    fig, ax = plt.subplots(len(rows), n, figsize=(2.2 * n, 1.8 * len(rows)), sharex=True, sharey=True)
    for r, (name, X) in enumerate(rows.items()):
        for j in range(n):
            ax[r, j].plot(h, X[j], color=REAL_COLOR if r == 0 else PALETTE[(r - 1) % len(PALETTE)], lw=1)
            ax[r, j].axhline(0, color="#999999", lw=0.5)
        ax[r, 0].set_ylabel(name, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "example_profiles.png", dpi=150)
    plt.close(fig)


def training_curves(runs: Path, models, out: Path):
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, model in enumerate(m for m in models if not m.startswith("baseline_")):
        for j, f in enumerate(sorted((runs / model).glob("seed*/val.csv"))):
            v = pd.read_csv(f)
            col = "val_rfd_ema" if v["val_rfd_ema"].notna().any() else "val_rfd"
            ax.plot(v.iteration, v[col], color=PALETTE[i % len(PALETTE)], alpha=0.8, lw=1,
                    label=LABELS.get(model, model) if j == 0 else None)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("Generator iterations")
    ax.set_ylabel("Validation rFD (mean over labels)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "training_curves.png", dpi=150)
    plt.close(fig)


def label_heatmaps(runs: Path, models, out: Path, metrics=("profile.energy", "kpi.near_peak.w1_std")):
    from .report import collect

    df = collect(runs)
    for metric in metrics:
        sub = df[(df.metric == metric) & (df.model.isin(models))]
        per = sub.groupby(["model", "label"]).value.mean().reset_index()
        vmax = np.nanquantile(per.value, 0.98)
        fig, ax = plt.subplots(1, len(models), figsize=(3.2 * len(models), 6), sharey=True)
        for a, model in zip(np.atleast_1d(ax), models):
            grid = np.full((2 * len(GROUP_NAMES), 4), np.nan)
            for _, r in per[per.model == model].iterrows():
                lab = int(r.label)
                d, g, s = lab // 32, (lab // 4) % 8, lab % 4
                grid[d * len(GROUP_NAMES) + g, s] = r.value
            im = a.imshow(grid, vmin=0, vmax=vmax, cmap="viridis", aspect="auto")
            a.set_title(LABELS.get(model, model), fontsize=9)
            a.set_xticks(range(4), [s[:2] for s in SEASON_NAMES])
            a.set_yticks(range(2 * len(GROUP_NAMES)),
                         [f"{DAYTYPE_NAMES[i // len(GROUP_NAMES)][:3]} {GROUP_NAMES[i % len(GROUP_NAMES)]}"
                          for i in range(2 * len(GROUP_NAMES))], fontsize=7)
        fig.colorbar(im, ax=ax, shrink=0.8, label=metric)
        fig.savefig(out / f"heatmap_{metric.replace('.', '_')}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="results/figures")
    ap.add_argument("--models", nargs="+", default=["cgan_eb", "cwgan_gp", "baseline_copula"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default="configs/base.yaml")
    args = ap.parse_args(argv)
    runs, out = Path(args.runs), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data_cfg = load_config(args.config)["data"]
    real = get_splits(data_cfg)["test"]
    syn = {m: _source(runs, m, args.seed).sample(real.label, 2025 + args.seed) for m in args.models}
    mean_profiles(real, syn, out)
    kpi_distributions(real, syn, out)
    condition_effects(real, syn, out)
    correlation_matrices(real, syn, out)
    example_profiles(real, syn, out)
    training_curves(runs, args.models, out)
    label_heatmaps(runs, args.models, out)
    print(f"figures written to {out}")


if __name__ == "__main__":
    main()
