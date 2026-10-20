"""Aggregate evaluation outputs into paper tables.

    python -m cgan.report --runs runs --out results

Reads every ``runs/<model>/seed<k>/eval_<split>*.csv`` and writes:

    results/all_metrics_long.csv      model, seed, label, metric, value
    results/summary_<table>.{csv,md,tex}
        cell = mean +/- std over seeds of the label-averaged (macro) value
    results/significance.csv          paired Wilcoxon signed-rank tests over
        labels (seed-averaged per-label values), Holm-corrected per metric
    results/by_group_<metric>.csv     macro value per technology group
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from .data.prepare import GROUP_NAMES
from .data.profiles import decode_label

# (metric, direction): -1 lower is better, +1 higher is better, 0 closer to target
TABLES = {
    "fidelity": [
        ("profile.fd", -1), ("profile.mmd2", -1), ("profile.energy", -1), ("profile.c2st_auc", -1),
        ("profile.precision", +1), ("profile.recall", +1), ("profile.density", 0), ("profile.coverage", +1),
    ],
    "temporal": [
        ("temporal.mean_profile_rmse", -1), ("temporal.std_profile_rmse", -1), ("temporal.q90_profile_rmse", -1),
        ("temporal.corr_mad", -1), ("temporal.acf_mae", -1), ("temporal.hourly_w1", -1),
        ("temporal.hourly_ks", -1), ("temporal.ramp_w1", -1),
    ],
    "kpi_price": [
        (f"kpi.{k}.{m}", -1)
        for k in ["near_peak", "near_base", "high_load_duration", "rise_time", "fall_time", "daily_energy"]
        for m in ["w1_std", "ks", "jsd"]
    ],
    "kpi_reference_style": [  # comparable with the statistics reported by Li et al. (2025)
        (f"kpi.{k}.{m}", -1)
        for k in ["ref_peak", "ref_base", "ref_duration", "ref_rise", "ref_fall"]
        for m in ["mean_diff.abs", "std_diff.abs", "kl", "fd1"]
    ],
    "conditional": [
        (f"cond.{kind}.{f}", +1) for kind in ["recovery", "tstr"]
        for f in ["label", "pv", "hp", "ev", "daytype", "season"]
    ] + [(f"cond.real_reference.{f}", +1) for f in ["label", "pv", "hp", "ev", "daytype", "season"]],
    "aggregate": [
        ("aggregate.agg_peak_w1_std", -1), ("aggregate.agg_peak_p95_rel", 0), ("aggregate.agg_cf_mean_diff", 0),
        ("kpi.evening_energy.w1_std", -1), ("kpi.export_energy.w1_std", -1),
    ],
    "privacy_physical": [
        ("privacy.dcr_q05", 0), ("privacy.dcr_ratio_med", 0), ("privacy.dcr_copies", -1),
        ("physical.neg_share", -1), ("physical.above_train_max", -1),
    ],
}
TARGET = {"profile.density": 1.0, "privacy.dcr_q05": 0.05, "privacy.dcr_ratio_med": 1.0,
          "aggregate.agg_peak_p95_rel": 0.0, "aggregate.agg_cf_mean_diff": 0.0}
SIGNIFICANCE_METRICS = ["profile.fd", "profile.mmd2", "profile.energy", "profile.c2st_auc", "profile.coverage",
                        "temporal.corr_mad", "temporal.hourly_w1", "kpi.near_peak.w1_std", "kpi.daily_energy.w1_std"]


def collect(runs: Path, split: str = "test") -> pd.DataFrame:
    frames = []
    for csv in sorted(runs.glob(f"*/seed*/eval_{split}*.csv")):
        model_dir, seed_dir = csv.parent.parent.name, csv.parent.name
        m = re.fullmatch(rf"eval_{split}(?:_(\w+))?", csv.stem)
        ckpt = m.group(1) if m else None
        model = model_dir if ckpt in (None, "best") else f"{model_dir}[{ckpt}]"
        df = pd.read_csv(csv)
        df.insert(0, "seed", int(seed_dir.removeprefix("seed")))
        df.insert(0, "model", model)
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"no eval_{split}*.csv under {runs}")
    return pd.concat(frames, ignore_index=True)


def macro(df: pd.DataFrame) -> pd.DataFrame:
    """Per (model, seed, metric): mean over labels (global metrics pass through).

    Signed differences (``*_diff``) are also averaged in absolute value
    (``*_diff.abs``) so that errors of opposite sign do not cancel.
    """
    signed = df[df.metric.str.endswith("_diff")].copy()
    signed["metric"] = signed.metric + ".abs"
    signed["value"] = signed.value.abs()
    full = pd.concat([df, signed], ignore_index=True)
    return full.groupby(["model", "seed", "metric"]).value.mean().reset_index()


def summary_table(mac: pd.DataFrame, metrics) -> pd.DataFrame:
    names = [m for m, _ in metrics]
    sub = mac[mac.metric.isin(names)]
    agg = sub.groupby(["model", "metric"]).value.agg(["mean", "std", "count"]).reset_index()
    mean = agg.pivot(index="model", columns="metric", values="mean").reindex(columns=names)
    std = agg.pivot(index="model", columns="metric", values="std").reindex(columns=names)
    n = agg.pivot(index="model", columns="metric", values="count").reindex(columns=names)
    return mean, std, n


def _fmt(mean, std, digits=3):
    out = mean.copy().astype(object)
    for c in mean.columns:
        for r in mean.index:
            mu, sd = mean.loc[r, c], std.loc[r, c]
            if pd.isna(mu):
                out.loc[r, c] = ""
            elif pd.isna(sd):
                out.loc[r, c] = f"{mu:.{digits}g}"
            else:
                out.loc[r, c] = f"{mu:.{digits}g} ± {sd:.2g}"
    return out


def significance(df: pd.DataFrame, ours: str, metrics=SIGNIFICANCE_METRICS) -> pd.DataFrame:
    per_label = df[df.label >= 0].groupby(["model", "metric", "label"]).value.mean().reset_index()
    rows = []
    models = sorted(set(per_label.model) - {ours})
    for metric in metrics:
        direction = dict(m for t in TABLES.values() for m in t).get(metric, -1)
        a = per_label[(per_label.model == ours) & (per_label.metric == metric)].set_index("label").value
        pvals = []
        for other in models:
            b = per_label[(per_label.model == other) & (per_label.metric == metric)].set_index("label").value
            common = a.index.intersection(b.index)
            if len(common) < 5:
                continue
            x, y = a[common].to_numpy(), b[common].to_numpy()
            better = (x < y) if direction < 0 else (x > y)
            try:
                p = wilcoxon(x, y).pvalue
            except ValueError:
                p = np.nan
            rows.append({"metric": metric, "ours": ours, "other": other, "n_labels": len(common),
                         "median_ours": np.median(x), "median_other": np.median(y),
                         "win_rate_ours": better.mean(), "p_wilcoxon": p})
            pvals.append(len(rows) - 1)
        # Holm correction within metric
        idx = [i for i in pvals if not np.isnan(rows[i]["p_wilcoxon"])]
        order = sorted(idx, key=lambda i: rows[i]["p_wilcoxon"])
        running = 0.0
        for rank, i in enumerate(order):
            adj = min(1.0, (len(order) - rank) * rows[i]["p_wilcoxon"])
            running = max(running, adj)
            rows[i]["p_holm"] = running
    return pd.DataFrame(rows)


def compute_table(runs: Path) -> pd.DataFrame:
    """Training cost per model: wall-clock, generator updates, real samples seen, parameters."""
    import json

    rows = []
    for meta_path in sorted(runs.glob("*/seed*/meta.json")):
        meta = json.loads(meta_path.read_text())
        if "seconds" not in meta or "iterations" not in meta:
            continue
        t = meta["config"]["train"]
        rows.append({
            "model": meta_path.parent.parent.name, "seed": meta["seed"], "minutes": meta["seconds"] / 60,
            "generator_updates": meta["iterations"],
            "real_samples_seen": meta["iterations"] * int(t.get("n_critic", 1)) * int(t["batch_size"]),
            "params_g": meta["params_g"], "params_d": meta["params_d"], "device": meta["env"]["device"],
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.groupby("model").agg(seeds=("seed", "count"), minutes_mean=("minutes", "mean"),
                                   minutes_std=("minutes", "std"), generator_updates=("generator_updates", "first"),
                                   real_samples_seen=("real_samples_seen", "first"), params_g=("params_g", "first"),
                                   params_d=("params_d", "first"), device=("device", "first"))


def metric_agreement(mac: pd.DataFrame, metrics=("profile.energy", "profile.mmd2", "profile.fd", "profile.c2st_auc",
                                                 "profile.coverage", "kpi.near_peak.w1_std",
                                                 "temporal.hourly_w1")) -> pd.DataFrame:
    """Kendall tau between model rankings induced by different metrics (seed-averaged macro values)."""
    from scipy.stats import kendalltau

    m = mac[mac.metric.isin(metrics)].groupby(["model", "metric"]).value.mean().unstack()
    if "profile.coverage" in m:
        m["profile.coverage"] = -m["profile.coverage"]  # orient: lower is better for all
    cols = [c for c in metrics if c in m]
    out = pd.DataFrame(index=cols, columns=cols, dtype=float)
    for a in cols:
        for b in cols:
            out.loc[a, b] = kendalltau(m[a], m[b], nan_policy="omit").statistic
    return out


def by_group(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    sub = df[(df.metric == metric) & (df.label >= 0)].copy()
    sub["group"] = [GROUP_NAMES[int(decode_label(l)[1])] for l in sub.label]
    g = sub.groupby(["model", "seed", "group"]).value.mean().reset_index()
    agg = g.groupby(["model", "group"]).value.agg(["mean", "std"]).reset_index()
    mean = agg.pivot(index="model", columns="group", values="mean").reindex(columns=GROUP_NAMES)
    std = agg.pivot(index="model", columns="group", values="std").reindex(columns=GROUP_NAMES)
    return _fmt(mean, std)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="results")
    ap.add_argument("--split", default="test")
    ap.add_argument("--ours", default="cwgan_gp")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = collect(Path(args.runs), args.split)
    df.to_csv(out / "all_metrics_long.csv", index=False)
    mac = macro(df)
    mac.to_csv(out / "macro_by_seed.csv", index=False)
    for name, metrics in TABLES.items():
        mean, std, n = summary_table(mac, metrics)
        mean.to_csv(out / f"summary_{name}_mean.csv")
        std.to_csv(out / f"summary_{name}_std.csv")
        table = _fmt(mean, std)
        table.columns = [c.split(".", 1)[1] if c.count(".") else c for c in table.columns]
        table.insert(0, "seeds", n.max(axis=1).astype("Int64"))
        (out / f"summary_{name}.md").write_text(table.to_markdown())
        (out / f"summary_{name}.tex").write_text(table.to_latex(escape=True))
    if args.ours in set(df.model):
        significance(df, args.ours).to_csv(out / "significance.csv", index=False)
    comp = compute_table(Path(args.runs))
    if len(comp):
        comp.to_csv(out / "compute.csv")
        (out / "compute.md").write_text(comp.to_markdown())
    if df.model.nunique() >= 3:
        metric_agreement(mac).to_csv(out / "metric_agreement_kendall.csv")
    for metric in ["profile.energy", "profile.coverage", "kpi.near_peak.w1_std", "temporal.hourly_w1"]:
        by_group(df, metric).to_csv(out / f"by_group_{metric.replace('.', '_')}.csv")
    print(f"wrote tables for {df.model.nunique()} models to {out}")


if __name__ == "__main__":
    main()
