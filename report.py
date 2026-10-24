"""Evaluate every finished run on the test households and write the result tables and figures.

    python report.py --models faithful final --figures faithful:last final:best \
        --ablation faithful abl_shuffle abl_quantile abl_wgan_refnets abl_wide abl_ema improved improved_big final \
        --selection dev_improved_nofloor improved_diff improved_big improved_bigdiff final \
        --paper-protocol faithful:last final:best --curves faithful final

Outputs in ``results/``:

    summary.md / .csv          mean +/- std over seeds of the label-averaged metrics on the test
                               households, checkpoints ``best`` (validation-selected) and ``last``
                               (final iterate, as the reference), next to the real-data benchmarks
    per_seed.md                the accuracy of every seed
    comparison.md              final vs the GitHub model per metric: direction, means, and whether every
                               final seed is better than every GitHub value (complete separation)
    paper_criteria.md          share of (label, key parameter) pairs meeting the levels quoted by
                               Li et al. (2025): |difference of mean| and of std below 1 (kWh or h),
                               KL below 0.3, FID below 5; with the paper's definitions and a proper
                               KL, and exactly as the cGAN_EB code computes them
    paper_protocol.md          the same criteria with the paper's protocol: generated days against
                               the *training* households (in-sample), 2,000 days per label
    ablation.md                validation-selected checkpoint of every step of the ablation ladder, with
                               the seed-matched difference to the previous step
    selection.md               validation accuracy of the model-selection candidates (no test data), as
                               used by the selection rule and re-scored with 5 fresh noise draws
    table1_labels.md           days and households per label and split (counterpart of Table 1)
    figures/<model>_<ckpt>/    the paper's Figs. 3-12 (``utils.stats_plot``, both definition sets),
                               seed 0, 2,000 generated days per label, and the generated days
    figures/compare/           Figs. 3-4 with the real days and every generator in each panel
    figures/validation_curves.png   validation accuracy during training, every finished seed; the
                               dotted line is the real benchmark on the validation households
                               (``evaluate.py --real train --split val``)

Runs are evaluated only when finished (``train_summary.json``).  Results are cached
in ``eval_test_<ckpt>.json`` with a fingerprint of the checkpoint, the run config and
the evaluation code; a cache whose fingerprint does not match is recomputed.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data_prep import DAYTYPE_NAMES, GROUP_NAMES, N_LABELS, SEASON_NAMES, Split, decode_label, load_profiles
from evaluate import EVAL_SEED, HEADLINE, evaluate, real_reference, summarise

ROOT = Path(__file__).resolve().parent
COLUMNS = {
    "accuracy": "Accuracy", "mean_profile": "Mean profile", "std_profile": "Std profile",
    "kpi_mean": "KP mean", "kpi_std": "KP std", "hourly_dist": "Hourly 1-KS", "kpi_dist": "KP 1-KS",
    "accuracy.min_label": "Worst label", "accuracy.share_labels_ge_85": "Labels >= 85%",
    "paper.fid": "FID (5 KP)", "paper.kl.near_peak": "KL near-peak", "paper.kl.near_base": "KL near-base",
    "profile.rfd": "rFD", "profile.corr_mae": "Corr. MAE", "profile.c2st_auc": "C2ST AUC",
    "profile.precision": "Precision", "profile.recall": "Recall", "profile.coverage": "Coverage",
    "profile.dcr_ratio": "DCR ratio", "physical.neg_share": "Neg. share (no PV)",
}
DISPLAY = {"faithful": "GitHub cGAN", "final": "Final"}
# What each step of the ablation ladder changes with respect to the previous row.
STEPS = {
    "faithful": "reference cGAN_EB (BCE, G 64-128-64, D 64-128-256, min-max, data order, 10,000 epochs)",
    "abl_shuffle": "+ minibatches reshuffled every epoch",
    "abl_quantile": "+ per-hour Gaussianising value transform",
    "abl_wgan_refnets": "+ WGAN-GP training recipe with the reference networks: loss, critic without sigmoid / "
                        "dropout, 5 critic steps, Adam lr 1e-4 both, betas (0.5, 0.9), batch 1,024, 400 critic epochs",
    "abl_wide": "+ wider networks (3 x 256 LeakyReLU) and 32-d noise",
    "abl_ema": "+ EMA (0.999) of the generator weights",
    "improved": "+ net load of labels without PV floored at 0 kWh",
    "improved_big": "+ 4 x 512 networks, 64-d noise",
    "final": "+ critic input with hour-to-hour differences (= improved_bigdiff, selected on validation)",
}


def code_hash() -> str:
    h = hashlib.sha1()
    for f in ("metrics.py", "evaluate.py", "data_prep.py"):
        h.update((ROOT / f).read_bytes())
    return h.hexdigest()[:12]


def fingerprint(run_dir: Path, ckpt: str) -> dict:
    state = torch.load(run_dir / f"{ckpt}.pt", map_location="cpu", weights_only=True)
    return {"ckpt_epoch": int(state["epoch"]), "ckpt_val_accuracy": float(state["val_accuracy"]),
            "ckpt_mtime_ns": (run_dir / f"{ckpt}.pt").stat().st_mtime_ns,
            "config_sha1": hashlib.sha1((run_dir / "config.json").read_bytes()).hexdigest()[:12],
            "code": code_hash(), "eval_seed": EVAL_SEED}


def evaluate_run(run_dir: Path, ckpt: str, splits_cache: dict) -> dict:
    out = run_dir / f"eval_test_{ckpt}.json"
    fp = fingerprint(run_dir, ckpt)
    if out.exists():
        cached = json.loads(out.read_text())
        if cached.get("_fingerprint") == fp:
            return cached
    from train import load_run

    wrapped, cfg = load_run(run_dir, ckpt)
    key = (cfg["target"], cfg["split_seed"])
    if key not in splits_cache:
        splits_cache[key] = load_profiles(cfg["data"], *key)
    splits = splits_cache[key]
    gen = wrapped.sample_kwh(splits["test"].label, EVAL_SEED + cfg["seed"])
    df = evaluate(gen, splits["test"], splits["train"])
    df.to_csv(run_dir / f"eval_test_{ckpt}.csv", index=False)
    summary = {**summarise(df), "_fingerprint": fp}
    out.write_text(json.dumps(summary, indent=2))
    print(f"{run_dir} [{ckpt}] accuracy {summary['accuracy']:.4f}", flush=True)
    return summary


def reference_row(runs: Path, splits, src: str) -> dict:
    """Real days of ``src`` households scored against the test households (cached with a code fingerprint)."""
    path = runs / "real_reference" / f"eval_test_{src}.json"
    fp = {"code": code_hash()}
    if path.exists():
        cached = json.loads(path.read_text())
        if cached.get("_fingerprint") == fp:
            return cached
    gen, groups = real_reference(splits, src, "test")
    df = evaluate(gen, splits["test"], splits["train"], gen_groups=groups)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path.with_suffix(".csv"), index=False)
    summary = {**summarise(df), "_fingerprint": fp}
    path.write_text(json.dumps(summary, indent=2))
    return summary


def finished_runs(runs: Path, model: str):
    for run_dir in sorted(Path(runs, model).glob("seed*")):
        if (run_dir / "train_summary.json").exists():
            yield run_dir
        else:
            print(f"skip {run_dir}: training not finished", flush=True)


CRITERIA = [("mean_diff_lt1", "|Δ mean| < 1"), ("std_diff_lt1", "|Δ std| < 1"), ("kl_lt0.3", "KL < 0.3"),
            ("fid_lt5", "FID < 5"), ("mean_kl", "mean KL"), ("mean_fid", "mean FID")]


def criteria_table(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    keys = [(f"criteria.{p}.{k}", f"{name} ({title})", k in ("mean_kl", "mean_fid"))
            for p, title in (("paper", "paper definitions, proper KL"), ("paper_code", "as in cGAN_EB code"))
            for k, name in CRITERIA if f"criteria.{p}.{k}" in df]
    means = df.groupby("model", sort=False)[[k for k, _, _ in keys]].mean()
    return pd.DataFrame({name: [f"{v:.3g}" if raw else f"{100 * v:.0f}%" for v in means[k]]
                         for k, name, raw in keys}, index=means.index)


def table(rows: list[dict], steps: bool = False) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    keys = [k for k in HEADLINE if k in df.columns]
    g = df.groupby("model", sort=False)
    mean, std, n = g[keys].mean(), g[keys].std(), g.size()

    def cell(m, s, k):
        pct = k.startswith("accuracy") or k in ("mean_profile", "std_profile", "kpi_mean", "kpi_std",
                                                 "hourly_dist", "kpi_dist")
        if pct:
            return f"{100 * m:.1f}" + (f" ± {100 * s:.1f}" if np.isfinite(s) else "")
        return f"{m:.3g}" + (f" ± {s:.2g}" if np.isfinite(s) else "")

    out = pd.DataFrame({COLUMNS[k]: [cell(mean.loc[i, k], std.loc[i, k], k) for i in mean.index] for k in keys},
                       index=mean.index)
    out.insert(0, "Seeds", n.loc[mean.index].values)
    if steps:
        out.insert(0, "Change", [STEPS.get(i.split(" [")[0], "") for i in mean.index])
    return out


# Direction of every reported metric: +1 higher is better, -1 lower is better, ("abs", t) closer to t,
# None descriptive only (no "better" direction).
DIRECTION = {
    "accuracy": +1, "mean_profile": +1, "std_profile": +1, "kpi_mean": +1, "kpi_std": +1, "hourly_dist": +1,
    "kpi_dist": +1, "accuracy.min_label": +1, "accuracy.share_labels_ge_85": +1, "paper.fid": -1,
    "paper.kl.near_peak": -1, "paper.kl.near_base": -1, "profile.rfd": -1, "profile.corr_mae": -1,
    "profile.c2st_auc": ("abs", 0.5), "profile.precision": +1, "profile.recall": +1, "profile.coverage": +1,
    "profile.dcr_ratio": ("log", 1.0), "physical.neg_share": -1,
    "criteria.paper.mean_diff_lt1": +1, "criteria.paper.std_diff_lt1": +1, "criteria.paper.kl_lt0.3": +1,
    "criteria.paper.fid_lt5": +1, "criteria.paper.mean_kl": -1, "criteria.paper.mean_fid": -1,
    "criteria.paper_code.mean_diff_lt1": +1, "criteria.paper_code.std_diff_lt1": +1,
    "criteria.paper_code.fid_lt5": +1, "criteria.paper_code.mean_fid": -1,
    "criteria.paper_code.kl_lt0.3": None, "criteria.paper_code.mean_kl": None,
}


def _badness(values, direction):
    v = np.asarray(values, float)
    if direction == +1:
        return -v
    if direction == -1:
        return v
    kind, target = direction
    return np.abs(v - target) if kind == "abs" else np.abs(np.log(v / target))


def comparison_table(rows: list[dict], final="final", faithful="faithful") -> pd.DataFrame:
    """Per metric: final (mean +/- SD over seeds) vs every GitHub value (best and last checkpoints, all seeds);
    'complete separation' when every final seed is better than every GitHub value."""
    df = pd.DataFrame(rows)
    fin = df[df.model == f"{final} [best]"]
    git = df[df.model.isin([f"{faithful} [best]", f"{faithful} [last]"])]
    out = []
    for k, direction in DIRECTION.items():
        if k not in df or fin[k].isna().all() or git[k].isna().all():
            continue
        name = COLUMNS.get(k, k.replace("criteria.", ""))
        row = {"metric": name, "better": {+1: "higher", -1: "lower"}.get(direction, "descriptive")
               if not isinstance(direction, tuple) else f"closer to {direction[1]}",
               "final": f"{fin[k].mean():.4g} ± {fin[k].std():.2g}",
               "GitHub [last]": f"{git[git.model.str.endswith('[last]')][k].mean():.4g}",
               "GitHub [best]": f"{git[git.model.str.endswith('[best]')][k].mean():.4g}"}
        if direction is None:
            row["final better"] = "descriptive only"
        else:
            bf, bg = _badness(fin[k], direction), _badness(git[k], direction)
            row["final better"] = ("yes, complete separation" if bf.max() < bg.min() else
                                   "on average" if bf.mean() < bg.mean() else "no")
        out.append(row)
    return pd.DataFrame(out)


def ablation_deltas(abl: list[dict], order: list[str]) -> dict[str, str]:
    """Seed-matched difference of test accuracy to the previous step (percentage points, mean +/- SD, n)."""
    df = pd.DataFrame(abl)
    out = {}
    for prev, cur in zip(order[:-1], order[1:]):
        a = df[df.model == prev].set_index("seed")["accuracy"]
        b = df[df.model == cur].set_index("seed")["accuracy"]
        common = a.index.intersection(b.index)
        if len(common):
            d = 100 * (b[common] - a[common])
            out[cur] = f"{d.mean():+.2f}" + (f" ± {d.std():.2f}" if len(d) > 1 else "") + f" (n={len(d)})"
    return out


def label_table(splits) -> pd.DataFrame:
    """Days and households per label and split (counterpart of Table 1 of the paper)."""
    rows = []
    for lab in range(N_LABELS):
        d, g, se = (int(v) for v in decode_label(lab))
        row = {"Label": lab, "Group": GROUP_NAMES[g], "Season": SEASON_NAMES[se], "Day type": DAYTYPE_NAMES[d]}
        for name in ("train", "val", "test"):
            m = splits[name].label == lab
            row[f"days {name}"] = int(m.sum())
            row[f"households {name}"] = int(len(np.unique(splits[name].meter[m])))
        row["days all"] = row["days train"] + row["days val"] + row["days test"]
        rows.append(row)
    return pd.DataFrame(rows)


def per_seed_table(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame([r for r in rows if not r["model"].startswith("Real")])
    if df.empty:
        return df
    t = df.pivot_table(index="model", columns="seed", values="accuracy", sort=False)
    t = (100 * t).round(2)
    t["min"], t["mean"] = t.min(axis=1), t.mean(axis=1).round(2)
    return t


def validation_curves(runs: Path, models: list[str], path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = ["#3B5BA9", "#008F8C", "#B45309", "#7E22CE"]
    for color, model in zip(colors, models):
        for k, run_dir in enumerate(finished_runs(runs, model)):
            v = pd.read_csv(run_dir / "val.csv")
            epochs = json.loads((run_dir / "config.json").read_text())["epochs"]
            ax.plot(100 * v.epoch / epochs, 100 * v.val_accuracy, color=color, lw=1.5, alpha=0.85,
                    label=f"{DISPLAY.get(model, model)}" if k == 0 else None)
    ax.axhline(85, color="#C2410C", ls="--", lw=1.2, label="target 85%")
    ref = runs / "real_reference" / "eval_val_train.json"
    if ref.exists():
        acc = 100 * json.loads(ref.read_text())["accuracy"]
        ax.axhline(acc, color="#15803D", ls=":", lw=1.2, label=f"real benchmark: training vs validation households {acc:.1f}%")
    ax.set_xlabel("training progress (% of planned epochs)")
    ax.set_ylabel("validation accuracy (%)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def paper_protocol_rows(runs: Path, specs: list[tuple[str, str]], splits, n_per_label=2000, seed=0) -> list[dict]:
    """Criteria with the paper's protocol: generated days against days of the training households."""
    from train import load_run

    rng = np.random.default_rng(seed)
    tr = splits["train"]
    idx = np.concatenate([rng.permutation(np.flatnonzero(tr.label == lab))[:n_per_label] for lab in range(N_LABELS)])
    real = Split(tr.X[idx], tr.label[idx], tr.meter[idx], tr.day[idx])
    rows = []
    for model, ckpt in specs:
        for run_dir in finished_runs(runs, model):
            wrapped, cfg = load_run(run_dir, ckpt)
            gen = wrapped.sample_kwh(real.label, EVAL_SEED + cfg["seed"])
            summary = summarise(evaluate(gen, real, tr, profile=False))
            rows.append({"model": f"{model} [{ckpt}]", "seed": run_dir.name, **summary})
            print(f"paper protocol {run_dir} [{ckpt}] accuracy {summary['accuracy']:.4f}", flush=True)
    return rows


def parse_spec(spec: str, default="best") -> tuple[str, str]:
    model, _, ckpt = spec.partition(":")
    return model, ckpt or default


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="results")
    ap.add_argument("--models", nargs="+", default=["faithful", "final"])
    ap.add_argument("--ckpts", nargs="+", default=["best", "last"])
    ap.add_argument("--figures", nargs="*", default=[], help="model[:ckpt] (seed 0), e.g. faithful:last final:best")
    ap.add_argument("--ablation", nargs="*", default=[])
    ap.add_argument("--selection", nargs="*", default=[],
                    help="model-selection candidates: best validation accuracy of each (no test data)")
    ap.add_argument("--paper-protocol", nargs="*", default=[], help="model[:ckpt], e.g. faithful:last final:best")
    ap.add_argument("--curves", nargs="*", default=[], help="models for the validation-curve figure")
    ap.add_argument("--data", default="data/processed/fluvius_hourly.npz")
    args = ap.parse_args(argv)
    runs, out = Path(args.runs), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.selection:
        from train import Validator, load_run

        val_splits = load_profiles(args.data)
        sel = []
        for name in args.selection:
            for run_dir in finished_runs(runs, name):
                s = json.loads((run_dir / "train_summary.json").read_text())
                cfg = json.loads((run_dir / "config.json").read_text())
                wrapped, _ = load_run(run_dir, "best")
                fresh = [Validator(val_splits["val"], cfg["val_max_per_label"], 50_000 + k).score(wrapped)[0]
                         for k in range(5)]
                sel.append({"candidate": name, "seed": run_dir.name, "hidden x depth": f"{cfg['hidden_size']} x "
                            f"{cfg.get('depth', '-')}", "noise": cfg["input_size"],
                            "critic sees hour differences": cfg.get("critic_diff", False),
                            "no-PV floor": cfg.get("nonneg_without_pv", False),
                            "best validation accuracy (selection draw)": f"{100 * s['best_val_accuracy']:.2f}%",
                            "best epoch": s["best_epoch"],
                            "same checkpoint, 5 fresh noise draws": f"{100 * np.mean(fresh):.2f} ± "
                                                                    f"{100 * np.std(fresh, ddof=1):.2f}%"})
                print(sel[-1], flush=True)
        sel = pd.DataFrame(sel)
        (out / "selection.md").write_text(sel.to_markdown(index=False) + "\n", encoding="utf-8")
        print(sel.to_markdown(index=False))
    if args.curves:
        validation_curves(runs, args.curves, out / "figures" / "validation_curves.png")

    cache = {}
    rows = []
    for model in args.models:
        for run_dir in finished_runs(runs, model):
            for ckpt in args.ckpts:
                rows.append({"model": f"{model} [{ckpt}]", "seed": run_dir.name,
                             **evaluate_run(run_dir, ckpt, cache)})
    splits = cache.get(("net", 2024)) or load_profiles(args.data)
    labels = label_table(splits)
    labels.to_csv(out / "table1_labels.csv", index=False)
    totals = {d: int(labels.loc[labels["Day type"] == d, "days all"].sum()) for d in DAYTYPE_NAMES}
    (out / "table1_labels.md").write_text(labels.to_markdown(index=False) + "\n\nTotal days: "
                                          + ", ".join(f"{d} {n:,}" for d, n in totals.items()) + "\n",
                                          encoding="utf-8")
    for src, name in (("val", "Real: validation households"), ("train", "Real: training households")):
        rows.append({"model": name, "seed": "-", **reference_row(runs, splits, src)})
    pd.DataFrame(rows).drop(columns="_fingerprint").to_csv(out / "all_runs.csv", index=False)
    t = table(rows)
    t.to_csv(out / "summary.csv")
    (out / "summary.md").write_text(t.to_markdown() + "\n", encoding="utf-8")
    print(t.to_markdown())
    p = per_seed_table(rows)
    (out / "per_seed.md").write_text(p.to_markdown() + "\n", encoding="utf-8")
    print(p.to_markdown())
    if any(r["model"].startswith("faithful") for r in rows) and any(r["model"].startswith("final") for r in rows):
        cmp_ = comparison_table(rows)
        (out / "comparison.md").write_text(cmp_.to_markdown(index=False) + "\n", encoding="utf-8")
        print(cmp_.to_markdown(index=False))
    c = criteria_table(rows)
    c.to_csv(out / "paper_criteria.csv")
    (out / "paper_criteria.md").write_text(c.to_markdown() + "\n", encoding="utf-8")
    print(c.to_markdown())

    if args.paper_protocol:
        pr = paper_protocol_rows(runs, [parse_spec(sp) for sp in args.paper_protocol], splits)
        c = criteria_table(pr)
        c.insert(0, "Accuracy (in-sample)", [f"{100 * v:.1f}" for v in
                                             pd.DataFrame(pr).groupby("model", sort=False)["accuracy"].mean()])
        (out / "paper_protocol.md").write_text(c.to_markdown() + "\n", encoding="utf-8")
        print(c.to_markdown())

    if args.ablation:
        abl = [{"model": model, "seed": run_dir.name, **evaluate_run(run_dir, "best", cache)}
               for model in args.ablation for run_dir in finished_runs(runs, model)]
        t = table(abl + [row for row in rows if row["model"].startswith("Real")], steps=True)
        deltas = ablation_deltas(abl, args.ablation)
        t.insert(2, "Δ accuracy vs previous step (pt)", [deltas.get(i, "") for i in t.index])
        t.to_csv(out / "ablation.csv")
        (out / "ablation.md").write_text(t.to_markdown() + "\n", encoding="utf-8")
        print(t.to_markdown())

    if args.figures:
        from train import load_run
        from utils import compare_profiles, generate_results, real_results, stats_plot

        # The last model listed (normally `final`) fixes the example labels of Figs. 7-8 for all models.
        examples, folders = None, {}
        for model, ckpt in reversed([parse_spec(sp) for sp in args.figures]):
            wrapped, cfg = load_run(runs / model / "seed0", ckpt)
            folder = out / "figures" / f"{model}_{ckpt}"
            real_results(splits["test"], folder)
            generate_results(wrapped, 2000, folder, seed=EVAL_SEED)
            paper = stats_plot(folder, "paper", examples and examples["paper"])
            code = stats_plot(folder, "code", examples and examples["code"])
            examples = examples or {"paper": paper["examples"], "code": code["examples"]}
            folders[f"{DISPLAY.get(model, model)} [{ckpt}]"] = folder
            print(f"figures in {folder}", flush=True)
        if len(folders) > 1:   # the last model listed (normally `final`) gets the main line style
            compare_profiles(folders, out / "figures" / "compare")


if __name__ == "__main__":
    main()
