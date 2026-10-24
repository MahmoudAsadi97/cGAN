"""Evaluate generated profiles against real days of held-out households.

    python evaluate.py --run runs/faithful/seed0 --ckpt best --split test
    python evaluate.py --real val --split test        # real-data reference

For every label the generator produces as many days as the held-out split has
for that label, and every metric of ``metrics.py`` is computed per label.
Results: ``<run>/eval_<split>_<ckpt>.csv`` (label, metric, value) and a JSON
with the label-averaged values.  The real-data benchmark scores real days of
other households (the validation or the training households) against the
held-out households.  It is a benchmark, not an upper bound: it depends on how
many and which households are in the pool, and a generator that represents the
population well can exceed it.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data_prep import decode_label, load_profiles
from metrics import PAPER_KEYS, accuracy, key_parameters, paper_code_metrics, paper_metrics, profile_metrics

EVAL_SEED = 2025


def evaluate(gen: np.ndarray, real_split, train_split, labels=None, profile=True, seed=EVAL_SEED,
             gen_groups=None) -> pd.DataFrame:
    """Per-label metrics of ``gen`` (rows aligned with ``real_split.label``).

    ``gen_groups``: household of every generated row when ``gen`` are real days (references).
    """
    rng = np.random.default_rng(seed)
    mu, sd = train_split.X.mean(0), train_split.X.std(0)
    rows = []
    for lab in np.unique(real_split.label) if labels is None else labels:
        m = real_split.label == lab
        r, g = real_split.X[m].astype(np.float64), np.asarray(gen[m], np.float64)
        kr, kg = key_parameters(r), key_parameters(g)
        res = accuracy(r, g, kr, kg)
        res.update({f"paper.{k}": v for k, v in paper_metrics(kr, kg).items()})
        res.update({f"paper_code.{k}": v for k, v in paper_code_metrics(r, g).items()})
        if profile:
            tr = train_split.X[train_split.label == lab]
            res.update({f"profile.{k}": v for k, v in
                        profile_metrics((r - mu) / sd, (g - mu) / sd, (tr - mu) / sd, rng,
                                        real_groups=real_split.meter[m],
                                        gen_groups=None if gen_groups is None else gen_groups[m]).items()})
        if decode_label(lab)[1] & 1 == 0:
            res["physical.neg_share"] = float(np.mean(g < -1e-6))
        rows += [(int(lab), k, float(v)) for k, v in res.items()]
    return pd.DataFrame(rows, columns=["label", "metric", "value"])


def summarise(df: pd.DataFrame) -> dict:
    per_label = df.pivot(index="label", columns="metric", values="value")
    out = per_label.mean().to_dict()
    out["accuracy.min_label"] = float(per_label["accuracy"].min())
    out["accuracy.share_labels_ge_85"] = float((per_label["accuracy"] >= 0.85).mean())
    out.update(paper_criteria(per_label))
    return out


REF_KEYS = ["Peak_load", "Base_load", "Peak_duration", "Rise", "Fall"]


def paper_criteria(per_label: pd.DataFrame) -> dict:
    """Share of (label, key parameter) pairs that meet the acceptance levels quoted by Li et al. (2025):
    |difference of mean| and |difference of std| below 1 (kWh for loads, h for times), KL below 0.3 and
    FID below 5; for the paper's text definitions with a proper KL (``paper.``) and for the definitions
    and "KLD" of the cGAN_EB code that produced the published figures (``paper_code.``)."""
    out = {}
    for prefix, keys, kl in (("paper", PAPER_KEYS, "kl"), ("paper_code", REF_KEYS, "kld")):
        cols = lambda m: [f"{prefix}.{m}.{k}" for k in keys if f"{prefix}.{m}.{k}" in per_label]
        if not cols("mean_diff"):
            continue
        out[f"criteria.{prefix}.mean_diff_lt1"] = float((per_label[cols("mean_diff")].abs() < 1).to_numpy().mean())
        out[f"criteria.{prefix}.std_diff_lt1"] = float((per_label[cols("std_diff")].abs() < 1).to_numpy().mean())
        out[f"criteria.{prefix}.kl_lt0.3"] = float((per_label[cols(kl)] < 0.3).to_numpy().mean())
        out[f"criteria.{prefix}.fid_lt5"] = float((per_label[cols("fid")] < 5).to_numpy().mean())
        out[f"criteria.{prefix}.mean_kl"] = float(np.nanmean(per_label[cols(kl)].to_numpy()))
        out[f"criteria.{prefix}.mean_fid"] = float(np.nanmean(per_label[cols("fid")].to_numpy()))
    return out


def real_reference(splits, source="val", target="test", seed=0):
    """Real days of ``source`` households, one per day of ``target`` (same labels), without repeats where
    possible; returns the days and their households."""
    rng = np.random.default_rng(seed)
    src, tgt = splits[source], splits[target]
    out = np.empty_like(tgt.X)
    meters = np.empty(len(tgt), np.int64)
    for lab in np.unique(tgt.label):
        idx = np.flatnonzero(tgt.label == lab)
        pool = np.flatnonzero(src.label == lab)
        take = np.concatenate([rng.permutation(pool) for _ in range(int(np.ceil(len(idx) / len(pool))))])
        out[idx] = src.X[take[:len(idx)]]
        meters[idx] = src.meter[take[:len(idx)]]
    return out, meters


HEADLINE = ["accuracy", "mean_profile", "std_profile", "kpi_mean", "kpi_std", "hourly_dist", "kpi_dist",
            "accuracy.min_label", "accuracy.share_labels_ge_85", "paper.fid", "paper.kl.near_peak",
            "paper.kl.near_base", "profile.rfd", "profile.corr_mae", "profile.c2st_auc", "profile.precision",
            "profile.recall", "profile.coverage", "profile.dcr_ratio", "physical.neg_share"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run")
    g.add_argument("--real", choices=["val", "train"])
    ap.add_argument("--ckpt", default="best")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--data", default="data/processed/fluvius_hourly.npz")
    ap.add_argument("--out", default="runs/real_reference")
    args = ap.parse_args(argv)

    if args.run:
        from train import load_run

        model, cfg = load_run(Path(args.run), args.ckpt)
        splits = load_profiles(args.data, cfg["target"], cfg["split_seed"])
        real = splits[args.split]
        gen = model.sample_kwh(real.label, EVAL_SEED + cfg["seed"])
        out_dir, tag = Path(args.run), f"eval_{args.split}_{args.ckpt}"
    else:
        splits = load_profiles(args.data)
        real = splits[args.split]
        gen, gen_groups = real_reference(splits, args.real, args.split)
        out_dir, tag = Path(args.out), f"eval_{args.split}_{args.real}"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = evaluate(gen, real, splits["train"], gen_groups=None if args.run else gen_groups)
    df.to_csv(out_dir / f"{tag}.csv", index=False)
    summary = summarise(df)
    (out_dir / f"{tag}.json").write_text(json.dumps(summary, indent=2))
    for k in HEADLINE:
        if k in summary:
            print(f"{k:32s} {summary[k]:.4f}")


if __name__ == "__main__":
    main()
