"""Evaluate a generator or baseline against held-out real profiles.

    python -m cgan.evaluate --run runs/cgan_eb/seed0 --ckpt best
    python -m cgan.evaluate --baseline runs/baseline_copula/seed0

For every condition label the synthetic sample size equals the number of
real test days with that label.  Results are written in long format to
``<run>/eval_<split>_<ckpt>.csv`` (columns: label, metric, value; label -1 =
global) together with a JSON of label-averaged values.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data.profiles import N_LABELS, ProfileConfig, decode_label, load_profiles
from .eval.conditional import LabelClassifier, balanced_subsample, factor_accuracy
from .eval.kpis import DISCRETE, all_kpis
from .eval.metrics import aggregate_metrics, dcr_metrics, profile_metrics, scalar_metrics, temporal_metrics
from .utils import load_config, write_json


# ---------------------------------------------------------------------------
# synthetic sources
# ---------------------------------------------------------------------------
class GeneratorSource:
    def __init__(self, run_dir, ckpt="best", use_ema=True, device=None):
        from .sample import load_generator

        self.G, self.scaler, self.state = load_generator(run_dir, ckpt, use_ema, device)
        self.config = self.state["config"]

    def sample(self, labels, seed):
        from .sample import generate

        return generate(self.G, self.scaler, labels, seed)


class BaselineSource:
    def __init__(self, run_dir):
        from .baselines import PerLabelBaseline

        run_dir = Path(run_dir)
        self.model = PerLabelBaseline.load(run_dir / "model.npz")
        self.config = json.loads((run_dir / "meta.json").read_text())["config"]

    def sample(self, labels, seed):
        return self.model.sample(labels, seed)


class RealSource:
    """Draws real profiles of the requested split (e.g. train-vs-test noise floor)."""

    def __init__(self, split, config):
        self.split = split
        self.config = config

    def sample(self, labels, seed):
        rng = np.random.default_rng(seed)
        out = np.empty((len(labels), self.split.X.shape[1]), np.float32)
        for lab in np.unique(labels):
            idx = np.flatnonzero(labels == lab)
            pool = rng.permutation(np.flatnonzero(self.split.label == lab))
            # Every pool day once before any repetition (duplicates bias U-statistics).
            reps = int(np.ceil(len(idx) / len(pool)))
            take = np.concatenate([pool] + [rng.permutation(pool) for _ in range(reps - 1)])[:len(idx)]
            out[idx] = self.split.X[take]
        return out


# ---------------------------------------------------------------------------
_SPLIT_CACHE: dict[str, dict] = {}


def data_key(data_cfg: dict) -> str:
    return hashlib.sha1(json.dumps(data_cfg, sort_keys=True).encode()).hexdigest()[:12]


def get_splits(data_cfg: dict):
    key = data_key(data_cfg)
    if key not in _SPLIT_CACHE:
        _SPLIT_CACHE[key] = load_profiles(ProfileConfig.from_dict(data_cfg))
    return _SPLIT_CACHE[key]


def real_classifier(splits, data_cfg, cache_dir: Path, device, n_per_label=1500, seed=0) -> LabelClassifier:
    mu, sd = standardiser(splits["train"])
    path = cache_dir / f"clf_real_{data_key(data_cfg)}_n{n_per_label}_s{seed}.pt"
    tr = splits["train"]
    clf = LabelClassifier(tr.X.shape[1], seed=seed, device=device)
    if path.exists():
        clf.net.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        clf.net.eval()
        return clf
    idx = balanced_subsample(tr.label, n_per_label, seed)
    clf.fit((tr.X[idx] - mu) / sd, tr.label[idx])
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(clf.net.state_dict(), path)
    return clf


def standardiser(train):
    mu = train.X.mean(0).astype(np.float64)
    sd = train.X.std(0).astype(np.float64)
    return mu, np.where(sd > 0, sd, 1.0)


def evaluate(source, splits, data_cfg: dict, split: str = "test", gen_seed: int = 2025, device=None,
             c2st: bool = True, n_cap: int = 3000, dcr_cap: int = 1000, cache_dir: Path = Path("runs/_cache"),
             clf_n_per_label: int = 1500, log=print, save_samples: Path | None = None) -> pd.DataFrame:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    real = splits[split]
    train = splits["train"]
    T = real.X.shape[1]
    dt = 24.0 / T if data_cfg.get("target", "net") != "import_export" else 48.0 / T
    mu, sd = standardiser(train)
    syn = source.sample(real.label, gen_seed).astype(np.float64)
    if save_samples is not None:
        # Kept for household-bootstrap analyses (cgan.bootstrap) without regeneration.
        np.savez_compressed(save_samples, X=syn.astype(np.float32), label=real.label, seed=gen_seed)
    rows = []
    rng = np.random.default_rng(gen_seed)

    for lab in np.unique(real.label):
        m = real.label == lab
        r, s = real.X[m].astype(np.float64), syn[m]
        group = int(decode_label(lab)[1])
        kr, ks = all_kpis(r, dt), all_kpis(s, dt)
        for k in kr:
            for name, v in scalar_metrics(kr[k], ks[k], discrete=k in DISCRETE).items():
                rows.append((int(lab), f"kpi.{k}.{name}", v))
        sel = rng.choice(len(r), n_cap, replace=False) if len(r) > n_cap else np.arange(len(r))
        for name, v in profile_metrics((r[sel] - mu) / sd, (s[sel] - mu) / sd, device, c2st, seed=gen_seed).items():
            rows.append((int(lab), f"profile.{name}", v))
        for name, v in temporal_metrics(r, s).items():
            rows.append((int(lab), f"temporal.{name}", v))
        tr_lab = train.X[train.label == lab].astype(np.float64)
        ds = s[:dcr_cap]
        for name, v in dcr_metrics((ds - mu) / sd, (tr_lab - mu) / sd, (r - mu) / sd, device).items():
            rows.append((int(lab), f"privacy.{name}", v))
        for name, v in aggregate_metrics(r, s, seed=gen_seed + int(lab)).items():
            rows.append((int(lab), f"aggregate.{name}", v))
        if group & 1 == 0:  # no PV: net load cannot be negative
            rows.append((int(lab), "physical.neg_share", float(np.mean(s < -1e-6))))
        rows.append((int(lab), "physical.above_train_max", float(np.mean(s > train.X.max()))))
        rows.append((int(lab), "n_real", float(len(r))))

    clf = real_classifier(splits, data_cfg, cache_dir, device, clf_n_per_label)
    for f, v in factor_accuracy(clf.predict((syn - mu) / sd), real.label).items():
        rows.append((-1, f"cond.recovery.{f}", v))
    for f, v in factor_accuracy(clf.predict((real.X - mu) / sd), real.label).items():
        rows.append((-1, f"cond.real_reference.{f}", v))
    tstr_labels = np.repeat(np.arange(N_LABELS), clf_n_per_label)
    syn_tr = source.sample(tstr_labels, gen_seed + 1).astype(np.float64)
    clf_syn = LabelClassifier(T, seed=0, device=device).fit((syn_tr - mu) / sd, tstr_labels)
    for f, v in factor_accuracy(clf_syn.predict((real.X - mu) / sd), real.label).items():
        rows.append((-1, f"cond.tstr.{f}", v))

    df = pd.DataFrame(rows, columns=["label", "metric", "value"])
    log(f"evaluated {len(np.unique(real.label))} labels in {time.time() - t0:.0f}s")
    return df


def summarise(df: pd.DataFrame) -> dict[str, float]:
    per_label = df[df.label >= 0].groupby("metric").value.mean()
    glob = df[df.label < 0].set_index("metric").value
    return {**per_label.to_dict(), **glob.to_dict()}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", help="GAN run directory")
    g.add_argument("--baseline", help="baseline run directory")
    g.add_argument("--real", help="evaluate real profiles of this split against --split (noise floor)")
    ap.add_argument("--config", default="configs/base.yaml", help="data config for --real")
    ap.add_argument("--ckpt", default="best", choices=["best", "best_raw", "last", "last_raw"])
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--seed", type=int, default=2025, help="generation seed")
    ap.add_argument("--no-c2st", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    if args.run:
        src = GeneratorSource(args.run, args.ckpt, not args.no_ema)
        out_dir, tag = Path(args.run), f"eval_{args.split}_{args.ckpt}"
    elif args.baseline:
        src = BaselineSource(args.baseline)
        out_dir, tag = Path(args.baseline), f"eval_{args.split}"
    data_cfg = (src.config if not args.real else load_config(args.config)).get("data", {})
    splits = get_splits(data_cfg)
    if args.real:
        src = RealSource(splits[args.real], {"data": data_cfg})
        out_dir, tag = Path(args.out or f"runs/real_{args.real}") / f"seed{args.seed}", f"eval_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = evaluate(src, splits, data_cfg, args.split, args.seed, c2st=not args.no_c2st,
                  save_samples=out_dir / f"samples_{tag.removeprefix('eval_')}.npz")
    df.to_csv(out_dir / f"{tag}.csv", index=False)
    write_json(out_dir / f"{tag}.json", summarise(df))
    print(f"wrote {out_dir / (tag + '.csv')}")


if __name__ == "__main__":
    main()
