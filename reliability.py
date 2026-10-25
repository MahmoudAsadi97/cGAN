"""Reliability check: five headline metrics with confidence intervals over households.

    python reliability.py --models final:best faithful:last --reps 20 --workers 6

The five metrics, each computed per label on the test households and averaged over
the 64 labels:

    c2st_strong   ROC-AUC of a gradient-boosting classifier separating real from generated
                  days, on the values, the step-to-step changes and both sorted (values
                  rounded to the meter resolution of 0.001 kWh first); folds grouped by
                  household.  0.5 = indistinguishable
    accuracy      statistical accuracy (``metrics.accuracy``), the 85 % target
    kp_ks         Kolmogorov-Smirnov distance of the key parameters (near-peak, near-base,
                  peak duration, rise, fall, daily energy), averaged; 0 = identical
    coverage      share of real days with a generated day among their 5 nearest real
                  neighbours (Naeem et al. 2020); real-vs-real gives about 0.8
    dcr_ratio     median distance of generated days to the nearest training day of the label,
                  divided by the same for the real test days; about 1 expected, << 1 = copying

Confidence intervals: the test households of every technology group are split at random
into two halves ``reps`` times; every metric is computed on both halves, and the standard
error of the full-sample value is SD(half A - half B) / 2 (as ``uncertainty.py``).  For a
model with several seeds the seed-averaged value is used and the seed SD is added:
SE_total = sqrt(SE_households^2 + SD_seeds^2 / n_seeds).  The same splits are used for all
rows, so differences to the real-data benchmarks are paired.

Runs on the CPU (``CUDA_VISIBLE_DEVICES=-1``) at below-normal priority, so it does
not disturb training; CPU noise differs from the GPU draws of ``report.py``.
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"           # no GPU (an empty value is dropped on Windows)
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import ctypes  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import zlib  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data_prep import N_LABELS, decode_label, load_profiles  # noqa: E402
from evaluate import EVAL_SEED, real_reference  # noqa: E402
from metrics import KEYS, _ks, accuracy, c2st_auc, key_parameters  # noqa: E402

METRICS = ["c2st_strong", "accuracy", "kp_ks", "coverage", "dcr_ratio"]
IDEAL = {"c2st_strong": "0.5", "accuracy": "higher", "kp_ks": "lower", "coverage": "higher", "dcr_ratio": "about 1"}
N_MAX = 1000          # days per side and label for the classifier, coverage and DCR
_S = {}


def _below_normal():
    if sys.platform == "win32":
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)


def _init(test_X, test_label, test_meter, train_X, train_label, mu, sd, sources, labels):
    import torch

    _below_normal()
    torch.set_num_threads(1)
    _S.update(labels=labels, X=test_X, label=test_label, meter=test_meter, tX=train_X, tlabel=train_label, mu=mu, sd=sd,
              sources=sources)


def _features(x):
    x = np.round(x, 3)                                 # meter resolution
    d = np.diff(x, axis=1)
    return np.hstack([x, d, np.sort(x, 1), np.sort(np.abs(d), 1)])


def _nn_dist(a, b, exclude_self=False):
    import torch

    d = torch.cdist(torch.as_tensor(a), torch.as_tensor(b))
    if exclude_self:
        d.fill_diagonal_(float("inf"))
    return d


def _label_metrics(lab, rows, name, rng):
    """The five metrics of one label on the given test rows."""
    X, meter = _S["X"], _S["meter"]
    gen, gen_meter = _S["sources"][name]
    r = rows[_S["label"][rows] == lab]
    real, fake = X[r], gen[r]
    out = {}
    acc = accuracy(real, fake, key_parameters(real), key_parameters(fake))
    out["accuracy"] = acc["accuracy"]
    out["kp_ks"] = float(np.mean([1.0 - acc[f"kpi_dist.{k}"] for k in KEYS]))
    pick = rng.permutation(len(r))[:N_MAX]
    rr, gg = real[pick], fake[pick]
    out["c2st_strong"] = c2st_auc(_features(rr), _features(gg), meter[r][pick], gen_meter[r][pick])
    zs = lambda a: ((a - _S["mu"]) / _S["sd"]).astype(np.float64)       # noqa: E731
    zr, zg = zs(rr), zs(gg)
    rad = _nn_dist(zr, zr, exclude_self=True).kthvalue(5, 1).values
    out["coverage"] = float((_nn_dist(zr, zg).min(1).values < rad).double().mean())
    tr = _S["tX"][_S["tlabel"] == lab]
    tr = zs(tr[rng.permutation(len(tr))[:5000]])
    d_g = _nn_dist(zg, tr).min(1).values.median()
    d_r = _nn_dist(zr, tr).min(1).values.median()
    out["dcr_ratio"] = float(d_g / max(float(d_r), 1e-12))
    return out


def _evaluate(task):
    """Label-averaged metrics of one source on one set of test rows (full sample or a half)."""
    name, rep, half, rows = task
    rng = np.random.default_rng(zlib.crc32(f"{name}|{rep}|{half}".encode()))
    per = [_label_metrics(lab, rows, name, rng) for lab in _S["labels"]]
    return name, rep, half, {m: float(np.mean([p[m] for p in per])) for m in METRICS}


def _halves(meter, label, rep):
    rng = np.random.default_rng(10_000 + rep)                       # the splits of uncertainty.py
    group = decode_label(label)[1]
    a = np.zeros(len(label), bool)
    for g in np.unique(group):
        houses = rng.permutation(np.unique(meter[group == g]))
        a |= np.isin(meter, houses[: len(houses) // 2]) & (group == g)
    return np.flatnonzero(a), np.flatnonzero(~a)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="results")
    ap.add_argument("--models", nargs="+", default=["final:best", "faithful:last"],
                    help="model:ckpt; every finished seed is used")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--data", default="data/processed/fluvius_hourly.npz")
    ap.add_argument("--labels", type=int, default=N_LABELS, help="first n labels only (quick test)")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    _below_normal()
    from train import load_run

    splits = load_profiles(args.data)
    test, train = splits["test"], splits["train"]
    sources, groups = {}, {}
    for spec in args.models:
        model, ckpt = spec.split(":")
        for run_dir in sorted(Path(args.runs, model).glob("seed*")):
            if not (run_dir / "train_summary.json").exists():
                continue
            wrapped, cfg = load_run(run_dir, ckpt)
            name = f"{model} [{ckpt}] {run_dir.name}"
            sources[name] = (wrapped.sample_kwh(test.label, EVAL_SEED + cfg["seed"]).astype(np.float32), test.meter)
            groups.setdefault(f"{model} [{ckpt}]", []).append(name)
    for src, title in (("val", "real: validation households"), ("train", "real: training households")):
        days, meters = real_reference(splits, src, "test")
        sources[title] = (days.astype(np.float32), meters)
        groups[title] = [title]
    print(f"{len(sources)} generated sets, {args.reps} replicates", flush=True)

    mu, sd = train.X.mean(0), train.X.std(0) + 1e-6
    tasks = [(name, -1, "full", np.arange(len(test))) for name in sources]
    for rep in range(args.reps):
        a, b = _halves(test.meter, test.label, rep)
        tasks += [(name, rep, half, rows) for name in sources for half, rows in (("A", a), ("B", b))]
    results = {}
    with ProcessPoolExecutor(args.workers, initializer=_init,
                             initargs=(test.X, test.label, test.meter, train.X, train.label, mu, sd, sources,
                                       list(range(args.labels)))) as ex:
        for k, (name, rep, half, vals) in enumerate(ex.map(_evaluate, tasks, chunksize=1)):
            results[(name, rep, half)] = vals
            if k % 20 == 0:
                print(f"{k + 1}/{len(tasks)} done", flush=True)

    rows, diffs = [], {}
    for title, names in groups.items():
        row = {"model": title, "seeds": len(names)}
        for m in METRICS:
            full = np.array([results[(n, -1, "full")][m] for n in names])
            d = np.array([np.mean([results[(n, rep, "A")][m] - results[(n, rep, "B")][m] for n in names])
                          for rep in range(args.reps)])
            diffs[(title, m)] = d
            se = np.sqrt((d.std(ddof=1) / 2) ** 2 + (full.std(ddof=1) ** 2 / len(full) if len(full) > 1 else 0))
            est = full.mean()
            row[m] = f"{est:.3f} ({est - 1.96 * se:.3f} to {est + 1.96 * se:.3f})"
            row[f"_{m}"] = est
        rows.append(row)
    table = pd.DataFrame(rows)
    paired = []
    for title in groups:
        if title.startswith("real"):
            continue
        row = {"comparison": f"{title} - real: validation households"}
        for m in METRICS:
            d = diffs[(title, m)] - diffs[("real: validation households", m)]
            est = table.set_index("model").loc[title, f"_{m}"] - table.set_index("model").loc[
                "real: validation households", f"_{m}"]
            se = d.std(ddof=1) / 2
            row[m] = f"{est:+.3f} ({est - 1.96 * se:+.3f} to {est + 1.96 * se:+.3f})"
        paired.append(row)
    shown = table[["model", "seeds"] + METRICS]
    note = (f"\nTest households, label-averaged; in brackets the 95 % interval over households (stratified "
            f"half-sampling, {args.reps} replicates) including the seed spread.  Ideal: "
            + ", ".join(f"{m} {IDEAL[m]}" for m in METRICS) + ".  CPU sampling.\n")
    text = (shown.to_markdown(index=False) + "\n" + note + "\nPaired differences to the real validation-household"
            " benchmark (same household splits):\n\n" + pd.DataFrame(paired).to_markdown(index=False) + "\n")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "reliability.md").write_text(text, encoding="utf-8")
    pd.DataFrame([{"source": k[0], "rep": k[1], "half": k[2], **v} for k, v in results.items()]).to_csv(
        out / "reliability_replicates.csv", index=False)
    print(text)


if __name__ == "__main__":
    main()
