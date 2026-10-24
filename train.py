"""Train the cGAN on the Fluvius 2024 daily profiles (reference: cGAN_EB ``train.py``).

    python train.py --config faithful --seed 0
    python train.py --config faithful --seed 0 --name faithful_e100 --set epochs=100

The reference script reads all profiles, drops days with values < 0 or
> 100 kWh, min-max scales with the global minimum and maximum, trains for
10,000 epochs with batch size 2,000 and writes 2,000 generated days per label.
Here the data come from ``data_prep.py`` and the households are split into
train / validation / test: the scaler and the model see only the training
households, the validation households select the checkpoint (statistical
accuracy, see ``metrics.py``) and the test households are used only by
``evaluate.py`` / ``report.py`` for the reported evaluation (see README,
Protocol notes).  The < 0 filter is not applied because the target is net
load, which is negative whenever PV exports.

Long runs can be interrupted: after every validation the full training state
(weights, optimiser states, EMA, loss history, best checkpoint so far and the
random-number generator states) is written to ``resume.pt``, and
``--resume`` continues the run from there (at most ``val_every`` epochs are
repeated).  ``run_jobs.py`` runs a list of jobs this way.

A run directory ``runs/<name>/seed<k>/`` receives ``config.json``,
``history.csv`` (losses) and ``val.csv`` (validation accuracy), both written while
training runs (``dashboard.py`` shows them live), ``best.pt`` and ``last.pt``.
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from data_prep import N_LABELS, decode_label, load_profiles
from metrics import accuracy

CONFIGS = {
    # Reference hyper-parameters (cGAN_EB train.py / Table 2 of the paper).
    "faithful": {
        "model": "cGAN", "input_size": 24, "hidden_size": 64, "lr": 1e-4,
        "epochs": 10000, "batch_size": 2000, "shuffle": False, "scaler": "minmax",
    },
    # Improved variant (see README): Wasserstein loss with gradient penalty, wider networks,
    # shuffled batches, per-hour Gaussianising value transform, EMA generator.
    "improved": {
        "model": "cWGAN", "input_size": 32, "hidden_size": 256, "depth": 3, "lr": 1e-4, "n_critic": 5,
        "gp_lambda": 10.0, "ema": 0.999, "epochs": 400, "batch_size": 1024, "shuffle": True,
        "scaler": "quantile", "val_every": 10, "nonneg_without_pv": True, "critic_diff": False,
    },
}
# Ablation ladder from the reference to `improved`.  abl_shuffle and abl_quantile each add one change;
# abl_wgan_refnets switches to the WGAN-GP training recipe as a whole (loss, critic without sigmoid and
# dropout, 5 critic steps, Adam lr 1e-4 for both networks with betas (0.5, 0.9), batch 1,024, 400 critic
# epochs, validation every 10 epochs) while keeping the reference networks; abl_wide widens the networks
# (3 x 256) and the noise (32-d); abl_ema adds the EMA read-out; `improved` adds the no-PV floor.
CONFIGS["abl_shuffle"] = {**CONFIGS["faithful"], "shuffle": True}
CONFIGS["abl_quantile"] = {**CONFIGS["abl_shuffle"], "scaler": "quantile"}
CONFIGS["abl_wgan_refnets"] = {**CONFIGS["improved"], "ref_nets": True, "input_size": 24, "hidden_size": 64,
                               "ema": 0.0, "nonneg_without_pv": False}
CONFIGS["abl_wide"] = {**CONFIGS["improved"], "ema": 0.0, "nonneg_without_pv": False}
CONFIGS["abl_ema"] = {**CONFIGS["improved"], "nonneg_without_pv": False}
# Final model, chosen on the validation households among `improved` and three variants
# (wider networks, critic input with hour-to-hour differences, both); see README.
CONFIGS["final"] = {**CONFIGS["improved"], "hidden_size": 512, "depth": 4, "input_size": 64, "critic_diff": True}
DEFAULTS = {"target": "net", "split_seed": 2024, "val_every": 50, "val_max_per_label": 2000, "val_seed": 12345}


# ---------------------------------------------------------------------------
class Scaler:
    """Value transform fitted on the training households.

    ``minmax``    (x - min) / (max - min), global extremes (reference)
    ``zscore``    per-hour standardisation
    ``quantile``  per-hour Gaussianisation: x -> Phi^-1(F_t(x)) with F_t the empirical CDF of hour t
                  (all labels pooled), inverted by interpolation between ``n_quantiles`` training
                  quantiles; generated values therefore stay inside the observed range of each hour.
    """

    def __init__(self, kind="minmax", **state):
        self.kind = kind
        self.state = state

    def fit(self, X, n_quantiles=2001):
        X = np.asarray(X, np.float64)
        if self.kind == "minmax":
            self.state = {"min": float(X.min()), "max": float(X.max())}
        elif self.kind == "zscore":
            self.state = {"mean": X.mean(0).tolist(), "std": X.std(0).tolist()}
        elif self.kind == "quantile":
            p = np.linspace(0, 1, n_quantiles)
            self.state = {"p": p.tolist(), "q": np.quantile(X, p, axis=0).T.tolist()}
        else:
            raise ValueError(f"unknown scaler {self.kind!r}")
        return self

    def transform(self, X):
        from scipy.stats import norm

        s, X = self.state, np.asarray(X, np.float64)
        if self.kind == "minmax":
            return (X - s["min"]) / (s["max"] - s["min"])
        if self.kind == "zscore":
            return (X - np.array(s["mean"])) / np.array(s["std"])
        p, q = np.array(s["p"]), np.array(s["q"])
        eps = 1e-6
        u = np.column_stack([np.interp(X[:, t], q[t], p) for t in range(X.shape[1])])
        return norm.ppf(np.clip(u, eps, 1 - eps))

    def inverse(self, Z):
        from scipy.stats import norm

        s, Z = self.state, np.asarray(Z, np.float64)
        if self.kind == "minmax":
            return Z * (s["max"] - s["min"]) + s["min"]
        if self.kind == "zscore":
            return Z * np.array(s["std"]) + np.array(s["mean"])
        p, q = np.array(s["p"]), np.array(s["q"])
        u = norm.cdf(Z)
        return np.column_stack([np.interp(u[:, t], p, q[t]) for t in range(Z.shape[1])])

    def to_dict(self):
        return {"kind": self.kind, **self.state}


def build_model(cfg, data_dim):
    import cGAN as models

    if cfg["model"] == "cGAN":
        return models.cGAN(cfg["input_size"], N_LABELS, cfg["hidden_size"], data_dim, cfg["lr"],
                           seed=cfg["seed"])
    if cfg["model"] == "cWGAN":
        return models.cWGAN(cfg["input_size"], N_LABELS, cfg["hidden_size"], data_dim, cfg["lr"],
                            depth=cfg["depth"], n_critic=cfg["n_critic"], gp_lambda=cfg["gp_lambda"],
                            ema=cfg["ema"], critic_diff=cfg.get("critic_diff", False),
                            ref_nets=cfg.get("ref_nets", False), seed=cfg["seed"])
    raise ValueError(f"unknown model {cfg['model']!r}")


class Wrapped:
    """Model + scaler: generates profiles in kWh.

    With ``nonneg_without_pv`` the net load of labels without PV is floored at 0 kWh:
    meters without PV register no injection (release legend), so they cannot export.
    """

    def __init__(self, model, scaler, nonneg_without_pv=False):
        self.model, self.scaler, self.nonneg_without_pv = model, scaler, nonneg_without_pv

    def sample_kwh(self, labels, seed):
        x = self.scaler.inverse(self.model.sample(labels, seed))
        if self.nonneg_without_pv:
            no_pv = decode_label(np.asarray(labels))[1] & 1 == 0
            x[no_pv] = np.maximum(x[no_pv], 0.0)
        return x


def load_run(run_dir: Path, ckpt="best"):
    cfg = json.loads((run_dir / "config.json").read_text())
    state = torch.load(run_dir / f"{ckpt}.pt", map_location="cpu", weights_only=True)
    model = build_model(cfg, state["data_dim"])
    model.load_state_dict(state["model"])
    return Wrapped(model, Scaler(**state["scaler"]), cfg.get("nonneg_without_pv", False)), cfg


class Validator:
    """Statistical accuracy on the validation households, with fixed noise."""

    def __init__(self, val, n_max, seed):
        self.seed = seed
        self.real = {}
        labels = []
        for lab in np.unique(val.label):
            r = val.X[val.label == lab]
            self.real[int(lab)] = r
            labels.append(np.full(min(len(r), n_max), lab))
        self.labels = np.concatenate(labels)

    def score(self, wrapped):
        g = wrapped.sample_kwh(self.labels, self.seed)
        accs = [accuracy(self.real[lab], g[self.labels == lab])["accuracy"] for lab in self.real]
        return float(np.mean(accs)), float(np.min(accs))


def _rng_state():
    name, keys, pos, has_gauss, gauss = np.random.get_state()
    return {"torch": torch.get_rng_state(), "numpy": [name, keys.tolist(), pos, has_gauss, gauss],
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _set_rng_state(s):
    torch.set_rng_state(s["torch"])
    name, keys, pos, has_gauss, gauss = s["numpy"]
    np.random.set_state((name, np.array(keys, np.uint32), pos, has_gauss, gauss))
    if torch.cuda.is_available() and s["cuda"]:
        torch.cuda.set_rng_state_all(s["cuda"])


def _truncate_csv(path: Path, epoch: int):
    """Keep the header and the rows up to ``epoch`` (rows written after the last checkpoint are repeated)."""
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    keep = [rows[0]] + [r for r in rows[1:] if r and int(float(r[0])) <= epoch]
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(keep)


def train(cfg, run_dir: Path, resume: bool = False):
    run_dir.mkdir(parents=True, exist_ok=True)
    splits = load_profiles(cfg["data"], cfg["target"], cfg["split_seed"])
    tr = splits["train"]
    scaler = Scaler(cfg["scaler"]).fit(tr.X)
    data = torch.tensor(np.column_stack([scaler.transform(tr.X), tr.label]))  # data + labels, as the reference
    model = build_model(cfg, tr.X.shape[1])
    wrapped = Wrapped(model, scaler, cfg.get("nonneg_without_pv", False))
    validator = Validator(splits["val"], cfg["val_max_per_label"], cfg["val_seed"])
    best = {"acc": -1.0}
    start_epoch, offset = 0, 0.0
    if resume:
        state = torch.load(run_dir / "resume.pt", map_location="cpu", weights_only=True)
        model.load_training_state(state["training"])
        best = state["best"]
        start_epoch, offset = state["epoch"], state["seconds"]
        _set_rng_state(state["rng"])
        for name in ("val.csv", "history.csv"):
            _truncate_csv(run_dir / name, start_epoch)
        print(f"resuming {run_dir} at epoch {start_epoch} (best {best['acc']:.4f})", flush=True)
    else:
        (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))
        for name, header in (("val.csv", ["epoch", "val_accuracy", "val_min_label", "seconds"]),
                             ("history.csv", ["epoch", "loss_G", "loss_D", "seconds"])):
            with open(run_dir / name, "w", newline="") as f:
                csv.writer(f).writerow(header)
    val_f = open(run_dir / "val.csv", "a", newline="")
    val_w = csv.writer(val_f)
    hist_f = open(run_dir / "history.csv", "a", newline="")
    hist_w = csv.writer(hist_f)
    t0 = time.time() - offset

    def save(name, epoch, acc):
        torch.save({"model": model.state_dict(), "scaler": scaler.to_dict(), "epoch": epoch, "val_accuracy": acc,
                    "data_dim": tr.X.shape[1]}, run_dir / f"{name}.pt")

    def callback(m, epoch):
        h = m.history[-1]
        hist_w.writerow([h["epoch"], f"{h['loss_G']:.6g}", f"{h['loss_D']:.6g}", h["seconds"]])
        hist_f.flush()
        acc, worst = validator.score(wrapped)
        val_w.writerow([epoch, f"{acc:.5f}", f"{worst:.5f}", round(time.time() - t0, 1)])
        val_f.flush()
        if acc > best["acc"]:
            best.update(acc=acc, epoch=epoch)
            save("best", epoch, acc)
        print(f"  val accuracy {acc:.4f} (worst label {worst:.4f}; best {best['acc']:.4f} @ {best['epoch']})",
              flush=True)
        # Full training state for --resume; written to a temporary file first so that an
        # interruption while saving cannot corrupt the previous checkpoint.
        torch.save({"training": m.training_state(), "best": best, "epoch": epoch,
                    "seconds": round(time.time() - t0, 1), "rng": _rng_state()}, run_dir / "resume.tmp")
        os.replace(run_dir / "resume.tmp", run_dir / "resume.pt")

    model.train(data, cfg["epochs"], cfg["batch_size"], shuffle=cfg["shuffle"], callback=callback,
                callback_every=cfg["val_every"], start_epoch=start_epoch, time_offset=offset)
    save("last", cfg["epochs"], validator.score(wrapped)[0])
    hist_f.close()
    val_f.close()
    summary = {"best_epoch": best["epoch"], "best_val_accuracy": best["acc"], "seconds": round(time.time() - t0, 1),
               "n_train": len(tr), "n_val": len(splits["val"]), "n_test": len(splits["test"])}
    (run_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def keep_awake():
    """Keep Windows from sleeping while this process trains (no effect elsewhere)."""
    if sys.platform == "win32":
        import ctypes

        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def parse_value(v):
    try:
        return json.loads(v)
    except json.JSONDecodeError:
        return v


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="faithful", choices=sorted(CONFIGS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default=None)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--data", default="data/processed/fluvius_hourly.npz")
    ap.add_argument("--set", nargs="*", default=[], help="overrides, e.g. epochs=100 shuffle=true")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing run directory")
    ap.add_argument("--resume", action="store_true",
                    help="continue an interrupted run from its resume.pt (start it if it does not exist)")
    args = ap.parse_args(argv)
    keep_awake()
    cfg = {**DEFAULTS, **CONFIGS[args.config], "config": args.config, "seed": args.seed, "data": args.data}
    for kv in args.set:
        k, v = kv.split("=", 1)
        if k not in cfg or k in ("seed", "data", "config"):
            raise SystemExit(f"--set: unknown or reserved key {k!r} (use --seed / --data / --config)")
        v = parse_value(v)
        if isinstance(cfg[k], bool) and not isinstance(v, bool):
            raise SystemExit(f"--set {k}: expected true or false")
        cfg[k] = v
    name = args.name or args.config
    if args.set and not args.name:
        raise SystemExit("--set changes the configuration: give the run its own --name")
    run_dir = Path(args.out) / name / f"seed{args.seed}"
    if args.resume and (run_dir / "train_summary.json").exists():
        print(f"{run_dir} is finished; nothing to resume")
        return
    if args.resume and (run_dir / "resume.pt").exists():
        saved = json.loads((run_dir / "config.json").read_text())
        if saved != cfg:
            diff = sorted(k for k in set(saved) | set(cfg) if saved.get(k) != cfg.get(k))
            raise SystemExit(f"{run_dir}: configuration differs from the saved run ({diff}); not resuming")
        print(json.dumps(train(cfg, run_dir, resume=True), indent=2))
        return
    if (run_dir / "config.json").exists():
        if not (args.overwrite or args.resume):
            raise SystemExit(f"{run_dir} exists; use another --name, --overwrite or --resume")
        # Remove everything the previous run left behind, so no stale result can be read later.
        for f in run_dir.iterdir():
            if f.is_file():
                f.unlink()
    print(json.dumps(train(cfg, run_dir), indent=2))


if __name__ == "__main__":
    main()
