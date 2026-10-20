"""Non-adversarial baselines, each fitted independently per condition label.

``gaussian``  multivariate normal N(mu_k, S_k) of the D-dim daily profile.
``copula``    Gaussian copula with empirical marginals per time step: rank ->
              normal scores, correlation of the scores, sample, map back
              through the empirical quantile function.  Keeps skewed,
              sign-changing marginals exactly.
``gmm``       full-covariance Gaussian mixture; the number of components is
              chosen per label by validation log-likelihood.
``resample``  draws training-set profiles of the same label with replacement.
              This is a copy baseline: it bounds attainable fidelity and
              shows what memorisation looks like in the privacy metrics.

Fitted parameters are stored as plain arrays (``model.npz``).

    python -m cgan.baselines --kind copula --out runs
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.special import ndtr, ndtri
from sklearn.mixture import GaussianMixture

from .data.profiles import ProfileConfig, Split, load_profiles
from .utils import load_config, write_json

KINDS = ("gaussian", "copula", "gmm", "resample")


class PerLabelBaseline:
    def __init__(self, kind: str, gmm_components=(1, 2, 4, 8, 16), seed: int = 0):
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        self.kind = kind
        self.gmm_components = tuple(gmm_components)
        self.seed = seed
        self.params: dict[int, dict[str, np.ndarray]] = {}
        self.info: dict[int, dict] = {}

    # ------------------------------------------------------------------ api
    def fit(self, train: Split, val: Split | None = None) -> "PerLabelBaseline":
        for lab in np.unique(train.label):
            x = train.X[train.label == lab].astype(np.float64)
            xv = None if val is None else val.X[val.label == lab].astype(np.float64)
            self.params[int(lab)] = getattr(self, f"_fit_{self.kind}")(x, xv, int(lab))
        return self

    def sample(self, labels: np.ndarray, seed: int = 0) -> np.ndarray:
        labels = np.asarray(labels)
        dim = next(iter(self.params.values()))["dim"].item()
        out = np.empty((len(labels), dim), np.float32)
        for lab in np.unique(labels):
            idx = np.flatnonzero(labels == lab)
            rng = np.random.default_rng([seed, int(lab)])
            out[idx] = getattr(self, f"_sample_{self.kind}")(self.params[int(lab)], len(idx), rng)
        return out

    def save(self, path: str | Path) -> None:
        flat = {f"{lab}/{k}": v for lab, p in self.params.items() for k, v in p.items()}
        np.savez_compressed(path, __kind__=np.array(self.kind), **flat)

    @classmethod
    def load(cls, path: str | Path) -> "PerLabelBaseline":
        z = np.load(path, allow_pickle=False)
        obj = cls(str(z["__kind__"]))
        for key in z.files:
            if key == "__kind__":
                continue
            lab, name = key.split("/", 1)
            obj.params.setdefault(int(lab), {})[name] = z[key]
        return obj

    # ------------------------------------------------------------- gaussian
    def _fit_gaussian(self, x, xv, lab):
        return {"dim": np.array(x.shape[1]), "mu": x.mean(0), "cov": np.cov(x, rowvar=False)}

    def _sample_gaussian(self, p, n, rng):
        return rng.multivariate_normal(p["mu"], p["cov"], size=n, method="eigh")

    # --------------------------------------------------------------- copula
    def _fit_copula(self, x, xv, lab):
        n = len(x)
        ranks = np.argsort(np.argsort(x, axis=0, kind="stable"), axis=0, kind="stable") + 1
        z = ndtri(ranks / (n + 1.0))
        return {"dim": np.array(x.shape[1]), "sorted": np.sort(x, axis=0), "corr": np.corrcoef(z, rowvar=False)}

    def _sample_copula(self, p, n, rng):
        xs, corr = p["sorted"], p["corr"]
        u = ndtr(rng.multivariate_normal(np.zeros(len(corr)), corr, size=n, method="eigh"))
        q = np.linspace(0.0, 1.0, len(xs))
        return np.stack([np.interp(u[:, j], q, xs[:, j]) for j in range(xs.shape[1])], 1)

    # ------------------------------------------------------------------ gmm
    def _fit_gmm(self, x, xv, lab):
        best, best_ll, tried = None, -np.inf, {}
        for k in self.gmm_components:
            if k * (x.shape[1] + 1) >= len(x):
                continue
            g = GaussianMixture(k, covariance_type="full", reg_covar=1e-6, random_state=self.seed, max_iter=300)
            g.fit(x)
            ll = g.score(xv) if xv is not None and len(xv) else -g.bic(x)
            tried[k] = float(ll)
            if ll > best_ll:
                best, best_ll = g, ll
        self.info[lab] = {"components": int(best.n_components), "val_loglik": tried}
        return {"dim": np.array(x.shape[1]), "weights": best.weights_, "means": best.means_,
                "covs": best.covariances_}

    def _sample_gmm(self, p, n, rng):
        comp = rng.choice(len(p["weights"]), size=n, p=p["weights"] / p["weights"].sum())
        out = np.empty((n, p["means"].shape[1]))
        for c in np.unique(comp):
            idx = np.flatnonzero(comp == c)
            out[idx] = rng.multivariate_normal(p["means"][c], p["covs"][c], size=len(idx), method="eigh")
        return out

    # ------------------------------------------------------------- resample
    def _fit_resample(self, x, xv, lab):
        return {"dim": np.array(x.shape[1]), "data": x.astype(np.float32)}

    def _sample_resample(self, p, n, rng):
        # Without replacement whenever possible: duplicated days would bias U-statistics.
        return p["data"][rng.choice(len(p["data"]), size=n, replace=n > len(p["data"]))]


def fit_baseline(kind: str, cfg: dict, out: str | Path, seed: int = 0, splits=None) -> Path:
    splits = splits or load_profiles(ProfileConfig.from_dict(cfg.get("data")))
    model = PerLabelBaseline(kind, seed=seed).fit(splits["train"], splits["val"])
    run_dir = Path(out) / f"baseline_{kind}" / f"seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    model.save(run_dir / "model.npz")
    write_json(run_dir / "meta.json", {"kind": kind, "seed": seed, "config": cfg, "info": model.info})
    return run_dir


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", required=True, choices=KINDS)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    run_dir = fit_baseline(args.kind, load_config(args.config), args.out, args.seed)
    print(f"fitted {args.kind} baseline -> {run_dir}")


if __name__ == "__main__":
    main()
