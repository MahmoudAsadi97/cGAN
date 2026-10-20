"""Generate synthetic daily profiles (kWh) from a trained run.

    python -m cgan.sample --run runs/cgan_eb/seed0 --ckpt best --n-per-label 2000 --out samples.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .data.profiles import N_LABELS, label_names
from .data.scaling import Scaler
from .eval.fast import generate_scaled
from .models.nets import build_nets


def resolve_ckpt(spec: str) -> tuple[str, bool]:
    """Checkpoint spec -> (file stem, use EMA weights).

    ``best``      best.pt, EMA weights if the run tracked an EMA (selected on the EMA score)
    ``best_raw``  best_raw.pt, raw weights (selected on the raw score)
    ``last``      last.pt, EMA weights if available;  ``last_raw``: last.pt, raw weights
    """
    if spec == "best_raw":
        return "best_raw", False
    if spec.endswith("_raw"):
        return spec.removesuffix("_raw"), False
    return spec, True


def load_generator(run_dir: str | Path, ckpt: str = "best", use_ema: bool = True, device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    stem, ema_ok = resolve_ckpt(ckpt)
    use_ema = use_ema and ema_ok
    state = torch.load(Path(run_dir) / f"{stem}.pt", map_location=device, weights_only=True)
    G, _ = build_nets(state["config"]["model"], state["data_dim"])
    weights = state["g_ema"] if (use_ema and state.get("g_ema") is not None) else state["g"]
    G.load_state_dict(weights)
    G.to(device).eval()
    return G, Scaler.from_dict(state["scaler"]), state


def generate(G, scaler: Scaler, labels: np.ndarray, seed: int = 0, device: str | None = None) -> np.ndarray:
    """Profiles in kWh for each entry of ``labels`` (int array), reproducible given ``seed``."""
    device = device or next(G.parameters()).device
    gen = torch.Generator(device=device).manual_seed(int(seed))
    y = torch.as_tensor(np.asarray(labels, np.int64), device=device)
    x = scaler.inverse_torch(generate_scaled(G, y, gen).double())
    return x.cpu().numpy().astype(np.float32)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="best", choices=["best", "best_raw", "last", "last_raw"])
    ap.add_argument("--n-per-label", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    G, scaler, _ = load_generator(args.run, args.ckpt, not args.no_ema)
    labels = np.repeat(np.arange(N_LABELS), args.n_per_label)
    X = generate(G, scaler, labels, args.seed)
    np.savez_compressed(args.out, X=X, label=labels, label_names=np.array(label_names()))
    print(f"wrote {X.shape} to {args.out}")


if __name__ == "__main__":
    main()
