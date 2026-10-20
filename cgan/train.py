"""Train a conditional GAN on daily load profiles.

    python -m cgan.train --config configs/cgan_eb.yaml --seed 0
    python -m cgan.train --config configs/cwgan_gp.yaml --seed 0 --set train.iterations=20000

A run directory ``runs/<name>/seed<k>/`` receives the resolved config,
environment info, loss history, validation curve and two checkpoints:
``last.pt`` (final iterate) and ``best.pt`` (lowest validation score).
"""
from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np
import torch

from .data.profiles import N_LABELS, ProfileConfig, load_profiles
from .data.scaling import Scaler
from .eval.fast import ValScorer
from .models.losses import d_loss, g_loss, gradient_penalty
from .models.nets import build_nets
from .utils import environment_info, git_revision, load_config, seed_everything, write_json


def prepare_data(cfg: dict):
    splits = load_profiles(ProfileConfig.from_dict(cfg.get("data")))
    sc_cfg = cfg.get("scaler", {"kind": "minmax"})
    scaler = Scaler(sc_cfg.get("kind", "minmax")).fit(splits["train"].X, sc_cfg.get("asinh_c"))
    return splits, scaler


def _batches(X, Y, batch_size, shuffle, drop_last, gen):
    n = X.shape[0]
    stop = n - batch_size + 1 if drop_last else n
    while True:
        order = torch.randperm(n, device=X.device, generator=gen) if shuffle else torch.arange(n, device=X.device)
        for s in range(0, stop, batch_size):
            idx = order[s:s + batch_size]
            yield X[idx], Y[idx]


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float):
        import copy

        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        src = list(model.parameters())
        dst = list(self.model.parameters())
        torch._foreach_lerp_(dst, src, 1.0 - self.decay)
        for b_dst, b_src in zip(self.model.buffers(), model.buffers()):
            b_dst.copy_(b_src)


def train(cfg: dict, seed: int, run_dir: Path, device: str | None = None, quiet: bool = False) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(seed, cfg.get("deterministic", True))
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tcfg = cfg["train"]

    splits, scaler = prepare_data(cfg)
    train_split = splits["train"]
    data_dim = train_split.X.shape[1]
    Xs = scaler.transform(train_split.X).astype(np.float32)
    if cfg["model"].get("out_bound") is not None:
        # Generator output range = training range (scaled space) widened by out_bound x range.
        lo, hi = float(Xs.min()), float(Xs.max())
        margin = float(cfg["model"]["out_bound"]) * (hi - lo)
        cfg["model"]["out_range"] = [lo - margin, hi + margin]
    X = torch.from_numpy(Xs).to(device)
    Y = torch.from_numpy(train_split.label).to(device)

    G, D = build_nets(cfg["model"], data_dim)
    G.to(device)
    D.to(device)
    cap = device.startswith("cuda")
    opt_g = torch.optim.Adam(G.parameters(), lr=tcfg["lr_g"], betas=tuple(tcfg.get("betas_g", (0.9, 0.999))),
                             capturable=cap)
    opt_d = torch.optim.Adam(D.parameters(), lr=tcfg["lr_d"], betas=tuple(tcfg.get("betas_d", (0.9, 0.999))),
                             capturable=cap)
    ema = EMA(G, tcfg["ema"]) if tcfg.get("ema", 0) else None

    loss_kind = tcfg["loss"]
    n_critic = int(tcfg.get("n_critic", 1))
    gp_lambda = float(tcfg.get("gp_lambda", 0.0))
    batch_size = int(tcfg["batch_size"])
    steps_per_epoch = math.ceil(len(X) / batch_size) if not tcfg.get("drop_last", False) else len(X) // batch_size
    if "iterations" in tcfg:
        total = int(tcfg["iterations"])
    else:
        total = int(tcfg["epochs"]) * steps_per_epoch // n_critic
    g_labels = tcfg.get("g_labels", "batch")
    noise_dim = G.noise_dim

    if n_critic > 1 and not tcfg.get("drop_last", False):
        raise ValueError("n_critic > 1 requires train.drop_last: true")
    data_gen = torch.Generator(device=device).manual_seed(seed)
    batches = _batches(X, Y, batch_size, tcfg.get("shuffle", True), tcfg.get("drop_last", False), data_gen)

    scorer = ValScorer(splits["val"], scaler, n_max=cfg.get("val", {}).get("n_per_label", 1000),
                       seed=cfg.get("val", {}).get("seed", 12345), device=device)
    eval_every = int(tcfg.get("eval_every", max(total // 50, 1)))
    log_every = int(tcfg.get("log_every", max(eval_every // 10, 1)))

    meta = {
        "config": cfg, "seed": seed, "git": git_revision(Path(__file__).resolve().parents[1]),
        "env": environment_info(), "n_train": len(X), "n_val": len(splits["val"]), "n_test": len(splits["test"]),
        "data_dim": data_dim, "steps_per_epoch": steps_per_epoch, "iterations": total,
        "scaler": scaler.to_dict(),
        "params_g": sum(p.numel() for p in G.parameters()), "params_d": sum(p.numel() for p in D.parameters()),
    }
    write_json(run_dir / "meta.json", meta)

    hist_f = open(run_dir / "history.csv", "w", newline="")
    hist = csv.writer(hist_f)
    hist.writerow(["iteration", "epoch", "loss_d", "loss_g", "d_real", "d_fake", "gp", "seconds"])
    val_f = open(run_dir / "val.csv", "w", newline="")
    valw = csv.writer(val_f)
    valw.writerow(["iteration", "val_rfd", "val_rfd_ema", "seconds"])

    def ckpt(it, score):
        return {
            "g": G.state_dict(), "g_ema": ema.model.state_dict() if ema else None, "d": D.state_dict(),
            "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(), "iteration": it, "val_score": score,
            "scaler": scaler.to_dict(), "config": cfg, "seed": seed, "data_dim": data_dim,
        }

    acc = torch.zeros(5, device=device)

    def step(reals: torch.Tensor, ys: torch.Tensor) -> None:
        """One generator iteration: n_critic D updates, then one G update.

        Noise comes from the default CUDA generator so that the step can be
        captured in a CUDA graph (graph-safe RNG).  Gradients are zeroed in
        place so the captured graph always writes to the same buffers.
        """
        for k in range(n_critic):
            real, y = reals[k], ys[k]
            z = torch.randn(real.shape[0], noise_dim, device=device)
            with torch.no_grad():
                fake = G(z, y)
            opt_d.zero_grad(set_to_none=False)
            real_logit = D(real, y)
            fake_logit = D(fake, y)
            loss_d = d_loss(loss_kind, real_logit, fake_logit)
            gp = torch.zeros((), device=device)
            if gp_lambda > 0:
                gp = gradient_penalty(D, real, fake, y)
                loss_d = loss_d + gp_lambda * gp
            loss_d.backward()
            opt_d.step()
        if g_labels == "uniform":
            y = torch.randint(0, N_LABELS, (ys.shape[1],), device=device)
        z = torch.randn(y.shape[0], noise_dim, device=device)
        opt_g.zero_grad(set_to_none=False)
        loss_g = g_loss(loss_kind, D(G(z, y), y))
        loss_g.backward()
        opt_g.step()
        if ema:
            ema.update(G)
        acc.add_(torch.stack([loss_d.detach(), loss_g.detach(), real_logit.detach().mean(),
                              fake_logit.detach().mean(), gp.detach()]))

    use_graph = bool(tcfg.get("cuda_graph", True)) and device.startswith("cuda")
    s_real = torch.zeros(n_critic, batch_size, data_dim, device=device)
    s_y = torch.zeros(n_critic, batch_size, dtype=torch.long, device=device)
    graph = None
    warmup = 3

    def next_group():
        pairs = [next(batches) for _ in range(n_critic)]
        return torch.stack([p[0] for p in pairs]), torch.stack([p[1] for p in pairs])

    n_acc = 0
    best = best_raw = float("inf")
    vals = [float("nan")] * 5
    t0 = time.time()
    D.train()
    G.train()
    side = torch.cuda.Stream() if use_graph else None
    for it in range(1, total + 1):
        if n_critic == 1:
            real, y = next(batches)
            reals, ys = real[None], y[None]
        else:
            reals, ys = next_group()
        full = reals.shape[1] == batch_size
        if use_graph and full and graph is None and it > warmup:
            # Capture (does not execute); this iteration's batch is then replayed below.
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                step(s_real, s_y)
            acc.zero_()  # capture records but does not run; keep accumulator clean
            n_acc = 0
        if graph is not None and full:
            s_real.copy_(reals)
            s_y.copy_(ys)
            graph.replay()
        elif use_graph and graph is None:
            # Warm-up iterations run eagerly on a side stream (required before capture).
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                step(reals, ys)
            torch.cuda.current_stream().wait_stream(side)
        else:
            step(reals, ys)  # eager path (CPU, or a short last batch of an epoch)
        n_acc += 1
        if it % log_every == 0 or it == total:
            vals = (acc / n_acc).tolist()
            hist.writerow([it, round(it * n_critic / steps_per_epoch, 3), *[f"{v:.6g}" for v in vals],
                           round(time.time() - t0, 1)])
            hist_f.flush()
            acc.zero_()
            n_acc = 0
        if it % eval_every == 0 or it == total:
            s_raw = scorer.score(G)
            s_ema = scorer.score(ema.model) if ema else float("nan")
            score = s_ema if ema else s_raw
            valw.writerow([it, f"{s_raw:.6g}", f"{s_ema:.6g}", round(time.time() - t0, 1)])
            val_f.flush()
            if score < best:
                best = score
                torch.save(ckpt(it, score), run_dir / "best.pt")
            if tcfg.get("save_every") and it % int(tcfg["save_every"]) == 0:
                torch.save(ckpt(it, score), run_dir / f"it{it}.pt")
            if ema and s_raw < best_raw:
                # Best raw (non-EMA) generator, for the EMA ablation without retraining.
                best_raw = s_raw
                torch.save(ckpt(it, s_raw), run_dir / "best_raw.pt")
            if not quiet:
                print(f"[{cfg.get('name')} s{seed}] it {it}/{total} loss_d {vals[0]:.4f} loss_g {vals[1]:.4f} "
                      f"val {score:.4f} (best {best:.4f}) {time.time() - t0:.0f}s", flush=True)
            G.train()
    torch.save(ckpt(total, score), run_dir / "last.pt")
    hist_f.close()
    val_f.close()
    meta.update(best_val=best, seconds=round(time.time() - t0, 1))
    write_json(run_dir / "meta.json", meta)
    return meta


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--name", default=None, help="run name (defaults to config name)")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. train.iterations=1000")
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)
    name = args.name or cfg.get("name", Path(args.config).stem)
    cfg["name"] = name
    train(cfg, args.seed, Path(args.out) / name / f"seed{args.seed}", args.device)


if __name__ == "__main__":
    main()
