"""Reference GAN optimization, fixed validation selection and resumable runs.

Checkpoints are local run artifacts containing only tensors and primitive data;
all loads use PyTorch's weights-only reader. Test profiles are never accessed.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time

import numpy as np
import torch
from torch import nn

from .data import load_split
from .models import Discriminator, Generator


DEFAULT_CONFIG = {
    "epochs": 10_000, "batch_size": 2000, "latent_dim": 24,
    "seed": 11, "num_threads": 1, "unconditional": False, "device": "cpu",
    "validation_per_condition": 1024, "validation_every": 5,
    "max_train_samples": None,
}
SCHEMA_VERSION = 1


def _configuration(config):
    if not isinstance(config, dict) or set(config) - set(DEFAULT_CONFIG):
        raise ValueError(f"Unknown config parameters: {set(config) - set(DEFAULT_CONFIG) if isinstance(config, dict) else config}")
    result = {**DEFAULT_CONFIG, **config}
    for key in ("epochs", "batch_size", "latent_dim", "num_threads", "validation_per_condition", "validation_every"):
        if type(result[key]) is not int or result[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if result["latent_dim"] != 24 or result["validation_per_condition"] > 1024:
        raise ValueError("Reference latent_dim is 24; validation_per_condition must be <=1024")
    if type(result["seed"]) is not int or not 0 <= result["seed"] < 2**32:
        raise ValueError("seed must be an integer in [0,2**32)")
    if type(result["unconditional"]) is not bool:
        raise ValueError("unconditional must be boolean")
    limit = result["max_train_samples"]
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("max_train_samples must be a positive integer for smoke runs only")
    if result["device"] not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if result["device"] == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    return result


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _read_data(directory):
    names = ("train.npz", "val.npz", "scaler.json", "metadata.json")
    hashes = {name: _sha256(directory / name) for name in names}
    training, validation = load_split(directory / "train.npz"), load_split(directory / "val.npz")
    scaler = json.loads((directory / "scaler.json").read_text())
    metadata = json.loads((directory / "metadata.json").read_text())
    if training["x"].shape[1] != 24 or validation["x"].shape[1] != 24:
        raise ValueError("Reference training requires physical hourly [N,24] profiles")
    if metadata.get("resolution", 24) != 24 or metadata.get("condition_dim", 64) != 64:
        raise ValueError("Prepared metadata must describe 24 slots and 64 conditions")
    for split in ("train", "val"):
        declared = metadata.get("splits", {}).get(split, {}).get("sha256")
        if declared is not None and declared != hashes[f"{split}.npz"]:
            raise ValueError(f"Prepared {split} hash disagrees with metadata")
    minimum, maximum = float(scaler["minimum"]), float(scaler["maximum"])
    if (not math.isfinite(minimum) or not math.isfinite(maximum) or minimum > maximum
            or scaler.get("fitted_on", "train") != "train"
            or minimum != float(training["x"].min()) or maximum != float(training["x"].max())):
        raise ValueError("scaler must be fitted on the complete train split only")
    train_conditions, val_conditions = np.unique(training["y"]), np.unique(validation["y"])
    if not np.array_equal(train_conditions, val_conditions):
        raise ValueError("Every training condition requires validation support and vice versa")
    # Leakage between the two consumed splits is detectable without touching test.
    if set(training["meter_id"]) & set(validation["meter_id"]):
        raise ValueError("Train/validation meter support must be disjoint")
    if hashes != {name: _sha256(directory / name) for name in names}:
        raise ValueError("Prepared data changed while loading")
    return training, validation, scaler, metadata, hashes


def _selection(training, validation, config):
    from .evaluation import projection_directions
    seed = 2027
    rng = np.random.default_rng(seed)
    conditions = np.unique(training["y"]).tolist()
    indices = {}
    for condition in conditions:
        available = np.flatnonzero(validation["y"] == condition)
        indices[str(condition)] = rng.choice(available, min(len(available), config["validation_per_condition"]), replace=False).tolist()
    return {"conditions": conditions, "validation_indices": indices,
            "index_seed": seed, "noise_seed": 2028,
            "directions_seed": 2026, "directions_count": 128,
            "directions_sha256": hashlib.sha256(projection_directions(24, count=128, seed=2026).astype('<f8').tobytes()).hexdigest(),
            "metric": "condition-macro normalized empirical sliced Wasserstein-1",
            "sampling": "fixed validation rows without replacement; fixed generated draws per condition",
            "max_per_condition": config["validation_per_condition"]}


def _validation(model, validation, scaler, selection, device):
    from .evaluation import projection_directions, sliced_wasserstein
    directions = projection_directions(24, count=128, seed=2026)
    noise = torch.Generator(device="cpu").manual_seed(selection["noise_seed"])
    scale = scaler["maximum"] - scaler["minimum"] or 1.0
    scores = []
    model.eval()
    with torch.no_grad():
        for condition in selection["conditions"]:
            indices = selection["validation_indices"][str(condition)]
            real = (validation["x"][indices] - scaler["minimum"]) / scale
            labels = torch.full((len(indices),), condition, dtype=torch.long, device=device)
            latent = torch.randn(len(indices), 24, generator=noise).to(device)
            fake = model(latent, labels).cpu().numpy()
            scores.append(float(sliced_wasserstein(real, fake, directions)))
    model.train()
    score = float(np.mean(scores))
    if not math.isfinite(score):
        raise FloatingPointError("Nonfinite validation SW1")
    return score


def train(data_dir, run_dir, config, resume=False):
    """Train all prepared rows; an optional row cap is explicitly a smoke run.

    Epochs is the total target, including completed epochs when resuming. All
    parameters other than that target, and all consumed input hashes, must match.
    CPU replay is exact within a fixed PyTorch/runtime/thread configuration.
    Returns the completed run manifest; each validation epoch prints progress.
    """
    config = _configuration(config)
    directory, output = Path(data_dir), Path(run_dir)
    if not resume and output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Refusing to overwrite nonempty run directory: {output}")
    training, validation, scaler, metadata, hashes = _read_data(directory)
    selection = _selection(training, validation, config)
    saved = None
    if resume:
        saved = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
        if saved.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Incompatible checkpoint schema")
        if saved["data_hashes"] != hashes:
            raise ValueError("Resume data hashes differ from checkpoint")
        old_config = {k: v for k, v in saved["config"].items() if k != "epochs"}
        if old_config != {k: v for k, v in config.items() if k != "epochs"}:
            raise ValueError("Resume config differs from checkpoint")
        if config["epochs"] < saved["epoch"]:
            raise ValueError("Resume epochs cannot precede completed checkpoint epoch")
        if json.loads((output / "history.json").read_text()) != saved["history"]:
            raise ValueError("Run history differs from checkpoint history")
        if selection != saved["selection"]:
            raise ValueError("Resume selection differs from checkpoint")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": SCHEMA_VERSION, "config": config,
        "nondefault_parameters": {k: v for k, v in config.items() if v != DEFAULT_CONFIG[k]},
        "data_hashes": hashes, "selection": selection,
        "source_metadata": metadata, "scaler": scaler,
        "runtime": {"python": platform.python_version(), "torch": str(torch.__version__),
                    "numpy": str(np.__version__), "device": config["device"], "num_threads": config["num_threads"]},
        "sampling": {"mode": "smoke" if config["max_train_samples"] is not None or metadata.get("sampling", {}).get("mode") == "smoke" else "full",
                     "max_train_samples": config["max_train_samples"], "available_train_rows": len(training["x"])},
        "optimizer": {"generator_lr": .0001, "discriminator_lr": .0002, "betas": [.9, .999],
                      "discriminator_loss": "mean(real_BCEWithLogits, fake_BCEWithLogits)",
                      "updates_per_batch": {"generator": 1, "discriminator": 1}},
        "status": "running",
    }
    if saved is not None and saved["runtime"] != manifest["runtime"]:
        raise ValueError("Resume runtime differs from checkpoint; exact replay is unsupported")
    if saved is not None:
        manifest.update(completed_epochs=saved["epoch"], best_epoch=saved["best_epoch"], best_validation_sw1=saved["best_score"])
    selected = np.arange(len(training["x"]))
    if config["max_train_samples"] is not None and len(selected) > config["max_train_samples"]:
        selected = np.sort(np.random.default_rng(config["seed"] + 300_003).choice(selected, config["max_train_samples"], replace=False))
        if not np.array_equal(np.unique(training["y"][selected]), np.unique(training["y"])):
            raise ValueError("Smoke subsampling removed training condition support; increase max_train_samples")
    manifest["sampling"]["used_train_rows"] = len(selected)
    manifest["sampling"]["selection_seed"] = config["seed"] + 300_003
    manifest["sampling"]["selected_index_sha256"] = hashlib.sha256(selected.astype('<i8').tobytes()).hexdigest()
    _json(output / "manifest.json", manifest)
    device = torch.device(config["device"])
    torch.set_num_threads(config["num_threads"])
    cuda_devices = [torch.cuda.current_device()] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(config["seed"])
            if cuda_devices:
                torch.cuda.manual_seed_all(config["seed"])
            generator = Generator(num_classes=0 if config["unconditional"] else 64).to(device)
            discriminator = Discriminator(num_classes=0 if config["unconditional"] else 64).to(device)
            optimizer_g = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(.9, .999))
            optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(.9, .999))
            loader_rng = torch.Generator(device="cpu").manual_seed(config["seed"] + 1)
            noise_rng = torch.Generator(device="cpu").manual_seed(config["seed"] + 2)
            history, best_score, best_epoch, start = [], math.inf, 0, 0
            if saved is not None:
                generator.load_state_dict(saved["generator"])
                discriminator.load_state_dict(saved["discriminator"])
                optimizer_g.load_state_dict(saved["optimizer_g"])
                optimizer_d.load_state_dict(saved["optimizer_d"])
                torch.set_rng_state(saved["model_rng_state"])
                if cuda_devices:
                    torch.cuda.set_rng_state_all(saved["cuda_rng_states"])
                loader_rng.set_state(saved["loader_rng_state"])
                noise_rng.set_state(saved["noise_rng_state"])
                history, best_score, best_epoch, start = saved["history"], saved["best_score"], saved["best_epoch"], saved["epoch"]
            scale = scaler["maximum"] - scaler["minimum"] or 1.0
            x = torch.from_numpy((training["x"][selected] - scaler["minimum"]) / scale).to(device)
            y = torch.from_numpy(training["y"][selected]).to(device)
            loss_fn = nn.BCEWithLogitsLoss()
            for epoch in range(start + 1, config["epochs"] + 1):
                began = time.perf_counter()
                order = torch.randperm(len(x), generator=loader_rng).to(device)
                loss_g_sum, loss_d_sum, batches = 0., 0., 0
                for indices in order.split(config["batch_size"]):
                    real, labels = x[indices], y[indices]
                    ones = torch.ones((len(real), 1), device=device)
                    optimizer_d.zero_grad(set_to_none=True)
                    latent = torch.randn(len(real), 24, generator=noise_rng).to(device)
                    fake = generator(latent, labels).detach()
                    fake_logits = discriminator(fake, labels)
                    real_logits = discriminator(real, labels)
                    loss_d = (loss_fn(real_logits, ones) + loss_fn(fake_logits, torch.zeros_like(ones))) / 2
                    if not torch.isfinite(loss_d):
                        raise FloatingPointError("Nonfinite discriminator loss")
                    loss_d.backward()
                    optimizer_d.step()
                    optimizer_g.zero_grad(set_to_none=True)
                    latent = torch.randn(len(real), 24, generator=noise_rng).to(device)
                    # D stays in training mode: retain the reference dropout
                    # draws while avoiding unused D parameter gradients.
                    discriminator.requires_grad_(False)
                    try:
                        loss_g = loss_fn(discriminator(generator(latent, labels), labels), ones)
                        if not torch.isfinite(loss_g):
                            raise FloatingPointError("Nonfinite generator loss")
                        loss_g.backward()
                        optimizer_g.step()
                    finally:
                        discriminator.requires_grad_(True)
                    loss_g_sum += loss_g.item()
                    loss_d_sum += loss_d.item()
                    batches += 1
                # Epoch one always selects a usable checkpoint. Later selection
                # follows a schedule independent of the requested total epochs.
                score = _validation(generator, validation, scaler, selection, device) if epoch == 1 or epoch % config["validation_every"] == 0 else None
                improved = score is not None and score < best_score
                if improved:
                    best_score, best_epoch = score, epoch
                history.append({"epoch": epoch, "loss_g": loss_g_sum / batches,
                                "loss_d": loss_d_sum / batches, "validation_sw1": score,
                                "seconds": time.perf_counter() - began})
                state = {
                    "schema_version": SCHEMA_VERSION, "epoch": epoch, "config": config,
                    "generator": generator.state_dict(), "discriminator": discriminator.state_dict(),
                    "optimizer_g": optimizer_g.state_dict(), "optimizer_d": optimizer_d.state_dict(),
                    "model_rng_state": torch.get_rng_state(),
                    "cuda_rng_states": torch.cuda.get_rng_state_all() if cuda_devices else [],
                    "loader_rng_state": loader_rng.get_state(), "noise_rng_state": noise_rng.get_state(),
                    "scaler": scaler, "data_hashes": hashes, "selection": selection,
                    "history": history, "best_score": best_score, "best_epoch": best_epoch,
                    "runtime": manifest["runtime"],
                }
                if improved:
                    _save(output / "best.pt", state)
                _save(output / "last.pt", state)
                _json(output / "history.json", history)
                manifest.update(completed_epochs=epoch, best_epoch=best_epoch, best_validation_sw1=best_score)
                _json(output / "manifest.json", manifest)
                if score is not None:
                    print(f"Epoch {epoch}: G={loss_g_sum / batches:.6f} D={loss_d_sum / batches:.6f} validation_SW1={score:.6f}", flush=True)
        manifest["artifact_hashes"] = {
            name: _sha256(output / name) for name in ("best.pt", "last.pt", "history.json")
        }
        manifest["status"] = "completed"
        _json(output / "manifest.json", manifest)
        return manifest
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        _json(output / "manifest.json", manifest)
        raise


def generate(checkpoint, labels, seed, batch_size=2048):
    """Generate physical hourly kWh using an isolated reporting RNG on CPU."""
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be a nonnegative integer below 2**63")
    labels = np.asarray(labels)
    if labels.ndim != 1 or labels.dtype.kind not in "iu" or np.any((labels < 0) | (labels >= 64)):
        raise ValueError("labels must be an integer vector in [0,64)")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Incompatible checkpoint schema")
    outputs = []
    with torch.random.fork_rng(devices=[]):
        model = Generator(num_classes=0 if saved["config"]["unconditional"] else 64)
        model.load_state_dict(saved["generator"])
        model.eval()
        # NumPy's local stream consumes the same normals across chunk sizes;
        # torch.randn switches algorithms for certain small remainder batches.
        noise = np.random.default_rng(seed)
        with torch.no_grad():
            for start in range(0, len(labels), batch_size):
                batch = torch.as_tensor(labels[start:start + batch_size].astype(np.int64))
                latent = torch.from_numpy(noise.standard_normal((len(batch), 24), dtype=np.float32))
                values = model(latent, batch).numpy()
                scaler = saved["scaler"]
                values = values * (scaler["maximum"] - scaler["minimum"] or 1.0) + scaler["minimum"]
                outputs.append(values)
    result = np.concatenate(outputs) if outputs else np.empty((0, 24), dtype=np.float32)
    if not np.isfinite(result).all():
        raise FloatingPointError("Generated profiles are nonfinite")
    return result
