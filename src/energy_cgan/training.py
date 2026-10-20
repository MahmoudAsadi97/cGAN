"""Reproducible, validation-selected training for daily interval-energy profiles.

Only training rows fit the scaler or update parameters.  Validation rows select
the best checkpoint; test rows never enter scaling, optimization, or selection.
Checkpoints include optimizer states for inspection, but this module does not
implement exact resumption (DataLoader/RNG state is intentionally not claimed).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
from typing import Any

import numpy as np
import scipy
from scipy.stats import wasserstein_distance
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .models import Discriminator, Generator


DEFAULT_CONFIG = {
    "epochs": 100, "batch_size": 64, "seed": 42, "latent_dim": 24,
    "hidden_dim": 64, "lr_g": 1e-4, "lr_d": 2e-4, "device": "cpu",
    "output_activation": "linear", "validation_every": 1, "patience": 20,
    "max_validation_samples": 2048, "cpu_threads": 1,
}


def _config(config: dict[str, Any]) -> dict[str, Any]:
    supplied = dict(config)
    lr = supplied.pop("lr", None)
    if lr is not None:
        if isinstance(lr, dict):
            if set(lr) != {"generator", "discriminator"}:
                raise ValueError("lr mapping requires generator and discriminator entries")
            supplied.setdefault("lr_g", lr["generator"])
            supplied.setdefault("lr_d", lr["discriminator"])
        else:
            supplied.setdefault("lr_g", lr)
            supplied.setdefault("lr_d", 2 * lr)
    unknown = set(supplied) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"unknown training config keys: {sorted(unknown)}")
    normalized = {**DEFAULT_CONFIG, **supplied}
    for key in ("epochs", "batch_size", "latent_dim", "hidden_dim", "validation_every", "cpu_threads"):
        value = normalized[key]
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
        normalized[key] = int(value)
    for key in ("seed", "patience"):
        value = normalized[key]
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
        normalized[key] = int(value)
    if normalized["seed"] > 2 ** 64 - 5:
        raise ValueError("seed must be at most 2**64 - 5 to allow independent RNG streams")
    cap = normalized["max_validation_samples"]
    if cap is not None and (isinstance(cap, bool) or not isinstance(cap, (int, np.integer)) or cap < 1):
        raise ValueError("max_validation_samples must be a positive integer or null")
    normalized["max_validation_samples"] = None if cap is None else int(cap)
    for key in ("lr_g", "lr_d"):
        value = normalized[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be a finite positive number")
        normalized[key] = float(value)
    if normalized["output_activation"] not in {"linear", "softplus"}:
        raise ValueError("output_activation must be linear or softplus")
    if normalized["device"] == "auto":
        normalized["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(normalized["device"])
    if device.type not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be cpu, cuda, or mps")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device requested, but CUDA is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS device requested, but MPS is unavailable")
    normalized["device"] = str(device)
    return normalized


def _read_bundle(data_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(data_dir / "dataset.npz", allow_pickle=False) as bundle:
        required = {"x", "y", "split", "meter_id", "date"}
        if not required.issubset(bundle.files):
            raise ValueError(f"dataset.npz is missing keys: {sorted(required - set(bundle.files))}")
        arrays = {key: bundle[key] for key in required}
    metadata = json.loads((data_dir / "metadata.json").read_text())
    classes = metadata.get("classes")
    if not isinstance(classes, list) or not classes or not all(isinstance(item, str) and item for item in classes) or len(set(classes)) != len(classes):
        raise ValueError("metadata classes must be a nonempty list of unique strings")
    if metadata.get("unit") != "kWh":
        raise ValueError("dataset unit must be interval energy in kWh")
    if metadata.get("interval_hours") not in (0.25, 1.0):
        raise ValueError("interval_hours must be 0.25 or 1.0")
    x, y = arrays["x"], arrays["y"]
    if x.ndim != 2 or not x.shape[0] or not x.shape[1] or x.dtype.kind not in {"i", "u", "f"}:
        raise ValueError("x must be a nonempty real numeric (N, T) array")
    if x.shape[1] * metadata["interval_hours"] != 24:
        raise ValueError("each profile must contain exactly 24 hours of intervals")
    if not np.isfinite(x).all():
        raise ValueError("all profile values must be finite")
    if any(value.ndim != 1 or len(value) != len(x) for key, value in arrays.items() if key != "x"):
        raise ValueError("labels, splits, meter IDs, and dates must be one-dimensional with N rows")
    if not np.issubdtype(y.dtype, np.integer) or np.any(y < 0) or np.any(y >= len(classes)):
        raise ValueError("labels must be integers indexing metadata classes")
    for field in ("split", "meter_id", "date"):
        if arrays[field].dtype.kind not in {"U", "S"}:
            raise ValueError(f"{field} must contain strings")
        arrays[field] = arrays[field].astype(str)
        if np.any(np.char.str_len(arrays[field]) == 0):
            raise ValueError(f"{field} must not contain empty strings")
    if set(arrays["split"]) - {"train", "val", "test"}:
        raise ValueError("split values must be train, val, or test")
    splits = {name: arrays["split"] == name for name in ("train", "val", "test")}
    if not splits["train"].any() or not splits["val"].any():
        raise ValueError("nonempty train and val splits are required for model selection")
    meter_sets = {name: set(arrays["meter_id"][mask]) for name, mask in splits.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if meter_sets[a] & meter_sets[b]:
            raise ValueError(f"meter leakage between {a} and {b} splits")
    trained = set(y[splits["train"]].tolist())
    if trained != set(range(len(classes))):
        raise ValueError("every metadata label must have at least one training example")
    arrays["x"] = x.astype(np.float32, copy=False)
    if not np.isfinite(arrays["x"]).all():
        raise ValueError("profile values must remain finite in float32")
    arrays["y"] = y.astype(np.int64, copy=False)
    return arrays, metadata


def _fit_scaler(training: np.ndarray, activation: str) -> dict[str, Any]:
    minimum, maximum = float(training.min()), float(training.max())
    if activation == "softplus":
        if minimum < 0:
            raise ValueError("softplus requires nonnegative training energy values")
        minimum = 0.0
    scale = maximum - minimum
    return {"kind": "zero_fixed_max" if activation == "softplus" else "global_minmax",
            "min": minimum, "max": maximum, "scale": scale if scale > 0 else 1.0,
            "fitted_split": "train", "unit": "kWh"}


def _inverse(values: np.ndarray, scaler: dict[str, Any]) -> np.ndarray:
    result = (values * scaler["scale"] + scaler["min"]).astype(np.float32)
    if not np.isfinite(result).all():
        raise FloatingPointError("generated physical-unit profiles are nonfinite")
    return result


def _validation_indices(labels: np.ndarray, cap: int | None, seed: int) -> np.ndarray:
    if cap is None or len(labels) <= cap:
        return np.arange(len(labels))
    classes = np.unique(labels)
    if cap < len(classes):
        raise ValueError("max_validation_samples must cover every validation label")
    rng = np.random.default_rng(seed)
    groups = [rng.permutation(np.flatnonzero(labels == value)).tolist() for value in classes]
    # Round-robin selection prevents a frequent class from hiding a rare class.
    selected = []
    while len(selected) < cap:
        for group in groups:
            if group and len(selected) < cap:
                selected.append(group.pop())
    return np.array(sorted(selected), dtype=np.int64)


def _validation_score(real: np.ndarray, fake: np.ndarray, labels: np.ndarray) -> float:
    per_class = [float(np.mean([
        wasserstein_distance(real[labels == label, slot], fake[labels == label, slot])
        for slot in range(real.shape[1])
    ])) for label in np.unique(labels)]
    score = float(np.mean(per_class))
    if not math.isfinite(score):
        raise FloatingPointError("validation metric is nonfinite")
    return score


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest(data_dir: Path, arrays: dict[str, np.ndarray], selected: np.ndarray) -> dict[str, Any]:
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2],
            stderr=subprocess.DEVNULL, text=True, timeout=5,
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=Path(__file__).resolve().parents[2],
            stderr=subprocess.DEVNULL, text=True, timeout=5,
        ).strip())
    except (OSError, subprocess.SubprocessError):
        revision, dirty = None, None
    return {
        "dataset_sha256": _sha256(data_dir / "dataset.npz"),
        "metadata_sha256": _sha256(data_dir / "metadata.json"),
        "git_revision": revision, "git_dirty": dirty,
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(),
                        "numpy": str(np.__version__), "scipy": str(scipy.__version__),
                        "torch": str(torch.__version__), "cuda": torch.version.cuda},
        "split_counts": {name: int(np.sum(arrays["split"] == name)) for name in ("train", "val", "test")},
        "validation_indices_within_split": selected.tolist(),
        "validation_labels": np.unique(arrays["y"][arrays["split"] == "val"][selected]).tolist(),
        "test_used_for_training_or_selection": False,
        "exact_resume_supported": False,
    }


def _finite_update(loss: torch.Tensor, model: nn.Module, optimizer: torch.optim.Optimizer, name: str) -> None:
    if not torch.isfinite(loss).all():
        raise FloatingPointError(f"{name} loss became nonfinite")
    loss.backward()
    for parameter in model.parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise FloatingPointError(f"{name} gradients became nonfinite")
    optimizer.step()
    if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise FloatingPointError(f"{name} parameters became nonfinite")


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _write_json(payload: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def train(data_dir: str | Path, output_dir: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    """Train a cGAN and return paths and validation-selection results.

    ``patience`` counts validation checks without improvement; zero disables
    early stopping. The final epoch is validated even between scheduled checks.
    ``lr`` accepts a G learning rate (D is twice it), or a mapping with generator
    and discriminator keys. Explicit ``lr_g`` and ``lr_d`` take precedence.
    """
    config = _config(config)
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError("output_dir is nonempty; choose a new directory")
    arrays, metadata = _read_bundle(data_dir)
    if config["output_activation"] == "softplus" and metadata.get("target") == "net":
        raise ValueError("softplus is incompatible with a signed net energy target")
    training_mask, val_mask = arrays["split"] == "train", arrays["split"] == "val"
    scaler = _fit_scaler(arrays["x"][training_mask], config["output_activation"])
    train_x = ((arrays["x"][training_mask] - scaler["min"]) / scaler["scale"]).astype(np.float32)
    if not np.isfinite(train_x).all():
        raise FloatingPointError("normalized training profiles are nonfinite")
    train_y = arrays["y"][training_mask]
    val_x, val_y = arrays["x"][val_mask], arrays["y"][val_mask]
    selected = _validation_indices(val_y, config["max_validation_samples"], config["seed"] + 1)
    val_x, val_y = val_x[selected], val_y[selected]
    manifest = _manifest(data_dir, arrays, selected)
    missing_validation_labels = sorted(set(range(len(metadata["classes"]))) - set(val_y.tolist()))
    manifest["missing_validation_labels"] = missing_validation_labels
    manifest["validation_warnings"] = (
        ["Checkpoint selection excludes classes absent from validation: " +
         ", ".join(metadata["classes"][label] for label in missing_validation_labels)]
        if missing_validation_labels else []
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(manifest, output_dir / "manifest.json")
    _write_json(config, output_dir / "config.json")
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(config["cpu_threads"])
        return _run_training(train_x, train_y, val_x, val_y, metadata, scaler, manifest, output_dir, config)
    finally:
        torch.set_num_threads(previous_threads)


def _run_training(train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray,
                  metadata: dict[str, Any], scaler: dict[str, Any], manifest: dict[str, Any],
                  output_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    seed = config["seed"]
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(config["device"])
    model_args = {"profile_length": train_x.shape[1], "num_classes": len(metadata["classes"]),
                  "hidden_dim": config["hidden_dim"]}
    generator = Generator(**model_args, latent_dim=config["latent_dim"],
                          output_activation=config["output_activation"]).to(device)
    discriminator = Discriminator(**model_args).to(device)
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=config["lr_g"], betas=(0.9, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=config["lr_d"], betas=(0.9, 0.999))
    shuffle_rng = torch.Generator().manual_seed(seed + 2)
    noise_rng = torch.Generator().manual_seed(seed + 3)
    validation_rng = torch.Generator().manual_seed(seed + 4)
    loader = DataLoader(TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
                        batch_size=config["batch_size"], shuffle=True, generator=shuffle_rng,
                        num_workers=0, drop_last=False)
    val_noise = torch.randn(len(val_y), config["latent_dim"], generator=validation_rng)
    val_labels = torch.from_numpy(val_y)
    criterion = nn.BCEWithLogitsLoss()
    history: list[dict[str, Any]] = []
    best_score, best_epoch, no_improvement = math.inf, None, 0
    for epoch in range(1, config["epochs"] + 1):
        generator.train()
        discriminator.train()
        d_total, g_total = 0.0, 0.0
        for real, labels in loader:
            real, labels = real.to(device), labels.to(device)
            noise = torch.randn(len(real), config["latent_dim"], generator=noise_rng).to(device)
            with torch.no_grad():
                fake = generator(noise, labels)
            optimizer_d.zero_grad(set_to_none=True)
            real_logits = discriminator(real, labels)
            fake_logits = discriminator(fake.detach(), labels)
            loss_d = 0.5 * (criterion(real_logits, torch.ones_like(real_logits)) +
                            criterion(fake_logits, torch.zeros_like(fake_logits)))
            _finite_update(loss_d, discriminator, optimizer_d, "discriminator")
            optimizer_g.zero_grad(set_to_none=True)
            discriminator.requires_grad_(False)
            try:
                generator_noise = torch.randn(len(real), config["latent_dim"], generator=noise_rng).to(device)
                fake = generator(generator_noise, labels)
                generated_logits = discriminator(fake, labels)
                loss_g = criterion(generated_logits, torch.ones_like(generated_logits))
                _finite_update(loss_g, generator, optimizer_g, "generator")
            finally:
                discriminator.requires_grad_(True)
            d_total += float(loss_d.detach().cpu()) * len(real)
            g_total += float(loss_g.detach().cpu()) * len(real)
        validate_now = epoch % config["validation_every"] == 0 or epoch == config["epochs"]
        score = None
        improved = False
        if validate_now:
            generator.eval()
            predictions = []
            with torch.no_grad():
                for offset in range(0, len(val_y), config["batch_size"]):
                    predictions.append(generator(
                        val_noise[offset:offset + config["batch_size"]].to(device),
                        val_labels[offset:offset + config["batch_size"]].to(device),
                    ).cpu().numpy())
            generated = _inverse(np.concatenate(predictions), scaler)
            score = _validation_score(val_x, generated, val_y)
            improved = score < best_score
            if improved:
                best_score, best_epoch, no_improvement = score, epoch, 0
            else:
                no_improvement += 1
        history.append({"epoch": epoch, "generator_loss": g_total / len(train_x),
                        "discriminator_loss": d_total / len(train_x),
                        "validation_wasserstein_kwh": score})
        checkpoint = {
            "format_version": 1, "epoch": epoch, "config": config,
            "model_args": model_args, "metadata": metadata, "scaler": scaler,
            "manifest": manifest, "generator_state_dict": generator.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            "optimizer_g": optimizer_g.state_dict(), "optimizer_d": optimizer_d.state_dict(),
            "validation_metric": score, "best_validation_metric": best_score if best_epoch else None,
            "best_epoch": best_epoch,
            "selection_metric": "mean_per_condition_per_slot_wasserstein_kwh",
            "history": history,
        }
        if improved:
            _atomic_torch_save(checkpoint, output_dir / "best.pt")
        _atomic_torch_save(checkpoint, output_dir / "last.pt")
        _write_json(history, output_dir / "history.json")
        if validate_now and config["patience"] and no_improvement >= config["patience"]:
            break
    return {"best_checkpoint": str((output_dir / "best.pt").resolve()),
            "last_checkpoint": str((output_dir / "last.pt").resolve()),
            "history_path": str((output_dir / "history.json").resolve()),
            "best_epoch": best_epoch, "best_validation_wasserstein_kwh": best_score,
            "epochs_completed": len(history), "validation_samples": len(val_y),
            "selection_metric": "mean_per_condition_per_slot_wasserstein_kwh"}


def validate_checkpoint_data(checkpoint_path: str | Path, data_dir: str | Path) -> dict[str, Any]:
    """Reject evaluation bundles differing from the checkpoint's source bundle.

    Both files are matched byte-for-byte by SHA-256. Metadata is also compared
    with the embedded class order, units, interval resolution, and target so a
    checkpoint cannot be paired with another label dictionary accidentally.
    This verifies provenance consistency, not authenticity of untrusted files.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("unsupported checkpoint format")
    data_dir = Path(data_dir)
    manifest = checkpoint.get("manifest", {})
    hashes = {}
    for name, filename in (("dataset", "dataset.npz"), ("metadata", "metadata.json")):
        key = name + "_sha256"
        actual = _sha256(data_dir / filename)
        if manifest.get(key) != actual:
            raise ValueError(f"{name} SHA-256 differs from the checkpoint training bundle")
        hashes[key] = actual
    metadata = json.loads((data_dir / "metadata.json").read_text())
    embedded = checkpoint["metadata"]
    for field in ("classes", "unit", "interval_hours", "target"):
        if metadata.get(field) != embedded.get(field):
            raise ValueError(f"metadata {field} differs from the checkpoint")
    if len(metadata["classes"]) != checkpoint["model_args"]["num_classes"]:
        raise ValueError("metadata classes differ from checkpoint model dimensions")
    if checkpoint["model_args"]["profile_length"] * metadata["interval_hours"] != 24:
        raise ValueError("checkpoint profile dimensions must cover 24 hours")
    return {"verified": True, **hashes}


def generate(checkpoint_path: str | Path, labels: np.ndarray, seed: int = 42,
             batch_size: int = 1024) -> np.ndarray:
    """Generate interval kWh on CPU from integer condition labels.

    Repeated calls with the same checkpoint, seed, and batch size are
    deterministic within the same numerical environment. Linear-output models
    may produce negative energy; values are never silently clipped.
    """
    labels = np.asarray(labels)
    if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("labels must be a one-dimensional integer array")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if seed > 2 ** 64 - 1:
        raise ValueError("seed must be at most 2**64 - 1")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("unsupported checkpoint format")
    model_args, config = checkpoint["model_args"], checkpoint["config"]
    if labels.size and (labels.min() < 0 or labels.max() >= model_args["num_classes"]):
        raise ValueError("labels are outside the trained class range")
    generator = Generator(**model_args, latent_dim=config["latent_dim"],
                          output_activation=config["output_activation"])
    generator.load_state_dict(checkpoint["generator_state_dict"])
    generator.eval()
    if len(labels) == 0:
        return np.empty((0, model_args["profile_length"]), dtype=np.float32)
    noise_rng = torch.Generator().manual_seed(int(seed))
    # Draw once so batch partitioning does not change the chosen noise vectors.
    noise = torch.randn(len(labels), config["latent_dim"], generator=noise_rng)
    conditions = torch.from_numpy(labels.astype(np.int64, copy=False))
    result = []
    with torch.no_grad():
        for offset in range(0, len(labels), batch_size):
            result.append(generator(noise[offset:offset + batch_size],
                                    conditions[offset:offset + batch_size]).numpy())
    return _inverse(np.concatenate(result), checkpoint["scaler"])
