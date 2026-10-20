"""Reproducible BCE cGAN training with validation-only checkpoint selection.

Prepared profiles are in original kWh/interval.  Only train.npz and val.npz are
opened; the test split is reserved for a separate final evaluation command.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path
import platform
import re
import subprocess
import tempfile
import time

import numpy as np
from scipy.stats import wasserstein_distance
import torch
from torch import nn
from torch.nn import functional as F

from .models import Discriminator, Generator


DEFAULT_CONFIG = {
    "epochs": 100, "batch_size": 256, "seed": 42, "latent_dim": 24,
    "hidden_dim": 64, "lr_g": 1e-4, "lr_d": 2e-4, "device": "cpu",
    "num_threads": 1, "unconditional": False, "validation_samples": 2048,
    "max_train_samples": None, "patience": None, "normalization": "minmax",
}
VALIDATION_METRIC = "mean_interval_wasserstein_normalized"
CHECKPOINT_FORMAT = 1


def _positive_int(name, value, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _config(config):
    if not isinstance(config, dict):
        raise ValueError("config must be a dictionary")
    unknown = set(config) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"unknown config fields: {sorted(unknown)}")
    result = {**DEFAULT_CONFIG, **config}
    for name in ("epochs", "batch_size", "latent_dim", "hidden_dim", "num_threads", "validation_samples"):
        result[name] = _positive_int(name, result[name])
    result["seed"] = _positive_int("seed", result["seed"], 0)
    if result["seed"] >= 2**32:
        raise ValueError("seed must be smaller than 2**32")
    for name in ("max_train_samples", "patience"):
        if result[name] is not None:
            result[name] = _positive_int(name, result[name])
    for name in ("lr_g", "lr_d"):
        value = result[name]
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
        result[name] = float(value)
    if not isinstance(result["unconditional"], bool):
        raise ValueError("unconditional must be a boolean")
    if result["normalization"] not in ("minmax", "standard"):
        raise ValueError("normalization must be minmax or standard")
    if result["device"] == "auto":
        result["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    if not isinstance(result["device"], str) or not re.fullmatch(r"cpu|cuda(?::\d+)?", result["device"]):
        raise ValueError("device must be auto, cpu, cuda, or cuda:N")
    device = torch.device(result["device"])
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError("CUDA device index is unavailable")
    return result


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_snapshot(path):
    raw = Path(path).read_bytes()
    try:
        value = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON must contain an object: {path}")
    return value, hashlib.sha256(raw).hexdigest()


def _read_split(path):
    """Validate original-unit arrays and bind loaded data to an exact file hash."""
    before = _sha256(path)
    with np.load(path, allow_pickle=False) as archive:
        required = {"x", "y", "meter_id", "date"}
        if not required.issubset(archive.files):
            raise ValueError(f"{path.name}: missing NPZ arrays {sorted(required - set(archive.files))}")
        data = {key: archive[key] for key in required}
    if _sha256(path) != before:
        raise ValueError(f"{path.name} changed while loading")
    x, y, meters, dates = (data[key] for key in ("x", "y", "meter_id", "date"))
    if x.ndim != 2 or not len(x) or x.shape[1] not in (24, 96):
        raise ValueError(f"{path.name}: x must be nonempty (N, 24) or (N, 96)")
    if x.dtype.kind not in "fiu" or not np.isfinite(x).all():
        raise ValueError(f"{path.name}: x must be finite numeric values")
    if y.shape != (len(x),) or y.dtype.kind not in "iu" or np.any((y < 0) | (y >= 64)):
        raise ValueError(f"{path.name}: y must be integer labels in [0, 63]")
    for name, array in (("meter_id", meters), ("date", dates)):
        if array.shape != (len(x),) or array.dtype.kind != "U" or np.any(array == ""):
            raise ValueError(f"{path.name}: {name} must be a nonempty Unicode vector of length N")
    try:
        days = dates.astype("datetime64[D]")
    except ValueError as exc:
        raise ValueError(f"{path.name}: dates must be ISO calendar dates") from exc
    if np.isnat(days).any() or np.any(np.datetime_as_string(days, unit="D") != dates):
        raise ValueError(f"{path.name}: dates must be exact YYYY-MM-DD calendar dates")
    months = days.astype("datetime64[M]").astype(np.int64) % 12 + 1
    weekends = ((days.astype(np.int64) + 3) % 7 >= 5).astype(np.int64)
    if np.any(y % 8 != weekends * 4 + (months % 12) // 3):
        raise ValueError(f"{path.name}: labels disagree with date weekend/season")
    order = np.lexsort((dates, meters))
    same_meter = meters[order][1:] == meters[order][:-1]
    if np.any(same_meter & (dates[order][1:] == dates[order][:-1])):
        raise ValueError(f"{path.name}: duplicate meter/date profiles")
    if np.any(same_meter & (y[order][1:] // 8 != y[order][:-1] // 8)):
        raise ValueError(f"{path.name}: technology conditions change within a meter")
    data["y"] = y.astype(np.int64, copy=False)
    return data, before


def _load_prepared(prepared_dir):
    directory = Path(prepared_dir)
    metadata, metadata_hash = _json_snapshot(directory / "metadata.json")
    scaler, scaler_hash = _json_snapshot(directory / "scaler.json")
    train_data, train_hash = _read_split(directory / "train.npz")
    val_data, val_hash = _read_split(directory / "val.npz")
    if np.intersect1d(train_data["meter_id"], val_data["meter_id"]).size:
        raise ValueError("train and validation meter IDs overlap")
    resolution = metadata.get("resolution")
    if isinstance(resolution, bool) or resolution not in (24, 96):
        raise ValueError("metadata resolution must be 24 or 96")
    if any(data["x"].shape[1] != resolution for data in (train_data, val_data)):
        raise ValueError("metadata resolution disagrees with profile width")
    if metadata.get("target") not in ("net", "import", "injection"):
        raise ValueError("metadata target must be net, import, or injection")
    if metadata["target"] != "net" and any(np.any(data["x"] < 0) for data in (train_data, val_data)):
        raise ValueError("observed import/injection profiles cannot have negative energy")
    if metadata.get("units") != "kWh/interval":
        raise ValueError("metadata units must be kWh/interval")
    if not isinstance(metadata.get("source_sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", metadata["source_sha256"]):
        raise ValueError("metadata source_sha256 must be a SHA256 hex digest")
    _positive_int("metadata split_seed", metadata.get("split_seed"), 0)
    splits = metadata.get("splits")
    if not isinstance(splits, dict):
        raise ValueError("metadata splits must contain train/val counts")
    for name, data in (("train", train_data), ("val", val_data)):
        recorded = splits.get(name)
        if not isinstance(recorded, dict) or recorded.get("profiles") != len(data["x"]) or recorded.get("meters") != len(np.unique(data["meter_id"])):
            raise ValueError(f"metadata {name} counts disagree with prepared profiles")
    minimum, maximum = scaler.get("minimum"), scaler.get("maximum")
    if any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) for v in (minimum, maximum)):
        raise ValueError("scaler minimum and maximum must be finite numbers")
    if minimum > maximum or not math.isfinite(maximum - minimum):
        raise ValueError("invalid scaler range")
    # Deliberately fit over the complete prepared training split before smoke subsampling.
    if minimum != float(train_data["x"].min()) or maximum != float(train_data["x"].max()):
        raise ValueError("scaler does not match the training-only minimum and maximum")
    hashes = {"train.npz": train_hash, "val.npz": val_hash,
              "metadata.json": metadata_hash, "scaler.json": scaler_hash}
    return train_data, val_data, metadata, scaler, hashes


def _atomic_json(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=".tmp-", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_checkpoint(path, checkpoint):
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".tmp-", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        # Use a file handle: some PyTorch versions reject dot-prefixed filenames.
        with temporary.open("wb") as handle:
            torch.save(checkpoint, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _provenance(device):
    root = Path(__file__).resolve().parents[2]
    result = {"git_commit": None, "git_commit_tree": None, "git_dirty": None,
              "python": platform.python_version(),
              "numpy": str(np.__version__), "torch": str(torch.__version__),
              "hardware": {"platform": platform.platform(), "machine": platform.machine(),
                           "processor": platform.processor(), "cpu_count": os.cpu_count(),
                           "device": str(device), "cuda_device": (
                               torch.cuda.get_device_name(device) if device.type == "cuda" else None)}}
    try:
        result["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        result["git_commit_tree"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD^{tree}"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        result["git_dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return result


@contextmanager
def _deterministic(seed, num_threads, device):
    old_threads = torch.get_num_threads()
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    old_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    old_cudnn_benchmark = torch.backends.cudnn.benchmark
    old_cudnn_deterministic = torch.backends.cudnn.deterministic
    old_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    cuda_devices = list(range(torch.cuda.device_count())) if device.type == "cuda" else []
    try:
        # Required by deterministic CUDA GEMMs; set before this function initializes CUDA.
        if device.type == "cuda":
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.set_num_threads(num_threads)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(seed)
            yield
    finally:
        torch.set_num_threads(old_threads)
        torch.use_deterministic_algorithms(old_deterministic, warn_only=old_warn_only)
        torch.backends.cudnn.benchmark = old_cudnn_benchmark
        torch.backends.cudnn.deterministic = old_cudnn_deterministic
        if old_workspace is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = old_workspace


def _conditions(labels, condition_dim, device):
    if not condition_dim:
        return torch.empty((len(labels), 0), dtype=torch.float32, device=device)
    return F.one_hot(torch.as_tensor(labels, dtype=torch.int64, device=device), num_classes=condition_dim).float()


def _scaler_parameters(scaler):
    """Return the affine offset/scale, including legacy min/max checkpoints."""
    minimum, maximum = scaler.get("minimum"), scaler.get("maximum")
    if any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v)
           for v in (minimum, maximum)) or maximum < minimum or not math.isfinite(maximum - minimum):
        raise ValueError("invalid checkpoint scaler")
    kind = scaler.get("kind", "minmax")
    if kind == "minmax":
        offset, scale = minimum, maximum - minimum
    elif kind == "standard":
        offset, scale = scaler.get("mean"), scaler.get("std")
        if any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v)
               for v in (offset, scale)) or scale < 0:
            raise ValueError("invalid standard scaler mean/std")
    else:
        raise ValueError("invalid checkpoint scaler kind")
    return offset, scale if scale else 1.0


def _normalized(x, scaler):
    offset, scale = _scaler_parameters(scaler)
    values = ((x.astype(np.float64) - offset) / scale).astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("normalization produced nonfinite values")
    return values


def _validation_score(generator, normalized, labels, noise, condition_dim, device, batch_size):
    """Mean_t W1(val[:,t], G(fixed_z, matching_val_labels)[:,t]), normalized units.

    This marginal score is a transparent checkpoint criterion, not a joint-profile
    fidelity or privacy measure.  The independent evaluation reports richer metrics.
    """
    was_training = generator.training
    generator.eval()
    pieces = []
    with torch.no_grad():
        for start in range(0, len(labels), batch_size):
            stop = start + batch_size
            c = _conditions(labels[start:stop], condition_dim, device)
            pieces.append(generator(noise[start:stop].to(device), c).cpu().numpy())
    generator.train(was_training)
    generated = np.concatenate(pieces)
    if not np.isfinite(generated).all():
        raise ValueError("generator produced nonfinite validation profiles")
    return float(np.mean([wasserstein_distance(normalized[:, t], generated[:, t])
                          for t in range(normalized.shape[1])]))


@contextmanager
def _run_directory(path):
    """Atomically reserve an empty experiment so concurrent jobs cannot mix it."""
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError("output directory must be empty to avoid mixing experiments")
    path.mkdir(parents=True, exist_ok=True)
    lock = path / ".training.lock"
    try:
        with lock.open("x") as handle:
            handle.write(str(os.getpid()))
    except FileExistsError as exc:
        raise ValueError("an active training run already reserved this output directory") from exc
    try:
        if any(child != lock for child in path.iterdir()):
            raise ValueError("output directory must be empty to avoid mixing experiments")
        yield path
    finally:
        lock.unlink(missing_ok=True)


def train(prepared_dir, output_dir, config: dict) -> dict:
    """Train a seeded cGAN and select best.pt solely on a fixed validation sample.

    ``output_dir`` must be missing or empty.  Reusing a populated experiment is
    rejected; checkpoint resume and test-based tuning are intentionally unsupported.
    """
    started = time.perf_counter()
    config = _config(config)
    with _run_directory(output_dir) as directory:
        return _train(prepared_dir, directory, config, started)


def _train(prepared_dir, output_dir, config, started):
    full_train, val, metadata, scaler, hashes = _load_prepared(prepared_dir)
    if config["normalization"] == "standard":
        # Fit a single population mean/std over all prepared TRAIN values, before
        # smoke subsampling. Keep the verified original min/max and file hashes.
        values = full_train["x"].astype(np.float64)
        scaler = {**scaler, "kind": "standard", "mean": float(values.mean()),
                  "std": float(values.std())}
        del values
    subset_rng = np.random.default_rng(config["seed"])
    count = len(full_train["x"])
    selected = np.arange(count, dtype=np.int64)
    if config["max_train_samples"] is not None and config["max_train_samples"] < count:
        selected = np.sort(subset_rng.choice(count, config["max_train_samples"], replace=False))
    labels = full_train["y"][selected]
    supported_labels = sorted(int(v) for v in np.unique(labels))
    train_x = torch.from_numpy(_normalized(full_train["x"][selected], scaler))
    eligible = np.flatnonzero(np.isin(val["y"], supported_labels))
    if len(eligible) < len(val["x"]) and len(selected) == count:
        missing = sorted(set(val["y"].tolist()) - set(supported_labels))
        raise ValueError(f"validation has unsupported labels absent from full training data: {missing}")
    if not len(eligible):
        raise ValueError("validation has no labels supported by the actual training subset")
    validation_rng = np.random.default_rng(config["seed"] + 1)
    val_indices = np.sort(validation_rng.choice(eligible, min(len(eligible), config["validation_samples"]), replace=False))
    val_x = _normalized(val["x"][val_indices], scaler)
    val_labels = val["y"][val_indices]
    condition_dim = 0 if config["unconditional"] else 64
    device = torch.device(config["device"])
    validation_noise = torch.randn((len(val_indices), config["latent_dim"]),
                                   generator=torch.Generator().manual_seed(config["seed"] + 2))
    provenance = _provenance(device)
    selection = {
        "train_profiles_available": count, "train_profiles_used": len(selected),
        "train_indices_sha256": hashlib.sha256(selected.astype("<i8").tobytes()).hexdigest(),
        "validation_profiles_available": len(val["x"]),
        "validation_profiles_eligible": len(eligible),
        "validation_profiles_used": len(val_indices),
        "validation_unsupported_profiles": len(val["x"]) - len(eligible),
        "validation_indices": val_indices.tolist(),
        "validation_labels_sha256": hashlib.sha256(val_labels.astype("<i8").tobytes()).hexdigest(),
        "validation_noise_sha256": hashlib.sha256(validation_noise.numpy().tobytes()).hexdigest(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_dir / "config.json", config)
    history = []
    best_score, best_epoch, stale_epochs = math.inf, 0, 0
    with _deterministic(config["seed"], config["num_threads"], device):
        generator = Generator(config["latent_dim"], condition_dim, config["hidden_dim"], metadata["resolution"]).to(device)
        discriminator = Discriminator(metadata["resolution"], condition_dim, config["hidden_dim"]).to(device)
        optimizer_g = torch.optim.Adam(generator.parameters(), lr=config["lr_g"])
        optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=config["lr_d"])
        objective = nn.BCEWithLogitsLoss()
        shuffle_rng = torch.Generator().manual_seed(config["seed"] + 3)
        for epoch in range(1, config["epochs"] + 1):
            epoch_started = time.perf_counter()
            generator.train()
            discriminator.train()
            permutation = torch.randperm(len(selected), generator=shuffle_rng)
            total_g = total_d = 0.0
            for start in range(0, len(selected), config["batch_size"]):
                batch_indices = permutation[start:start + config["batch_size"]]
                real = train_x[batch_indices].to(device)
                c = _conditions(labels[batch_indices.numpy()], condition_dim, device)
                ones = torch.ones((len(real), 1), device=device)
                zeros = torch.zeros_like(ones)
                optimizer_d.zero_grad(set_to_none=True)
                noise = torch.randn((len(real), config["latent_dim"]), device=device)
                # No G graph is retained for the discriminator step.
                with torch.no_grad():
                    fake = generator(noise, c)
                loss_d = (objective(discriminator(real, c), ones) +
                          objective(discriminator(fake.detach(), c), zeros)) / 2
                if not torch.isfinite(loss_d):
                    raise ValueError("nonfinite discriminator loss")
                loss_d.backward()
                optimizer_d.step()
                optimizer_g.zero_grad(set_to_none=True)
                discriminator.requires_grad_(False)
                try:
                    noise = torch.randn((len(real), config["latent_dim"]), device=device)
                    loss_g = objective(discriminator(generator(noise, c), c), ones)
                    if not torch.isfinite(loss_g):
                        raise ValueError("nonfinite generator loss")
                    loss_g.backward()
                    optimizer_g.step()
                finally:
                    discriminator.requires_grad_(True)
                total_g += float(loss_g.detach()) * len(real)
                total_d += float(loss_d.detach()) * len(real)
            score = _validation_score(generator, val_x, val_labels, validation_noise,
                                      condition_dim, device, config["batch_size"])
            if not math.isfinite(score):
                raise ValueError("nonfinite validation score")
            history.append({"epoch": epoch, "loss_g": total_g / len(selected),
                            "loss_d": total_d / len(selected), "validation_score": score,
                            "elapsed_seconds": time.perf_counter() - epoch_started})
            improved = score < best_score
            if improved:
                best_score, best_epoch, stale_epochs = score, epoch, 0
            else:
                stale_epochs += 1
            checkpoint = {
                "format_version": CHECKPOINT_FORMAT, "epoch": epoch,
                "config": config, "condition_dim": condition_dim,
                "output_dim": metadata["resolution"], "scaler": scaler,
                "metadata": metadata, "data_hashes": hashes,
                "supported_labels": supported_labels, "selection": selection,
                "provenance": provenance, "validation_metric": VALIDATION_METRIC,
                "validation_score": score, "best_epoch": best_epoch,
                "best_validation_score": best_score,
                "generator_state_dict": generator.state_dict(),
                "discriminator_state_dict": discriminator.state_dict(),
                "optimizer_g_state_dict": optimizer_g.state_dict(),
                "optimizer_d_state_dict": optimizer_d.state_dict(),
            }
            if improved:
                _atomic_checkpoint(output_dir / "best.pt", checkpoint)
            _atomic_checkpoint(output_dir / "last.pt", checkpoint)
            _atomic_json(output_dir / "history.json", history)
            print(f"Epoch {epoch}/{config['epochs']}: G={history[-1]['loss_g']:.6f} "
                  f"D={history[-1]['loss_d']:.6f} val_W1={score:.6f} "
                  f"elapsed={history[-1]['elapsed_seconds']:.2f}s", flush=True)
            if config["patience"] is not None and stale_epochs >= config["patience"]:
                break
    summary = {
        "best_epoch": best_epoch, "best_validation_score": best_score,
        "validation_metric": VALIDATION_METRIC, "epochs_completed": len(history),
        "stopped_early": len(history) < config["epochs"],
        "supported_labels": supported_labels, "condition_dim": condition_dim,
        "resolution": metadata["resolution"], "target": metadata["target"],
        "units": metadata["units"], "selection": selection,
        "data_hashes": hashes, "source_sha256": metadata["source_sha256"],
        "provenance": provenance, "scaler": scaler,
        "elapsed_seconds": time.perf_counter() - started,
        "best_checkpoint": "best.pt", "last_checkpoint": "last.pt",
        "test_used": False,
    }
    _atomic_json(output_dir / "summary.json", summary)
    return summary


def generate(checkpoint, labels: np.ndarray, seed=42, batch_size=1024) -> np.ndarray:
    """Generate original-unit profiles; reject labels absent from actual training.

    Uses safe ``weights_only`` deserialization and CPU inference.  Fixed seed and
    batch size replay exactly in the same software/hardware environment.
    """
    seed = _positive_int("seed", seed, 0)
    if seed >= 2**32:
        raise ValueError("seed must be smaller than 2**32")
    batch_size = _positive_int("batch_size", batch_size)
    labels = np.asarray(labels)
    if labels.ndim != 1 or labels.dtype.kind not in "iu" or np.any((labels < 0) | (labels >= 64)):
        raise ValueError("labels must be a one-dimensional integer array in [0, 63]")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(saved, dict) or saved.get("format_version") != CHECKPOINT_FORMAT:
        raise ValueError("unsupported checkpoint format")
    # A CUDA-trained checkpoint remains usable on a CPU-only inference machine.
    config = _config({**saved["config"], "device": "cpu"})
    condition_dim = 0 if config["unconditional"] else 64
    if saved.get("condition_dim") != condition_dim or saved.get("output_dim") not in (24, 96):
        raise ValueError("checkpoint model dimensions disagree with config")
    support = saved.get("supported_labels")
    if not isinstance(support, list) or not support or any(isinstance(v, bool) or not isinstance(v, int) or not 0 <= v < 64 for v in support):
        raise ValueError("invalid checkpoint supported_labels")
    missing = sorted(set(labels.tolist()) - set(support))
    if missing:
        raise ValueError(f"unsupported generation labels absent from training: {missing}")
    offset, scale = _scaler_parameters(saved["scaler"])
    if saved["scaler"].get("kind", "minmax") != config["normalization"]:
        raise ValueError("checkpoint scaler kind disagrees with normalization config")
    with _deterministic(seed, config["num_threads"], torch.device("cpu")):
        generator = Generator(config["latent_dim"], condition_dim, config["hidden_dim"], saved["output_dim"])
        generator.load_state_dict(saved["generator_state_dict"], strict=True)
        generator.eval()
        # Draw once so latent vectors do not depend on how inference is batched.
        noise = torch.randn((len(labels), config["latent_dim"]), generator=torch.Generator().manual_seed(seed))
        result = np.empty((len(labels), saved["output_dim"]), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, len(labels), batch_size):
                stop = start + batch_size
                conditions = _conditions(labels[start:stop], condition_dim, torch.device("cpu"))
                result[start:stop] = generator(noise[start:stop], conditions).numpy()
    result = (result.astype(np.float64) * scale + offset).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("generator produced nonfinite profiles")
    return result
