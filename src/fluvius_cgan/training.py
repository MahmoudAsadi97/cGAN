"""Reproducible cGAN training with validation-only checkpoint selection.

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
import warnings

import numpy as np
from scipy.stats import wasserstein_distance
import torch
from torch.nn import functional as F

from .models import Discriminator, Generator
from .residual import ResidualGenerator, fit_mean_profiles
from .distribution_loss import conditional_sliced_wasserstein_loss
from .temporal_loss import derivative_sliced_wasserstein_loss, fit_derivative_scale
from .release import RELEASE_SHA256


DEFAULT_CONFIG = {
    "epochs": 100, "batch_size": 256, "seed": 42, "latent_dim": 24,
    "hidden_dim": 64, "lr_g": 1e-4, "lr_d": 2e-4, "device": "cpu",
    "num_threads": 1, "unconditional": False, "validation_samples": 2048,
    "max_train_samples": None, "patience": None, "normalization": "minmax",
    "objective": "bce", "adam_beta1": 0.9, "adam_beta2": 0.999,
    "discriminator_hidden_dim": None, "conditional_mean_residual": False,
    "conditional_quantile_decoder": False,
    "batch_sampling": "shuffle", "conditions_per_batch": 4,
    "swd_weight": 0.0, "swd_projections": 128,
    "derivative_swd_weight": 0.0,
    "validation_selection": "pooled_w1",
    "allow_uncurated_data": False,
}
VALIDATION_METRIC = "mean_interval_wasserstein_normalized"
CONDITIONAL_VALIDATION_METRIC = "mean_condition_interval_wasserstein_normalized"
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
    for name in ("epochs", "batch_size", "latent_dim", "hidden_dim", "num_threads", "validation_samples",
                 "conditions_per_batch", "swd_projections"):
        result[name] = _positive_int(name, result[name])
    result["seed"] = _positive_int("seed", result["seed"], 0)
    if result["seed"] >= 2**32:
        raise ValueError("seed must be smaller than 2**32")
    for name in ("max_train_samples", "patience", "discriminator_hidden_dim"):
        if result[name] is not None:
            result[name] = _positive_int(name, result[name])
    for name in ("lr_g", "lr_d"):
        value = result[name]
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
        result[name] = float(value)
    for name in ("adam_beta1", "adam_beta2"):
        value = result[name]
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not 0 <= value < 1:
            raise ValueError(f"{name} must be finite and in [0, 1)")
        result[name] = float(value)
    for name in ("unconditional", "conditional_mean_residual", "conditional_quantile_decoder", "allow_uncurated_data"):
        if not isinstance(result[name], bool):
            raise ValueError(f"{name} must be a boolean")
    if result["normalization"] not in ("minmax", "standard"):
        raise ValueError("normalization must be minmax or standard")
    if result["objective"] not in ("bce", "hinge_sn"):
        raise ValueError("objective must be bce or hinge_sn")
    if result["batch_sampling"] not in ("shuffle", "balanced_conditions"):
        raise ValueError("batch_sampling must be shuffle or balanced_conditions")
    if result["validation_selection"] not in ("pooled_w1", "conditional_w1"):
        raise ValueError("validation_selection must be pooled_w1 or conditional_w1")
    for name in ("swd_weight", "derivative_swd_weight"):
        weight = result[name]
        if isinstance(weight, bool) or not isinstance(weight, Real) or not math.isfinite(weight) or weight < 0:
            raise ValueError(f"{name} must be nonnegative and finite")
        result[name] = float(weight)
    if result["conditional_quantile_decoder"]:
        if result["unconditional"] or result["normalization"] != "standard" or result["conditional_mean_residual"]:
            raise ValueError("conditional_quantile_decoder requires conditional standard-normalized training without mean residuals")
        if result["derivative_swd_weight"]:
            raise ValueError("conditional_quantile_decoder requires derivative_swd_weight=0")
    auxiliary_enabled = result["swd_weight"] or result["derivative_swd_weight"]
    if result["derivative_swd_weight"] and result["normalization"] != "standard":
        raise ValueError("positive derivative_swd_weight requires standard normalization")
    if result["batch_sampling"] == "balanced_conditions":
        if result["batch_size"] % result["conditions_per_batch"]:
            raise ValueError("balanced batch_size must be divisible by conditions_per_batch")
        if auxiliary_enabled and result["batch_size"] // result["conditions_per_batch"] < 32:
            raise ValueError("SWD needs at least 32 samples per sampled condition")
    elif auxiliary_enabled:
        raise ValueError("positive SWD weights require balanced_conditions sampling")
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


def _training_curation_guard(prepared_dir, config):
    """Qualify new training inputs without opening or hashing reserved TEST."""
    directory = Path(prepared_dir)
    metadata, metadata_hash = _json_snapshot(directory / "metadata.json")
    if ("derived" in metadata or "reserved_parent_test96" in metadata or
            (directory / "derivation_manifest.json").exists()):
        from .derived import resolve_parent_directory, verify_hourly_development
        parent = resolve_parent_directory(directory, metadata.get("derived"))
        verified = verify_hourly_development(parent, directory)
        if verified["metadata_sha256"] != metadata_hash:
            raise ValueError("derived metadata changed during the training provenance check")
        return verified
    pinned = metadata.get("source_sha256") == RELEASE_SHA256
    status = {"pinned_release": pinned, "metadata_sha256": metadata_hash}
    if "curation" not in metadata:
        if not pinned:
            return {**status, "status": "not_required_for_unpinned_source",
                    "qualification": "No pinned-release curation requirement applied; no trace-independence claim."}
        qualification = ("Exploratory reproduction/software check on uncurated pinned-source data; "
                         "exact trace aliases can cross splits. Results are not confirmatory.")
        if not config["allow_uncurated_data"]:
            raise ValueError(
                "Pinned-release training requires identity-curated v2 data. Run "
                "python scripts/curate_prepared.py --prepared ORIGINAL_DIR --output NEW_CURATED_DIR, "
                "then train from NEW_CURATED_DIR. Explicit allow_uncurated_data=true is reserved "
                "for qualified historical reproduction or software checks."
            )
        warnings.warn(qualification, UserWarning, stacklevel=3)
        return {**status, "status": "uncurated_explicit_override", "qualification": qualification}
    from .curation import FINGERPRINT_ALGORITHM, INPUT_FILES, SPLIT_PRIORITY
    curation = metadata["curation"]
    priority = list(SPLIT_PRIORITY)
    canonical_order = "lexicographic within the earliest original split"
    if (not isinstance(curation, dict) or type(curation.get("schema_version")) is not int or
            curation["schema_version"] != 1 or type(curation.get("fingerprint_version")) is not int or
            curation["fingerprint_version"] != 2 or curation.get("fingerprint_algorithm") != FINGERPRINT_ALGORITHM or
            curation.get("split_priority") != priority or curation.get("canonical_id_order") != canonical_order):
        raise ValueError("malformed or unsupported claimed curation algorithm/version/split priority")
    required_flags = {"signed_zero_canonicalized_for_identity_only": True, "nonzero_rounding_or_tolerance": False,
                      "byte_comparison_verified": True, "original_split_assignments_preserved": True,
                      "is_new_random_split": False, "development_traces_transferred_to_test": False}
    if any(curation.get(name) is not expected for name, expected in required_flags.items()):
        raise ValueError("malformed claimed curation identity/split-preservation flags")
    try:
        manifest, manifest_hash = _json_snapshot(directory / "curation_manifest.json")
    except OSError as error:
        raise ValueError("claimed curation requires curation_manifest.json") from error
    if (type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1 or
            type(manifest.get("fingerprint_version")) is not int or manifest["fingerprint_version"] != 2 or
            manifest.get("split_priority") != priority or manifest.get("canonical_id_order") != canonical_order):
        raise ValueError("malformed curation manifest version/split priority")
    original_hashes = curation.get("input_files_sha256")
    if (not isinstance(original_hashes, dict) or set(original_hashes) != set(INPUT_FILES) or
            any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value) for value in original_hashes.values()) or
            manifest.get("input_files_sha256") != original_hashes):
        raise ValueError("curation manifest disagrees with original-input fingerprints")
    output_hashes = manifest.get("output_files_sha256")
    if not isinstance(output_hashes, dict) or set(output_hashes) != set(INPUT_FILES):
        raise ValueError("curation manifest must bind all prepared files")
    verified = {}
    for name in ("metadata.json", "scaler.json", "train.npz", "val.npz"):
        actual = metadata_hash if name == "metadata.json" else _sha256(directory / name)
        if output_hashes.get(name) != actual:
            raise ValueError(f"curation manifest hash does not match {name}")
        verified[name] = actual
    return {**status, "status": "verified_curation_v2", "manifest_sha256": manifest_hash,
            "verified_file_hashes": verified,
            "qualification": "Identity-curated v2; validation remains exploratory and exact-trace uniqueness does not establish independent households."}


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
    derived_status = None
    if ("derived" in metadata or "reserved_parent_test96" in metadata or
            (directory / "derivation_manifest.json").exists()):
        derived_status = _training_curation_guard(directory, {"allow_uncurated_data": False})
        if derived_status["metadata_sha256"] != metadata_hash:
            raise ValueError("derived metadata changed before loading")
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
    if derived_status is not None:
        if hashes != derived_status["verified_file_hashes"]:
            raise ValueError("derived development files changed after transformation verification")
        if _sha256(directory / "derivation_manifest.json") != derived_status["derived_manifest_sha256"]:
            raise ValueError("derivation manifest changed after transformation verification")
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
              "source_files_sha256": {
                  str(path.relative_to(root)): _sha256(path)
                  for path in sorted(Path(__file__).resolve().parent.glob("*.py"))},
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


def _quantile_decoder_source_binding(selection, hashes, scaler, provenance):
    """Bind a selected-TRAIN fit to existing immutable run records."""
    return {
        "fitting_scope": "actual_gradient_training_subset_only",
        "train_npz_sha256": hashes["train.npz"],
        "train_indices_sha256": selection["train_indices_sha256"],
        "train_labels_sha256": selection["train_labels_sha256"],
        "train_profiles_available": selection["train_profiles_available"],
        "train_profiles_used": selection["train_profiles_used"],
        "data_hashes": dict(hashes), "scaler": dict(scaler),
        "source_files_sha256": dict(provenance["source_files_sha256"]),
    }


def load_quantile_decoded_generator(saved, device="cpu"):
    """Reconstruct and verify an embedded TRAIN-only decoder without fitting.

    Source hashes bind the historical implementation recorded at training time;
    they are not a claim that an arbitrary later checkout has identical code.
    No prepared-data file is opened by this function.
    """
    from .models import QuantileDecodedGenerator
    from .quantile_decoder import ConditionalQuantileDecoder

    if not isinstance(saved, dict) or saved.get("format_version") != CHECKPOINT_FORMAT:
        raise ValueError("unsupported decoder checkpoint format")
    try:
        config = _config({**saved["config"], "device": "cpu"})
        if not config["conditional_quantile_decoder"]:
            raise ValueError("checkpoint does not enable conditional_quantile_decoder")
        if saved["condition_dim"] != 64 or saved["output_dim"] not in (24, 96):
            raise ValueError("decoder checkpoint model dimensions disagree with config")
        scaler, selection, provenance = saved["scaler"], saved["selection"], saved["provenance"]
        if scaler.get("kind", "minmax") != "standard":
            raise ValueError("decoder checkpoint requires a standard scaler")
        offset, scale = _scaler_parameters(scaler)
        decoder = ConditionalQuantileDecoder.from_state_dict(saved["generator_state_dict"], prefix="decoder.").to("cpu")
        fit = decoder.fit_metadata
        if saved.get("quantile_decoder") != fit or provenance.get("quantile_decoder") != fit:
            raise ValueError("decoder checkpoint fit metadata disagrees with state/provenance")
        binding = _quantile_decoder_source_binding(selection, saved["data_hashes"], scaler, provenance)
        if fit["source_binding"] != binding:
            raise ValueError("decoder checkpoint source binding disagrees with run records")
        required_sources = {f"src/fluvius_cgan/{name}.py" for name in ("training", "models", "quantile_decoder")}
        sources = binding["source_files_sha256"]
        if (not required_sources.issubset(sources) or not sources or
                any(not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
                    for digest in sources.values())):
            raise ValueError("decoder checkpoint source fingerprints are incomplete")
        hashes = binding["data_hashes"]
        if (not {"train.npz", "val.npz", "metadata.json", "scaler.json"}.issubset(hashes) or
                "test.npz" in hashes or
                any(not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
                    for digest in hashes.values())):
            raise ValueError("decoder checkpoint development-data fingerprints are invalid")
        used = _positive_int("decoder train_profiles_used", selection["train_profiles_used"])
        available = _positive_int("decoder train_profiles_available", selection["train_profiles_available"])
        if (used > available or saved["metadata"]["splits"]["train"]["profiles"] != available or
                used != min(available, config["max_train_samples"] or available)):
            raise ValueError("decoder checkpoint selected TRAIN size disagrees with config/metadata")
        if (fit["selected_profiles"] != used or fit["output_dim"] != saved["output_dim"] or
                fit["condition_dim"] != 64 or fit["supported_labels"] != saved["supported_labels"] or
                fit["selected_indices_sha256"] != selection["train_indices_sha256"] or
                fit["selected_labels_sha256"] != selection["train_labels_sha256"] or
                fit["selected_profiles_sha256"] != selection["train_profiles_sha256"] or
                fit["condition_counts"] != selection["train_profiles_per_condition"]):
            raise ValueError("decoder checkpoint selected TRAIN support/counts/hashes disagree")
        base = Generator(config["latent_dim"], 64, config["hidden_dim"], saved["output_dim"])
        if any(not isinstance(value, torch.Tensor) or value.dtype != torch.float32
               for name, value in saved["generator_state_dict"].items() if name.startswith("base.")):
            raise ValueError("decoder checkpoint base weights must retain float32 training dtype")
        for name, expected in (("normalization_offset", offset), ("normalization_scale", scale)):
            buffer = saved["generator_state_dict"][name]
            if (not isinstance(buffer, torch.Tensor) or buffer.dtype != torch.float64 or
                    buffer.shape != () or buffer.item() != expected):
                raise ValueError("decoder checkpoint normalization buffers disagree with scaler")
        generator = QuantileDecodedGenerator(base, decoder, offset=offset, scale=scale)
        generator.load_state_dict(saved["generator_state_dict"], strict=True)
        decoder.validate_state()
        if (generator.normalization_offset.dtype != torch.float64 or
                generator.normalization_scale.dtype != torch.float64 or
                generator.normalization_offset.numel() != 1 or generator.normalization_scale.numel() != 1 or
                generator.normalization_offset.item() != offset or generator.normalization_scale.item() != scale):
            raise ValueError("decoder checkpoint normalization buffers disagree with scaler")
        if any(not torch.isfinite(parameter).all() for parameter in generator.parameters()):
            raise ValueError("decoder checkpoint learned parameters must be finite")
        return generator.to(device)
    except (KeyError, TypeError, RuntimeError, AttributeError) as exc:
        raise ValueError("malformed quantile decoder checkpoint") from exc


def _validation_score(generator, normalized, labels, noise, condition_dim, device, batch_size,
                      selection="pooled_w1"):
    """Fixed-sample interval W1 in normalized units, pooled or condition-macro.

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
    return _profile_validation_score(normalized, generated, labels, selection)


def _profile_validation_score(real, generated, labels, selection):
    if selection == "pooled_w1":
        return float(np.mean([wasserstein_distance(real[:, t], generated[:, t])
                              for t in range(real.shape[1])]))
    groups = [labels == label for label in np.unique(labels)]
    return float(np.mean([np.mean([wasserstein_distance(real[group, t], generated[group, t])
                                  for t in range(real.shape[1])]) for group in groups]))


def _validation_indices(labels, supported_labels, sample_count, rng, selection):
    if selection == "pooled_w1":
        eligible = np.flatnonzero(np.isin(labels, supported_labels))
        return np.sort(rng.choice(eligible, min(len(eligible), sample_count), replace=False))
    if sample_count % len(supported_labels) or sample_count // len(supported_labels) < 2:
        raise ValueError("conditional validation_samples must be divisible by support size and allow at least 2 per condition")
    cap = sample_count // len(supported_labels)
    pieces = []
    for label in supported_labels:
        eligible = np.flatnonzero(labels == label)
        if len(eligible) < 2:
            raise ValueError("conditional validation needs at least 2 available profiles per supported condition")
        pieces.append(rng.choice(eligible, min(len(eligible), cap), replace=False))
    return np.sort(np.concatenate(pieces))


def _training_batches(count, batch_size, generator, *, condition_indices=None, conditions_per_batch=4):
    """Yield actual-training-subset indices; balanced draws use replacement."""
    if condition_indices is None:
        yield from torch.randperm(count, generator=generator).split(batch_size)
        return
    per_condition = batch_size // conditions_per_batch
    for _ in range(math.ceil(count / batch_size)):
        chosen = torch.randperm(len(condition_indices), generator=generator)[:conditions_per_batch]
        yield torch.cat([condition_indices[int(label)][torch.randint(
            len(condition_indices[int(label)]), (per_condition,), generator=generator)] for label in chosen])


def _discriminator_loss(real_logits, fake_logits, objective):
    if objective == "hinge_sn":
        return F.relu(1 - real_logits).mean() + F.relu(1 + fake_logits).mean()
    return (F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits)) +
            F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))) / 2


def _generator_loss(fake_logits, objective):
    if objective == "hinge_sn":
        return -fake_logits.mean()
    return F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))


@contextmanager
def _frozen_discriminator(discriminator, *, freeze_buffers=False):
    """Keep gradients to generated profiles while holding discriminator fixed.

    For spectral normalization, eval mode also freezes the power-iteration
    vectors during G's update. BCE keeps its original training-mode dropout.
    """
    was_training = discriminator.training
    discriminator.requires_grad_(False)
    if freeze_buffers:
        discriminator.eval()
    try:
        yield
    finally:
        discriminator.requires_grad_(True)
        discriminator.train(was_training)


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
        curation_status = _training_curation_guard(prepared_dir, config)
        return _train(prepared_dir, directory, config, started, curation_status)


def _train(prepared_dir, output_dir, config, started, curation_status):
    full_train, val, metadata, scaler, hashes = _load_prepared(prepared_dir)
    if (hashes["metadata.json"] != curation_status["metadata_sha256"] or
            any(hashes.get(name) != digest for name, digest in curation_status.get("verified_file_hashes", {}).items())):
        raise ValueError("prepared files changed after the training curation check")
    if curation_status.get("manifest_sha256"):
        manifest_hash = _sha256(Path(prepared_dir) / "curation_manifest.json")
        if manifest_hash != curation_status["manifest_sha256"]:
            raise ValueError("curation manifest changed after verification")
        hashes["curation_manifest.json"] = manifest_hash
    if curation_status.get("derived_manifest_sha256"):
        manifest_hash = _sha256(Path(prepared_dir) / "derivation_manifest.json")
        if manifest_hash != curation_status["derived_manifest_sha256"]:
            raise ValueError("derivation manifest changed after verification")
        hashes["derivation_manifest.json"] = manifest_hash
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
    condition_indices = None
    if config["batch_sampling"] == "balanced_conditions":
        if len(supported_labels) < config["conditions_per_batch"]:
            raise ValueError("actual training support is smaller than conditions_per_batch")
        condition_indices = [torch.from_numpy(np.flatnonzero(labels == label)) for label in supported_labels]
    train_x = torch.from_numpy(_normalized(full_train["x"][selected], scaler))
    eligible = np.flatnonzero(np.isin(val["y"], supported_labels))
    if len(eligible) < len(val["x"]) and len(selected) == count:
        missing = sorted(set(val["y"].tolist()) - set(supported_labels))
        raise ValueError(f"validation has unsupported labels absent from full training data: {missing}")
    if not len(eligible):
        raise ValueError("validation has no labels supported by the actual training subset")
    validation_rng = np.random.default_rng(config["seed"] + 1)
    val_indices = _validation_indices(val["y"], supported_labels, config["validation_samples"],
                                      validation_rng, config["validation_selection"])
    val_x = _normalized(val["x"][val_indices], scaler)
    val_labels = val["y"][val_indices]
    condition_dim = 0 if config["unconditional"] else 64
    device = torch.device(config["device"])
    validation_noise = torch.randn((len(val_indices), config["latent_dim"]),
                                   generator=torch.Generator().manual_seed(config["seed"] + 2))
    provenance = _provenance(device)
    validation_metric = (VALIDATION_METRIC if config["validation_selection"] == "pooled_w1"
                         else CONDITIONAL_VALIDATION_METRIC)
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
        "validation_selection": config["validation_selection"], "validation_selection_version": 1,
        "validation_samples_requested": config["validation_samples"],
        "validation_per_condition_cap": (config["validation_samples"] // len(supported_labels)
                                          if config["validation_selection"] == "conditional_w1" else None),
        "validation_profiles_per_condition": {str(label): int(np.sum(val_labels == label)) for label in supported_labels},
        "batch_sampling": config["batch_sampling"],
        "swd_conditioning": "pooled" if config["unconditional"] else "conditional",
    }
    derivative_scale = None
    if config["derivative_swd_weight"]:
        derivative_scale = {
            **fit_derivative_scale(train_x, labels),
            "fitting_scope": "actual_gradient_training_subset_only",
            "train_indices_sha256": selection["train_indices_sha256"],
            "train_npz_sha256": hashes["train.npz"],
            "normalization": {"kind": "standard", "mean": scaler["mean"], "std": scaler["std"],
                              "fitting_scope": "full_prepared_train_before_subset_selection"},
        }
    provenance["derivative_swd"] = {
        "enabled": bool(config["derivative_swd_weight"]), "weight": config["derivative_swd_weight"],
        "scale": derivative_scale,
        "projection_seed": config["seed"] + 5 if config["derivative_swd_weight"] else None,
        "num_projections": config["swd_projections"] if config["derivative_swd_weight"] else None,
        "conditioning": "pooled" if config["unconditional"] else "equal_observed_condition_macro",
        "difference_policy": "signed adjacent within-day differences; no midnight wrap or centering",
    }
    quantile_decoder = None
    quantile_decoder_metadata = None
    if config["conditional_quantile_decoder"]:
        from .quantile_decoder import fit_quantile_decoder
        selection["train_labels_sha256"] = hashlib.sha256(labels.astype("<i8").tobytes()).hexdigest()
        selection["train_profiles_per_condition"] = {
            str(label): int(np.sum(labels == label)) for label in supported_labels}
        # Fit physical knots condition by condition; the module selects canonical
        # TRAIN rows itself. No validation values or generated samples enter fit.
        quantile_decoder = fit_quantile_decoder(
            full_train["x"], full_train["y"], selected_indices=selected, condition_dim=64,
            split="train",
            source_binding=_quantile_decoder_source_binding(selection, hashes, scaler, provenance))
        quantile_decoder_metadata = quantile_decoder.fit_metadata
        selection["train_profiles_sha256"] = quantile_decoder_metadata["selected_profiles_sha256"]
        provenance["quantile_decoder"] = quantile_decoder_metadata
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_dir / "config.json", config)
    history = []
    generator_updates = discriminator_updates = 0
    samples_seen = 0
    class_exposure_counts = {str(label): 0 for label in supported_labels}
    best_score, best_epoch, stale_epochs = math.inf, 0, 0
    with _deterministic(config["seed"], config["num_threads"], device):
        generator = Generator(config["latent_dim"], condition_dim, config["hidden_dim"], metadata["resolution"]).to(device)
        if config["conditional_mean_residual"]:
            # Fixed offsets use only the normalized rows actually trained on.
            generator = ResidualGenerator(generator, fit_mean_profiles(train_x, labels, condition_dim))
        if quantile_decoder is not None:
            from .models import QuantileDecodedGenerator
            offset, scale = _scaler_parameters(scaler)
            generator = QuantileDecodedGenerator(generator, quantile_decoder.to(device), offset=offset, scale=scale)
        discriminator = Discriminator(metadata["resolution"], condition_dim,
                                      config["discriminator_hidden_dim"] or config["hidden_dim"],
                                      objective=config["objective"]).to(device)
        betas = (config["adam_beta1"], config["adam_beta2"])
        optimizer_g = torch.optim.Adam(generator.parameters(), lr=config["lr_g"], betas=betas)
        optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=config["lr_d"], betas=betas)
        shuffle_rng = torch.Generator().manual_seed(config["seed"] + 3)
        swd_rng = (torch.Generator(device=device).manual_seed(config["seed"] + 4)
                   if config["swd_weight"] else None)
        derivative_rng = (torch.Generator(device=device).manual_seed(config["seed"] + 5)
                          if config["derivative_swd_weight"] else None)
        for epoch in range(1, config["epochs"] + 1):
            epoch_started = time.perf_counter()
            generator.train()
            discriminator.train()
            total_g = total_d = total_adversarial = total_swd = 0.0
            total_derivative_swd = 0.0
            epoch_samples_seen = 0
            epoch_class_exposure_counts = {str(label): 0 for label in supported_labels}
            epoch_generator_updates = epoch_discriminator_updates = 0
            for batch_indices in _training_batches(len(selected), config["batch_size"], shuffle_rng,
                    condition_indices=condition_indices, conditions_per_batch=config["conditions_per_batch"]):
                real = train_x[batch_indices].to(device)
                batch_labels = labels[batch_indices.numpy()]
                c = _conditions(batch_labels, condition_dim, device)
                optimizer_d.zero_grad(set_to_none=True)
                noise = torch.randn((len(real), config["latent_dim"]), device=device)
                # No G graph is retained for the discriminator step.
                with torch.no_grad():
                    fake = generator(noise, c)
                loss_d = _discriminator_loss(discriminator(real, c), discriminator(fake.detach(), c),
                                             config["objective"])
                if not torch.isfinite(loss_d):
                    raise ValueError("nonfinite discriminator loss")
                loss_d.backward()
                optimizer_d.step()
                discriminator_updates += 1
                epoch_discriminator_updates += 1
                optimizer_g.zero_grad(set_to_none=True)
                with _frozen_discriminator(discriminator, freeze_buffers=config["objective"] == "hinge_sn"):
                    noise = torch.randn((len(real), config["latent_dim"]), device=device)
                    fake = generator(noise, c)
                    loss_adversarial = _generator_loss(discriminator(fake, c), config["objective"])
                    loss_g = loss_adversarial
                    swd_value = 0.0
                    derivative_swd_value = 0.0
                    if config["swd_weight"]:
                        swd_labels = torch.as_tensor(batch_labels, dtype=torch.int64, device=device)
                        if config["unconditional"]:
                            swd_labels = torch.zeros_like(swd_labels)
                        loss_swd = conditional_sliced_wasserstein_loss(
                            real, fake, swd_labels, swd_labels, p=2,
                            num_projections=config["swd_projections"], min_samples_per_condition=32,
                            generator=swd_rng)
                        loss_g = loss_adversarial + config["swd_weight"] * loss_swd
                        swd_value = float(loss_swd.detach())
                    if config["derivative_swd_weight"]:
                        derivative_labels = torch.as_tensor(batch_labels, dtype=torch.int64, device=device)
                        if config["unconditional"]:
                            derivative_labels = torch.zeros_like(derivative_labels)
                        loss_derivative_swd = derivative_sliced_wasserstein_loss(
                            real, fake, derivative_labels, derivative_labels, derivative_scale["used_float32"],
                            num_projections=config["swd_projections"], min_samples_per_condition=32,
                            generator=derivative_rng)
                        loss_g = loss_g + config["derivative_swd_weight"] * loss_derivative_swd
                        derivative_swd_value = float(loss_derivative_swd.detach())
                    if not torch.isfinite(loss_g):
                        raise ValueError("nonfinite generator loss")
                    loss_g.backward()
                    optimizer_g.step()
                    generator_updates += 1
                    epoch_generator_updates += 1
                total_g += float(loss_g.detach()) * len(real)
                total_d += float(loss_d.detach()) * len(real)
                total_adversarial += float(loss_adversarial.detach()) * len(real)
                total_swd += swd_value * len(real)
                total_derivative_swd += derivative_swd_value * len(real)
                samples_seen += len(real)
                epoch_samples_seen += len(real)
                for label, exposure in zip(*np.unique(batch_labels, return_counts=True)):
                    class_exposure_counts[str(label)] += int(exposure)
                    epoch_class_exposure_counts[str(label)] += int(exposure)
            score_args = (generator, val_x, val_labels, validation_noise, condition_dim, device, config["batch_size"])
            score = (_validation_score(*score_args) if config["validation_selection"] == "pooled_w1"
                     else _validation_score(*score_args, selection="conditional_w1"))
            if not math.isfinite(score):
                raise ValueError("nonfinite validation score")
            history.append({"epoch": epoch, "loss_g": total_g / epoch_samples_seen,
                            "loss_d": total_d / epoch_samples_seen, "validation_score": score,
                            "loss_g_adversarial": total_adversarial / epoch_samples_seen,
                            "loss_swd": total_swd / epoch_samples_seen,
                            "loss_derivative_swd": total_derivative_swd / epoch_samples_seen,
                            "samples_seen": epoch_samples_seen,
                            "class_exposure_counts": epoch_class_exposure_counts,
                            "generator_updates": epoch_generator_updates,
                            "discriminator_updates": epoch_discriminator_updates,
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
                "data_curation": curation_status,
                "derivative_swd_scale": derivative_scale,
                "supported_labels": supported_labels, "selection": selection,
                "provenance": provenance, "validation_metric": validation_metric,
                "validation_score": score, "best_epoch": best_epoch,
                "best_validation_score": best_score,
                "generator_updates": generator_updates, "discriminator_updates": discriminator_updates,
                "samples_seen": samples_seen, "class_exposure_counts": dict(class_exposure_counts),
                "epoch_metrics": history[-1],
                "generator_state_dict": generator.state_dict(),
                "discriminator_state_dict": discriminator.state_dict(),
                "optimizer_g_state_dict": optimizer_g.state_dict(),
                "optimizer_d_state_dict": optimizer_d.state_dict(),
            }
            if quantile_decoder_metadata is not None:
                checkpoint["quantile_decoder"] = quantile_decoder_metadata
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
        "validation_metric": validation_metric, "epochs_completed": len(history),
        "generator_updates": generator_updates, "discriminator_updates": discriminator_updates,
        "samples_seen": samples_seen, "class_exposure_counts": class_exposure_counts,
        "last_epoch_metrics": history[-1],
        "stopped_early": len(history) < config["epochs"],
        "supported_labels": supported_labels, "condition_dim": condition_dim,
        "resolution": metadata["resolution"], "target": metadata["target"],
        "units": metadata["units"], "selection": selection,
        "data_hashes": hashes, "source_sha256": metadata["source_sha256"],
        "data_curation": curation_status,
        "derivative_swd_scale": derivative_scale,
        "provenance": provenance, "scaler": scaler,
        "elapsed_seconds": time.perf_counter() - started,
        "best_checkpoint": "best.pt", "last_checkpoint": "last.pt",
        "test_used": False,
    }
    if quantile_decoder_metadata is not None:
        summary["quantile_decoder"] = quantile_decoder_metadata
    _atomic_json(output_dir / "summary.json", summary)
    return summary


def generate(checkpoint, labels: np.ndarray, seed=42, batch_size=1024) -> np.ndarray:
    """Generate original-unit profiles; reject labels absent from actual training.

    Uses safe ``weights_only`` deserialization and CPU inference.  Fixed seed and
    batch size replay exactly in the same software/hardware environment.
    """
    return _generate(checkpoint, labels, seed, batch_size)


def generate_quantile_decoder_diagnostics(checkpoint, labels: np.ndarray, seed=42, batch_size=1024):
    """Return the same physical draw plus its latent probabilities and fit record.

    This validation helper allocates an extra float64 probability array. Ordinary
    ``generate`` avoids that allocation, including for full-TRAIN utility draws.
    No quantiles or normalization parameters are fitted during either path.
    """
    return _generate(checkpoint, labels, seed, batch_size, quantile_diagnostics=True)


def _generate(checkpoint, labels, seed, batch_size, *, quantile_diagnostics=False):
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
    if quantile_diagnostics and not config["conditional_quantile_decoder"]:
        raise ValueError("quantile decoder diagnostics require an enabled decoder checkpoint")
    if not config["conditional_quantile_decoder"] and (
            saved.get("quantile_decoder") is not None or saved.get("provenance", {}).get("quantile_decoder") is not None):
        raise ValueError("decoder fit metadata disagrees with disabled decoder config")
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
        if config["conditional_quantile_decoder"]:
            generator = load_quantile_decoded_generator(saved)
        else:
            generator = Generator(config["latent_dim"], condition_dim, config["hidden_dim"], saved["output_dim"])
            if config["conditional_mean_residual"]:
                generator = ResidualGenerator(generator, saved["generator_state_dict"]["mean_profiles"])
            # Restore the residual support mask as well as its means and base weights.
            generator.load_state_dict(saved["generator_state_dict"], strict=True)
        generator.eval()
        # Draw once so latent vectors do not depend on how inference is batched.
        noise = torch.randn((len(labels), config["latent_dim"]), generator=torch.Generator().manual_seed(seed))
        result = np.empty((len(labels), saved["output_dim"]), dtype=np.float32)
        probabilities = np.empty(result.shape, dtype=np.float64) if quantile_diagnostics else None
        with torch.no_grad():
            for start in range(0, len(labels), batch_size):
                stop = start + batch_size
                conditions = _conditions(labels[start:stop], condition_dim, torch.device("cpu"))
                if quantile_diagnostics:
                    physical, probability = generator.decode_with_probabilities(noise[start:stop], conditions)
                    result[start:stop] = physical.numpy()
                    probabilities[start:stop] = probability.numpy()
                elif config["conditional_quantile_decoder"]:
                    result[start:stop] = generator.decode_physical(noise[start:stop], conditions).numpy()
                else:
                    result[start:stop] = generator(noise[start:stop], conditions).numpy()
    if not config["conditional_quantile_decoder"]:
        result = (result.astype(np.float64) * scale + offset).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("generator produced nonfinite profiles")
    if quantile_diagnostics:
        if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
            raise ValueError("decoder produced invalid latent probabilities")
        return result, probabilities, saved["quantile_decoder"]
    return result
