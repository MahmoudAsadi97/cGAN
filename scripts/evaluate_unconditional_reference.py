"""Supplemental native-96 raw unconditional BCE validation diagnostics.

This companion does not replace the pinned primary validation_report.json. It
uses the explicitly recorded current evaluator/copula interpolation and never
fits, loads or applies a conditional calibrator. TEST has no command option.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import scipy
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from compare_checkpoints import _checkpoint, _digest, _validate_manifest
from evaluate_calibration import _battery_diagnostics, _generate_verified, _profile_digest
from fluvius_cgan.battery import simulate_battery
from fluvius_cgan.copula import fit_conditional_copula
from fluvius_cgan.evaluation import evaluate_conditional, mean_baseline, resample_baseline
from fluvius_cgan.joint import conditional_projected_wasserstein
from fluvius_cgan.release import RELEASE_SHA256, sha256_file
from fluvius_cgan.training import (
    CONDITIONAL_VALIDATION_METRIC, _config, _load_prepared, _provenance,
    _training_curation_guard, _validation_indices,
)

SEEDS = (11, 22, 33, 44, 55)
TRAIN_PROFILES, EPOCHS, UPDATES_PER_EPOCH, UPDATES = 606060, 26, 1184, 30784
EVALUATION_SEED, QUANTILE_KNOTS = 2026, 1025
SOURCE_COMMIT = "1ca62d0fbe1d77f916f8904e812db7d720f8cf53"
SOURCE_TREE = "55b35f8204241eaabc58f5196cb14fff5b158cad"
PREPARED_FILES = ("metadata.json", "scaler.json", "train.npz", "val.npz")
RUN_FILES = ("best.pt", "last.pt", "config.json", "summary.json", "history.json", "validation_report.json")
METHODS = ("raw_unconditional_cgan", "conditional_mean", "conditional_resampling",
           "gaussian_copula", "independent_marginals")
# Exact complete Python-source inventory recorded by the declared training pin.
TRAINING_SOURCE_HASHES = {
    "src/fluvius_cgan/__init__.py": "6492d4cfc58c39594f849ec7274ca0d995d12b15fd0108006f27f87e490df548",
    "src/fluvius_cgan/__main__.py": "6d8b7d7846a845059d7a3107143f11131f63c5511d669b44085b15ec5e3d2279",
    "src/fluvius_cgan/battery.py": "f7c7437de23c5b3a9dbc266bd25f1345731ec0fb2112aeaf830332cef730c8e3",
    "src/fluvius_cgan/calibration.py": "5b4bfe252500e8e11a96f2102822842d6a4e60ef107df8d66c878b7b44a7598f",
    "src/fluvius_cgan/cli.py": "825994a8082e06add40f303d32af4e573dc76bdfb4252c3b4944a3db9b3a56ad",
    "src/fluvius_cgan/copula.py": "5739c8d0cb438ebd3fa06bc67ba79d6037493f8406a4e7ac1ba50ebba29d6939",
    "src/fluvius_cgan/curation.py": "b250d1471f688daedd7ca1ea8652811baac89514b1cf22d96b9388d21892db27",
    "src/fluvius_cgan/data.py": "824f6f2d8ca0e1bfff9c27af6825c5f47739b22b40023585c8bafbc5de3c96b5",
    "src/fluvius_cgan/distribution_loss.py": "37e810c829d18569325e9b95592a65a7268d100e9a51196a7baa0f399c41ac0a",
    "src/fluvius_cgan/evaluation.py": "3f59f817abda7fbd7d4d043b5bee81b1c2c700db219f9af35acf94b7c910281e",
    "src/fluvius_cgan/joint.py": "1c81d551acf1a627cf46659530c07ddb211d036f3df73eaf2d635ee799944941",
    "src/fluvius_cgan/models.py": "a47a2321e1229aa5a053fac16b6300b3f3a2504a008ade2fe861e9333bc87da7",
    "src/fluvius_cgan/release.py": "728ca2c24802e0458937e7818f712b5a1e770e5ca40f5fedccfed5b0c75ababe",
    "src/fluvius_cgan/residual.py": "e50a55ddfb7b74c62b3717d9b3924879c2bb109198325f6b39f062aa0e789b80",
    "src/fluvius_cgan/temporal_loss.py": "498d817001dd009caae0924682a7e0549bfc39897e3592d0fcd90bac3f80207a",
    "src/fluvius_cgan/training.py": "1eabd6ee18c381ba16a2af659247400d0d08557c8fcf7a66dcd3c6fc16b5236a",
    "src/fluvius_cgan/uncertainty.py": "468c8abcb3ad85c4b03d7508b13a41a9adaa289fbba1a66c5a3257eea5b3e214",
    "src/fluvius_cgan/utility.py": "65db77468ea664844426bfd62041dded781302e195820bb52a11d88a8feb5133",
}
SUPPLEMENTAL_NOTE = (
    "Separately named supplemental raw unconditional reference diagnostics. The original primary "
    "validation_report.json and its declared 1ca62d0 training/native evaluator attribution are retained. "
    "This report uses the current separately fingerprinted evaluator and copula interpolation; its "
    "statistical-control numbers must not silently replace pinned native-primary results. No "
    "conditional calibration map is fitted, loaded or applied to the unconditional generator."
)


def _check(condition, message):
    if not condition:
        raise ValueError(message)


def _same(left, right):
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def _finite(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _json(path):
    raw = Path(path).read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def _preflight(prepared, paths):
    """Check pathname and inode aliases without reading reserved TEST contents."""
    reserved = Path(prepared) / "test.npz"
    for path in paths:
        path = Path(path)
        if path.resolve() == reserved.resolve() or (
                path.exists() and reserved.exists() and path.samefile(reserved)):
            raise ValueError(f"input/output aliases reserved TEST: {path.name}")


def expected_config(training_seed):
    _check(type(training_seed) is int and training_seed in SEEDS, "training_seed must be one of 11,22,33,44,55")
    return _config({
        "epochs": 26, "batch_size": 512, "seed": training_seed, "latent_dim": 24,
        "hidden_dim": 64, "discriminator_hidden_dim": 64, "objective": "bce",
        "lr_g": .0001, "lr_d": .0002, "adam_beta1": .9, "adam_beta2": .999,
        "normalization": "minmax", "conditional_mean_residual": False,
        "swd_weight": 0., "derivative_swd_weight": 0., "swd_projections": 128,
        "batch_sampling": "balanced_conditions", "conditions_per_batch": 4,
        "validation_selection": "conditional_w1", "validation_samples": 4096,
        "max_train_samples": None, "patience": None, "device": "cpu", "num_threads": 1,
        "unconditional": True, "allow_uncurated_data": False,
    })


def _verify_run_contract(config, summary, history, best, last, *, training_seed, train_count):
    """Validate completion separately from the possibly earlier selected checkpoint."""
    expected = expected_config(training_seed)
    _check(train_count == TRAIN_PROFILES and type(train_count) is int, "full canonical TRAIN requires 606060 profiles")
    _check(all(_same(value, expected) for value in (config, best.get("config"), last.get("config"))),
           "run is not the declared unconditional BCE96 configuration/seed")
    _check(summary.get("epochs_completed") == EPOCHS and summary.get("stopped_early") is False
           and summary.get("test_used") is False and last.get("epoch") == EPOCHS,
           "completed 26-epoch validation-only run required")
    _check(isinstance(history, list) and len(history) == EPOCHS and
           all(isinstance(row, dict) and type(row.get("epoch")) is int and _finite(row.get("validation_score"))
               for row in history) and
           [row.get("epoch") for row in history] == list(range(1, EPOCHS + 1)), "incomplete/nonfinite training history")
    for record, keys in ((summary, ("epochs_completed", "best_epoch")), (best, ("epoch", "best_epoch")),
                         (last, ("epoch", "best_epoch"))):
        _check(all(type(record.get(key)) is int for key in (*keys, "generator_updates", "discriminator_updates", "samples_seen")),
               "training budget counters must be integers")
    selected = min(history, key=lambda row: row["validation_score"])
    best_epoch = selected["epoch"]
    _check(summary.get("best_epoch") == best.get("epoch") == best.get("best_epoch") == last.get("best_epoch") == best_epoch,
           "selected checkpoint does not match earliest minimum validation score")
    _check(all(_finite(value) and value == selected["validation_score"] for value in (
        summary.get("best_validation_score"), best.get("validation_score"),
        best.get("best_validation_score"), last.get("best_validation_score"))), "best validation score binding mismatch")
    _check(last.get("validation_score") == history[-1]["validation_score"], "last checkpoint score/history mismatch")
    for record in (best, last, summary):
        _check(record.get("condition_dim") == 0 and
               record.get("validation_metric") == CONDITIONAL_VALIDATION_METRIC,
               "unconditional model dimension/conditional validation selection mismatch")
        _check(record.get("resolution", record.get("output_dim")) == 96, "native96 resolution required")
        _check(record.get("supported_labels") == list(range(64)), "all 64 conditions must remain supported")
        _check(record.get("derivative_swd_scale") is None, "BCE reference must not fit derivative scale")
    _check(summary.get("target") == "net" and summary.get("units") == "kWh/interval", "signed net-energy target required")
    _check(summary.get("source_sha256") == RELEASE_SHA256, "source release differs from declaration")
    for key in ("data_hashes", "selection", "scaler", "data_curation", "provenance"):
        _check(_same(best.get(key), last.get(key)) and _same(summary.get(key), best.get(key)),
               f"best/last/summary binding mismatch: {key}")
    _check(set(best["data_hashes"]) == set(PREPARED_FILES) | {"curation_manifest.json"},
           "native training must bind exactly the four development files and curation manifest, without TEST hashes")
    provenance = best["provenance"]
    _check(provenance.get("git_commit") == SOURCE_COMMIT and provenance.get("git_commit_tree") == SOURCE_TREE
           and provenance.get("git_dirty") is False
           and provenance.get("source_files_sha256") == TRAINING_SOURCE_HASHES,
           "training must retain clean declared 1ca62d0 source/tree/file fingerprints")
    selected_rows = best["selection"]
    _check(selected_rows.get("train_profiles_available") == selected_rows.get("train_profiles_used") == TRAIN_PROFILES
           and selected_rows.get("train_indices_sha256") == _digest(np.arange(TRAIN_PROFILES), "<i8"),
           "reference must use the complete ordered canonical TRAIN rows")
    _check(selected_rows.get("validation_selection") == "conditional_w1"
           and selected_rows.get("validation_samples_requested") == 4096
           and selected_rows.get("batch_sampling") == "balanced_conditions"
           and selected_rows.get("swd_conditioning") == "pooled", "training selection policy mismatch")
    totals = dict.fromkeys(map(str, range(64)), 0)
    best_totals = None
    for row in history:
        epoch = row["epoch"]
        _check(all(_finite(row.get(key)) for key in ("loss_g", "loss_d", "validation_score")), "nonfinite training history")
        _check(row.get("generator_updates") == row.get("discriminator_updates") == UPDATES_PER_EPOCH,
               "per-epoch adversary update budget differs")
        _check(row.get("samples_seen") == UPDATES_PER_EPOCH * 512 and
               row.get("loss_swd", 0.) == row.get("loss_derivative_swd", 0.) == 0.,
               "reference exposure/loss history differs")
        exposure = row.get("class_exposure_counts", {})
        _check(set(exposure) == set(totals) and all(type(v) is int and v > 0 and v % 128 == 0 for v in exposure.values())
               and sum(exposure.values()) == row["samples_seen"], "invalid balanced condition exposure accounting")
        for label, count in exposure.items():
            totals[label] += count
        if epoch == best_epoch:
            best_totals = dict(totals)
    for record, epoch, exposure in ((best, best_epoch, best_totals), (last, EPOCHS, totals), (summary, EPOCHS, totals)):
        _check(record.get("generator_updates") == record.get("discriminator_updates") == epoch * UPDATES_PER_EPOCH
               and record.get("samples_seen") == epoch * UPDATES_PER_EPOCH * 512
               and record.get("class_exposure_counts") == exposure, "checkpoint/summary cumulative update/exposure mismatch")
    _check(summary.get("last_epoch_metrics") == last.get("epoch_metrics") == history[-1]
           and best.get("epoch_metrics") == history[best_epoch - 1], "checkpoint epoch metrics/history mismatch")
    return {"training_seed": training_seed, "epochs_completed": EPOCHS,
            "generator_updates": UPDATES, "discriminator_updates": UPDATES,
            "best_epoch": best_epoch, "selected_checkpoint_updates": best_epoch * UPDATES_PER_EPOCH,
            "train_profiles": TRAIN_PROFILES, "full_train": True,
            "training_source_commit": SOURCE_COMMIT, "training_source_tree": SOURCE_TREE}


def _load_run(checkpoint, train, val, metadata, scaler, hashes, guard, training_seed):
    checkpoint = Path(checkpoint)
    _check(checkpoint.name == "best.pt", "input must be the declared selected best.pt")
    directory = checkpoint.parent
    paths = {name: directory / name for name in RUN_FILES}
    artifacts = {name: sha256_file(path) for name, path in paths.items()}
    config, _ = _json(paths["config.json"])
    summary, _ = _json(paths["summary.json"])
    history, _ = _json(paths["history.json"])
    primary, _ = _json(paths["validation_report.json"])
    bound_hashes = {**hashes, "curation_manifest.json": guard["manifest_sha256"]}
    details, indices = _checkpoint(checkpoint, train, metadata, scaler, bound_hashes)
    _checkpoint(paths["last.pt"], train, metadata, scaler, bound_hashes)
    best = torch.load(checkpoint, map_location="cpu", weights_only=True)
    last = torch.load(paths["last.pt"], map_location="cpu", weights_only=True)
    evidence = _verify_run_contract(config, summary, history, best, last,
                                    training_seed=training_seed, train_count=len(train["x"]))
    _check(best["data_curation"] == guard, "checkpoint curation verification differs from actual prepared lineage")
    selected = _validation_indices(val["y"], list(range(64)), 4096,
                                   np.random.default_rng(training_seed + 1), "conditional_w1")
    recorded = best["selection"]
    _check(recorded.get("validation_indices") == selected.tolist()
           and recorded.get("validation_labels_sha256") == _digest(val["y"][selected], "<i8")
           and recorded.get("validation_profiles_available") == recorded.get("validation_profiles_eligible") == len(val["x"])
           and recorded.get("validation_profiles_used") == len(selected)
           and recorded.get("validation_unsupported_profiles") == 0,
           "checkpoint selection does not bind the actual validation rows")
    noise = torch.randn((len(selected), 24), generator=torch.Generator().manual_seed(training_seed + 2))
    _check(recorded.get("validation_noise_sha256") == _digest(noise.numpy(), "<f4"), "selection noise fingerprint mismatch")
    _check(primary.get("split") == "val" and primary.get("test_used", False) is False
           and primary.get("seed") == 2026 and primary.get("samples_per_condition_cap") == 100
           and primary.get("checkpoint_sha256") == artifacts["best.pt"]
           and primary.get("train_sha256") == hashes["train.npz"]
           and primary.get("evaluation_sha256") == hashes["val.npz"]
           and primary.get("units") == "kWh/interval" and primary.get("target") == "net"
           and isinstance(primary.get("methods"), dict)
           and {"cgan", "conditional_mean", "conditional_resampling"} <= set(primary["methods"])
           and primary.get("training_profiles") == {"prepared": TRAIN_PROFILES, "model": TRAIN_PROFILES, "baselines": TRAIN_PROFILES},
           "original primary validation report is absent, incompatible, or not bound to this completed run")
    evidence["primary_raw_report"] = {
        "filename": "validation_report.json", "sha256": artifacts["validation_report.json"],
        "evaluation_seed": 2026, "samples_per_condition_cap": 100,
        "declared_native_evaluator_source_commit": SOURCE_COMMIT,
        "attribution_scope": "Protocol-declared pin; legacy raw report lacks its own evaluator source fingerprints.",
        "superseded": False,
    }
    _check(all(sha256_file(paths[name]) == digest for name, digest in artifacts.items()),
           "run artifacts changed during verification")
    return details, indices, evidence, artifacts


def _text_digest(values):
    digest = hashlib.sha256()
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def _source_fingerprints():
    paths = [*sorted((ROOT / "src" / "fluvius_cgan").glob("*.py")),
             ROOT / "scripts" / "compare_checkpoints.py", ROOT / "scripts" / "evaluate_calibration.py",
             Path(__file__)]
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in paths}


def evaluate(prepared, checkpoint, *, training_seed, samples_per_condition=100, seed=2026):
    """Raw-only native-quarter-hour diagnostics; no calibration or TEST option."""
    expected_config(training_seed)
    _check(type(samples_per_condition) is int and samples_per_condition in (100, 1000), "cap must be primary100 or precision1000")
    _check(type(seed) is int and seed == EVALUATION_SEED, "evaluation seed is fixed at 2026")
    prepared, checkpoint = Path(prepared).resolve(), Path(checkpoint).resolve()
    paths = [prepared / name for name in (*PREPARED_FILES, "curation_manifest.json")]
    paths += [checkpoint.parent / name for name in RUN_FILES]
    _preflight(prepared, paths)
    started = time.perf_counter()
    sources = _source_fingerprints()
    guard = _training_curation_guard(prepared, {"allow_uncurated_data": False})
    _check(guard["status"] == "verified_curation_v2" and guard["pinned_release"] is True,
           "native identity-curated pinned96 data required")
    train, val, metadata, scaler, hashes = _load_prepared(prepared)
    _check(metadata.get("resolution") == 96 and metadata.get("target") == "net"
           and metadata.get("units") == "kWh/interval" and metadata.get("source_sha256") == RELEASE_SHA256
           and "derived" not in metadata, "only native curated96 signed net energy is accepted")
    _check(hashes == guard["verified_file_hashes"], "prepared files changed after curation verification")
    _validate_manifest(metadata, hashes, train, val)
    _check(np.array_equal(np.unique(train["y"]), np.arange(64)) and
           np.array_equal(np.unique(val["y"]), np.arange(64)), "all64 TRAIN and VAL conditions are required")
    details, indices, evidence, artifacts = _load_run(
        checkpoint, train, val, metadata, scaler, hashes, guard, training_seed)
    _check(np.array_equal(indices, np.arange(len(train["x"]))), "reference controls require full ordered TRAIN")
    tx, ty = train["x"], train["y"]
    rng = np.random.default_rng(seed)
    selected = np.concatenate([
        rng.choice(np.flatnonzero(val["y"] == label), min(samples_per_condition, int(np.sum(val["y"] == label))), replace=False)
        for label in range(64)
    ])
    real, labels = val["x"][selected], val["y"][selected]
    meters, dates = val["meter_id"][selected], val["date"][selected]
    loaded_at = time.perf_counter()
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        context, thread_limit = nullcontext(), None
    else:
        context, thread_limit = threadpool_limits(limits=1, user_api="blas"), 1
    metrics, joint, battery, profile_hashes, timings = {}, {}, {}, {}, {}
    real_battery = simulate_battery(real, .25)
    fixed_battery = {"version": 1, "capacity_kwh": 10., "charge_power_kw": 4., "discharge_power_kw": 4.,
                     "eta_charge": float(np.sqrt(.9)), "eta_discharge": float(np.sqrt(.9)),
                     "interval_hours": .25, "episode_hours": 24., "initial_soc_kwh": 0.,
                     "grid_charging_allowed": False, "battery_export_allowed": False}
    _check(all(real_battery["scenario"].get(key) == value for key, value in fixed_battery.items()),
           "battery scenario differs from the declared native96 diagnostic")

    def copula_profiles(shrinkage):
        model = fit_conditional_copula(tx, ty, quantile_knots=QUANTILE_KNOTS, shrinkage=shrinkage)
        return model.generate(labels, seed=seed)

    builders = {
        "raw_unconditional_cgan": lambda: _generate_verified(checkpoint, details, labels, seed),
        "conditional_mean": lambda: mean_baseline(tx, ty, labels),
        "conditional_resampling": lambda: resample_baseline(tx, ty, labels, seed=seed),
        "gaussian_copula": lambda: copula_profiles(.01),
        "independent_marginals": lambda: copula_profiles(1.),
    }
    with context:
        # Retain only one generated method at a time, including at precision cap1000.
        for name, build in builders.items():
            method_started = time.perf_counter()
            values = build()
            profile_hashes[name] = _profile_digest(values)
            metrics[name] = evaluate_conditional(real, labels, values, labels, interval_hours=.25)
            joint[name] = conditional_projected_wasserstein(real, labels, values, labels)
            _check(joint[name]["directions"]["count"] == 128 and joint[name]["directions"]["seed"] == 9182
                   and joint[name]["directions"]["n_features"] == 96, "projection settings differ from declared native96 diagnostic")
            battery[name] = _battery_diagnostics(real_battery, values, labels, .25)
            timings[name] = time.perf_counter() - method_started
            del values
    _check(len({row["directions"]["sha256"] for row in joint.values()}) == 1
           and len({row["scenario_sha256"] for row in battery.values()}) == 1,
           "methods must share identical projection directions and battery scenario")
    _preflight(prepared, paths)
    _check(all(sha256_file(prepared / name) == digest for name, digest in hashes.items())
           and sha256_file(prepared / "curation_manifest.json") == guard["manifest_sha256"],
           "prepared lineage changed during supplemental evaluation")
    _check(all(sha256_file(checkpoint.parent / name) == digest for name, digest in artifacts.items()),
           "run artifacts changed during supplemental evaluation")
    _check(_source_fingerprints() == sources, "evaluation implementation changed during execution")
    provenance = _provenance(torch.device("cpu"))
    provenance.update({"scipy": str(scipy.__version__), "blas_thread_limit": thread_limit,
                       "supplemental_evaluation_source_files_sha256": sources,
                       "evaluation_script_sha256": sources[str(Path(__file__).relative_to(ROOT))]})
    return {
        "schema_version": 1, "report_kind": "supplemental_native96_raw_unconditional_reference",
        "scope": "Exploratory validation diagnostics; no uncertainty interval, final TEST result or checkpoint selection.",
        "supplemental_numerical_version_disclosure": SUPPLEMENTAL_NOTE,
        "split": "val", "test_used": False, "conditional_calibrator_used": False,
        "supersedes_primary_validation_report": False,
        "source_sha256": RELEASE_SHA256, "prepared_data_hashes": hashes,
        "curation_manifest_sha256": guard["manifest_sha256"], "data_curation": guard,
        "checkpoint": details, "completed_training_run": evidence, "run_artifacts_sha256": artifacts,
        "training_seed": training_seed, "seed": seed, "units": "kWh/interval", "target": "net",
        "resolution": 96, "interval_hours": .25,
        "evaluation_role": "primary_cap100_supplement" if samples_per_condition == 100 else "precision_cap1000_supplement",
        "training_profiles": {"prepared": len(tx), "model": len(indices), "baseline": len(tx)},
        "training_reference": {"selection": details["training_selection"], "labels_sha256": _digest(ty, "<i8"),
                               "profiles_sha256": _profile_digest(tx),
                               "condition_counts": {str(label): int(np.sum(ty == label)) for label in range(64)},
                               "fitting_scope": "Complete actual ordered canonical TRAIN; no VAL fitting."},
        "validation_selection": {
            "policy": "Uniform without replacement within each condition in sorted order; seed2026.",
            "samples_per_condition_cap": samples_per_condition, "available_profiles": len(val["x"]),
            "available_meters": len(np.unique(val["meter_id"])), "selected_profiles": len(real),
            "selected_meters": len(np.unique(meters)), "selected_indices_sha256": _digest(selected, "<i8"),
            "conditions": list(range(64)), "condition_counts": {str(label): int(np.sum(labels == label)) for label in range(64)},
        },
        "frozen_samples": {
            "hash_encoding": "Profiles: little-endian float64 C-order; labels/indices: little-endian int64 C-order.",
            "string_hash_encoding": "Ordered UTF-8 strings, each prefixed by its byte length as unsigned little-endian int64.",
            "profile_shape": list(real.shape), "labels_sha256": _digest(labels, "<i8"),
            "meter_ids_sha256": _text_digest(meters), "dates_sha256": _text_digest(dates),
            "real_profiles_sha256": _profile_digest(real), "profiles_sha256": profile_hashes,
            "pairing": "Shared real validation rows and condition strata; generated rows are not paired household days.",
            "postprocessing": "None on raw generator output; no calibration, clipping, ranking or rescaling.",
        },
        "baselines": {
            "conditional_mean": {"reference": "Full actual canonical TRAIN"},
            "conditional_resampling": {"reference": "Full actual canonical TRAIN", "seed": seed},
            "gaussian_copula": {"quantile_knots": 1025, "shrinkage": .01, "seed": seed},
            "independent_marginals": {"quantile_knots": 1025, "shrinkage": 1., "seed": seed},
            "conditioning_note": "Statistical controls use TRAIN labels; the raw generator has condition_dim=0 and ignores supplied evaluation strata.",
        },
        "methods": metrics, "joint_diagnostics": joint, "battery_diagnostics": battery,
        "battery_diagnostics_status": "computed_on_same_frozen_signed_net_profiles",
        "uncertainty": {"computed": False, "reason": "Matched conditioning uncertainty is a separate declared runner."},
        "timings_seconds": {"load_and_verify": loaded_at - started, "per_method": timings,
                            "total_before_serialization": time.perf_counter() - started},
        "provenance": provenance,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--training-seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--samples-per-condition", type=int, choices=(100, 1000), default=100)
    parser.add_argument("--seed", type=int, choices=(2026,), default=2026)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output)
    _check(output.suffix == ".json" and output.name not in RUN_FILES,
           "choose a separately named supplemental JSON, not an existing run artifact name")
    _preflight(Path(args.prepared), [output])
    if os.path.lexists(output):
        raise FileExistsError(f"supplemental output already exists: {output}")
    result = evaluate(args.prepared, args.checkpoint, training_seed=args.training_seed,
                      samples_per_condition=args.samples_per_condition, seed=args.seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"Wrote separately attributed supplemental validation report: {output}")


if __name__ == "__main__":
    main()
