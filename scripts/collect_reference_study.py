"""Collect the complete declared reference study without reading profile arrays.

The input is an explicit JSON inventory; missing arms, seeds or evidence roles
fail closed. Checkpoint weights and calibration metadata are verified, but no
prepared/generated profile array, inference, fitting or TEST content is used.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
from zipfile import ZipFile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from collect_calibrated_study import _details, _preparation_qualification, _same
from compare_reference_uncertainty import _declared_config, _source_contract
from compare_resolutions import AGGREGATION
from evaluate_unconditional_reference import SOURCE_COMMIT, SOURCE_TREE, TRAINING_SOURCE_HASHES
from fluvius_cgan.derived import OPERATION, QUALIFICATION
from fluvius_cgan.models import Generator, Discriminator
from fluvius_cgan.release import RELEASE_SHA256, sha256_file
from fluvius_cgan.training import CHECKPOINT_FORMAT, CONDITIONAL_VALIDATION_METRIC, _config
from fluvius_cgan.utility import _feature_names

ARMS = ("candidate96", "bce96", "unconditional96", "bce24")
SEEDS = (11, 22, 33, 44, 55)
SUPPORT = tuple(map(str, range(64)))
CONTRASTS = ("conditioning", "candidate_reference", "resolution_raw", "resolution_calibrated")
NATIVE_ROLES = ("raw_primary", "primary", "precision", "utility", "utility_precision")
HOURLY_ROLES = ("primary", "precision", "utility", "utility_precision")
PREPARED = ("train.npz", "val.npz", "metadata.json", "scaler.json")
RUN_FILES = ("config.json", "summary.json", "history.json", "best.pt", "last.pt")
TRAIN_ROWS, VAL_ROWS, EPOCHS, UPDATES_EPOCH = 606060, 128492, 26, 1184
METRICS = ("pointwise_wasserstein_kwh", "daily_energy_wasserstein_kwh",
           "peak_power_wasserstein_kw", "ramp_wasserstein_kw", "lag_one_acf_wasserstein")
CORE = ("training.py", "models.py", "residual.py", "distribution_loss.py")


def check(value, message):
    if not value:
        raise ValueError(message)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def read_json(path):
    def invalid(value):
        raise ValueError(f"nonfinite JSON literal: {value}")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            check(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=invalid, object_pairs_hook=unique)


def integer(value, wanted, name):
    check(type(value) is int and value == wanted, f"{name} must be {wanted}")


def numeric_tree(value, path=""):
    """Finite numeric leaves for descriptive seed summaries; no CI aggregation."""
    result = {}
    if isinstance(value, dict):
        for key, item in value.items():
            result.update(numeric_tree(item, f"{path}.{key}" if path else key))
    elif type(value) in (float, int):
        check(math.isfinite(value), f"nonfinite result at {path}")
        result[path] = value
    return result


class Inputs:
    """Preflight every artifact against known profile paths before content access."""
    def __init__(self, base, prepared96, prepared24):
        self.base = Path(base).resolve()
        self.prepared = {96: self.path(prepared96), 24: self.path(prepared24)}
        self.forbidden = [directory / name for directory in self.prepared.values()
                          for name in ("train.npz", "val.npz", "test.npz")]
        self.records = {}

    def path(self, value):
        check(isinstance(value, (str, Path)) and str(value).strip(), "artifact path required")
        path = Path(value)
        return (path if path.is_absolute() else self.base / path).resolve()

    def preflight(self, value):
        path = self.path(value)
        check(path.name not in ("train.npz", "val.npz", "test.npz"), "profile array path forbidden")
        for reserved in self.forbidden:
            check(path != reserved.resolve(), "artifact aliases reserved profile file")
            if path.exists() and reserved.exists():
                check(not os.path.samefile(path, reserved), "artifact hardlinks reserved profile file")
        return path

    def record(self, value):
        path = self.preflight(value)
        check(path.is_file(), f"incomplete reference study: missing {path}")
        key = str(path)
        if key not in self.records:
            self.records[key] = {"path": key, "sha256": sha256_file(path), "bytes": path.stat().st_size}
        return self.records[key]

    def json(self, value):
        record = self.record(value)
        return read_json(record["path"])

    def finish(self):
        for path, record in self.records.items():
            self.preflight(path)
            check(Path(path).stat().st_size == record["bytes"] and sha256_file(path) == record["sha256"],
                  f"artifact changed during collection: {path}")


def inventory(spec):
    check(isinstance(spec, dict) and set(spec) == {"schema_version", "candidate_kind", "prepared96",
          "prepared24", "runs", "development_evidence"}, "invalid reference inventory fields")
    integer(spec["schema_version"], 1, "inventory schema")
    check(spec["candidate_kind"] in ("swd10", "capacity"), "explicit swd10/capacity candidate required")
    check(set(spec["runs"]) == set(ARMS), "exactly four arms required")
    evidence = spec["development_evidence"]
    check(isinstance(evidence, dict) and set(evidence) == {"native", "common_hourly", "paired", "additional"},
          "invalid development evidence coverage")
    check(set(evidence["native"]) == set(ARMS), "native evidence requires every arm")
    seeds = set(map(str, SEEDS))
    slots = []
    for arm in ARMS:
        check(set(spec["runs"][arm]) == seeds and set(evidence["native"][arm]) == seeds,
              f"{arm} requires all five declared seeds")
        for seed in SEEDS:
            entry = evidence["native"][arm][str(seed)]
            check(set(entry) == set(NATIVE_ROLES), f"{arm}/{seed} native evidence roles incomplete")
            slots.extend((f"native.{arm}.{seed}.{role}", path) for role, path in entry.items())
    check(set(evidence["common_hourly"]) == seeds, "common-hourly evidence requires all seeds")
    for seed in SEEDS:
        entry = evidence["common_hourly"][str(seed)]
        check(set(entry) == set(HOURLY_ROLES), f"hourly/{seed} evidence roles incomplete")
        slots.extend((f"common_hourly.{seed}.{role}", path) for role, path in entry.items())
    check(set(evidence["paired"]) == set(CONTRASTS), "all four paired contrasts required")
    for contrast in CONTRASTS:
        check(set(evidence["paired"][contrast]) == seeds, f"{contrast} requires all seeds")
        slots.extend((f"paired.{contrast}.{seed}", path) for seed, path in evidence["paired"][contrast].items())
    check(isinstance(evidence["additional"], list), "additional evidence must be a list")
    slots.extend((f"additional.{i}", path) for i, path in enumerate(evidence["additional"]))
    return slots


def preparation(inputs):
    result = {}
    for width in (96, 24):
        directory = inputs.prepared[width]
        metadata, scaler = (inputs.json(directory / name) for name in ("metadata.json", "scaler.json"))
        check(metadata.get("resolution") == width and metadata.get("target") == "net" and
              metadata.get("units") == "kWh/interval" and metadata.get("source_sha256") == RELEASE_SHA256,
              "prepared metadata resolution/target/source differs")
        hashes = {name: inputs.record(directory / name)["sha256"] for name in ("metadata.json", "scaler.json")}
        for split, count in (("train", TRAIN_ROWS), ("val", VAL_ROWS)):
            row = metadata["splits"][split]
            integer(row["profiles"], count, f"{width}/{split} rows")
            integer(row.get("meters"), 1665 if split == "train" else 353, f"{width}/{split} meters")
            check(isinstance(row.get("meter_ids"), list) and len(row["meter_ids"]) == row["meters"] and
                  len(set(row["meter_ids"])) == row["meters"], "metadata canonical meter roster incomplete")
            check(digest(row["sha256"]), "recorded development hash missing")
            hashes[split + ".npz"] = row["sha256"]
        check(not set(metadata["splits"]["train"]["meter_ids"]) & set(metadata["splits"]["val"]["meter_ids"]),
              "recorded development meter membership overlaps")
        manifest_name = "curation_manifest.json" if width == 96 else "derivation_manifest.json"
        manifest = inputs.json(directory / manifest_name)
        manifest_hash = inputs.record(directory / manifest_name)["sha256"]
        check(all(manifest.get("output_files_sha256", {}).get(name) == value for name, value in hashes.items()),
              "metadata/manifests disagree on development fingerprints")
        result[width] = {"metadata": metadata, "scaler": scaler, "hashes": hashes,
                         "manifest": manifest, "manifest_name": manifest_name, "manifest_sha256": manifest_hash}
    parent, child = result[96], result[24]
    check(parent["metadata"].get("curation", {}).get("fingerprint_version") == 2 and
          parent["metadata"].get("split_seed") == child["metadata"].get("split_seed") == 42,
          "original curated-v2 split metadata required")
    derived = child["metadata"].get("derived", {})
    check(derived.get("schema_version") == 1 and derived.get("operation") == OPERATION and
          derived.get("qualification") == QUALIFICATION and derived.get("test24_materialized") is False and
          "curation" not in child["metadata"], "invalid recorded hourly derivation")
    binding = derived.get("parent", {})
    check(binding.get("development_files_sha256") == parent["hashes"] and
          binding.get("curation_manifest_sha256") == parent["manifest_sha256"] and
          child["manifest"].get("derived") == derived, "hourly parent lineage mismatch")
    check(binding.get("path_policy") == "relative_to_child_prepared_directory" and
          (inputs.prepared[24] / binding.get("prepared_dir", "")).resolve() == inputs.prepared[96],
          "hourly recorded parent directory differs")
    for split in ("train", "val"):
        for field in ("profiles", "meters", "meter_ids"):
            check(parent["metadata"]["splits"][split].get(field) == child["metadata"]["splits"][split].get(field),
                  f"hourly canonical {split} membership differs")
    return result


def training_run(inputs, directory, arm, seed, kind, prep):
    directory = inputs.path(directory)
    files = {name: inputs.record(directory / name) for name in RUN_FILES}
    config, summary, history = (inputs.json(directory / name) for name in RUN_FILES[:3])
    best, last = (torch.load(files[name]["path"], map_location="cpu", weights_only=True) for name in ("best.pt", "last.pt"))
    expected = _declared_config(seed, unconditional=arm == "unconditional96",
                                candidate_kind=kind if arm == "candidate96" else None)
    check(_config(config) == expected and all(_config(s["config"]) == expected for s in (best, last)),
          f"{arm}/{seed}: declared configuration differs")
    width, condition_dim = (24 if arm == "bce24" else 96), (0 if arm == "unconditional96" else 64)
    data = prep[width]
    full_hashes = {**data["hashes"], data["manifest_name"]: data["manifest_sha256"]}
    integer(summary.get("epochs_completed"), EPOCHS, "completed epochs")
    check(summary.get("stopped_early") is False and summary.get("test_used") is False, "incomplete or TEST-used training")
    check(isinstance(history, list) and len(history) == EPOCHS and
          [row.get("epoch") for row in history] == list(range(1, EPOCHS + 1)), "history budget incomplete")
    check(all(finite(row.get("validation_score")) for row in history), "nonfinite checkpoint selection scores")
    minimum = min(history, key=lambda row: row["validation_score"])
    best_epoch = minimum["epoch"]
    totals = dict.fromkeys(SUPPORT, 0)
    prefix = None
    for row in history:
        for key in ("generator_updates", "discriminator_updates"):
            integer(row.get(key), UPDATES_EPOCH, "epoch " + key)
        integer(row.get("samples_seen"), UPDATES_EPOCH * 512, "epoch draws")
        check(all(finite(row.get(key)) for key in ("loss_g", "loss_d", "validation_score")), "nonfinite training loss")
        check(finite(row.get("loss_swd", 0.)), "nonfinite SWD training loss")
        check(row.get("loss_derivative_swd", 0.) == 0., "reference/capacity arm must not use derivative loss")
        if arm != "candidate96":
            check(row.get("loss_swd", 0.) == 0., "BCE arm must not use SWD")
        exposure = row.get("class_exposure_counts", {})
        check(set(exposure) == set(SUPPORT) and all(type(v) is int and v > 0 and v % 128 == 0 for v in exposure.values())
              and sum(exposure.values()) == row["samples_seen"], "invalid balanced exposure accounting")
        totals = {label: totals[label] + exposure[label] for label in SUPPORT}
        if row["epoch"] == best_epoch:
            prefix = dict(totals)
    canonical_indices = hashlib.sha256(np.arange(TRAIN_ROWS, dtype="<i8").tobytes()).hexdigest()
    for state, epoch, exposure in ((best, best_epoch, prefix), (last, EPOCHS, totals), (summary, EPOCHS, totals)):
        check(state.get("data_hashes") == full_hashes, "checkpoint/summary data binding mismatch")
        for key in ("selection", "scaler", "data_curation", "provenance"):
            check(state.get(key) == best.get(key), f"best/last/summary {key} differs")
        integer(state.get("best_epoch"), best_epoch, "best epoch")
        check(state.get("best_validation_score") == minimum["validation_score"], "selected score differs")
        for key in ("generator_updates", "discriminator_updates"):
            integer(state.get(key), epoch * UPDATES_EPOCH, "cumulative " + key)
        integer(state.get("samples_seen"), epoch * UPDATES_EPOCH * 512, "cumulative draws")
        check(state.get("class_exposure_counts") == exposure, "cumulative exposure differs")
        check(state.get("supported_labels") == list(range(64)) and state.get("condition_dim") == condition_dim and
              state.get("resolution", state.get("output_dim")) == width and
              state.get("validation_metric") == CONDITIONAL_VALIDATION_METRIC and state.get("derivative_swd_scale") is None,
              "training resolution/conditioning/selection differs")
    for state, epoch in ((best, best_epoch), (last, EPOCHS)):
        integer(state.get("format_version"), CHECKPOINT_FORMAT, "checkpoint format")
        integer(state.get("epoch"), epoch, "checkpoint epoch")
        check(state.get("metadata") == data["metadata"] and state.get("epoch_metrics") == history[epoch - 1] and
              state.get("validation_score") == history[epoch - 1]["validation_score"], "checkpoint/history/metadata differs")
        for cls, name in ((Generator, "generator"), (Discriminator, "discriminator")):
            model = (cls(expected["latent_dim"], condition_dim, expected["hidden_dim"], width) if name == "generator" else
                     cls(width, condition_dim, expected["discriminator_hidden_dim"], objective=expected["objective"]))
            weights = state.get(name + "_state_dict")
            check(isinstance(weights, dict) and all(torch.is_tensor(v) and torch.isfinite(v).all() for v in weights.values()),
                  "invalid/nonfinite model weights")
            model.load_state_dict(weights, strict=True)
    check(summary.get("last_epoch_metrics") == history[-1] and summary.get("source_sha256") == RELEASE_SHA256 and
          summary.get("target") == "net" and summary.get("units") == "kWh/interval", "summary final state differs")
    selection = best["selection"]
    check(selection.get("train_profiles_available") == selection.get("train_profiles_used") == TRAIN_ROWS and
          selection.get("train_indices_sha256") == canonical_indices and
          selection.get("validation_selection") == "conditional_w1" and
          selection.get("validation_samples_requested") == 4096 and selection.get("batch_sampling") == "balanced_conditions",
          "full canonical TRAIN or selection rule differs")
    check(selection.get("swd_conditioning") == ("pooled" if condition_dim == 0 else "conditional"), "auxiliary conditioning record differs")
    scaler = best.get("scaler", {})
    if expected["normalization"] == "minmax":
        check(scaler == data["scaler"], "reference scaler differs from prepared TRAIN min/max")
    else:
        check(scaler.get("kind") == "standard" and scaler.get("fitted_on") == "train" and
              finite(scaler.get("mean")) and finite(scaler.get("std")) and scaler["std"] > 0 and
              all(scaler.get(key) == data["scaler"].get(key) for key in ("minimum", "maximum")),
              "candidate TRAIN standard scaler record differs")
    if width == 96:
        guard = best.get("data_curation", {})
        check(guard.get("status") == "verified_curation_v2" and guard.get("pinned_release") is True and
              guard.get("metadata_sha256") == data["hashes"]["metadata.json"] and
              guard.get("manifest_sha256") == data["manifest_sha256"], "native checkpoint curation verification missing")
    provenance = best.get("provenance", {})
    check(isinstance(provenance.get("source_files_sha256"), dict) and all(
        digest(provenance["source_files_sha256"].get("src/fluvius_cgan/" + name)) for name in CORE), "missing training source hashes")
    if arm in ("bce96", "unconditional96"):
        check(provenance.get("git_commit") == SOURCE_COMMIT and provenance.get("git_commit_tree") == SOURCE_TREE and
              provenance.get("git_dirty") is False and provenance["source_files_sha256"] == TRAINING_SOURCE_HASHES,
              "native BCE training source differs from declared clean pin")
    result = {"directory": str(directory), "arm": arm, "training_seed": seed, "width": width,
              "config": expected, "best_epoch": best_epoch, "epochs_completed": EPOCHS,
              "generator_updates": EPOCHS * UPDATES_EPOCH, "discriminator_updates": EPOCHS * UPDATES_EPOCH,
              "checkpoint_sha256": files["best.pt"]["sha256"], "last_checkpoint_sha256": files["last.pt"]["sha256"],
              "artifacts": files, "prepared_data_hashes": data["hashes"], "provenance": provenance,
              "training_selection": selection, "state": best, "summary": summary}
    del last
    return result


def map_metadata(inputs, run):
    path = Path(run["directory"]) / "calibration.npz"
    artifact = inputs.record(path)
    # Only the scalar metadata member is decoded. Quantiles/profile arrays are
    # neither loaded nor reconstructed; the entire archive is SHA-bound.
    with ZipFile(path) as archive:
        info = archive.getinfo("metadata.npy")
        check(info.file_size < 2_000_000, "unreasonable calibration metadata member")
        value = np.lib.format.read_array(io.BytesIO(archive.read(info)), allow_pickle=False)
    check(value.shape == () and value.dtype.kind in "US", "invalid calibration metadata scalar")
    meta = json.loads(str(value.item()))
    provenance = meta.get("provenance", {})
    check(meta.get("schema_version") == 1 and meta.get("n_features") == run["width"] and meta.get("quantile_knots") == 1025,
          "calibration metadata width/knots differs")
    check(provenance.get("checkpoint_sha256") == run["checkpoint_sha256"] and
          provenance.get("prepared_data_hashes") == run["prepared_data_hashes"] and
          provenance.get("train_indices_sha256") == run["training_selection"]["train_indices_sha256"] and
          provenance.get("calibration_seed") == 7301 and provenance.get("calibration_samples_per_condition") == 4096 and
          provenance.get("calibration_profiles_total") == 64 * 4096 and provenance.get("quantile_knots") == 1025,
          "saved map protocol/checkpoint/TRAIN binding differs")
    return {**artifact, "metadata": meta}


def validation(report, legacy=False):
    check(report.get("split") == "val" and report.get("test_used", False if legacy else None) is False,
          "evidence must be validation-only")
    check(report.get("units") == "kWh/interval" and report.get("target", "net") == "net", "invalid evidence target/units")


def bound(report, run):
    validation(report)
    checkpoint_details(report["checkpoint"], run, "native evidence")
    check(report.get("prepared_data_hashes") == run["prepared_data_hashes"] and
          report.get("source_sha256") == RELEASE_SHA256, "native evidence data/source differs")


def checkpoint_details(details, run, name):
    state = run["state"]
    _details(details, state, run["checkpoint_sha256"], name)
    check(details.get("training_provenance") == state["provenance"] and details.get("supported_labels") == list(range(64)),
          f"{name}: recorded training provenance/support differs")
    selected = details["training_selection"]
    check(selected.get("training_seed") == run["training_seed"] and selected.get("max_train_samples") is None,
          f"{name}: recorded training selection differs")
    check(details.get("checkpoint_selection") == {key: state[key] for key in
        ("validation_metric", "validation_score", "best_epoch", "best_validation_score")}, f"{name}: recorded checkpoint selection differs")


def selection(report, cap, utility=False, hourly=False):
    row = report["validation_selection"]
    observed_cap = report.get("samples_per_condition") if hourly and not utility else row.get("samples_per_condition_cap")
    integer(observed_cap, cap, "validation cap")
    check(row.get("condition_counts") == {key: cap for key in SUPPORT}, "all 64 conditions and declared counts required")
    check(digest(row.get("selected_indices_sha256")), "selected row fingerprint missing")
    labels = row.get("labels_sha256") if utility else report["frozen_samples"].get("labels_sha256")
    real = row.get("profiles_sha256") if utility else report["frozen_samples"].get("real_hourly_sha256" if hourly else "real_profiles_sha256")
    check(digest(labels) and digest(real), "frozen real/label fingerprints missing")
    expected_labels = hashlib.sha256(np.repeat(np.arange(64, dtype="<i8"), cap).tobytes()).hexdigest()
    check(labels == expected_labels, "frozen labels differ from declared sorted-condition support")
    if hourly:
        indices = row.get("indices", [])
        check(len(indices) == 64 * cap and all(type(i) is int and 0 <= i < VAL_ROWS for i in indices) and
              len(set(indices)) == len(indices), "hourly selected indices are not unique canonical VAL rows")
        check(hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest() == row["selected_indices_sha256"] and
              row.get("labels") == np.repeat(np.arange(64), cap).tolist(), "hourly selection lists/fingerprints differ")
        for key in ("meter_ids", "dates"):
            values = row.get(key, [])
            check(len(values) == len(indices) and all(isinstance(v, str) and v for v in values), "hourly row identities missing")
            if utility:
                check(row.get(key + "_sha256") == hashlib.sha256(json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest(),
                      "hourly row identity fingerprint differs")
        if utility:
            check(row.get("selected_meters") == len(set(row["meter_ids"])), "hourly selected meter count differs")
    return {"indices": row["selected_indices_sha256"], "labels": labels, "real": real}


def physical_methods(methods, width, cap):
    check(isinstance(methods, dict) and methods, "physical methods missing")
    for name, result in methods.items():
        check(set(result.get("per_condition", {})) == set(SUPPORT), f"{name}: condition omitted")
        for row in result["per_condition"].values():
            check(row.get("n_real") == row.get("n_generated") == cap and row.get("n_intervals") == width and
                  row.get("interval_hours") == 24 / width, f"{name}: physical count/units differs")
        for metric in METRICS:
            def value(row):
                return row.get("autocorrelation", {}).get("1", {}).get("wasserstein") if metric == METRICS[-1] else row.get(metric)
            values = [value(row) for row in result["per_condition"].values()]
            check(all(finite(v) for v in values) and finite(value(result["macro"])), f"{name}: invalid {metric}")
            check(math.isclose(statistics.mean(values), value(result["macro"]), rel_tol=1e-10, abs_tol=1e-12),
                  f"{name}: equal-condition macro differs")
        numeric_tree(result)


def provenance(report):
    value = report.get("provenance")
    check(isinstance(value, dict) and isinstance(value.get("source_files_sha256"), dict) and
          digest(value["source_files_sha256"].get("src/fluvius_cgan/evaluation.py")) and
          digest(value["source_files_sha256"].get("src/fluvius_cgan/copula.py")), "evaluation source fingerprints missing")
    return value


def bind_evaluator(groups, report, family, *, current=False):
    """Refuse mixed numerical versions within a declared comparison family."""
    source = provenance(report)
    modules = ["evaluation.py", "models.py", "residual.py", "calibration.py", "copula.py"]
    utility = "utility" in family
    modules += ["utility.py"] if utility else ["joint.py", "battery.py"] if family != "paired" else ["uncertainty.py"]
    if family == "common_utility":
        modules.append("utility_uncertainty.py")
    hashes = source["source_files_sha256"]
    numerical = {"src/fluvius_cgan/" + name: hashes.get("src/fluvius_cgan/" + name) for name in modules}
    check(all(digest(value) for value in numerical.values()), "numerical evaluator fingerprints incomplete")
    if family.startswith("native_physical"):
        runner = {key: source.get(key) for key in ("evaluation_script_sha256", "comparison_helper_sha256")}
    elif family.startswith("native_utility"):
        runner = source.get("source_file_sha256", {})
        check(all(key in runner for key in ("scripts/evaluate_utility.py", "scripts/compare_checkpoints.py")),
              "native utility helper fingerprints missing")
    elif family == "supplemental":
        runner = source.get("supplemental_evaluation_source_files_sha256", {})
        check(all(key in runner for key in ("scripts/evaluate_unconditional_reference.py", "scripts/compare_checkpoints.py",
                                           "scripts/evaluate_calibration.py")), "supplemental helper fingerprints missing")
        check(all(runner.get(key) == value for key, value in numerical.items()), "supplemental package fingerprints conflict")
    elif family == "common_physical":
        runner = {key: source.get(key) for key in ("runner_sha256", "comparison_helper_sha256", "diagnostic_helper_sha256")}
    else:
        helper = source.get("helper_files_sha256", {})
        required = {"scripts/compare_checkpoints.py", "scripts/compare_resolutions.py", "scripts/evaluate_calibration.py"}
        if utility:
            required.add("scripts/evaluate_utility.py")
        check(required <= set(helper), "comparison helper fingerprints missing")
        runner = {"runner_sha256": source.get("runner_sha256"), **helper}
    check(runner and all(digest(value) for value in runner.values()), "evaluator runner/helper fingerprints incomplete")
    signature = {"numerical": numerical, "runner_helpers": runner}
    if family in groups:
        check(groups[family] == signature, f"mixed evaluator source versions in {family}")
    groups[family] = signature
    if current:
        common = groups.setdefault("current_numerical_overlap", {})
        for key, value in numerical.items():
            check(key not in common or common[key] == value, "common-hourly/paired/supplemental numerical versions differ")
            common[key] = value


def supplemental_raw_binding(report, run, raw_artifact, cap):
    records = {name: record["sha256"] for name, record in run["artifacts"].items()}
    records["validation_report.json"] = raw_artifact["sha256"]
    check(report.get("run_artifacts_sha256") == records, "supplemental six-artifact binding differs")
    primary = report.get("completed_training_run", {}).get("primary_raw_report", {})
    check(primary.get("sha256") == raw_artifact["sha256"] and primary.get("filename") == "validation_report.json" and
          primary.get("evaluation_seed") == 2026 and primary.get("samples_per_condition_cap") == 100 and
          primary.get("declared_native_evaluator_source_commit") == SOURCE_COMMIT and primary.get("superseded") is False and
          isinstance(primary.get("attribution_scope"), str) and primary["attribution_scope"], "original raw report attribution missing")
    check(report.get("evaluation_role") == ("primary_cap100_supplement" if cap == 100 else "precision_cap1000_supplement"),
          "supplemental role differs")


def native_physical(report, run, cap, map_record=None):
    bound(report, run)
    integer(report.get("seed"), 2026, "native draw seed")
    check(report.get("resolution") == run["width"] and report.get("interval_hours") == 24 / run["width"], "native resolution differs")
    frozen = selection(report, cap)
    check(report.get("training_profiles") == {"prepared": TRAIN_ROWS, "model": TRAIN_ROWS, "baseline": TRAIN_ROWS},
          "native physical baselines must use full TRAIN")
    physical_methods(report["methods"], run["width"], cap)
    sample_hashes = report["frozen_samples"].get("profiles_sha256", {})
    check(set(sample_hashes) == set(report["methods"]) and all(digest(value) for value in sample_hashes.values()),
          "native method sample fingerprints incomplete")
    source = provenance(report)
    if run["arm"] == "unconditional96":
        check(report.get("report_kind") == "supplemental_native96_raw_unconditional_reference" and
              report.get("conditional_calibrator_used") is False and report.get("supersedes_primary_validation_report") is False,
              "unconditional evidence must be separately attributed raw supplement")
        check("calibration" not in report and "saved_maps" not in report, "unconditional primary cannot contain a conditional map")
        check(set(report["methods"]) == {"raw_unconditional_cgan", "conditional_mean", "conditional_resampling", "gaussian_copula", "independent_marginals"},
              "unconditional method coverage differs or contains a map")
        check(report.get("training_seed") == run["training_seed"] and report.get("curation_manifest_sha256") ==
              run["state"]["data_hashes"]["curation_manifest.json"], "unconditional seed/curation differs")
        for name, record in run["artifacts"].items():
            check(report.get("run_artifacts_sha256", {}).get(name) == record["sha256"], "supplemental run-artifact binding differs")
    else:
        check(set(report["methods"]) == {"raw_cgan", "calibrated_cgan", "gaussian_copula", "independent_marginals", "exact_marginal_copula", "conditional_resampling"},
              "conditional native method coverage differs")
        calibration = report["calibration"]
        check(calibration["saved_model"]["sha256"] == map_record["sha256"] and
              calibration["provenance"] == map_record["metadata"]["provenance"] and
              calibration.get("samples_per_condition") == 4096 and calibration.get("quantile_knots") == 1025 and
              report.get("calibration_seed") == 7301, "native fixed map binding differs")
        check(report.get("seeds") == {"calibration_reference": 7301, "validation_selection": 2026,
              "evaluation_generation": 2026, "baseline_generation": 2026, "nearest_neighbor_subsampling": 2026},
              "native calibration draw seeds differ")
        _preparation_qualification(report, run["state"], run["summary"])
        if run["width"] == 96:
            check(source["source_files_sha256"]["src/fluvius_cgan/copula.py"] == TRAINING_SOURCE_HASHES["src/fluvius_cgan/copula.py"],
                  "native96 calibration must retain declared original copula interpolation")
    for name, shrinkage in (("gaussian_copula", .01), ("independent_marginals", 1.)):
        check(report["baselines"][name]["quantile_knots"] == 1025 and report["baselines"][name]["shrinkage"] == shrinkage,
              "native matched-knot control settings differ")
    diagnostics(report, set(report["methods"]), run["width"])
    return frozen


def diagnostics(report, method_names, width):
    joint, battery = report.get("joint_diagnostics", {}), report.get("battery_diagnostics", {})
    check(set(joint) == set(battery) == method_names, "joint/battery method coverage missing")
    directions, scenario = None, None
    for name in method_names:
        j, b = joint[name], battery[name]
        check(set(j.get("per_condition", {})) == set(b.get("per_condition", {})) == set(SUPPORT), "joint/battery conditions missing")
        check(j.get("directions", {}).get("count") == 128 and j["directions"].get("seed") == 9182,
              "joint projections differ from declaration")
        check(j["directions"].get("n_features") == width and digest(j["directions"].get("sha256")), "joint projection width/hash differs")
        check(b["scenario"].get("interval_hours") == 24 / width and b["scenario"].get("capacity_kwh") == 10. and
              b["scenario"].get("charge_power_kw") == b["scenario"].get("discharge_power_kw") == 4. and
              b["scenario"].get("initial_soc_kwh") == 0., "battery scenario differs from declaration")
        scenario_values = b["scenario"]
        check(all(_same(scenario_values.get(key), value) for key, value in {
            "eta_charge": math.sqrt(.9), "eta_discharge": math.sqrt(.9), "round_trip_efficiency": .9,
            "episode_hours": 24., "grid_charging_allowed": False, "battery_export_allowed": False}.items()),
            "battery efficiency/boundary settings differ")
        if directions is None:
            directions, scenario = j["directions"], b["scenario"]
        check(j["directions"] == directions and b["scenario"] == scenario, "joint directions/battery scenarios differ")
        check(digest(b.get("scenario_sha256")) and b["scenario_sha256"] == hashlib.sha256(
            json.dumps(b["scenario"], sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest(), "battery scenario hash mismatch")
        check(finite(j.get("macro")), "joint macro missing/nonfinite")
        check(_same(j["macro"], statistics.mean(row["mean_projected_wasserstein_kwh"] for row in j["per_condition"].values())),
              "joint macro differs from condition mean")
        for label in SUPPORT:
            expected_count = report["methods"][name]["per_condition"][label]["n_real"]
            for row in (j["per_condition"][label], b["per_condition"][label]):
                check(row.get("n_real") == row.get("n_generated") == expected_count, "joint/battery condition count differs")
        for metric, value in numeric_tree(b["macro"]).items():
            values = [numeric_tree(row) for row in b["per_condition"].values()]
            check(all(metric in row for row in values) and _same(value, statistics.mean(row[metric] for row in values)),
                  "battery macro differs from condition mean")
        numeric_tree(j); numeric_tree(b)


def raw_primary(report, run):
    validation(report, legacy=True)
    check(report.get("checkpoint_sha256") == run["checkpoint_sha256"] and
          report.get("train_sha256") == run["prepared_data_hashes"]["train.npz"] and
          report.get("evaluation_sha256") == run["prepared_data_hashes"]["val.npz"], "legacy raw artifact binding differs")
    integer(report.get("seed"), 2026, "raw primary seed")
    integer(report.get("samples_per_condition_cap"), 100, "raw primary cap")
    check(report.get("training_profiles") == {"prepared": TRAIN_ROWS, "model": TRAIN_ROWS, "baselines": TRAIN_ROWS},
          "raw primary must use full TRAIN")
    check(set(report["methods"]) == {"cgan", "conditional_mean", "conditional_resampling"}, "raw primary methods differ")
    physical_methods(report["methods"], run["width"], 100)


def utility_methods(report, names, width, cap):
    check(set(report.get("methods", {})) == set(names), "utility method coverage differs")
    check(set(report.get("training", {})) == set(names) - {"persistence"}, "utility fitted-method coverage differs")
    task = report["task"]
    required = {"name": "within_day_observed_history_next_interval_linear_ridge", "resolution": width,
                "lags": width // 8, "forecast_horizon_intervals": 1, "forecast_horizon_hours": 24 / width,
                "ridge_alpha": 1., "target_slots_inclusive": [width // 8, width - 1],
                "feature_names": _feature_names(width // 8),
                "ridge_objective": "sum_squared_error_plus_alpha_nonintercept_squared_norm"}
    check(all(task.get(key) == value for key, value in required.items()), "utility task/horizon differs")
    selection_record = report["utility_training_selection"]
    counts = selection_record.get("condition_counts", {})
    check(set(counts) == set(SUPPORT) and all(type(v) is int and v > 0 for v in counts.values()) and
          sum(counts.values()) == TRAIN_ROWS and digest(selection_record.get("labels_sha256")), "full TRAIN label/count binding missing")
    check(selection_record.get("selected_indices_sha256") == hashlib.sha256(np.arange(TRAIN_ROWS, dtype="<i8").tobytes()).hexdigest(),
          "utility training rows must retain full canonical order")
    for name, result in report["methods"].items():
        check(set(result.get("per_condition", {})) == set(SUPPORT), "utility conditions missing")
        for metric in ("rmse_kwh", "mae_kwh"):
            values = [row.get(metric) for row in result["per_condition"].values()]
            check(all(finite(v) and v >= 0 for v in values) and
                  math.isclose(statistics.mean(values), result["macro"][metric], rel_tol=1e-10, abs_tol=1e-12), "utility macro differs")
        for row in result["per_condition"].values():
            check(row.get("n_profiles") == cap and row.get("n_windows") == cap * (width - width // 8), "utility windows/counts differ")
        if name != "persistence":
            fit = report["training"][name]
            check(fit.get("n_profiles") == TRAIN_ROWS and fit.get("resolution") == width and fit.get("lags") == width // 8 and
                  fit.get("n_windows") == TRAIN_ROWS * (width - width // 8) and fit.get("n_features") == width // 8 + 69 and
                  fit.get("ridge_alpha") == 1. and fit.get("ridge_objective") == required["ridge_objective"], "full TRAIN utility fit required")
            check(digest(fit.get("profiles_sha256")) and digest(fit.get("labels_sha256")), "utility training hashes missing")
            check(fit["labels_sha256"] == selection_record["labels_sha256"] and fit.get("condition_counts") == counts,
                  "utility fitted labels/counts differ from full TRAIN selection")
    check(selection_record.get("cap_applied") is False and selection_record.get("selected_profiles") == TRAIN_ROWS,
          "capped downstream training forbidden")


def native_utility(report, run, cap, physical_frozen, map_record):
    bound(report, run)
    check(report["config"].get("seed") == 2026 and report["config"].get("samples_per_condition") == cap and
          report["config"].get("max_train_profiles") is None and report["config"].get("copula") is True, "native utility settings differ")
    check(report.get("seeds") == {"validation_selection": 2026, "utility_training_cap": 2126,
          "cgan_generation": 2026, "conditional_resampling": 2026, "copula_generation": 2026,
          "calibration_reference": None if run["arm"] == "unconditional96" else 7301}, "native utility draw seeds differ")
    check(selection(report, cap, utility=True) == physical_frozen, "native utility real rows differ from physical report")
    names = {"training_real", "training_cgan", "training_resampling", "training_copula", "persistence"}
    if run["arm"] != "unconditional96":
        names.add("training_calibrated_cgan")
        check(report["calibration"].get("sha256") == map_record["sha256"] and
              report["calibration"].get("provenance") == map_record["metadata"]["provenance"], "utility map differs")
    else:
        check(report["calibration"].get("enabled") is False and report["config"].get("calibrator") is None,
              "unconditional utility must remain raw-only")
    utility_methods(report, names, run["width"], cap)
    pool = report["source_training_pool"]
    check(pool.get("actual_gan_profiles") == pool.get("prepared_profiles") == TRAIN_ROWS and
          pool.get("actual_indices_sha256") == run["training_selection"]["train_indices_sha256"], "utility source pool differs")
    check(pool.get("labels_sha256") == report["utility_training_selection"]["labels_sha256"] and
          pool.get("condition_counts") == report["utility_training_selection"]["condition_counts"] and
          pool.get("profiles_sha256") == report["training"]["training_real"]["profiles_sha256"], "native utility source pool/real fit differs")
    knots = None if run["arm"] == "unconditional96" else 1025
    check(report["copula"].get("enabled") is True and report["copula"].get("quantile_knots") == knots and
          report["copula"].get("shrinkage") == .01, "native utility Gaussian marginal approximation differs")
    source = provenance(report)
    if run["width"] == 96:
        check(source["source_files_sha256"]["src/fluvius_cgan/copula.py"] == TRAINING_SOURCE_HASHES["src/fluvius_cgan/copula.py"],
              "native96 utility interpolation differs from declared pin")
    return {"gaussian_quantile_knots": knots, "copula_source_sha256": source["source_files_sha256"]["src/fluvius_cgan/copula.py"],
            "shared_gaussian_across_arms": False,
            "scope": "Unconditional native utility uses every sorted TRAIN observation; calibrated arms use 1025 knots. Keep separately labeled."}


def fit_signature(report, name):
    fit = report["training"][name]
    keys = ("profiles_sha256", "labels_sha256", "coefficients_sha256", "n_profiles", "n_windows",
            "n_features", "resolution", "lags", "ridge_alpha", "condition_counts")
    check(all(key in fit for key in keys), "utility fit identity incomplete")
    check(all(digest(fit[key]) for key in keys[:3]), "invalid utility fit hashes")
    return {key: fit[key] for key in keys}


def fixed_fits(primary, precision):
    check(set(primary["training"]) == set(precision["training"]), "utility precision training methods differ")
    for name in primary["training"]:
        check(fit_signature(primary, name) == fit_signature(precision, name),
              f"utility cap changed fitted TRAIN/generation/coefficients: {name}")


def common_binding(report, runs, prep):
    validation(report)
    check(report.get("source_sha256") == RELEASE_SHA256 and report.get("prepared_data_hashes") ==
          {"parent96": prep[96]["hashes"], "derived24": prep[24]["hashes"]}, "common-hourly preparation binding differs")
    check(report.get("derived_provenance") == prep[24]["metadata"]["derived"], "common-hourly lineage differs")
    manifests = report["preparation_manifests"]
    check(manifests["parent_curation_manifest"]["sha256"] == prep[96]["manifest_sha256"] and
          manifests["hourly_derivation_manifest"]["sha256"] == prep[24]["manifest_sha256"], "common-hourly manifest differs")
    check(set(report["models"]) == {"candidate96", "bce96", "bce24"} and
          set(report["saved_maps"]) == set(report["models"]), "common-hourly must include all three models and maps")
    for arm, details in report["models"].items():
        checkpoint_details(details, runs[arm], "common-hourly " + arm)
        check(report["saved_maps"][arm]["sha256"] == runs[arm]["map"]["sha256"] and
              report["saved_maps"][arm]["provenance"] == runs[arm]["map"]["metadata"]["provenance"], "hourly saved map differs")
    for split, count in (("train", TRAIN_ROWS), ("val", VAL_ROWS)):
        row = report["row_bindings"][split]
        check(row.get("profiles") == count and row.get("same_order_and_values") is True and row.get("hourly_values_verified") is True,
              "common-hourly row identity verification missing")
        check(all(digest(row.get(key)) for key in ("labels_sha256", "meter_ids_sha256", "dates_sha256")), "hourly row fingerprints missing")
    provenance(report)


def hourly_pipelines(report, *, utility=False):
    check(report.get("aggregation") == AGGREGATION, "hourly aggregation/order differs")
    pipelines = report.get("pipelines", {})
    for arm in ("candidate96", "bce96", "bce24"):
        width = 24 if arm == "bce24" else 96
        for mode in ("raw", "calibrated"):
            row = pipelines.get(arm + "_" + mode, {})
            mapped = mode == "calibrated"
            check(row.get("native_resolution") == width and row.get("evaluated_resolution") == 24 and
                  row.get("saved_map") is mapped and row.get("aggregation") == (AGGREGATION if width == 96 else None),
                  "native map/aggregation pipeline differs")
            check(row.get("training_dtype" if utility else "evaluated_dtype") == ("float64" if width == 24 and mapped else "float32"),
                  "hourly pipeline output dtype differs")
            if mapped:
                order = ("native generator -> saved native map -> hourly sum" if width == 96 else
                         "hourly generator -> saved hourly map (float64 retained)" if utility else
                         "hourly generator -> saved hourly map (float64 map output retained)")
                check(row.get("order") == order, "calibration must precede hourly sum")
            if utility:
                keys = ("native_raw_profiles_sha256", "native_mapped_profiles_sha256") if mapped else ("native_profiles_sha256",)
                check(all(digest(row.get(key)) for key in keys), "native utility generation fingerprints missing")
                if mapped:
                    check(row["native_raw_profiles_sha256"] == pipelines[arm + "_raw"]["native_profiles_sha256"],
                          "mapped utility did not use the same raw generation")
    for name, shrinkage in (("gaussian_copula24", .01),) if utility else (("gaussian_copula24", .01), ("independent_marginals24", 1.)):
        row = pipelines.get(name, {})
        check(row.get("fitted_on") == "full TRAIN24" and row.get("seed") == 2026 and
              row.get("quantile_knots") == 1025 and row.get("shrinkage") == shrinkage, "hourly Gaussian/independent settings differ")


def common_physical(report, runs, prep, cap, native96_frozen, native24_frozen):
    common_binding(report, runs, prep)
    hourly_pipelines(report)
    integer(report.get("seed"), 2026, "common-hourly draw seed")
    check(report.get("resolution") == 24 and report.get("interval_hours") == 1., "common-hourly physical units differ")
    frozen = selection(report, cap, hourly=True)
    check(frozen == native24_frozen and all(frozen[k] == native96_frozen[k] for k in ("indices", "labels")),
          "common-hourly real rows do not match native evidence")
    expected = {f"{arm}_{mode}" for arm in ("candidate96", "bce96", "bce24") for mode in ("raw", "calibrated")}
    expected |= {"conditional_mean24", "conditional_resampling24", "gaussian_copula24", "independent_marginals24"}
    check(set(report["methods"]) == expected, "common-hourly physical method coverage differs")
    check(set(report["frozen_samples"].get("profiles_sha256", {})) == expected and
          all(digest(value) for value in report["frozen_samples"]["profiles_sha256"].values()), "hourly method sample fingerprints incomplete")
    physical_methods(report["methods"], 24, cap)
    diagnostics(report, expected, 24)
    check(report["training_profiles"] == {"parent96": TRAIN_ROWS, "derived24": TRAIN_ROWS,
          "controls": TRAIN_ROWS, "full_actual_train_required": True}, "common-hourly controls did not use full TRAIN")
    return frozen


def interval_contract(value, metrics, point_values, *, utility=False):
    integer(value.get("n_bootstrap"), 500, "bootstrap replicates")
    check(value.get("confidence") == .95 and value.get("seed") == (42027 if utility else 2026), "bootstrap confidence/seed differs")
    valid, invalid = value.get("n_valid_bootstrap"), value.get("n_invalid_bootstrap")
    check(type(valid) is int and type(invalid) is int and valid >= 0 and invalid >= 0 and valid + invalid == 500,
          "bootstrap draw accounting incomplete")
    complete = invalid == 0
    expected_status = "complete" if complete else ("withheld_missing_conditions" if utility else "unavailable_missing_condition_draws")
    check(value.get("interval_status") == expected_status, "bootstrap withheld-status accounting differs")
    first_name = "candidate" if utility else "generated"
    check(set(value.get("intervals", {})) == set(metrics), "bootstrap metric coverage differs")
    for i, name in enumerate((first_name, "comparator")):
        for metric in metrics:
            check(_same(value["point_estimates"][name][metric], point_values[i][metric]), "bootstrap point estimate differs from report metric")
    for metric, row in value["intervals"].items():
        check(_same(row.get("estimate"), point_values[0][metric] - point_values[1][metric]), "paired difference sign/value differs")
        if complete:
            check(finite(row.get("lower")) and finite(row.get("upper")) and row["lower"] <= row["upper"], "invalid interval endpoints")
        else:
            check(row.get("lower") is None and row.get("upper") is None, "invalid draws must withhold all endpoints")
    return complete


def bootstrap_design(value, selection_record, *, utility=False):
    check(value.get("condition_weighting") == "equal_over_original_observed_support" and
          value.get("resampling_scheme") == ("within_technology_real_meter" if utility else "within_stratum_real_meter"),
          "bootstrap must resample whole meters within technology")
    counts = value.get("stratum_meter_counts", {})
    check(set(counts) == set(map(str, range(8))) and all(type(x) is int and x > 0 for x in counts.values()) and
          sum(counts.values()) == value.get("n_meters") == selection_record.get("selected_meters"), "bootstrap meter/stratum accounting differs")
    missing = value.get("missing_condition_draws" if utility else "missing_condition_draw_counts", {})
    check(set(missing) == set(SUPPORT) and all(type(x) is int and 0 <= x <= value["n_invalid_bootstrap"] for x in missing.values()),
          "missing-condition bootstrap accounting differs")
    check((max(missing.values()) == 0) == (value["n_invalid_bootstrap"] == 0) and
          sum(missing.values()) >= value["n_invalid_bootstrap"], "missing-condition draw counts inconsistent")
    if utility:
        check(value.get("condition_counts") == selection_record["condition_counts"] and
              value.get("n_profiles") == sum(selection_record["condition_counts"].values()) and
              value.get("windows_per_profile") == 21, "paired utility row/window accounting differs")
    else:
        check(value.get("resampling_unit") == "real_meter" and value.get("generated_sample_policy") ==
              value.get("comparator_sample_policy") == "fixed_condition_reference", "physical bootstrap reference policy differs")


def common_utility(report, runs, prep, cap, frozen):
    common_binding(report, runs, prep)
    hourly_pipelines(report, utility=True)
    check(report.get("seeds") == {"training": runs["bce96"]["training_seed"], "validation_selection": 2026,
          "full_train_generation": 2026, "conditional_resampling": 2026, "gaussian_copula": 2026,
          "calibration_reference": 7301, "meter_bootstrap": 42027}, "common-hourly utility seed differs")
    check(selection(report, cap, utility=True, hourly=True) == frozen, "hourly utility and physical real rows differ")
    neural = {f"{arm}_{mode}" for arm in ("candidate96", "bce96", "bce24") for mode in ("raw", "calibrated")}
    names = neural | {"training_real24", "conditional_resampling24", "gaussian_copula24", "persistence"}
    utility_methods(report, names, 24, cap)
    check(report["utility_training_selection"]["labels_sha256"] == report["row_bindings"]["train"]["labels_sha256"] and
          report["utility_training_selection"]["real_profiles_sha256"] == report["training"]["training_real24"]["profiles_sha256"],
          "common-hourly utility training identity differs")
    expected_pairs = [("bce24_raw", "bce96_raw"), ("bce24_calibrated", "bce96_calibrated")]
    expected_pairs += [(name, control) for name in sorted(neural) if name.endswith("calibrated")
                       for control in ("training_real24", "conditional_resampling24")]
    check(set(report.get("paired_uncertainty", {})) == {a + "_minus_" + b for a, b in expected_pairs},
          "common-hourly paired utility coverage incomplete")
    for a, b in expected_pairs:
        pair = report["paired_uncertainty"][a + "_minus_" + b]
        check(pair.get("candidate") == a and pair.get("comparator") == b and
              digest(pair.get("candidate_errors_sha256")) and digest(pair.get("comparator_errors_sha256")) and
              pair.get("candidate_errors_sha256") == report["training"][a]["validation_errors_sha256"] and
              pair.get("comparator_errors_sha256") == report["training"][b]["validation_errors_sha256"], "utility pairing identity differs")
        interval_contract(pair["bootstrap"], ("rmse_kwh", "mae_kwh"),
                          [report["methods"][name]["macro"] for name in (a, b)], utility=True)
        bootstrap = pair["bootstrap"]
        bootstrap_design(bootstrap, report["validation_selection"], utility=True)
        check(set(bootstrap.get("per_condition_point_estimates", {})) == set(SUPPORT), "paired utility condition points missing")
        for label, points in bootstrap["per_condition_point_estimates"].items():
            for role, name in (("candidate", a), ("comparator", b)):
                check(set(points.get(role, {})) == {"rmse_kwh", "mae_kwh"} and all(
                    _same(points[role][metric], report["methods"][name]["per_condition"][label][metric])
                    for metric in ("rmse_kwh", "mae_kwh")), "paired utility condition point differs")


def compact_physical(row):
    return {key: (row["autocorrelation"]["1"]["wasserstein"] if key == METRICS[-1] else row[key]) for key in METRICS}


def paired_physical(report, contrast, runs, prep, kind, native_frozen, hourly_frozen):
    validation(report)
    check(report.get("contrast") == contrast and report.get("training_seed") == runs["bce96"]["training_seed"] and
          report.get("candidate_kind") == (kind if contrast == "candidate_reference" else None), "paired contrast/seed/kind differs")
    arms = (("bce96", "unconditional96") if contrast == "conditioning" else
            ("candidate96", "bce96") if contrast == "candidate_reference" else ("bce24", "bce96"))
    hourly = contrast.startswith("resolution_")
    calibrated = contrast in ("candidate_reference", "resolution_calibrated")
    width = 24 if hourly else 96
    check(report.get("resolution") == width and report.get("interval_hours") == 24 / width, "paired resolution differs")
    check(report.get("prepared_data_hashes") == {"parent96": prep[96]["hashes"], **({"derived24": prep[24]["hashes"]} if hourly else {})},
          "paired preparation fingerprints differ")
    for name, arm in zip(("candidate", "comparator"), arms):
        checkpoint_details(report[name], runs[arm], "paired " + name)
        if calibrated:
            check(report["saved_maps"][name]["sha256"] == runs[arm]["map"]["sha256"] and
                  report["saved_maps"][name]["provenance"] == runs[arm]["map"]["metadata"]["provenance"], "paired map differs")
    if not calibrated:
        check(report.get("saved_maps") == {}, "raw contrast must not use maps")
    check(report.get("source_sha256") == RELEASE_SHA256 and report.get("training_subsets_match") is True and
          report.get("canonical_full_train_selection_verified") is True and
          report.get("source_binding") == _source_contract(contrast, report["candidate"], report["comparator"]),
          "paired training-source attribution differs")
    manifests = report.get("preparation_manifests", {})
    check(manifests.get("parent_curation_manifest", {}).get("sha256") == prep[96]["manifest_sha256"], "paired parent manifest differs")
    if hourly:
        check(manifests.get("hourly_derivation_manifest", {}).get("sha256") == prep[24]["manifest_sha256"] and
              report.get("derived_provenance") == prep[24]["metadata"]["derived"], "paired hourly lineage differs")
        for split, count in (("train", TRAIN_ROWS), ("val", VAL_ROWS)):
            row = report["row_bindings"][split]
            check(row.get("profiles") == count and row.get("same_order_and_values") is True and row.get("hourly_values_verified") is True,
                  "paired hourly row verification missing")
    else:
        check(report.get("derived_provenance") is None, "native contrast cannot claim hourly lineage")
    check(set(report.get("pipelines", {})) == {"candidate", "comparator"}, "paired pipeline roles missing")
    for name, arm in zip(("candidate", "comparator"), arms):
        native_width = runs[arm]["width"]
        aggregate = hourly and native_width == 96
        pipeline = report["pipelines"][name]
        order = "native generator -> " + ("fixed native map -> " if calibrated else "") + ("hourly sum" if aggregate else "evaluation")
        check(pipeline.get("native_resolution") == native_width and pipeline.get("evaluated_resolution") == width and
              pipeline.get("saved_map") is calibrated and pipeline.get("aggregation") == (AGGREGATION if aggregate else None) and
              pipeline.get("order") == order, "paired fixed-map/aggregation order differs")
    report_frozen = selection(report, 100)
    check(report_frozen == (hourly_frozen if hourly else native_frozen), "paired selected real rows differ")
    check(report.get("seeds", {}).get("validation_selection") == report["seeds"].get("generation_both_methods") == 2026 and
          report.get("replicates") == 500 and report.get("metrics") == list(METRICS), "paired settings differ")
    physical_methods(report["methods"], width, 100)
    check(set(report["methods"]) == {"candidate", "comparator"}, "paired method coverage differs")
    bootstrap = report["paired_comparison"]
    point_values = [compact_physical(report["methods"][name]["macro"]) for name in ("candidate", "comparator")]
    interval_contract(bootstrap, METRICS, point_values)
    bootstrap_design(bootstrap, report["validation_selection"])
    check(bootstrap.get("n_conditions") == 64 and bootstrap.get("condition_support") == list(range(64)) and
          set(bootstrap.get("per_condition", {})) == set(SUPPORT), "paired uncertainty conditions missing")
    for label, condition in bootstrap["per_condition"].items():
        points = [compact_physical(report["methods"][name]["per_condition"][label]) for name in ("candidate", "comparator")]
        for method, expected in zip(("generated", "comparator"), points):
            check(_same(condition["point_estimates"][method], expected), "paired condition point estimates differ")
        check(set(condition.get("intervals", {})) == set(METRICS), "paired condition interval metrics missing")
        for metric, ci in condition["intervals"].items():
            check(_same(ci["estimate"], points[0][metric] - points[1][metric]), "paired condition difference differs")
            if bootstrap["n_invalid_bootstrap"]:
                check(ci["lower"] is None and ci["upper"] is None, "condition intervals must be withheld")
            else:
                check(finite(ci["lower"]) and finite(ci["upper"]) and ci["lower"] <= ci["upper"], "invalid condition interval")
    provenance(report)


def endpoint(report, artifact):
    fields = ("methods", "joint_diagnostics", "battery_diagnostics", "paired_uncertainty", "paired_comparison",
              "provenance", "validation_selection", "frozen_samples", "training", "task", "copula", "baselines",
              "source_binding", "method_names", "difference_definition", "interpretation", "pipelines",
              "supplemental_numerical_version_disclosure", "uncertainty_policy")
    return {"artifact": artifact, **{key: report[key] for key in fields if key in report}}


def stats(values):
    check(len(values) == 5 and all(finite(x) for x in values), "five finite seed values required")
    return {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values), "minimum": min(values),
            "maximum": max(values), "n_seeds": 5, "values_by_seed": dict(zip(map(str, SEEDS), values))}


def summarize(reports, neural):
    """Summarize endpoints only. CIs remain solely in their individual reports."""
    first = reports[0]
    check(all(set(r["methods"]) == set(first["methods"]) for r in reports), "seed method coverage differs")
    result = {}
    for name in first["methods"]:
        if name not in neural:
            # No repeat count is interpreted as additional baseline replication.
            shared = all(_same(r["methods"][name], first["methods"][name]) for r in reports)
            result[name] = {"shared_reference": shared, "independent_repetitions": 1 if shared else None,
                            "seed_summaries_computed": False,
                            "reports_by_seed": {str(s): r["methods"][name] for s, r in zip(SEEDS, reports)}}
            continue
        vectors = [numeric_tree(r["methods"][name]["macro"]) for r in reports]
        check(all(v.keys() == vectors[0].keys() for v in vectors), "seed macro metric schemas differ")
        per = {}
        for label in SUPPORT:
            rows = [numeric_tree(r["methods"][name]["per_condition"][label]) for r in reports]
            check(all(r.keys() == rows[0].keys() for r in rows), "seed condition metric schemas differ")
            per[label] = {key: stats([row[key] for row in rows]) for key in rows[0]}
        result[name] = {"macro": {key: stats([v[key] for v in vectors]) for key in vectors[0]}, "per_condition": per}
        for diagnostic in ("joint_diagnostics", "battery_diagnostics"):
            if diagnostic in first:
                values = [numeric_tree({"macro": r[diagnostic][name]["macro"],
                                        "per_condition": r[diagnostic][name]["per_condition"]}) for r in reports]
                check(all(v.keys() == values[0].keys() for v in values), "diagnostic metric schema differs")
                result[name][diagnostic] = {key: stats([v[key] for v in values]) for key in values[0]}
    return result


def summarize_paired_points(bootstraps, *, utility=False):
    """Describe paired point differences across seeds, never confidence bounds."""
    check(len(bootstraps) == len(SEEDS), "all five paired seed reports required")
    metrics = ("rmse_kwh", "mae_kwh") if utility else METRICS
    result = {"direction": "candidate minus comparator; negative favors candidate",
              "macro": {metric: stats([row["intervals"][metric]["estimate"] for row in bootstraps]) for metric in metrics},
              "per_condition": {}, "confidence_interval_aggregation": "none"}
    for label in SUPPORT:
        def difference(row, metric):
            if utility:
                values = row["per_condition_point_estimates"][label]
                return values["candidate"][metric] - values["comparator"][metric]
            return row["per_condition"][label]["intervals"][metric]["estimate"]
        result["per_condition"][label] = {metric: stats([difference(row, metric) for row in bootstraps]) for metric in metrics}
    return result


def collect(spec, *, base="."):
    slots = inventory(spec)
    inputs = Inputs(base, spec["prepared96"], spec["prepared24"])
    collector_sources = (Path(__file__), ROOT / "scripts/collect_calibrated_study.py",
                         ROOT / "scripts/compare_reference_uncertainty.py", ROOT / "scripts/compare_resolutions.py",
                         ROOT / "scripts/evaluate_unconditional_reference.py", *sorted((ROOT / "src/fluvius_cgan").glob("*.py")))
    # Check every explicit path before reading even the first report or checkpoint.
    for _, path in slots:
        resolved = inputs.preflight(path)
        check(resolved.suffix == ".json", "evidence entries must be JSON reports, not profile/binary archives")
    for arm in ARMS:
        for seed in SEEDS:
            directory = inputs.path(spec["runs"][arm][str(seed)])
            for name in (*RUN_FILES, *(("calibration.npz",) if arm != "unconditional96" else ())):
                inputs.preflight(directory / name)
    source_records = {str(path.relative_to(ROOT)): inputs.record(path)["sha256"] for path in collector_sources}
    prep = preparation(inputs)
    result = {"schema_version": 1, "kind": "completed_reference_validation_study", "split": "val", "test_used": False,
              "candidate_kind": spec["candidate_kind"], "training_seeds": list(SEEDS), "status": "complete",
              "scope": "Read-only saved-artifact consistency checks; no profile arrays, inference, fitting or TEST content access. No new performance gate.",
              "preparation_binding_scope": "Metadata/checkpoint/report consistency only; source profile bytes and hourly sums are not reaudited by this collector.",
              "preparation": {str(width): value for width, value in prep.items()},
              "native": {arm: {} for arm in ARMS}, "common_hourly": {}, "paired": {name: {} for name in CONTRASTS},
              "runs": {arm: {} for arm in ARMS}, "five_seed_summary": {"native": {}, "common_hourly": {}, "paired": {}}}
    reports_by_arm = {arm: {} for arm in ARMS}
    common_reports, previous_training_sources, controls, evaluators = {}, {}, {}, {}
    paired_reports = {contrast: [] for contrast in CONTRASTS}
    for key, path in slots:
        inputs.record(path)
    for seed in SEEDS:
        runs, physical_frozen = {}, {}
        for arm in ARMS:
            run = training_run(inputs, spec["runs"][arm][str(seed)], arm, seed, spec["candidate_kind"], prep)
            run["map"] = None if arm == "unconditional96" else map_metadata(inputs, run)
            sources = {name: run["provenance"]["source_files_sha256"]["src/fluvius_cgan/" + name] for name in CORE}
            if arm in previous_training_sources:
                check(sources == previous_training_sources[arm], f"{arm} core training source differs across seeds")
            previous_training_sources[arm] = sources
            runs[arm] = run
            native = {role: inputs.json(spec["development_evidence"]["native"][arm][str(seed)][role]) for role in NATIVE_ROLES}
            raw_primary(native["raw_primary"], run)
            raw_artifact = inputs.record(spec["development_evidence"]["native"][arm][str(seed)]["raw_primary"])
            for role, cap in (("primary", 100), ("precision", 1000)):
                if arm == "unconditional96":
                    supplemental_raw_binding(native[role], run, raw_artifact, cap)
                family = "supplemental" if arm == "unconditional96" else "native_physical" + str(run["width"])
                bind_evaluator(evaluators, native[role], family, current=arm == "unconditional96")
                frozen = native_physical(native[role], run, cap, run["map"])
                physical_frozen[(arm, cap)] = frozen
                utility_role = "utility" if cap == 100 else "utility_precision"
                bind_evaluator(evaluators, native[utility_role], "native_utility" + str(run["width"]))
                disclosure = native_utility(native[utility_role], run, cap, frozen, run["map"])
                if cap == 100:
                    raw_name = "raw_unconditional_cgan" if arm == "unconditional96" else "raw_cgan"
                    check(_same(native["raw_primary"]["methods"]["cgan"], native[role]["methods"][raw_name]) and
                          _same(native["raw_primary"]["methods"]["conditional_resampling"], native[role]["methods"]["conditional_resampling"]),
                          "raw primary differs from same-cap native generated/resampling metrics")
                for name in ("training_real", "training_resampling"):
                    signature = fit_signature(native[utility_role], name)
                    key = (run["width"], cap, name)
                    value = (signature, native[utility_role]["methods"][name])
                    if key in controls:
                        check(_same(value, controls[key]), "shared real/resampling utility control differs across arms/seeds")
                    controls[key] = value
            fixed_fits(native["utility"], native["utility_precision"])
            reports_by_arm[arm][seed] = native
            result["native"][arm][str(seed)] = {role: endpoint(report, inputs.record(
                spec["development_evidence"]["native"][arm][str(seed)][role])) for role, report in native.items()}
            result["native"][arm][str(seed)]["utility_reference_disclosure"] = disclosure
            result["runs"][arm][str(seed)] = {key: value for key, value in run.items() if key not in ("state", "summary")}
        for cap in (100, 1000):
            check(physical_frozen[("candidate96", cap)] == physical_frozen[("bce96", cap)] == physical_frozen[("unconditional96", cap)],
                  "native96 arms used different real rows")
        hourly = {role: inputs.json(spec["development_evidence"]["common_hourly"][str(seed)][role]) for role in HOURLY_ROLES}
        hourly_frozen = {}
        for role, cap in (("primary", 100), ("precision", 1000)):
            bind_evaluator(evaluators, hourly[role], "common_physical", current=True)
            frozen = common_physical(hourly[role], runs, prep, cap, physical_frozen[("bce96", cap)], physical_frozen[("bce24", cap)])
            hourly_frozen[cap] = frozen
            utility_role = "utility" if cap == 100 else "utility_precision"
            bind_evaluator(evaluators, hourly[utility_role], "common_utility", current=True)
            common_utility(hourly[utility_role], runs, prep, cap, frozen)
            check(all(reports_by_arm[arm][seed][utility_role]["source_training_pool"]["labels_sha256"] ==
                      hourly[utility_role]["row_bindings"]["train"]["labels_sha256"] for arm in ARMS),
                  "native/hourly utility full TRAIN label order differs")
            for common_name, native_name in (("training_real24", "training_real"), ("conditional_resampling24", "training_resampling")):
                signature = fit_signature(hourly[utility_role], common_name)
                check(_same((signature, hourly[utility_role]["methods"][common_name]), controls[(24, cap, native_name)]),
                      "common-hourly real/resampling utility differs from native24 reference")
        fixed_fits(hourly["utility"], hourly["utility_precision"])
        common_reports[seed] = hourly
        result["common_hourly"][str(seed)] = {role: endpoint(report, inputs.record(
            spec["development_evidence"]["common_hourly"][str(seed)][role])) for role, report in hourly.items()}
        for contrast in CONTRASTS:
            report = inputs.json(spec["development_evidence"]["paired"][contrast][str(seed)])
            bind_evaluator(evaluators, report, "paired", current=True)
            paired_physical(report, contrast, runs, prep, spec["candidate_kind"], physical_frozen[("bce96", 100)], hourly_frozen[100])
            if contrast.startswith("resolution_"):
                check(report["row_bindings"] == hourly["primary"]["row_bindings"], "paired/common-hourly row lineage differs")
                mode = "calibrated" if contrast == "resolution_calibrated" else "raw"
                for role, arm in (("candidate", "bce24"), ("comparator", "bce96")):
                    name = arm + "_" + mode
                    check(report["frozen_samples"][role]["evaluated_profiles_sha256"] ==
                          hourly["primary"]["frozen_samples"]["profiles_sha256"][name] and
                          _same(report["methods"][role], hourly["primary"]["methods"][name]),
                          "paired/common-hourly frozen method differs")
            paired_reports[contrast].append(report["paired_comparison"])
            result["paired"][contrast][str(seed)] = endpoint(report, inputs.record(spec["development_evidence"]["paired"][contrast][str(seed)]))
        # Only descriptive endpoint trees are needed for subsequent seed
        # summaries; release large calibration-diagnostic payloads per seed.
        for arm in ARMS:
            reports_by_arm[arm][seed] = {role: result["native"][arm][str(seed)][role] for role in NATIVE_ROLES}
        del runs
    for arm in ARMS:
        result["five_seed_summary"]["native"][arm] = {}
        for role in NATIVE_ROLES:
            neural = ({"training_cgan", "training_calibrated_cgan"} if "utility" in role else
                      {"cgan"} if role == "raw_primary" else {"raw_unconditional_cgan"} if arm == "unconditional96" else {"raw_cgan", "calibrated_cgan"})
            result["five_seed_summary"]["native"][arm][role] = summarize([reports_by_arm[arm][seed][role] for seed in SEEDS], neural)
    for role in HOURLY_ROLES:
        neural = {f"{arm}_{mode}" for arm in ("candidate96", "bce96", "bce24") for mode in ("raw", "calibrated")}
        result["five_seed_summary"]["common_hourly"][role] = summarize([common_reports[seed][role] for seed in SEEDS], neural)
        if "utility" in role:
            pairs = common_reports[SEEDS[0]][role]["paired_uncertainty"]
            result["five_seed_summary"]["common_hourly"][role]["paired_point_differences"] = {
                name: summarize_paired_points([common_reports[seed][role]["paired_uncertainty"][name]["bootstrap"] for seed in SEEDS], utility=True)
                for name in pairs}
    result["five_seed_summary"]["paired"] = {name: summarize_paired_points(reports) for name, reports in paired_reports.items()}
    evidence = deepcopy(spec["development_evidence"])
    for arm in ARMS:
        for seed in SEEDS:
            evidence["native"][arm][str(seed)] = {role: str(inputs.path(path)) for role, path in evidence["native"][arm][str(seed)].items()}
    for seed in SEEDS:
        evidence["common_hourly"][str(seed)] = {role: str(inputs.path(path)) for role, path in evidence["common_hourly"][str(seed)].items()}
    for contrast in CONTRASTS:
        evidence["paired"][contrast] = {seed: str(inputs.path(path)) for seed, path in evidence["paired"][contrast].items()}
    evidence["additional"] = [str(inputs.path(path)) for path in evidence["additional"]]
    result["development_evidence"] = evidence
    result["evaluator_source_groups"] = evaluators
    result["collector_sources_sha256"] = source_records
    result["aggregation"] = "Five fixed neural seeds, mean/sample SD/range; no profile pooling, no seed selection and no CI endpoint averaging. Repeated controls are not independent repetitions."
    result["limitations"] = ["Reused validation is exploratory; no TEST claim or new performance gate.",
        "Bootstrap intervals remain separate per seed and retain withheld intervals; no five-seed mean CI is constructed.",
        "Native unconditional Gaussian utility uses exact empirical marginals; other calibrated native Gaussian utility uses 1025 knots.",
        "Original native96 copula interpolation and supplemental/common-hourly source versions remain separately attributed.",
        "Legacy raw primary reports lack their own evaluator fingerprints; their source attribution is protocol-declared, not recovered from the report.",
        "Reference conditioning is an ablation within BCE, not within the selected candidate package.",
        "Exact trace/lineage checks do not establish household independence, privacy or external generalization.",
        "A complete explicit inventory cannot prove that unreported external attempts never occurred.",
        "Calibration archive metadata and whole-file hash are checked here; the freezer separately validates quantile arrays before use."]
    inputs.finish()
    result["artifacts"] = inputs.records
    return result


def evidence_index(report, report_path):
    path = Path(report_path).resolve()
    evidence = deepcopy(report["development_evidence"])
    evidence["additional"].append(str(path))
    artifact = {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
    return {"schema_version": 1, "kind": "reference_validation_evidence_index", "candidate_kind": report["candidate_kind"],
            "training_seeds": list(SEEDS), "development_evidence": evidence,
            "artifacts": {**report["artifacts"], str(path): artifact}, "collector_report": artifact,
            "scope": "Coverage and immutable artifact bindings for freezing; not approval, method selection or proof of performance."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence-index", type=Path, required=True)
    args = parser.parse_args(argv)
    for path in (args.output, args.evidence_index):
        check(path.suffix == ".json" and not os.path.lexists(path), "outputs must be new .json paths")
    check(args.output.resolve() != args.evidence_index.resolve(), "report and index output must differ")
    check(args.inventory.suffix == ".json" and args.inventory.resolve().suffix == ".json", "inventory must be a JSON manifest, not a profile archive")
    check(args.inventory.stat().st_nlink == 1, "inventory must not be a hardlinked file")
    snapshot = {"path": str(args.inventory.resolve()), "sha256": sha256_file(args.inventory), "bytes": args.inventory.stat().st_size}
    spec = read_json(args.inventory)
    guard = Inputs(args.inventory.parent, spec["prepared96"], spec["prepared24"])
    guard.preflight(args.inventory.resolve())
    guard.preflight(args.output.resolve()); guard.preflight(args.evidence_index.resolve())
    report = collect(spec, base=args.inventory.parent)
    check(sha256_file(args.inventory) == snapshot["sha256"] and args.inventory.stat().st_size == snapshot["bytes"],
          "inventory changed during collection")
    report["inventory"] = snapshot
    for path, value in ((args.output, report), (args.evidence_index, None)):
        path.parent.mkdir(parents=True, exist_ok=True)
        if value is None:
            value = evidence_index(report, args.output)
        with path.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    print(f"Collected complete validation reference study: {args.output}")


if __name__ == "__main__":
    main()
