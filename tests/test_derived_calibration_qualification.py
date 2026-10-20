"""Hourly inherited-audit qualifications and manifest bindings, synthetic only."""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from fluvius_cgan.calibration import ConditionalMarginalCalibrator, fit
from fluvius_cgan.derived import OPERATION, QUALIFICATION, derive_hourly
from fluvius_cgan.release import RELEASE_SHA256, sha256_file
from fluvius_cgan.training import CHECKPOINT_FORMAT, _config, _load_prepared, _training_curation_guard
from test_derived import curated96, _forbid_test_access
from test_evaluate_calibration import evaluation, synthetic_generated
from test_collect_calibrated_study import collector, make_run, forbid_profile_access
from test_evaluate_utility import evaluation as utility_evaluation
from test_compare_calibrated import comparison


def _digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def _derived_records():
    """Recorded schema for a report-only collector; these paths must never open."""
    hashes = {key: _digest(key) for key in collector.PREPARED_FILES}
    parent = {"prepared_dir": "../synthetic-do-not-open-parent96",
              "path_policy": "relative_to_child_prepared_directory",
              "development_files_sha256": {key: _digest("parent:" + key) for key in hashes},
              "curation_manifest_sha256": _digest("parent curation")}
    derived = {"schema_version": 1, "operation": copy.deepcopy(OPERATION), "parent": parent,
               "test24_materialized": False, "qualification": QUALIFICATION}
    reserved = {"resolution": 96, "access": "metadata_only_not_content_verified",
                "materialized_in_child": False, "split": {"sha256": _digest("reserved TEST metadata")}}
    manifest = _digest("derivation manifest")
    guard = {"status": "verified_hourly_derivation_v1", "pinned_release": True,
             "metadata_sha256": hashes["metadata.json"], "verified_file_hashes": hashes,
             "derived_manifest_sha256": manifest, "parent": parent, "qualification": QUALIFICATION}
    state = {"metadata": {"derived": derived, "reserved_parent_test96": reserved,
                           "source_sha256": RELEASE_SHA256, "resolution": 24, "interval_hours": 1},
             "data_hashes": {**hashes, "derivation_manifest.json": manifest}, "data_curation": guard}
    summary = {"data_curation": copy.deepcopy(guard)}
    report = {"data_curation_status": "verified_hourly_derivation_v1", "curation_metadata": None,
              "derivation_metadata": copy.deepcopy(derived), "derivation_manifest_sha256": manifest,
              "reserved_parent_test96": copy.deepcopy(reserved)}
    return state, summary, report


def _hourly_checkpoint(curated96, tmp_path, monkeypatch):
    _, parent = curated96
    _forbid_test_access(monkeypatch)
    child = tmp_path / "hourly24"
    derive_hourly(parent, child)
    train, _, metadata, scaler, hashes = _load_prepared(child)
    guard = _training_curation_guard(child, {"allow_uncurated_data": False})
    n = len(train["x"])
    state = {"format_version": CHECKPOINT_FORMAT, "epoch": 1,
             "config": _config({"seed": 7}), "metadata": metadata, "scaler": scaler, "output_dim": 24,
             "data_hashes": {**hashes, "derivation_manifest.json": guard["derived_manifest_sha256"]},
             "data_curation": guard, "supported_labels": np.unique(train["y"]).tolist(),
             "selection": {"train_profiles_available": n, "train_profiles_used": n,
                           "train_indices_sha256": hashlib.sha256(np.arange(n, dtype="<i8").tobytes()).hexdigest()}}
    checkpoint = tmp_path / "hourly.pt"
    torch.save(state, checkpoint)
    return child, checkpoint, state, hashes, train


def test_runner_verifies_hourly_data_and_binds_manifest_without_changing_map_hash_contract(
        curated96, tmp_path, monkeypatch):
    child, checkpoint, state, hashes, _ = _hourly_checkpoint(curated96, tmp_path, monkeypatch)
    metadata, guard = state["metadata"], state["data_curation"]
    monkeypatch.setattr(evaluation, "generate", lambda path, labels, seed: synthetic_generated(labels, seed))
    report = evaluation.evaluate(child, checkpoint, calibration_samples_per_condition=8,
                                 samples_per_condition=2, quantile_knots=5,
                                 calibration_output=tmp_path / "map.npz")
    assert report["data_curation_status"] == "verified_hourly_derivation_v1"
    assert report["curation_metadata"] is None
    assert report["derivation_metadata"] == metadata["derived"]
    assert report["reserved_parent_test96"] == metadata["reserved_parent_test96"]
    assert report["derivation_manifest_sha256"] == guard["derived_manifest_sha256"]
    assert "no new hourly identity audit" in report["data_integrity_note"]
    assert set(report["prepared_data_hashes"]) == set(collector.PREPARED_FILES)
    assert report["calibration"]["provenance"]["prepared_data_hashes"] == hashes
    assert report["checkpoint"]["data_hashes"] == state["data_hashes"]
    assert report["test_used"] is False

    # An otherwise matching checkpoint must bind the exact preparation manifest.
    state["data_hashes"]["derivation_manifest.json"] = "0" * 64
    torch.save(state, checkpoint)
    monkeypatch.setattr(evaluation, "generate", lambda *a, **kw: pytest.fail("must reject before generation"))
    with pytest.raises(ValueError, match="derivation_manifest.json"):
        evaluation.evaluate(child, checkpoint, calibration_samples_per_condition=8,
                            samples_per_condition=2, quantile_knots=5)


@pytest.mark.parametrize("runner", ["utility", "comparison"])
def test_hourly_utility_and_uncertainty_bind_manifest_and_retain_four_file_saved_map(
        curated96, tmp_path, monkeypatch, runner):
    child, checkpoint, state, hashes, train = _hourly_checkpoint(curated96, tmp_path, monkeypatch)
    labels = np.repeat(np.unique(train["y"]), 8)
    source = synthetic_generated(labels, 7301)
    provenance = {
        "checkpoint_sha256": sha256_file(checkpoint),
        "train_indices_sha256": state["selection"]["train_indices_sha256"],
        "prepared_data_hashes": hashes, "calibration_seed": 7301,
        "calibration_samples_per_condition": 8, "calibration_profiles_total": len(source),
        "calibration_generated_profiles_sha256": hashlib.sha256(source.astype("<f8").tobytes()).hexdigest(),
        "calibration_labels_sha256": hashlib.sha256(labels.astype("<i8").tobytes()).hexdigest(),
        "generator_inference_batch_size": 1024, "quantile_knots": 5,
    }
    saved_map = tmp_path / "saved-map.npz"
    fit(train["x"], train["y"], source, labels, quantile_knots=5, provenance=provenance).save(saved_map)
    monkeypatch.setattr(utility_evaluation, "_generate_verified",
                        lambda path, details, labels, seed: synthetic_generated(labels, seed))
    monkeypatch.setattr(comparison, "generate", lambda path, labels, seed: synthetic_generated(labels, seed))
    # The toy curation fixture has one meter per technology. Bootstrap arithmetic
    # has its own manual-oracle tests; this test isolates preparation provenance.
    monkeypatch.setattr(comparison, "paired_condition_macro_meter_bootstrap",
                        lambda *a, **kw: {"synthetic_provenance_fixture": True})

    def invoke():
        if runner == "utility":
            return utility_evaluation.evaluate(child, checkpoint, calibrator=saved_map, samples_per_condition=2)
        return comparison.compare(child, checkpoint, saved_map, baseline="conditional_resampling",
                                  samples_per_condition=2, replicates=2)

    report = invoke()
    assert report["data_curation_status"] == "verified_hourly_derivation_v1"
    assert report["derivation_metadata"] == state["metadata"]["derived"]
    assert report["derivation_manifest_sha256"] == state["data_hashes"]["derivation_manifest.json"]
    assert report["reserved_parent_test96"] == state["metadata"]["reserved_parent_test96"]
    assert report["prepared_data_hashes"] == hashes
    assert report["test_used"] is False and report["curation_metadata"] is None
    details = report["checkpoint" if runner == "utility" else "candidate"]
    assert details["data_hashes"] == state["data_hashes"]
    map_details = report["calibration" if runner == "utility" else "calibrator"]
    assert map_details["provenance"]["prepared_data_hashes"] == hashes
    helper_hash = sha256_file(Path(__file__).parents[1] / "scripts/evaluate_calibration.py")
    if runner == "utility":
        assert report["provenance"]["source_file_sha256"]["scripts/evaluate_calibration.py"] == helper_hash
    else:
        assert report["provenance"]["preparation_helper_sha256"] == helper_hash

    # An altered derivation claim fails before saved-map loading or generation.
    state["data_hashes"]["derivation_manifest.json"] = "0" * 64
    torch.save(state, checkpoint)
    with pytest.raises(ValueError, match="derivation_manifest.json"):
        invoke()


def test_shared_manifest_end_check_rejects_changed_bytes(tmp_path):
    path = tmp_path / "derivation_manifest.json"
    path.write_text("original synthetic manifest")
    qualification = {"derivation_manifest_sha256": sha256_file(path)}
    evaluation._check_preparation_unchanged(tmp_path, qualification)
    path.write_text("changed synthetic manifest")
    with pytest.raises(ValueError, match="manifest changed"):
        evaluation._check_preparation_unchanged(tmp_path, qualification)


def test_collector_accepts_recorded_derivation_without_accessing_parent_or_child(monkeypatch):
    state, summary, report = _derived_records()
    monkeypatch.setattr(Path, "open", lambda *a, **kw: pytest.fail("report-only qualification must not open files"))
    collector._preparation_qualification(report, state, summary)


@pytest.mark.parametrize("mutation", [
    lambda s, u, r: r.update(data_curation_status="unverified_no_curation_metadata"),
    lambda s, u, r: r.update(data_curation_status="curation_metadata_present_not_reaudited"),
    lambda s, u, r: r["derivation_metadata"].update(test24_materialized=True),
    lambda s, u, r: r.update(derivation_manifest_sha256="0" * 64),
    lambda s, u, r: r["reserved_parent_test96"].update(access="content_verified"),
    lambda s, u, r: s["data_hashes"].pop("derivation_manifest.json"),
    lambda s, u, r: s["data_hashes"].update(**{"test.npz": "0" * 64}),
    lambda s, u, r: s["metadata"].update(curation={"inherited": True}),
    lambda s, u, r: s["metadata"]["derived"]["operation"].update(new_identity_audit=True),
    lambda s, u, r: s["metadata"]["derived"]["parent"].update(curation_manifest_sha256=None),
    lambda s, u, r: s["metadata"]["derived"]["parent"].update(prepared_dir="/absolute/parent96"),
    lambda s, u, r: s["metadata"]["derived"]["parent"].update(prepared_dir="../parent//nested"),
    lambda s, u, r: s["metadata"]["derived"]["parent"].update(prepared_dir="."),
    lambda s, u, r: s["metadata"]["derived"]["parent"].update(prepared_dir="C:/parent96"),
    lambda s, u, r: s["metadata"]["derived"]["parent"].update(prepared_dir="../unnecessary/../parent96"),
    lambda s, u, r: s["metadata"]["derived"]["parent"].pop("path_policy"),
    lambda s, u, r: s["data_curation"].update(status="verified_curation_v2"),
    lambda s, u, r: s["data_curation"].update(pinned_release=False),
    lambda s, u, r: u["data_curation"].update(derived_manifest_sha256="0" * 64),
    lambda s, u, r: s["data_curation"]["verified_file_hashes"].update(**{"val.npz": "0" * 64}),
])
def test_collector_rejects_changed_or_overstated_inherited_claims(mutation):
    state, summary, report = _derived_records()
    mutation(state, summary, report)
    with pytest.raises(ValueError):
        collector._preparation_qualification(report, state, summary)


@pytest.mark.parametrize("curated", [False, True])
def test_ordinary_preparation_does_not_gain_a_derived_acceptance_path(curated):
    metadata = {"curation": {"schema_version": 1}} if curated else {}
    state = {"metadata": metadata, "data_hashes": {}}
    report = {"curation_metadata": metadata.get("curation"),
              "data_curation_status": "curation_metadata_present_not_reaudited" if curated
                                      else "unverified_no_curation_metadata"}
    collector._preparation_qualification(report, state, {})
    report.update(derivation_metadata={"schema_version": 1}, data_curation_status="verified_hourly_derivation_v1")
    with pytest.raises(ValueError, match="must not claim hourly"):
        collector._preparation_qualification(report, state, {})


def test_stripped_derived_marker_does_not_erase_reserved_lineage():
    state = {"metadata": {"reserved_parent_test96": {}}, "data_hashes": {}}
    report = {"curation_metadata": None, "data_curation_status": "unverified_no_curation_metadata"}
    with pytest.raises(ValueError, match="must not claim hourly"):
        collector._preparation_qualification(report, state, {})


def test_complete_collector_retains_derived_metadata_and_distinct_five_four_hash_layers(make_run, request):
    directory = make_run("hourly_recorded", curated=False, optional=True)
    state = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
    recorded, guard_summary, qualification = _derived_records()
    state["metadata"].update(recorded["metadata"])
    state["data_hashes"] = recorded["data_hashes"]
    state["data_curation"] = recorded["data_curation"]
    torch.save(state, directory / "best.pt")
    checkpoint_hash = sha256_file(directory / "best.pt")
    prepared_hashes = {key: state["data_hashes"][key] for key in collector.PREPARED_FILES}
    model = ConditionalMarginalCalibrator.load(directory / "calibration.npz")
    model.provenance.update(checkpoint_sha256=checkpoint_hash, prepared_data_hashes=prepared_hashes)
    model.save(directory / "calibration.npz")
    map_hash = sha256_file(directory / "calibration.npz")
    for name in ("summary.json", "validation_report.json", "calibration_validation.json",
                 "utility_validation.json", "uncertainty_gaussian.json"):
        path = directory / name
        value = json.loads(path.read_text())
        if name == "summary.json":
            value.update(guard_summary, data_hashes=state["data_hashes"], source_sha256=RELEASE_SHA256)
        elif name == "validation_report.json":
            value.update(checkpoint_sha256=checkpoint_hash, train_sha256=prepared_hashes["train.npz"],
                         evaluation_sha256=prepared_hashes["val.npz"])
        else:
            value.update(qualification, prepared_data_hashes=prepared_hashes, source_sha256=RELEASE_SHA256)
            value["checkpoint"].update(sha256=checkpoint_hash, data_hashes=state["data_hashes"])
            if name == "calibration_validation.json":
                value["calibration"]["provenance"] = model.provenance
                value["calibration"]["saved_model"]["sha256"] = map_hash
            elif name == "utility_validation.json":
                value["calibration"].update(provenance=model.provenance, sha256=map_hash)
            else:
                value["candidate"].update(sha256=checkpoint_hash, data_hashes=state["data_hashes"])
                value["calibrator"].update(provenance=model.provenance, sha256=map_hash)
        path.write_text(json.dumps(value))
    request.getfixturevalue("forbid_profile_access")
    result = collector.collect_run("hourly", directory)
    assert result["data_curation_status"] == "verified_hourly_derivation_v1"
    assert result["derivation_metadata"] == state["metadata"]["derived"]
    assert result["curation_metadata"] is None and result["identity_audit"] is None
    assert result["derivation_manifest_sha256"] == state["data_hashes"]["derivation_manifest.json"]
    assert result["reserved_parent_test96"] == state["metadata"]["reserved_parent_test96"]
    assert len(result["checkpoint_data_hashes"]) == 5
    assert result["prepared_data_hashes"] == result["calibration_provenance"]["prepared_data_hashes"] == prepared_hashes
    assert "does not reaudit" in result["preparation_binding_scope"]
    for row in [result["utility"], *result["uncertainty"]]:
        assert row["data_curation_status"] == "verified_hourly_derivation_v1"
        assert row["derivation_manifest_sha256"] == state["data_hashes"]["derivation_manifest.json"]
        assert row["derivation_metadata"] == state["metadata"]["derived"]
    for name in ("utility_validation.json", "uncertainty_gaussian.json"):
        path = directory / name
        original = path.read_text()
        changed = json.loads(original)
        changed["derivation_manifest_sha256"] = "0" * 64
        path.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match="derivation metadata/manifest"):
            collector.collect_run("hourly", directory)
        path.write_text(original)
