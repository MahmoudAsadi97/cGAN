"""Capacity contracts reuse the report-only temporal assessor and its gates."""
import hashlib
import json

import pytest
import torch

from test_assess_temporal_pilot import change_metric, digest, pilot, synthetic, verified


def capacity_inputs(arm="candidate", seed=11, *, derivative_metadata=True):
    values = synthetic(arm, seed)
    run, summary, history, _, _ = values
    run["config"] = pilot.expected_config(arm, seed, candidate_kind="capacity")
    # Synthetic recorded provenance follows the named execution contract;
    # historical baseline training remains separate from that candidate pin.
    for provenance, key in ((run["provenance"]["training_checkpoint"], "source_files_sha256"),
                            (run["provenance"]["calibration_evaluation"], "source_files_sha256"),
                            (run["utility"]["provenance"], "source_file_sha256")):
        provenance[key].update({path: value for path, value in pilot.CAPACITY_SOURCE_HASHES.items()
                                if path in provenance[key]})
        if arm == "candidate":
            provenance.update(git_commit=pilot.CAPACITY_COMMIT, git_dirty=False)
    if arm == "baseline":
        run["provenance"]["training_checkpoint"]["source_files_sha256"]["src/fluvius_cgan/training.py"] = digest("historical-training")
    evaluation = run["provenance"]["calibration_evaluation"]
    evaluation["evaluation_script_sha256"] = pilot.CAPACITY_SOURCE_HASHES["scripts/evaluate_calibration.py"]
    evaluation["comparison_helper_sha256"] = pilot.CAPACITY_SOURCE_HASHES["scripts/compare_checkpoints.py"]
    # Independent fixed-noise oracle; the assessor must verify these recorded
    # bytes, not compare unequal latent dimensions as if they shared draws.
    noise = torch.randn((4096, run["config"]["latent_dim"]), dtype=torch.float32,
                        generator=torch.Generator().manual_seed(seed + 2))
    run["training_selection"].update(
        validation_noise_sha256=hashlib.sha256(noise.numpy().astype("<f4").tobytes()).hexdigest(),
        validation_indices=list(range(4096)), validation_labels_sha256=digest("selection-labels"))
    provenance = run["provenance"]["training_checkpoint"]
    if derivative_metadata:
        provenance["derivative_swd"] = {"enabled": False, "weight": 0., "scale": None,
                                        "projection_seed": None, "num_projections": None}
        summary["derivative_swd_scale"] = None
        for row in history:
            row["loss_derivative_swd"] = 0.
    else:
        provenance.pop("derivative_swd", None)
        summary.pop("derivative_swd_scale", None)
    return values


def capacity_verified(arm, seed):
    return pilot._verify_protocol(*capacity_inputs(arm, seed), arm, seed, candidate_kind="capacity")


@pytest.fixture
def capacity_runs():
    return [capacity_verified(arm, seed) for arm in ("candidate", "baseline") for seed in pilot.SEEDS]


@pytest.fixture
def capacity_protocol(monkeypatch, tmp_path):
    protocol = tmp_path / "CAPACITY_REFINEMENT.md"
    protocol.write_text("Synthetic capacity protocol fixture.\n")
    monkeypatch.setitem(pilot.PROTOCOL_DOCUMENTS, "capacity", str(protocol))
    return protocol


def test_capacity_changes_only_width_and_latent_with_explicit_zero_derivative():
    candidate = pilot.expected_config("candidate", 11, candidate_kind="capacity")
    baseline = pilot.expected_config("baseline", 11, candidate_kind="capacity")
    differences = {key: (baseline[key], value) for key, value in candidate.items() if value != baseline[key]}
    assert differences == {"hidden_dim": (64, 128), "latent_dim": (24, 96)}
    assert candidate["derivative_swd_weight"] == baseline["derivative_swd_weight"] == 0.
    assert candidate["discriminator_hidden_dim"] == baseline["discriminator_hidden_dim"] == 128
    assert pilot.expected_config("candidate", 11)["derivative_swd_weight"] == 1.


def test_capacity_noise_is_bound_at_each_width_and_all_promotion_math_is_reused(capacity_runs):
    pilot._compatibility(capacity_runs, candidate_kind="capacity")
    for seed in pilot.SEEDS:
        pair = [run for run in capacity_runs if run["training_seed"] == seed]
        assert pair[0]["training_selection"]["validation_noise_sha256"] != pair[1]["training_selection"]["validation_noise_sha256"]
    temporal = [verified(arm, seed) for arm in ("candidate", "baseline") for seed in pilot.SEEDS]
    assert pilot.measured_gates(capacity_runs) == pilot.measured_gates(temporal)


@pytest.mark.parametrize("metadata", [False, True])
def test_absent_or_disabled_derivative_evidence_is_accepted(metadata):
    for arm in ("candidate", "baseline"):
        values = capacity_inputs(arm, derivative_metadata=metadata)
        assert pilot._verify_protocol(*values, arm, 11, candidate_kind="capacity")["arm"] == arm


@pytest.mark.parametrize("field,value", [
    ("hidden_dim", 64), ("latent_dim", 24), ("discriminator_hidden_dim", 256),
    ("derivative_swd_weight", 1.), ("derivative_swd_weight", False), ("swd_weight", 0.), ("lr_g", .0002),
    ("batch_size", 256), ("validation_selection", "pooled_w1"),
])
def test_capacity_rejects_extra_or_missing_architecture_and_training_changes(field, value):
    values = capacity_inputs()
    values[0]["config"][field] = value
    with pytest.raises(ValueError, match="configuration"):
        pilot._verify_protocol(*values, "candidate", 11, candidate_kind="capacity")


def test_capacity_requires_explicit_candidate_zero_but_accepts_historical_baseline_default():
    values = capacity_inputs()
    values[0]["config"].pop("derivative_swd_weight")
    with pytest.raises(ValueError, match="explicitly declare"):
        pilot._verify_protocol(*values, "candidate", 11, candidate_kind="capacity")
    baseline = capacity_inputs("baseline", derivative_metadata=False)
    baseline[0]["config"].pop("derivative_swd_weight")
    pilot._verify_protocol(*baseline, "baseline", 11, candidate_kind="capacity")


@pytest.mark.parametrize("mutation,match", [
    (lambda r,s,h: r["provenance"]["training_checkpoint"]["derivative_swd"].update(enabled=True), "zero-weight derivative"),
    (lambda r,s,h: r["provenance"]["training_checkpoint"]["derivative_swd"].update(weight=1.), "zero-weight derivative"),
    (lambda r,s,h: r["provenance"]["training_checkpoint"]["derivative_swd"].update(scale={"used_float32": 1.}), "zero-weight derivative"),
    (lambda r,s,h: r["provenance"]["training_checkpoint"]["derivative_swd"].update(projection_seed=16), "zero-weight derivative"),
    (lambda r,s,h: s.update(derivative_swd_scale={"used_float32": 1.}), "must not fit"),
    (lambda r,s,h: h[0].update(loss_derivative_swd=.01), "zero derivative loss"),
    (lambda r,s,h: r["training_selection"].update(validation_noise_sha256=digest("wrong-seed")), "noise differs"),
])
def test_capacity_rejects_derivative_execution_or_unbound_noise(mutation, match):
    values = capacity_inputs()
    mutation(*values[:3])
    with pytest.raises(ValueError, match=match):
        pilot._verify_protocol(*values, "candidate", 11, candidate_kind="capacity")


@pytest.mark.parametrize("field,value", [
    ("validation_indices", [7] * 4096),
    ("validation_labels_sha256", digest("different-labels")),
])
def test_capacity_noise_exception_does_not_relax_selection_rows_or_labels(capacity_runs, field, value):
    capacity_runs[0]["training_selection"][field] = value
    with pytest.raises(ValueError, match="checkpoint-selection"):
        pilot._compatibility(capacity_runs, candidate_kind="capacity")


def test_capacity_cross_arm_config_is_checked_even_after_individual_validation(capacity_runs):
    capacity_runs[0]["config"]["discriminator_hidden_dim"] = 256
    with pytest.raises(ValueError, match="undeclared cross-arm"):
        pilot._compatibility(capacity_runs, candidate_kind="capacity")


@pytest.mark.parametrize("mutation,match", [
    (lambda r: r["pilot_sources"]["evaluation"].update({"src/fluvius_cgan/copula.py": digest("different-source")}), "evaluation source"),
    (lambda r: r["frozen_samples"]["profiles_sha256"].update(gaussian_copula=digest("different-draw")), "statistical control"),
])
def test_capacity_keeps_evaluator_and_statistical_control_bindings_strict(capacity_runs, mutation, match):
    mutation(capacity_runs[0])
    with pytest.raises(ValueError, match=match):
        pilot._compatibility(capacity_runs, candidate_kind="capacity")


@pytest.mark.parametrize("stage,key,value,match", [
    ("training", "git_commit", "f" * 40, "clean pinned commit"),
    ("training", "git_dirty", True, "clean pinned commit"),
    ("training", "git_dirty", None, "clean pinned commit"),
    ("evaluation", "git_commit", "f" * 40, "recorded commit"),
    ("evaluation", "git_dirty", True, "dirty source tree"),
    ("utility", "git_commit", "f" * 40, "recorded commit"),
    ("utility", "git_dirty", True, "dirty source tree"),
])
def test_capacity_candidate_rejects_incompatible_recorded_execution_pin(stage, key, value, match):
    values = capacity_inputs()
    run = values[0]
    provenance = {"training": run["provenance"]["training_checkpoint"],
                  "evaluation": run["provenance"]["calibration_evaluation"],
                  "utility": run["utility"]["provenance"]}[stage]
    provenance[key] = value
    with pytest.raises(ValueError, match=match):
        pilot._verify_protocol(*values, "candidate", 11, candidate_kind="capacity")


def test_capacity_candidate_requires_pinned_training_hash_even_with_clean_commit():
    values = capacity_inputs()
    values[0]["provenance"]["training_checkpoint"]["source_files_sha256"]["src/fluvius_cgan/training.py"] = digest("wrong-content")
    with pytest.raises(ValueError, match="training source differs from pinned bytes"):
        pilot._verify_protocol(*values, "candidate", 11, candidate_kind="capacity")


def test_capacity_candidate_rejects_consistent_but_unpinned_evaluation_arithmetic():
    values = capacity_inputs()
    run = values[0]
    for provenance, key in ((run["provenance"]["calibration_evaluation"], "source_files_sha256"),
                            (run["utility"]["provenance"], "source_file_sha256")):
        provenance[key]["src/fluvius_cgan/copula.py"] = digest("different-interpolation")
    with pytest.raises(ValueError, match="evaluation source differs from pinned bytes"):
        pilot._verify_protocol(*values, "candidate", 11, candidate_kind="capacity")


def test_evaluation_git_metadata_can_be_absent_but_pinned_hashes_cannot():
    values = capacity_inputs()
    run = values[0]
    for provenance in (run["provenance"]["calibration_evaluation"], run["utility"]["provenance"]):
        provenance.pop("git_commit")
        provenance.pop("git_dirty")
    pilot._verify_protocol(*values, "candidate", 11, candidate_kind="capacity")
    run["provenance"]["calibration_evaluation"]["source_files_sha256"].pop("src/fluvius_cgan/copula.py")
    with pytest.raises(ValueError, match="source fingerprints"):
        pilot._verify_protocol(*values, "candidate", 11, candidate_kind="capacity")


@pytest.mark.parametrize("status,code", [("pass", 0), ("fail", 1), ("incomplete", 2)])
def test_capacity_cli_records_contract_and_preserves_all_outcomes(monkeypatch, tmp_path, capacity_protocol, status, code):
    def verify(arm, seed, directory, *, candidate_kind):
        assert candidate_kind == "capacity"
        if status == "incomplete" and arm == "candidate" and seed == 22:
            raise ValueError("required full-TRAIN utility missing")
        run = capacity_verified(arm, seed)
        if status == "fail" and arm == "candidate":
            change_metric(run, "ramp_wasserstein_kw", .99)
        return run
    monkeypatch.setattr(pilot, "verify_run", verify)
    output, markdown = tmp_path / "assessment.json", tmp_path / "assessment.md"
    args = ["--candidate-kind", "capacity", "--candidate-root", str(tmp_path / "candidate"),
            "--baseline-root", str(tmp_path / "baseline"), "--output", str(output), "--markdown", str(markdown)]
    assert pilot.main(args) == code
    report = json.loads(output.read_text())
    assert report["protocol"]["candidate_kind"] == "capacity"
    assert report["provenance"]["protocol_sha256"] == hashlib.sha256(capacity_protocol.read_bytes()).hexdigest()
    assert report["status"] == status and len(report["requested_runs"]) == 6
    assert report["promotion_eligible"] == (status == "pass")
    assert "# Capacity pilot assessment" in markdown.read_text()
    before = output.read_bytes()
    with pytest.raises(SystemExit):
        pilot.main(args)
    assert output.read_bytes() == before


def test_unknown_candidate_kind_is_rejected_before_collecting(monkeypatch, tmp_path):
    monkeypatch.setattr(pilot, "verify_run", lambda *a, **k: pytest.fail("must not collect"))
    with pytest.raises(ValueError, match="unknown pilot"):
        pilot.assess(tmp_path / "candidate", tmp_path / "baseline", candidate_kind="unregistered")
