"""Synthetic adversarial contracts for the sealed five-seed TEST evaluator.

The shared bundle is generated below from temporary fixtures only. Mutations
are restored after each check; no prepared research dataset is referenced.
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from fluvius_cgan import frozen_test as ft
from test_frozen_test import make_frozen_study


@pytest.fixture(scope="module")
def synthetic_study(tmp_path_factory):
    # Freeze once: these contracts do not repeat GAN generation or model fits.
    with pytest.MonkeyPatch.context() as patch:
        return make_frozen_study(tmp_path_factory.mktemp("frozen-contract"), patch)


def forbid_test_read(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("TEST was reached before preflight completed")
    monkeypatch.setattr(ft, "_read_verified_test", fail)


def assert_failed_attempt(study, output, *, test_access_started, ledger=True):
    failed = json.loads((output / "failure.json").read_text())
    assert failed["test_access_started"] is test_access_started
    assert not (output / "report.json").exists()
    assert not (output / "complete.json").exists()
    assert (output / "test_access_started.json").exists() is test_access_started
    events = study["bundle"] / "attempts"
    attempt_id = failed["attempt_id"]
    assert (events / f"{attempt_id}.start.json").exists() is ledger
    assert (events / f"{attempt_id}.failure.json").exists() is ledger
    assert not (events / f"{attempt_id}.complete.json").exists()
    if ledger:
        assert json.loads((events / f"{attempt_id}.failure.json").read_text()) == failed
    return failed


@contextmanager
def altered_manifest(study, change):
    path = study["bundle"] / "manifest.json"
    original = path.read_bytes()
    manifest = json.loads(original)
    change(manifest)
    altered = json.dumps(manifest, allow_nan=False).encode()
    try:
        path.write_bytes(altered)
        yield hashlib.sha256(altered).hexdigest()
    finally:
        path.write_bytes(original)


@pytest.mark.parametrize("seal", [None, "", "x" * 64, "0" * 63, True])
def test_valid_seal_argument_is_mandatory_before_output_creation(tmp_path, monkeypatch, seal):
    forbid_test_read(monkeypatch)
    output = tmp_path / "evaluation"
    with pytest.raises(ValueError, match="seal SHA256 is mandatory"):
        ft.evaluate_frozen_test(tmp_path / "nonexistent-bundle", seal, output)
    assert not output.exists()


def test_wrong_seal_records_failure_without_trusting_bundle_ledger(synthetic_study, tmp_path, monkeypatch):
    forbid_test_read(monkeypatch)
    output = tmp_path / "evaluation"
    with pytest.raises(ValueError, match="seal SHA256 mismatch"):
        ft.evaluate_frozen_test(synthetic_study["bundle"], "0" * 64, output)
    assert_failed_attempt(synthetic_study, output, test_access_started=False, ledger=False)


def test_changed_manifest_is_rejected_against_original_seal(synthetic_study, tmp_path, monkeypatch):
    forbid_test_read(monkeypatch)
    output = tmp_path / "evaluation"
    with altered_manifest(synthetic_study, lambda value: value.update(created_at="changed")):
        with pytest.raises(ValueError, match="seal SHA256 mismatch"):
            ft.evaluate_frozen_test(synthetic_study["bundle"], synthetic_study["seal"], output)
    assert_failed_attempt(synthetic_study, output, test_access_started=False, ledger=False)


@pytest.mark.parametrize("what", ["sources", "environment"])
def test_current_source_and_environment_must_equal_seal(synthetic_study, tmp_path, monkeypatch, what):
    forbid_test_read(monkeypatch)
    changed = deepcopy(synthetic_study["manifest"][what])
    changed["synthetic_contract_change"] = True
    monkeypatch.setattr(ft, "_" + what, lambda: changed)
    output = tmp_path / "evaluation"
    expected = "source" if what == "sources" else "environment"
    with pytest.raises(ValueError, match=f"evaluator {expected} differs"):
        ft.evaluate_frozen_test(synthetic_study["bundle"], synthetic_study["seal"], output)
    assert_failed_attempt(synthetic_study, output, test_access_started=False)


@pytest.mark.parametrize("change, message", [
    ("cap", "frozen test protocol differs"),
    ("seed", "sealed run list must contain all five seeds"),
    ("binding", "sealed TEST binding disagrees"),
    ("rationale", "decision rationale/protocol missing"),
])
def test_new_seal_cannot_override_protocol_or_curation(synthetic_study, tmp_path, monkeypatch, change, message):
    forbid_test_read(monkeypatch)
    def mutate(manifest):
        if change == "cap":
            manifest["protocol"]["samples_per_condition_cap"] = 100
        elif change == "seed":
            manifest["runs"][-1]["seed"] = 11
        elif change == "binding":
            manifest["test_binding"]["meter_ids"][0] = "unregistered"
        else:
            manifest["decision"]["rationale"] = "   "
    output = tmp_path / "evaluation"
    with altered_manifest(synthetic_study, mutate) as seal:
        with pytest.raises(ValueError, match=message):
            ft.evaluate_frozen_test(synthetic_study["bundle"], seal, output)
    assert_failed_attempt(synthetic_study, output, test_access_started=False)


def test_corrupt_generated_asset_fails_before_test(synthetic_study, tmp_path, monkeypatch):
    forbid_test_read(monkeypatch)
    filename = synthetic_study["manifest"]["runs"][0]["arrays"]["calibrated_cgan"]
    path = synthetic_study["bundle"] / "artifacts" / filename
    original = path.read_bytes()
    output = tmp_path / "evaluation"
    try:
        path.write_bytes(original + b"corrupt")
        with pytest.raises(ValueError, match="sealed artifact changed"):
            ft.evaluate_frozen_test(synthetic_study["bundle"], synthetic_study["seal"], output)
    finally:
        path.write_bytes(original)
    assert_failed_attempt(synthetic_study, output, test_access_started=False)


def test_rehashed_invalid_generated_dtype_fails_before_test(synthetic_study, tmp_path, monkeypatch):
    forbid_test_read(monkeypatch)
    filename = synthetic_study["manifest"]["runs"][0]["arrays"]["raw_cgan"]
    path = synthetic_study["bundle"] / "artifacts" / filename
    original = path.read_bytes()
    values = np.load(path, allow_pickle=False)
    output = tmp_path / "evaluation"
    try:
        with path.open("wb") as handle:
            np.save(handle, values.astype(np.float32), allow_pickle=False)
        def mutate(manifest):
            manifest["assets"][filename] = {"sha256": ft.sha256_file(path), "bytes": path.stat().st_size}
        with altered_manifest(synthetic_study, mutate) as seal:
            with pytest.raises(ValueError, match="invalid sealed generated profiles"):
                ft.evaluate_frozen_test(synthetic_study["bundle"], seal, output)
    finally:
        path.write_bytes(original)
    assert_failed_attempt(synthetic_study, output, test_access_started=False)


def test_existing_output_is_not_overwritten_or_read(synthetic_study, tmp_path, monkeypatch):
    forbid_test_read(monkeypatch)
    output = tmp_path / "existing"
    output.mkdir()
    (output / "sentinel.txt").write_text("preserve this output")
    before = set((synthetic_study["bundle"] / "attempts").iterdir())
    with pytest.raises(FileExistsError):
        ft.evaluate_frozen_test(synthetic_study["bundle"], synthetic_study["seal"], output)
    assert {p.name: p.read_text() for p in output.iterdir()} == {"sentinel.txt": "preserve this output"}
    assert set((synthetic_study["bundle"] / "attempts").iterdir()) == before


def test_changed_test_hash_has_failed_access_event_and_cannot_resume(synthetic_study, tmp_path):
    path = Path(synthetic_study["test_binding"]["path"])
    original = path.read_bytes()
    output = tmp_path / "evaluation"
    try:
        path.write_bytes(original + b"changed synthetic TEST bytes")
        with pytest.raises(ValueError, match="full TEST hash differs"):
            ft.evaluate_frozen_test(synthetic_study["bundle"], synthetic_study["seal"], output)
    finally:
        path.write_bytes(original)
    failed = assert_failed_attempt(synthetic_study, output, test_access_started=True)
    event = synthetic_study["bundle"] / "attempts" / f"{failed['attempt_id']}.test_access.json"
    assert event.exists()
    failure_bytes = (output / "failure.json").read_bytes()
    with pytest.raises(FileExistsError):
        ft.evaluate_frozen_test(synthetic_study["bundle"], synthetic_study["seal"], output)
    assert (output / "failure.json").read_bytes() == failure_bytes


@pytest.mark.parametrize("filename", ["../test.npz", "/tmp/test.npz", "a/b.npy", "", "..", ".hidden"])
def test_asset_names_cannot_escape_artifact_directory(tmp_path, filename):
    with pytest.raises(ValueError, match="artifact"):
        ft._safe_asset(tmp_path, filename)


def test_symlink_assets_are_rejected_even_when_bytes_match(tmp_path):
    (tmp_path / "artifacts").mkdir()
    target = tmp_path / "target.npy"
    target.write_bytes(b"synthetic bytes")
    (tmp_path / "artifacts" / "profiles.npy").symlink_to(target)
    with pytest.raises(ValueError, match="artifact"):
        ft._safe_asset(tmp_path, "profiles.npy")


@pytest.mark.parametrize("change, message", [
    ("extra_array", "invalid TEST arrays"),
    ("float64", "invalid TEST profiles/dtype/width"),
    ("nonfinite", "invalid TEST profiles/dtype/width"),
    ("width", "invalid TEST profiles/dtype/width"),
    ("label_dtype", "invalid TEST condition support/dtype"),
    ("label_support", "invalid TEST condition support/dtype"),
    ("meter_dtype", "invalid TEST meter/date vectors"),
    ("meter_membership", "TEST meter membership differs"),
    ("development_overlap", "TEST meter membership differs"),
    ("excluded_overlap", "TEST meter membership differs"),
    ("nat", "invalid TEST calendar format"),
    ("calendar_label", "TEST labels disagree with calendar"),
    ("duplicate_calendar", "TEST meter has missing/duplicate/unexpected dates"),
    ("technology", "TEST technology changes within a meter"),
    ("condition_counts", "TEST condition/technology counts differ"),
    ("technology_counts", "TEST condition/technology counts differ"),
])
def test_hash_matching_test_still_requires_structural_contract(synthetic_study, tmp_path, change, message):
    binding = deepcopy(synthetic_study["test_binding"])
    with np.load(binding["path"], allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    if change == "extra_array":
        arrays["unexpected"] = np.zeros(1)
    elif change == "float64":
        arrays["x"] = arrays["x"].astype(np.float64)
    elif change == "nonfinite":
        arrays["x"][0, 0] = np.nan
    elif change == "width":
        arrays["x"] = arrays["x"][:, :-1]
    elif change == "label_dtype":
        arrays["y"] = arrays["y"].astype(np.int32)
    elif change == "label_support":
        arrays["y"][0] = 64
    elif change == "meter_dtype":
        arrays["meter_id"] = arrays["meter_id"].astype("S")
    elif change == "meter_membership":
        arrays["meter_id"][arrays["meter_id"] == binding["meter_ids"][0]] = "foreign"
    elif change == "development_overlap":
        binding["development_meter_ids"].append(binding["meter_ids"][0])
    elif change == "excluded_overlap":
        binding["excluded_original_test_meter_ids"].append(binding["meter_ids"][0])
    elif change == "nat":
        arrays["date"][0] = "NaT"
    elif change == "calendar_label":
        arrays["y"][0] = (arrays["y"][0] + 1) % 8
    elif change == "duplicate_calendar":
        # Change its label as well so the calendar-completeness guard is reached.
        arrays["date"][0] = arrays["date"][1]
        arrays["y"][0] = arrays["y"][1]
    elif change == "technology":
        arrays["y"][0] += 8
    elif change == "condition_counts":
        binding["condition_counts"]["0"] += 1
    else:
        binding["technology_meter_counts"]["0"] += 1
    path = tmp_path / "synthetic-test.npz"
    np.savez(path, **arrays)
    binding.update(path=str(path), sha256=ft.sha256_file(path))
    with pytest.raises(ValueError, match=message):
        ft._read_verified_test(binding)


def test_test_path_must_not_be_symlink(synthetic_study, tmp_path):
    binding = deepcopy(synthetic_study["test_binding"])
    path = tmp_path / "synthetic-test-link.npz"
    path.symlink_to(binding["path"])
    binding["path"] = str(path)
    with pytest.raises(ValueError, match="TEST path cannot be a symlink"):
        ft._read_verified_test(binding)


def test_whole_meter_technology_swap_fails_despite_unchanged_counts(synthetic_study, tmp_path):
    binding = deepcopy(synthetic_study["test_binding"])
    with np.load(binding["path"], allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    first = next(meter for meter, tech in binding["technology_by_meter"].items() if tech == 0)
    second = next(meter for meter, tech in binding["technology_by_meter"].items() if tech == 1)
    original_labels = arrays["y"].copy()
    arrays["y"][arrays["meter_id"] == first] += 8
    arrays["y"][arrays["meter_id"] == second] -= 8
    # Every meter remains internally consistent, and marginal counts and each
    # date's season/weekend class are unchanged. Canonical identity is decisive.
    np.testing.assert_array_equal(np.bincount(arrays["y"]), np.bincount(original_labels))
    np.testing.assert_array_equal(arrays["y"] % 8, original_labels % 8)
    assert all(len(np.unique(arrays["y"][arrays["meter_id"] == meter] // 8)) == 1
               for meter in binding["meter_ids"])
    path = tmp_path / "synthetic-label-swap.npz"
    np.savez(path, **arrays)
    binding.update(path=str(path), sha256=ft.sha256_file(path))
    with pytest.raises(ValueError, match="TEST technology differs from canonical meter"):
        ft._read_verified_test(binding)


@pytest.mark.parametrize("input_role, alias_kind", [
    (role, kind)
    for role in ("checkpoint", "calibrator", "protocol", "evidence")
    for kind in ("direct", "symlink", "hardlink")
] + [
    (role, kind)
    for role in (*ft.PREPARED, "curation_manifest.json")
    for kind in ("symlink", "hardlink")
])
def test_freeze_rejects_reserved_test_aliases_before_content_read(
        synthetic_study, tmp_path, monkeypatch, input_role, alias_kind):
    reserved = Path(synthetic_study["test_binding"]["path"])
    prepared = synthetic_study["prepared"]
    runs = list(synthetic_study["runs"])
    protocol_files, evidence_files = [synthetic_study["protocol"]], []
    restore = None
    if input_role in (*ft.PREPARED, "curation_manifest.json"):
        alias = prepared / input_role
        restore = tmp_path / "saved-prepared-input"
        alias.rename(restore)
    elif alias_kind == "direct":
        alias = reserved
    else:
        alias = tmp_path / "innocent-input.dat"
    try:
        if alias_kind == "symlink":
            alias.symlink_to(reserved)
        elif alias_kind == "hardlink":
            alias.hardlink_to(reserved)
        if input_role == "checkpoint":
            runs[0] = (alias, runs[0][1])
        elif input_role == "calibrator":
            runs[0] = (runs[0][0], alias)
        elif input_role == "protocol":
            protocol_files = [alias]
        elif input_role == "evidence":
            evidence_files = [alias]

        original_open = Path.open
        def guarded_open(path, *args, **kwargs):
            if path.exists() and path.samefile(reserved):
                raise AssertionError("freeze opened reserved TEST through an alias")
            return original_open(path, *args, **kwargs)
        def advanced_past_preflight():
            raise AssertionError("freeze advanced past reserved-input preflight")

        output = tmp_path / "rejected-freeze"
        with monkeypatch.context() as patch:
            patch.setattr(Path, "open", guarded_open)
            patch.setattr(ft, "_sources", advanced_past_preflight)
            with pytest.raises(ValueError, match="reserved TEST"):
                ft.freeze_test(prepared, runs, output, decision_rationale="Synthetic alias rejection",
                    protocol_files=protocol_files, evidence_files=evidence_files,
                    utility=False, bootstrap_replicates=0)
        failed = json.loads((output / "freeze_failure.json").read_text())
        assert failed["test_accessed"] is False
        assert failed["error_type"] == "ValueError"
        assert not (output / "manifest.json").exists()
        assert not (output / "freeze_complete.json").exists()
        assert not (output / "artifacts").exists()
    finally:
        if restore is not None:
            alias.unlink(missing_ok=True)
            restore.rename(alias)
