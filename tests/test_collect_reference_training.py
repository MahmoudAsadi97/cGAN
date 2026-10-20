"""Small synthetic checkpoints exercise completion checks without profile data."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import collect_reference_study as collector


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def completed(tmp_path):
    def build(arm="bce96", seed=11, kind="swd10"):
        width, condition_dim = (24 if arm == "bce24" else 96), (0 if arm == "unconditional96" else 64)
        directory = tmp_path / arm
        directory.mkdir(exist_ok=True)
        prepared96, prepared24 = tmp_path / "prepared96", tmp_path / "prepared24"
        for prepared in (prepared96, prepared24):
            prepared.mkdir(exist_ok=True)
            for name in ("train.npz", "val.npz", "test.npz"):
                (prepared / name).write_bytes(b"FORBIDDEN SYNTHETIC PROFILE SENTINEL")
        config = collector._declared_config(seed, unconditional=condition_dim == 0,
                                            candidate_kind=kind if arm == "candidate96" else None)
        metadata = {"resolution": width, "target": "net", "units": "kWh/interval",
                    "source_sha256": collector.RELEASE_SHA256}
        scaler = {"minimum": -1., "maximum": 2., "fitted_on": "train"}
        training_scaler = {**scaler, "kind": "standard", "mean": .2, "std": .7} if arm == "candidate96" else scaler
        hashes = {name: hashlib.sha256(name.encode()).hexdigest() for name in collector.PREPARED}
        manifest_name = "derivation_manifest.json" if width == 24 else "curation_manifest.json"
        manifest_hash = "e" * 64
        prep = {width: {"metadata": metadata, "scaler": scaler, "hashes": hashes,
                        "manifest_name": manifest_name, "manifest_sha256": manifest_hash}}
        selection = {"train_profiles_available": collector.TRAIN_ROWS, "train_profiles_used": collector.TRAIN_ROWS,
                     "train_indices_sha256": hashlib.sha256(np.arange(collector.TRAIN_ROWS, dtype="<i8").tobytes()).hexdigest(),
                     "validation_selection": "conditional_w1", "validation_samples_requested": 4096,
                     "batch_sampling": "balanced_conditions", "swd_conditioning": "pooled" if condition_dim == 0 else "conditional"}
        guard = {"status": "verified_curation_v2", "pinned_release": True,
                 "metadata_sha256": hashes["metadata.json"], "manifest_sha256": manifest_hash}
        provenance = {"git_commit": collector.SOURCE_COMMIT, "git_commit_tree": collector.SOURCE_TREE,
                      "git_dirty": False, "source_files_sha256": deepcopy(collector.TRAINING_SOURCE_HASHES)}
        per_epoch = {str(label): 9472 for label in range(64)}
        history = [{"epoch": epoch, "generator_updates": 1184, "discriminator_updates": 1184,
                    "samples_seen": 606208, "class_exposure_counts": deepcopy(per_epoch),
                    "validation_score": .5 if epoch == 2 else 1. + epoch / 100.,
                    "loss_g": 1., "loss_d": .7, "loss_swd": .3 if arm == "candidate96" else 0.,
                    "loss_derivative_swd": 0.} for epoch in range(1, 27)]
        generator = collector.Generator(config["latent_dim"], condition_dim, config["hidden_dim"], width)
        discriminator = collector.Discriminator(width, condition_dim, config["discriminator_hidden_dim"], objective=config["objective"])
        common = {"format_version": collector.CHECKPOINT_FORMAT, "config": config, "metadata": metadata,
                  "condition_dim": condition_dim, "output_dim": width, "scaler": training_scaler,
                  "data_hashes": {**hashes, manifest_name: manifest_hash}, "data_curation": guard,
                  "selection": selection, "provenance": provenance, "supported_labels": list(range(64)),
                  "derivative_swd_scale": None, "best_epoch": 2, "best_validation_score": .5,
                  "validation_metric": collector.CONDITIONAL_VALIDATION_METRIC,
                  "generator_state_dict": generator.state_dict(), "discriminator_state_dict": discriminator.state_dict()}
        states = {}
        for name, epoch in (("best.pt", 2), ("last.pt", 26)):
            state = {**common, "epoch": epoch, "validation_score": history[epoch - 1]["validation_score"],
                     "epoch_metrics": history[epoch - 1], "generator_updates": 1184 * epoch,
                     "discriminator_updates": 1184 * epoch, "samples_seen": 606208 * epoch,
                     "class_exposure_counts": {label: count * epoch for label, count in per_epoch.items()}}
            states[name] = deepcopy(state)
            torch.save(state, directory / name)
        summary = {key: value for key, value in states["last.pt"].items()
                   if key not in ("generator_state_dict", "discriminator_state_dict", "config", "metadata")}
        summary.update(epochs_completed=26, stopped_early=False, test_used=False,
                       last_epoch_metrics=history[-1], source_sha256=collector.RELEASE_SHA256,
                       target="net", units="kWh/interval", resolution=width)
        for name, value in (("config.json", config), ("history.json", history), ("summary.json", summary)):
            write_json(directory / name, value)
        return {"directory": directory, "prep": prep, "states": states, "config": config,
                "summary": summary, "history": history, "arm": arm, "seed": seed, "kind": kind,
                "inputs": collector.Inputs(tmp_path, prepared96, prepared24)}
    return build


def run(value):
    return collector.training_run(value["inputs"], value["directory"], value["arm"], value["seed"], value["kind"], value["prep"])


@pytest.mark.parametrize("arm,kind", [("bce96", "swd10"), ("unconditional96", "swd10"),
                                      ("bce24", "swd10"), ("candidate96", "swd10"), ("candidate96", "capacity")])
def test_complete_declared_packages_without_profile_access(completed, monkeypatch, arm, kind):
    value = completed(arm, kind=kind)
    original = collector.sha256_file
    def protected(path):
        assert Path(path).name not in ("train.npz", "val.npz", "test.npz")
        return original(path)
    monkeypatch.setattr(collector, "sha256_file", protected)
    result = run(value)
    assert result["best_epoch"] == 2
    assert result["generator_updates"] == result["discriminator_updates"] == 30784
    assert result["training_selection"]["train_profiles_used"] == 606060
    assert set(result["artifacts"]) == set(collector.RUN_FILES)


@pytest.mark.parametrize("field,new", [("epochs_completed", 25), ("generator_updates", 30783),
                                       ("discriminator_updates", 30783), ("samples_seen", 1),
                                       ("best_epoch", 3), ("best_validation_score", .4),
                                       ("stopped_early", True), ("test_used", True)])
def test_incomplete_or_inconsistent_summary_rejected(completed, field, new):
    value = completed()
    value["summary"][field] = new
    write_json(value["directory"] / "summary.json", value["summary"])
    with pytest.raises(ValueError):
        run(value)


@pytest.mark.parametrize("mutation", ["missing_epoch", "wrong_epoch", "updates", "draws", "exposure", "nonfinite", "loss", "derivative"])
def test_history_corruption_rejected(completed, mutation):
    value = completed()
    history = value["history"]
    if mutation == "missing_epoch":
        history.pop()
    elif mutation == "wrong_epoch":
        history[-1]["epoch"] = 25
    elif mutation == "updates":
        history[-1]["generator_updates"] = 1183
    elif mutation == "draws":
        history[-1]["samples_seen"] -= 1
    elif mutation == "exposure":
        history[-1]["class_exposure_counts"]["0"] -= 1
    elif mutation == "nonfinite":
        history[-1]["validation_score"] = float("nan")
    elif mutation == "loss":
        history[-1]["loss_swd"] = 1.
    else:
        history[-1]["loss_derivative_swd"] = 1.
    write_json(value["directory"] / "history.json", history)
    with pytest.raises(ValueError):
        run(value)


@pytest.mark.parametrize("mutation", ["seed", "unconditional", "budget", "normalization"])
def test_config_deviation_rejected(completed, mutation):
    value = completed()
    field, wrong = {"seed": ("seed", 12), "unconditional": ("unconditional", True),
                    "budget": ("epochs", 27), "normalization": ("normalization", "standard")}[mutation]
    value["config"][field] = wrong
    write_json(value["directory"] / "config.json", value["config"])
    with pytest.raises(ValueError, match="configuration"):
        run(value)


@pytest.mark.parametrize("mutation", ["epoch", "history", "weights", "nonfinite_weights", "data", "source", "selection", "scaler"])
def test_checkpoint_corruption_rejected(completed, mutation):
    value = completed()
    state = value["states"]["last.pt"]
    if mutation == "epoch":
        state["epoch"] = 25
    elif mutation == "history":
        state["epoch_metrics"]["loss_d"] += .1
    elif mutation == "weights":
        state["generator_state_dict"].pop(next(iter(state["generator_state_dict"])))
    elif mutation == "nonfinite_weights":
        next(iter(state["generator_state_dict"].values())).reshape(-1)[0] = float("inf")
    elif mutation == "data":
        state["data_hashes"]["train.npz"] = "f" * 64
    elif mutation == "source":
        state["provenance"]["git_dirty"] = True
    elif mutation == "selection":
        state["selection"]["train_profiles_used"] -= 1
    else:
        state["scaler"]["maximum"] = 3.
    torch.save(state, value["directory"] / "last.pt")
    with pytest.raises((ValueError, RuntimeError)):
        run(value)


def test_clean_native_source_pin_is_mandatory_even_if_all_states_agree(completed):
    value = completed()
    for name, state in value["states"].items():
        state["provenance"]["git_commit"] = "0" * 40
        torch.save(state, value["directory"] / name)
    value["summary"]["provenance"]["git_commit"] = "0" * 40
    write_json(value["directory"] / "summary.json", value["summary"])
    with pytest.raises(ValueError, match="declared clean pin"):
        run(value)
