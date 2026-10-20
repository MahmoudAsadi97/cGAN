"""Small CPU checks for model, checkpoint, and evaluation isolation contracts."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from energy_cgan.models import Discriminator, Generator
from energy_cgan.training import generate, train, validate_checkpoint_data


@pytest.fixture(autouse=True)
def one_torch_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_bundle(path: Path, test_multiplier=1.0):
    path.mkdir()
    rng = np.random.default_rng(8)
    x = rng.uniform(0.2, 2.0, size=(24, 24)).astype(np.float32)
    x[16:] *= test_multiplier
    np.savez_compressed(
        path / "dataset.npz",
        x=x,
        y=np.tile([0, 1], 12).astype(np.int64),
        split=np.array(["train"] * 12 + ["val"] * 4 + ["test"] * 8),
        meter_id=np.array([f"meter-{i}" for i in range(24)]),
        date=np.array(["2025-01-01"] * 24),
    )
    (path / "metadata.json").write_text(json.dumps({
        "classes": ["weekday", "weekend"], "unit": "kWh",
        "interval_hours": 1.0, "target": "consumption", "provenance": {"source": "unit-test"},
    }))
    return path


def small_config(**updates):
    return dict(epochs=2, batch_size=4, seed=11, hidden_dim=8,
                latent_dim=3, device="cpu", validation_every=1,
                max_validation_samples=4, **updates)


def test_architecture_uses_logits_and_linear_baseline():
    generator = Generator(profile_length=8, num_classes=2)
    discriminator = Discriminator(profile_length=8, num_classes=2)
    linears = [layer for layer in generator.modules() if isinstance(layer, torch.nn.Linear)]
    assert [(layer.in_features, layer.out_features) for layer in linears] == [
        (26, 64), (64, 128), (128, 64), (64, 8),
    ]
    linears = [layer for layer in discriminator.modules() if isinstance(layer, torch.nn.Linear)]
    assert [(layer.in_features, layer.out_features) for layer in linears] == [
        (10, 64), (64, 128), (128, 256), (256, 1),
    ]
    assert not any(isinstance(layer, torch.nn.Sigmoid) for layer in discriminator.modules())
    assert sum(isinstance(layer, torch.nn.Dropout) for layer in discriminator.modules()) == 1
    assert generator(torch.randn(3, 24), torch.tensor([0, 1, 0])).shape == (3, 8)
    assert discriminator(torch.rand(3, 8), torch.tensor([0, 1, 0])).shape == (3, 1)


def test_cpu_training_checkpoint_and_reproducible_generation(tmp_path):
    bundle = make_bundle(tmp_path / "data")
    result = train(bundle, tmp_path / "model", small_config())
    checkpoint = torch.load(result["best_checkpoint"], map_location="cpu", weights_only=True)
    assert checkpoint["optimizer_g"] and checkpoint["optimizer_d"]
    assert checkpoint["manifest"]["dataset_sha256"]
    assert checkpoint["scaler"]["min"] == pytest.approx(float(np.load(bundle / "dataset.npz")["x"][:12].min()))
    assert len(json.loads((tmp_path / "model" / "history.json").read_text())) == 2
    labels = np.array([0, 1, 1, 0], dtype=np.int64)
    first = generate(result["best_checkpoint"], labels, seed=99)
    second = generate(result["best_checkpoint"], labels, seed=99)
    assert first.shape == (4, 24)
    assert first.dtype == np.float32
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, second)
    with pytest.raises(ValueError, match="label"):
        generate(result["best_checkpoint"], np.array([2]))


def test_test_values_cannot_affect_scaler_training_or_model_selection(tmp_path):
    first_data = make_bundle(tmp_path / "first-data")
    second_data = make_bundle(tmp_path / "second-data", test_multiplier=1_000_000)
    a = train(first_data, tmp_path / "first-model", small_config())
    b = train(second_data, tmp_path / "second-model", small_config())
    first = torch.load(a["best_checkpoint"], weights_only=True)
    second = torch.load(b["best_checkpoint"], weights_only=True)
    assert first["scaler"] == second["scaler"]
    assert first["epoch"] == second["epoch"]
    assert first["validation_metric"] == second["validation_metric"]
    for key in first["generator_state_dict"]:
        torch.testing.assert_close(first["generator_state_dict"][key], second["generator_state_dict"][key], rtol=0, atol=0)


def test_softplus_uses_physical_zero_and_returns_nonnegative_values(tmp_path):
    bundle = make_bundle(tmp_path / "data")
    result = train(bundle, tmp_path / "model", small_config(output_activation="softplus"))
    checkpoint = torch.load(result["best_checkpoint"], weights_only=True)
    assert checkpoint["scaler"]["min"] == 0.0
    assert (generate(result["best_checkpoint"], np.array([0, 1])) >= 0).all()


@pytest.mark.parametrize("field,value,message", [
    ("meter_id", "meter-0", "meter"),
    ("y", 9, "label"),
])
def test_rejects_meter_leakage_and_unknown_labels(tmp_path, field, value, message):
    bundle = make_bundle(tmp_path / "data")
    with np.load(bundle / "dataset.npz", allow_pickle=False) as stored:
        arrays = {key: stored[key] for key in stored.files}
    arrays[field][12] = value
    np.savez_compressed(bundle / "dataset.npz", **arrays)
    with pytest.raises(ValueError, match=message):
        train(bundle, tmp_path / "model", small_config())


def test_rejects_nonfinite_training_input(tmp_path):
    bundle = make_bundle(tmp_path / "data")
    with np.load(bundle / "dataset.npz", allow_pickle=False) as stored:
        arrays = {key: stored[key] for key in stored.files}
    arrays["x"][0, 0] = np.nan
    np.savez_compressed(bundle / "dataset.npz", **arrays)
    with pytest.raises(ValueError, match="finite"):
        train(bundle, tmp_path / "model", small_config())


def test_checkpoint_data_verification_rejects_changed_data_and_metadata(tmp_path):
    bundle = make_bundle(tmp_path / "data")
    other = make_bundle(tmp_path / "other", test_multiplier=2)
    result = train(bundle, tmp_path / "model", small_config())
    assert validate_checkpoint_data(result["best_checkpoint"], bundle)["verified"]
    with pytest.raises(ValueError, match="dataset"):
        validate_checkpoint_data(result["best_checkpoint"], other)
    metadata_path = bundle / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["classes"].reverse()
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="metadata"):
        validate_checkpoint_data(result["best_checkpoint"], bundle)


def test_softplus_rejects_net_target_even_when_nonnegative(tmp_path):
    bundle = make_bundle(tmp_path / "data")
    metadata_path = bundle / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["target"] = "net"
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="net"):
        train(bundle, tmp_path / "model", small_config(output_activation="softplus"))


def test_rejects_incomplete_day_and_existing_partial_results(tmp_path):
    bundle = make_bundle(tmp_path / "data")
    output = tmp_path / "model"
    output.mkdir()
    (output / "config.json").write_text("{}")
    with pytest.raises(FileExistsError, match="nonempty"):
        train(bundle, output, small_config())
    with np.load(bundle / "dataset.npz", allow_pickle=False) as stored:
        arrays = {key: stored[key] for key in stored.files}
    arrays["x"] = arrays["x"][:, :12]
    np.savez_compressed(bundle / "dataset.npz", **arrays)
    with pytest.raises(ValueError, match="24"):
        train(bundle, tmp_path / "fresh", small_config())


def test_auto_device_uses_cpu_when_cuda_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    bundle = make_bundle(tmp_path / "data")
    config = small_config()
    config["device"] = "auto"
    result = train(bundle, tmp_path / "model", config)
    checkpoint = torch.load(result["best_checkpoint"], weights_only=True)
    assert checkpoint["config"]["device"] == "cpu"
