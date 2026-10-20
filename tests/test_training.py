"""Training invariants that protect held-out data and reproducible experiments."""
import hashlib
import json

import numpy as np
import pytest
import torch

from fluvius_cgan.models import Discriminator, Generator
from fluvius_cgan.training import generate, train


def prepared(tmp_path):
    directory = tmp_path / "prepared"
    directory.mkdir()
    rng = np.random.default_rng(10)
    for split, n, offset in [("train", 8, 0), ("val", 6, 30)]:
        # Weekdays in winter; technology groups 0 and 1.
        x = rng.normal(offset, 2, size=(n, 24)).astype(np.float32)
        np.savez(directory / f"{split}.npz", x=x,
                 y=np.array([(i // 2 % 2) * 8 for i in range(n)], dtype=np.int64),
                 meter_id=np.array([f"{split}_{i // 2}" for i in range(n)]),
                 date=np.array(["2024-01-01", "2024-01-02"] * (n // 2)))
    # Deliberately unreadable: trainer must never load or hash test profiles.
    (directory / "test.npz").write_bytes(b"test must remain untouched")
    x = np.load(directory / "train.npz")["x"]
    (directory / "scaler.json").write_text(json.dumps({
        "minimum": float(x.min()), "maximum": float(x.max())}))
    (directory / "metadata.json").write_text(json.dumps({
        "resolution": 24, "target": "net", "units": "kWh/interval",
        "source_sha256": "a" * 64, "split_seed": 42,
        "timezone_policy": "source local CET/CEST wall-clock",
        "label_formula": "((HP*4+EV*2+PV)*2+weekend)*4+season",
        "splits": {"train": {"profiles": 8, "meters": 4},
                   "val": {"profiles": 6, "meters": 3},
                   "test": {"profiles": 999, "meters": 99}}}))
    return directory


def settings(**overrides):
    return {"epochs": 2, "batch_size": 3, "latent_dim": 4,
            "hidden_dim": 4, "validation_samples": 5,
            "num_threads": 1, "seed": 12, **overrides}


def change_npz(path, **updates):
    with np.load(path, allow_pickle=False) as data:
        arrays = dict(data)
    arrays.update(updates)
    np.savez(path, **arrays)


@pytest.mark.parametrize("condition_dim", [0, 64])
def test_model_shapes_gradients_and_logits(condition_dim):
    g = Generator(latent_dim=4, condition_dim=condition_dim, hidden_dim=4, output_dim=24)
    d = Discriminator(input_dim=24, condition_dim=condition_dim, hidden_dim=4)
    c = torch.zeros(3, condition_dim)
    x = g(torch.randn(3, 4), c)
    logits = d(x, c)
    assert x.shape == (3, 24) and logits.shape == (3, 1)
    torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.ones_like(logits)).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in g.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in d.parameters())
    assert not any(isinstance(m, torch.nn.Sigmoid) for m in d.modules())
    assert [m.out_features for m in g.modules() if isinstance(m, torch.nn.Linear)] == [4, 8, 4, 24]
    assert [m.out_features for m in d.modules() if isinstance(m, torch.nn.Linear)] == [4, 8, 16, 1]


def test_replay_isolation_scaler_provenance_and_artifacts(tmp_path):
    data = prepared(tmp_path)
    out1, out2 = tmp_path / "first", tmp_path / "second"
    first = train(data, out1, settings())
    second = train(data, out2, settings())
    assert first["best_validation_score"] == second["best_validation_score"]
    assert first["best_epoch"] == second["best_epoch"]
    assert first["epochs_completed"] == 2
    assert {p.name for p in out1.iterdir()} == {"best.pt", "last.pt", "config.json", "history.json", "summary.json"}
    checkpoint = torch.load(out1 / "best.pt", weights_only=True)
    assert checkpoint["scaler"] == json.loads((data / "scaler.json").read_text())
    assert checkpoint["supported_labels"] == [0, 8]
    assert "test.npz" not in checkpoint["data_hashes"]
    assert checkpoint["data_hashes"]["train.npz"] == hashlib.sha256((data / "train.npz").read_bytes()).hexdigest()
    assert checkpoint["metadata"]["source_sha256"] == "a" * 64
    labels = np.array([0, 8, 0], dtype=np.int64)
    a = generate(out1 / "best.pt", labels, seed=30)
    b = generate(out2 / "best.pt", labels, seed=30)
    np.testing.assert_array_equal(a, b)
    assert a.shape == (3, 24) and np.isfinite(a).all()
    assert not np.array_equal(a, generate(out1 / "best.pt", labels, seed=31))
    with pytest.raises(ValueError, match="unsupported"):
        generate(out1 / "best.pt", np.array([63]))
    with pytest.raises(ValueError, match="empty"):
        train(data, out1, settings())


@pytest.mark.parametrize("unconditional", [False, True])
def test_support_is_actual_smoke_subset_and_validation_noise_fixed(tmp_path, monkeypatch, unconditional):
    data = prepared(tmp_path)
    import fluvius_cgan.training as training
    observed = []
    original = training._validation_score

    def observe(generator, normalized, labels, noise, condition_dim, device, batch_size):
        observed.append((labels.copy(), noise.clone()))
        return original(generator, normalized, labels, noise, condition_dim, device, batch_size)

    monkeypatch.setattr(training, "_validation_score", observe)
    out = tmp_path / "run"
    train(data, out, settings(max_train_samples=1, unconditional=unconditional))
    checkpoint = torch.load(out / "best.pt", weights_only=True)
    assert len(checkpoint["supported_labels"]) == 1
    assert len(observed) == 2
    np.testing.assert_array_equal(observed[0][0], observed[1][0])
    assert torch.equal(observed[0][1], observed[1][1])
    assert set(observed[0][0].tolist()) == set(checkpoint["supported_labels"])
    missing = ({0, 8} - set(checkpoint["supported_labels"])).pop()
    with pytest.raises(ValueError, match="unsupported"):
        generate(out / "best.pt", np.array([missing]))


def test_validation_selects_best_checkpoint_and_patience(tmp_path, monkeypatch):
    data = prepared(tmp_path)
    scores = iter([0.3, 0.1, 0.2, 0.4])
    monkeypatch.setattr("fluvius_cgan.training._validation_score", lambda *args: next(scores))
    out = tmp_path / "run"
    summary = train(data, out, settings(epochs=8, patience=2))
    assert summary["best_epoch"] == 2
    assert summary["best_validation_score"] == 0.1
    assert summary["epochs_completed"] == 4
    assert torch.load(out / "best.pt", weights_only=True)["epoch"] == 2
    assert torch.load(out / "last.pt", weights_only=True)["epoch"] == 4


@pytest.mark.parametrize("violation", ["scaler", "overlap", "resolution", "counts", "source_hash", "nan", "labels", "duplicate_day"])
def test_provenance_and_input_mismatches_fail_closed(tmp_path, violation):
    data = prepared(tmp_path)
    if violation == "scaler":
        (data / "scaler.json").write_text('{"minimum": -1000, "maximum": 1000}')
    elif violation == "overlap":
        change_npz(data / "val.npz", meter_id=np.array(["train_0"] * 6))
    elif violation in {"resolution", "counts", "source_hash"}:
        meta = json.loads((data / "metadata.json").read_text())
        if violation == "resolution":
            meta["resolution"] = 96
        elif violation == "counts":
            meta["splits"]["train"]["profiles"] = 1
        else:
            meta["source_sha256"] = "not-a-hash"
        (data / "metadata.json").write_text(json.dumps(meta))
    elif violation == "nan":
        x = np.load(data / "train.npz")["x"]
        x[0, 0] = np.nan
        change_npz(data / "train.npz", x=x)
    elif violation == "labels":
        change_npz(data / "train.npz", y=np.array([64] * 8))
    elif violation == "duplicate_day":
        change_npz(data / "train.npz", date=np.array(["2024-01-01"] * 8))
    with pytest.raises(ValueError):
        train(data, tmp_path / "run", settings())


@pytest.mark.parametrize("key,value", [("epochs", 0), ("batch_size", -1), ("seed", -1),
    ("latent_dim", True), ("lr_g", float("nan")), ("device", "magic"),
    ("num_threads", 0), ("unconditional", "false"), ("validation_samples", 0),
    ("max_train_samples", 0), ("patience", 0), ("unknown", 1)])
def test_invalid_config_rejected(tmp_path, key, value):
    data = prepared(tmp_path)
    with pytest.raises(ValueError):
        train(data, tmp_path / "run", settings(**{key: value}))


def test_inverse_scaling_is_used_for_generation(tmp_path):
    data = prepared(tmp_path)
    out = tmp_path / "run"
    train(data, out, settings(epochs=1))
    checkpoint = torch.load(out / "best.pt", weights_only=True)
    for key, value in checkpoint["generator_state_dict"].items():
        checkpoint["generator_state_dict"][key] = torch.zeros_like(value)
    torch.save(checkpoint, out / "zero.pt")
    values = generate(out / "zero.pt", np.array([0, 8]))
    np.testing.assert_allclose(values, checkpoint["scaler"]["minimum"], rtol=0, atol=0)


def test_full_training_rejects_unsupported_validation_conditions(tmp_path):
    data = prepared(tmp_path)
    change_npz(data / "val.npz", y=np.array([0, 0, 16, 16, 0, 0], dtype=np.int64))
    with pytest.raises(ValueError, match="unsupported"):
        train(data, tmp_path / "run", settings())


@pytest.mark.parametrize("target", ["import", "injection"])
def test_target_metadata_rejects_observed_negative_channel_energy(tmp_path, target):
    data = prepared(tmp_path)
    meta = json.loads((data / "metadata.json").read_text())
    meta["target"] = target
    (data / "metadata.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="negative"):
        train(data, tmp_path / "run", settings())


def test_run_directory_is_reserved_before_data_loading(tmp_path, monkeypatch):
    data = prepared(tmp_path)
    import fluvius_cgan.training as training
    original = training._load_prepared
    output = tmp_path / "run"
    nested_attempted = False

    def load_while_another_call_starts(prepared_dir):
        nonlocal nested_attempted
        if not nested_attempted:
            nested_attempted = True
            with pytest.raises(ValueError, match="empty|active"):
                train(data, output, settings(epochs=1))
        return original(prepared_dir)

    monkeypatch.setattr(training, "_load_prepared", load_while_another_call_starts)
    train(data, output, settings(epochs=1))
    assert nested_attempted
    assert not (output / ".training.lock").exists()


def test_constant_scaler_inverse_preserves_raw_model_output(tmp_path):
    data = prepared(tmp_path)
    out = tmp_path/'run'
    train(data, out, settings(epochs=1))
    saved = torch.load(out/'best.pt', weights_only=True)
    saved['scaler'] = {'minimum': 2., 'maximum': 2.}
    for key, value in saved['generator_state_dict'].items():
        saved['generator_state_dict'][key] = torch.zeros_like(value)
    final_bias = [k for k in saved['generator_state_dict'] if k.endswith('bias')][-1]
    saved['generator_state_dict'][final_bias].fill_(1.)
    torch.save(saved, out/'constant.pt')
    # Forward normalization uses denominator1 if span0, so its inverse is y+min.
    np.testing.assert_array_equal(generate(out/'constant.pt', np.array([0])), np.full((1,24), 3.))
