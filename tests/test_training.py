"""Training invariants that protect held-out data and reproducible experiments."""
import hashlib
import json
from pathlib import Path
import subprocess

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


@pytest.mark.parametrize("normalization", ["minmax", "standard"])
@pytest.mark.parametrize("objective,adam_beta1", [("bce", 0.9), ("hinge_sn", 0.5)])
def test_replay_isolation_scaler_provenance_and_artifacts(tmp_path, normalization, objective, adam_beta1):
    data = prepared(tmp_path)
    out1, out2 = tmp_path / "first", tmp_path / "second"
    options = settings(normalization=normalization, objective=objective, adam_beta1=adam_beta1)
    first = train(data, out1, options)
    second = train(data, out2, options)
    assert first["best_validation_score"] == second["best_validation_score"]
    assert first["best_epoch"] == second["best_epoch"]
    assert first["epochs_completed"] == 2
    assert first["generator_updates"] == first["discriminator_updates"] == 6
    history = json.loads((out1 / "history.json").read_text())
    assert all(row["generator_updates"] == row["discriminator_updates"] == 3 for row in history)
    assert {p.name for p in out1.iterdir()} == {"best.pt", "last.pt", "config.json", "history.json", "summary.json"}
    checkpoint = torch.load(out1 / "best.pt", weights_only=True)
    for name in ("optimizer_g_state_dict", "optimizer_d_state_dict"):
        assert checkpoint[name]["param_groups"][0]["betas"] == (adam_beta1, 0.999)
    expected_scaler = json.loads((data / "scaler.json").read_text())
    if normalization == "standard":
        x = np.load(data / "train.npz")["x"].astype(np.float64)
        expected_scaler.update(kind="standard", mean=float(x.mean()), std=float(x.std()))
    assert checkpoint["scaler"] == expected_scaler
    assert checkpoint["supported_labels"] == [0, 8]
    assert "test.npz" not in checkpoint["data_hashes"]
    assert checkpoint["data_hashes"]["train.npz"] == hashlib.sha256((data / "train.npz").read_bytes()).hexdigest()
    assert checkpoint["data_hashes"]["scaler.json"] == hashlib.sha256((data / "scaler.json").read_bytes()).hexdigest()
    assert checkpoint["metadata"]["source_sha256"] == "a" * 64
    assert checkpoint["provenance"]["git_commit_tree"] == subprocess.check_output(
        ["git", "rev-parse", "HEAD^{tree}"], cwd=Path(__file__).resolve().parents[1], text=True).strip()
    root = Path(__file__).resolve().parents[1]
    assert checkpoint["provenance"]["source_files_sha256"] == {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((root / "src" / "fluvius_cgan").glob("*.py"))}
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
    ("max_train_samples", 0), ("patience", 0), ("unknown", 1),
    ("normalization", "robust"), ("normalization", None), ("normalization", []),
    ("objective", "wasserstein"), ("objective", None), ("objective", []),
    ("adam_beta1", -0.1), ("adam_beta1", True), ("adam_beta1", 1),
    ("adam_beta2", float("nan")), ("adam_beta2", float("inf")), ("adam_beta2", 1),
    ("discriminator_hidden_dim", 0), ("discriminator_hidden_dim", True),
    ("conditional_mean_residual", "false"), ("conditional_mean_residual", 1),
    ("conditional_mean_residual", None)])
def test_invalid_config_rejected(tmp_path, key, value):
    data = prepared(tmp_path)
    with pytest.raises(ValueError):
        train(data, tmp_path / "run", settings(**{key: value}))


def test_inverse_scaling_is_used_for_generation(tmp_path):
    data = prepared(tmp_path)
    out = tmp_path / "run"
    train(data, out, settings(epochs=1))
    checkpoint = torch.load(out / "best.pt", weights_only=True)
    # Checkpoints from before objective/normalization config fields still load.
    for name in ("objective", "normalization", "adam_beta1", "adam_beta2",
                 "discriminator_hidden_dim", "conditional_mean_residual"):
        checkpoint["config"].pop(name, None)
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


def test_standard_transform_fits_complete_train_and_applies_to_validation(tmp_path, monkeypatch):
    data = prepared(tmp_path)
    import fluvius_cgan.training as training
    observed = []
    original = training._normalized

    def observe(x, scaler):
        values = original(x, scaler)
        observed.append((x.copy(), values.copy(), scaler.copy()))
        return values

    monkeypatch.setattr(training, "_normalized", observe)
    out = tmp_path / "run"
    train(data, out, settings(epochs=1, max_train_samples=1, normalization="standard"))
    checkpoint = torch.load(out / "best.pt", weights_only=True)
    full_train = np.load(data / "train.npz")["x"].astype(np.float64)
    mean, std = float(full_train.mean()), float(full_train.std())
    assert checkpoint["scaler"]["mean"] == mean
    assert checkpoint["scaler"]["std"] == std
    assert observed[0][0].shape == (1, 24)
    assert float(observed[0][0].mean()) != pytest.approx(mean)
    assert len(observed) == 2
    for raw, normalized, scaler in observed:
        np.testing.assert_array_equal(normalized, ((raw.astype(np.float64) - mean) / std).astype(np.float32))
        assert scaler == checkpoint["scaler"]
    assert observed[1][1].min() > 1  # Validation is not fitted or clipped.


@pytest.mark.parametrize("constant", [False, True])
def test_standard_inverse_preserves_model_output_and_zero_std(tmp_path, constant):
    data = prepared(tmp_path)
    if constant:
        change_npz(data / "train.npz", x=np.full((8, 24), 2., dtype=np.float32))
        (data / "scaler.json").write_text('{"minimum": 2.0, "maximum": 2.0}')
    out = tmp_path / "run"
    train(data, out, settings(epochs=1, normalization="standard"))
    checkpoint = torch.load(out / "best.pt", weights_only=True)
    if constant:
        assert checkpoint["scaler"]["std"] == 0
    for key, value in checkpoint["generator_state_dict"].items():
        checkpoint["generator_state_dict"][key] = torch.zeros_like(value)
    final_bias = [key for key in checkpoint["generator_state_dict"] if key.endswith("bias")][-1]
    checkpoint["generator_state_dict"][final_bias].fill_(2.)
    torch.save(checkpoint, out / "fixed.pt")
    scaler = checkpoint["scaler"]
    expected = np.float32(scaler["mean"] + 2 * (scaler["std"] or 1.))
    np.testing.assert_array_equal(generate(out / "fixed.pt", np.array([0, 8])), np.full((2, 24), expected))


def test_hinge_losses_have_the_requested_scale_and_gradients():
    from fluvius_cgan.training import _discriminator_loss, _generator_loss
    real = torch.tensor([[-2.], [0.5], [2.]], requires_grad=True)
    fake = torch.tensor([[-2.], [-0.5], [2.]], requires_grad=True)
    loss = _discriminator_loss(real, fake, "hinge_sn")
    assert loss.item() == pytest.approx(7 / 3)
    loss.backward()
    torch.testing.assert_close(real.grad, torch.tensor([[-1 / 3], [-1 / 3], [0.]]))
    torch.testing.assert_close(fake.grad, torch.tensor([[0.], [1 / 3], [1 / 3]]))
    fake.grad = None
    loss_g = _generator_loss(fake, "hinge_sn")
    assert loss_g.item() == pytest.approx(1 / 6)
    loss_g.backward()
    torch.testing.assert_close(fake.grad, torch.full_like(fake, -1 / 3))
    expected_d = (torch.nn.functional.binary_cross_entropy_with_logits(real, torch.ones_like(real)) +
                  torch.nn.functional.binary_cross_entropy_with_logits(fake, torch.zeros_like(fake))) / 2
    torch.testing.assert_close(_discriminator_loss(real, fake, "bce"), expected_d, rtol=0, atol=0)
    torch.testing.assert_close(_generator_loss(fake, "bce"),
                               torch.nn.functional.binary_cross_entropy_with_logits(fake, torch.ones_like(fake)),
                               rtol=0, atol=0)


def test_spectral_discriminator_preserves_widths_and_freezes_buffers_for_generator():
    from fluvius_cgan.training import _frozen_discriminator
    g = Generator(latent_dim=4, hidden_dim=4, output_dim=24)
    d = Discriminator(input_dim=24, hidden_dim=4, objective="hinge_sn")
    linear = [module for module in d.modules() if isinstance(module, torch.nn.Linear)]
    assert [module.out_features for module in linear] == [4, 8, 16, 1]
    assert all(torch.nn.utils.parametrize.is_parametrized(module, "weight") for module in linear)
    assert all(module.p == 0 for module in d.modules() if isinstance(module, torch.nn.Dropout))
    baseline = Discriminator(input_dim=24, hidden_dim=4)
    assert not any(torch.nn.utils.parametrize.is_parametrized(module) for module in baseline.modules())
    assert [module.p for module in baseline.modules() if isinstance(module, torch.nn.Dropout)] == [0.4]
    buffers_before = {name: value.clone() for name, value in d.named_buffers()}
    with _frozen_discriminator(d, freeze_buffers=True):
        assert not d.training
        fake = g(torch.randn(3, 4), torch.zeros(3, 64))
        (-d(fake, torch.zeros(3, 64)).mean()).backward()
    assert d.training and all(parameter.requires_grad for parameter in d.parameters())
    assert all(parameter.grad is None for parameter in d.parameters())
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in g.parameters())
    assert any(torch.any(parameter.grad != 0) for parameter in g.parameters())
    for name, value in d.named_buffers():
        torch.testing.assert_close(value, buffers_before[name], rtol=0, atol=0)


def test_discriminator_width_can_change_without_generator_width(tmp_path):
    data = prepared(tmp_path)
    out = tmp_path / "run"
    train(data, out, settings(epochs=1, discriminator_hidden_dim=8))
    checkpoint = torch.load(out / "best.pt", weights_only=True)
    assert checkpoint["discriminator_state_dict"]["model.0.weight"].shape == (8, 88)
    assert checkpoint["generator_state_dict"]["model.0.weight"].shape == (4, 68)
    assert checkpoint["config"]["discriminator_hidden_dim"] == 8


@pytest.mark.parametrize("normalization", ["minmax", "standard"])
def test_residual_fits_selected_train_and_replays_inverse(tmp_path, normalization):
    data = prepared(tmp_path)
    options = settings(epochs=1, max_train_samples=3, normalization=normalization,
                       conditional_mean_residual=True)
    out, replay = tmp_path / "first", tmp_path / "replay"
    train(data, out, options)
    train(data, replay, options)
    checkpoint = torch.load(out / "best.pt", weights_only=True)
    with np.load(data / "train.npz") as saved_train:
        raw, labels = saved_train["x"].astype(np.float64), saved_train["y"]
    selected = np.sort(np.random.default_rng(options["seed"]).choice(len(raw), 3, replace=False))
    offset = raw.mean() if normalization == "standard" else raw.min()
    scale = raw.std() if normalization == "standard" else raw.max() - raw.min()
    normalized = ((raw - offset) / scale).astype(np.float32)
    state = checkpoint["generator_state_dict"]
    for label in np.unique(labels[selected]):
        mask = labels[selected] == label
        expected = normalized[selected][mask].astype(np.float64).mean(axis=0).astype(np.float32)
        np.testing.assert_array_equal(state["mean_profiles"][label].numpy(), expected)
        assert not np.array_equal(expected, normalized[labels == label].mean(axis=0))
    support = np.zeros(64, dtype=bool)
    support[np.unique(labels[selected])] = True
    np.testing.assert_array_equal(state["supported_conditions"].numpy(), support)
    generation_labels = np.array(checkpoint["supported_labels"] * 2)
    np.testing.assert_array_equal(generate(out / "best.pt", generation_labels, seed=31),
                                  generate(replay / "best.pt", generation_labels, seed=31))
    # Zero only the learned residual parameters; preserve fixed offsets/support.
    for name, value in state.items():
        if name.startswith("base."):
            state[name] = torch.zeros_like(value)
    torch.save(checkpoint, out / "zero_residual.pt")
    expected_raw = (state["mean_profiles"][generation_labels].numpy().astype(np.float64) * scale + offset).astype(np.float32)
    np.testing.assert_array_equal(generate(out / "zero_residual.pt", generation_labels), expected_raw)
    # Outer support metadata alone must not override the restored wrapper mask.
    absent = int(np.flatnonzero(~support)[0])
    checkpoint["supported_labels"].append(absent)
    torch.save(checkpoint, out / "altered_support.pt")
    with pytest.raises(ValueError, match="unsupported"):
        generate(out / "altered_support.pt", np.array([absent]))


def test_unconditional_residual_uses_one_empirical_global_mean_without_labels(tmp_path):
    data = prepared(tmp_path)
    # Unequal condition counts distinguish empirical global means from an equal
    # average of per-condition means. Technology remains constant within meters.
    change_npz(data / "train.npz", y=np.array([0, 0, 0, 0, 0, 0, 8, 8]))
    out = tmp_path / "run"
    train(data, out, settings(epochs=1, unconditional=True, conditional_mean_residual=True))
    checkpoint = torch.load(out / "best.pt", weights_only=True)
    raw = np.load(data / "train.npz")["x"].astype(np.float64)
    normalized = ((raw - raw.min()) / (raw.max() - raw.min())).astype(np.float32)
    expected = normalized.astype(np.float64).mean(axis=0).astype(np.float32)[None, :]
    np.testing.assert_array_equal(checkpoint["generator_state_dict"]["mean_profiles"].numpy(), expected)
    assert checkpoint["generator_state_dict"]["supported_conditions"].tolist() == [True]
    np.testing.assert_array_equal(generate(out / "best.pt", np.array([0, 8, 0]), seed=31),
                                  generate(out / "best.pt", np.array([8, 0, 8]), seed=31))
