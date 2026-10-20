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
    ("conditional_mean_residual", None), ("batch_sampling", "weighted"),
    ("conditions_per_batch", 0), ("conditions_per_batch", True),
    ("swd_weight", -1), ("swd_weight", float("nan")), ("swd_weight", True),
    ("derivative_swd_weight", -1), ("derivative_swd_weight", float("nan")),
    ("derivative_swd_weight", float("inf")), ("derivative_swd_weight", True),
    ("swd_projections", 0), ("swd_projections", True), ("validation_selection", "unknown"),
    ("allow_uncurated_data", "true"), ("allow_uncurated_data", 1)])
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
                 "discriminator_hidden_dim", "conditional_mean_residual", "batch_sampling",
                 "conditions_per_batch", "swd_weight", "swd_projections", "validation_selection",
                 "allow_uncurated_data", "derivative_swd_weight"):
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


def test_training_batches_preserve_shuffle_and_balance_distinct_conditions():
    from fluvius_cgan.training import _training_batches
    expected = torch.randperm(11, generator=torch.Generator().manual_seed(15))
    batches = list(_training_batches(11, 4, torch.Generator().manual_seed(15)))
    assert [len(batch) for batch in batches] == [4, 4, 3]
    torch.testing.assert_close(torch.cat(batches), expected, rtol=0, atol=0)
    labels = torch.tensor([0, 0, 8, 16, 16, 24, 32, 40, 40, 40, 40])
    pools = [torch.where(labels == label)[0] for label in labels.unique()]
    balanced = list(_training_batches(len(labels), 8, torch.Generator().manual_seed(15),
                                      condition_indices=pools, conditions_per_batch=4))
    assert len(balanced) == 2
    for batch in balanced:
        assert len(batch) == 8 and ((batch >= 0) & (batch < len(labels))).all()
        classes, counts = labels[batch].unique(return_counts=True)
        assert len(classes) == 4 and counts.tolist() == [2] * 4


def test_zero_swd_default_preserves_training_and_never_calls_auxiliary(tmp_path, monkeypatch):
    data = prepared(tmp_path)
    import fluvius_cgan.training as training
    monkeypatch.setattr(training, "conditional_sliced_wasserstein_loss",
                        lambda *args, **kwargs: pytest.fail("zero-weight SWD must not be evaluated"))
    one, two = tmp_path / "default", tmp_path / "explicit"
    first = train(data, one, settings())
    second = train(data, two, settings(batch_sampling="shuffle", conditions_per_batch=1,
                                      swd_weight=0., swd_projections=3))
    assert first["best_validation_score"] == second["best_validation_score"]
    a, b = (torch.load(path / "last.pt", weights_only=True) for path in (one, two))
    for name in ("generator_state_dict", "discriminator_state_dict"):
        for key in a[name]:
            torch.testing.assert_close(a[name][key], b[name][key], rtol=0, atol=0)
    assert first["samples_seen"] == 16
    assert first["class_exposure_counts"] == {"0": 8, "8": 8}
    history = json.loads((one / "history.json").read_text())
    assert all(row["loss_swd"] == 0 and row["loss_g_adversarial"] == row["loss_g"] for row in history)


def test_balanced_swd_uses_selected_rows_matching_labels_and_replays(tmp_path, monkeypatch):
    data = prepared(tmp_path)
    import fluvius_cgan.training as training
    original = training.conditional_sliced_wasserstein_loss
    options = settings(batch_sampling="balanced_conditions", conditions_per_batch=2,
                       batch_size=64, max_train_samples=3, normalization="standard",
                       swd_weight=0.1, swd_projections=4, conditional_mean_residual=True)
    with np.load(data / "train.npz") as archive:
        raw, labels = archive["x"].astype(np.float64), archive["y"]
    selected = np.sort(np.random.default_rng(options["seed"]).choice(len(raw), 3, replace=False))
    normalized = ((raw - raw.mean()) / raw.std()).astype(np.float32)
    observed = []

    def observe(real, generated, real_labels, generated_labels, **kwargs):
        torch.testing.assert_close(real_labels, generated_labels)
        assert real_labels.unique(return_counts=True)[1].tolist() == [32, 32]
        assert kwargs["p"] == 2 and kwargs["min_samples_per_condition"] == 32
        assert kwargs["num_projections"] == 4 and kwargs["generator"] is not None
        for label in real_labels.unique():
            allowed = torch.from_numpy(normalized[selected][labels[selected] == int(label)])
            assert ((real[real_labels == label, None, :] == allowed[None, :, :]).all(dim=2).any(dim=1)).all()
        loss = original(real, generated, real_labels, generated_labels, **kwargs)
        gradients = torch.autograd.grad(loss, generated, retain_graph=True)[0]
        assert torch.isfinite(gradients).all() and torch.any(gradients != 0)
        observed.append((real.detach().clone(), real_labels.clone()))
        return loss

    monkeypatch.setattr(training, "conditional_sliced_wasserstein_loss", observe)
    first, replay = tmp_path / "first", tmp_path / "replay"
    summary = train(data, first, options)
    repeat = train(data, replay, options)
    assert summary["best_validation_score"] == repeat["best_validation_score"]
    assert summary["samples_seen"] == 128
    assert summary["generator_updates"] == summary["discriminator_updates"] == 2
    assert summary["class_exposure_counts"] == {"0": 64, "8": 64}
    checkpoint = torch.load(first / "last.pt", weights_only=True)
    assert checkpoint["samples_seen"] == summary["samples_seen"]
    assert checkpoint["class_exposure_counts"] == summary["class_exposure_counts"]
    history = json.loads((first / "history.json").read_text())
    for row in history:
        assert row["samples_seen"] == 64
        assert row["class_exposure_counts"] == {"0": 32, "8": 32}
        assert row["loss_g"] == pytest.approx(row["loss_g_adversarial"] + .1 * row["loss_swd"])
    for before, after in zip(observed[:2], observed[2:]):
        for a, b in zip(before, after):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    np.testing.assert_array_equal(generate(first / "best.pt", np.array([0, 8])),
                                  generate(replay / "best.pt", np.array([0, 8])))


def test_unconditional_swd_pools_loss_labels_but_reports_sampled_groups(tmp_path, monkeypatch):
    data = prepared(tmp_path)
    import fluvius_cgan.training as training
    original = training.conditional_sliced_wasserstein_loss
    calls = []

    def observe(real, generated, real_labels, generated_labels, **kwargs):
        assert not real_labels.any() and not generated_labels.any()
        calls.append(len(real))
        return original(real, generated, real_labels, generated_labels, **kwargs)

    monkeypatch.setattr(training, "conditional_sliced_wasserstein_loss", observe)
    summary = train(data, tmp_path / "run", settings(epochs=1, unconditional=True,
                    batch_sampling="balanced_conditions", batch_size=64,
                    conditions_per_batch=2, swd_weight=.1, swd_projections=4))
    assert calls == [64]
    assert summary["class_exposure_counts"] == {"0": 32, "8": 32}


@pytest.mark.parametrize("overrides", [
    {"swd_weight": .1},
    {"batch_sampling": "balanced_conditions", "batch_size": 7, "conditions_per_batch": 2},
    {"batch_sampling": "balanced_conditions", "batch_size": 32, "conditions_per_batch": 2, "swd_weight": .1},
    {"batch_sampling": "balanced_conditions", "batch_size": 128, "conditions_per_batch": 4},
])
def test_invalid_distribution_batch_policy_rejected(tmp_path, overrides):
    with pytest.raises(ValueError):
        train(prepared(tmp_path), tmp_path / "run", settings(**overrides))


def test_conditional_validation_detects_swapped_conditions_hidden_by_pooling():
    from fluvius_cgan.training import _profile_validation_score
    labels = np.array([0, 0, 8, 8])
    real = np.repeat(np.array([[0.], [0.], [10.], [10.]]), 24, axis=1)
    generated = real[::-1].copy()
    assert _profile_validation_score(real, generated, labels, "pooled_w1") == 0
    assert _profile_validation_score(real, generated, labels, "conditional_w1") == 10
    # Every condition has equal importance even if its capped sample count differs.
    real = np.zeros((5, 24))
    generated = np.repeat(np.array([[1.], [1.], [1.], [9.], [9.]]), 24, axis=1)
    assert _profile_validation_score(real, generated, np.array([0, 0, 0, 8, 8]), "conditional_w1") == 5


def test_conditional_validation_sampling_and_metadata_are_fixed_across_arms(tmp_path):
    data = prepared(tmp_path)
    options = settings(batch_sampling="balanced_conditions", conditions_per_batch=2,
                       batch_size=64, validation_selection="conditional_w1",
                       validation_samples=4, swd_projections=4)
    control, auxiliary = tmp_path / "control", tmp_path / "auxiliary"
    summary = train(data, control, options)
    other = train(data, auxiliary, {**options, "swd_weight": .1})
    checkpoint = torch.load(control / "best.pt", weights_only=True)
    assert summary["validation_metric"] == checkpoint["validation_metric"] == "mean_condition_interval_wasserstein_normalized"
    selection = summary["selection"]
    assert selection["validation_selection"] == "conditional_w1"
    assert selection["validation_selection_version"] == 1
    assert selection["validation_samples_requested"] == 4
    assert selection["validation_per_condition_cap"] == 2
    assert selection["validation_profiles_per_condition"] == {"0": 2, "8": 2}
    assert len(set(selection["validation_indices"])) == 4
    assert selection == other["selection"]


@pytest.mark.parametrize("budget,missing_class", [(5, False), (2, False), (4, True)])
def test_conditional_validation_rejects_insufficient_or_indivisible_samples(tmp_path, budget, missing_class):
    data = prepared(tmp_path)
    if missing_class:
        change_npz(data / "val.npz", y=np.zeros(6, dtype=np.int64))
    with pytest.raises(ValueError, match="conditional validation"):
        train(data, tmp_path / "run", settings(validation_samples=budget, validation_selection="conditional_w1"))


def pinned_prepared(tmp_path, *, curate=False):
    from fluvius_cgan.release import RELEASE_SHA256
    data = prepared(tmp_path)
    metadata = json.loads((data / "metadata.json").read_text())
    metadata["source_sha256"] = RELEASE_SHA256
    if curate:
        with np.load(data / "val.npz") as archive:
            arrays = dict(archive)
        arrays["meter_id"] = np.array([value.replace("val", "test") for value in arrays["meter_id"]])
        arrays["x"] = arrays["x"] + 1000
        np.savez(data / "test.npz", **arrays)
        metadata["splits"]["test"] = {"profiles": 6, "meters": 3}
    (data / "metadata.json").write_text(json.dumps(metadata))
    if curate:
        from fluvius_cgan.curation import curate_prepared
        target = tmp_path / "curated"
        curate_prepared(data, target, expected_dates=np.array(["2024-01-01", "2024-01-02"]))
        return target
    return data


def test_pinned_training_requires_curation_before_loading_arrays(tmp_path, monkeypatch):
    data = pinned_prepared(tmp_path)
    monkeypatch.setattr("fluvius_cgan.training._load_prepared",
                        lambda *args: pytest.fail("uncurated source must be rejected before loading arrays"))
    with pytest.raises(ValueError, match="scripts/curate_prepared.py.*--prepared.*--output"):
        train(data, tmp_path / "run", settings(epochs=1))


def test_uncurated_override_warns_and_qualifies_saved_run_without_blocking_generation(tmp_path):
    data = pinned_prepared(tmp_path)
    output = tmp_path / "run"
    with pytest.warns(UserWarning, match="Exploratory"):
        summary = train(data, output, settings(epochs=1, allow_uncurated_data=True))
    checkpoint = torch.load(output / "best.pt", weights_only=True)
    assert summary["data_curation"]["status"] == "uncurated_explicit_override"
    assert checkpoint["data_curation"] == summary["data_curation"]
    assert "not confirmatory" in summary["data_curation"]["qualification"]
    # Generation of an older pinned-source checkpoint does not invoke the guard.
    checkpoint["config"].pop("allow_uncurated_data")
    torch.save(checkpoint, output / "legacy.pt")
    assert generate(output / "legacy.pt", np.array([0, 8])).shape == (2, 24)


def test_curated_guard_verifies_development_hashes_without_opening_test(tmp_path, monkeypatch):
    import fluvius_cgan.training as training
    data = pinned_prepared(tmp_path, curate=True)
    # Deliberately no longer matches the TEST hash; training must not check it.
    (data / "test.npz").write_bytes(b"reserved test must not be read or hashed")
    original_hash = training._sha256

    def guard_hash(path):
        assert Path(path).name != "test.npz"
        return original_hash(path)

    monkeypatch.setattr(training, "_sha256", guard_hash)
    output = tmp_path / "run"
    summary = train(data, output, settings(epochs=1))
    checkpoint = torch.load(output / "best.pt", weights_only=True)
    assert summary["data_curation"]["status"] == "verified_curation_v2"
    assert set(summary["data_curation"]["verified_file_hashes"]) == {"train.npz", "val.npz", "metadata.json", "scaler.json"}
    assert checkpoint["data_hashes"]["curation_manifest.json"] == original_hash(data / "curation_manifest.json")
    assert "test.npz" not in checkpoint["data_hashes"]


@pytest.mark.parametrize("violation", ["version", "algorithm", "priority", "flags", "manifest_version",
                                      "manifest_priority", "missing_manifest", "train.npz", "val.npz", "metadata.json", "scaler.json"])
def test_claimed_curation_fails_closed_for_malformed_fields_or_changed_hashes(tmp_path, violation):
    data = pinned_prepared(tmp_path, curate=True)
    metadata = json.loads((data / "metadata.json").read_text())
    manifest = json.loads((data / "curation_manifest.json").read_text())
    if violation in {"version", "algorithm", "priority", "flags"}:
        key, value = {"version": ("fingerprint_version", 1), "algorithm": ("fingerprint_algorithm", "unrecognized"),
                      "priority": ("split_priority", ["test", "val", "train"]),
                      "flags": ("development_traces_transferred_to_test", True)}[violation]
        metadata["curation"][key] = value
        (data / "metadata.json").write_text(json.dumps(metadata))
        manifest["output_files_sha256"]["metadata.json"] = hashlib.sha256((data / "metadata.json").read_bytes()).hexdigest()
    elif violation == "manifest_version":
        manifest["fingerprint_version"] = 1
    elif violation == "manifest_priority":
        manifest["split_priority"] = ["test", "val", "train"]
    elif violation == "missing_manifest":
        (data / "curation_manifest.json").unlink()
    else:
        with (data / violation).open("ab") as handle:
            handle.write(b"\n")
    if violation != "missing_manifest":
        (data / "curation_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="curation|manifest"):
        # An override never bypasses malformed claimed-curation validation.
        train(data, tmp_path / "run", settings(epochs=1, allow_uncurated_data=True))


def test_zero_derivative_weight_does_not_fit_or_evaluate_and_replays(tmp_path, monkeypatch):
    import fluvius_cgan.training as training
    data = prepared(tmp_path)
    monkeypatch.setattr(training, 'fit_derivative_scale', lambda *a, **k: pytest.fail('zero derivative weight must not fit'))
    monkeypatch.setattr(training, 'derivative_sliced_wasserstein_loss', lambda *a, **k: pytest.fail('zero derivative weight must not run'))
    options = settings(normalization='standard', batch_sampling='balanced_conditions',
                       conditions_per_batch=2, batch_size=64, swd_weight=10., swd_projections=4)
    train(data, tmp_path / 'implicit_zero', options)
    train(data, tmp_path / 'explicit_zero', {**options, 'derivative_swd_weight': 0.})
    before, after = (torch.load(tmp_path / name / 'last.pt', weights_only=True)
                     for name in ('implicit_zero', 'explicit_zero'))
    for key in ('generator_state_dict', 'discriminator_state_dict'):
        assert all(torch.equal(before[key][name], after[key][name]) for name in before[key])
    hb, ha = (json.loads((tmp_path / name / 'history.json').read_text())
              for name in ('implicit_zero', 'explicit_zero'))
    assert [{k: v for k, v in row.items() if k != 'elapsed_seconds'} for row in hb] == [
        {k: v for k, v in row.items() if k != 'elapsed_seconds'} for row in ha]
    assert before['derivative_swd_scale'] is None
    assert before['provenance']['derivative_swd']['projection_seed'] is None
    assert all(row['loss_derivative_swd'] == 0 for row in ha)


def test_derivative_scale_uses_exact_selected_train_rows_and_combined_loss_replays(tmp_path, monkeypatch):
    import fluvius_cgan.training as training
    data = prepared(tmp_path)
    options = settings(normalization='standard', batch_sampling='balanced_conditions', batch_size=64,
                       conditions_per_batch=2, max_train_samples=3, swd_weight=10., swd_projections=5,
                       derivative_swd_weight=1.)
    with np.load(data / 'train.npz', allow_pickle=False) as arrays:
        raw, labels = arrays['x'].astype(np.float64), arrays['y']
    selected = np.sort(np.random.default_rng(options['seed']).choice(len(raw), 3, replace=False))
    normalized = ((raw - raw.mean()) / raw.std()).astype(np.float32)[selected]
    selected_labels = labels[selected]
    differences = np.diff(normalized.astype(np.float64), axis=1)
    expected = np.sqrt(np.mean([np.mean(differences[selected_labels == label] ** 2)
                               for label in np.unique(selected_labels)]))
    original_fit, original_loss = training.fit_derivative_scale, training.derivative_sliced_wasserstein_loss
    fitting_calls, loss_calls = [], []

    def observe_fit(profiles, fit_labels, **kwargs):
        np.testing.assert_array_equal(profiles.numpy(), normalized)
        np.testing.assert_array_equal(fit_labels, selected_labels)
        fitting_calls.append(len(profiles))
        return original_fit(profiles, fit_labels, **kwargs)

    def observe_loss(real, fake, real_labels, generated_labels, scale, **kwargs):
        assert scale == float(np.float32(expected))
        assert kwargs['generator'].initial_seed() == options['seed'] + 5
        assert kwargs['num_projections'] == options['swd_projections']
        assert kwargs['min_samples_per_condition'] == 32
        assert torch.equal(real_labels, generated_labels)
        assert real_labels.unique(return_counts=True)[1].tolist() == [32, 32]
        result = original_loss(real, fake, real_labels, generated_labels, scale, **kwargs)
        gradient = torch.autograd.grad(result, fake, retain_graph=True)[0]
        assert torch.isfinite(gradient).all() and torch.any(gradient != 0)
        loss_calls.append(float(result.detach()))
        return result

    monkeypatch.setattr(training, 'fit_derivative_scale', observe_fit)
    monkeypatch.setattr(training, 'derivative_sliced_wasserstein_loss', observe_loss)
    first = train(data, tmp_path / 'derivative', options)
    second = train(data, tmp_path / 'replay', options)
    assert fitting_calls == [3, 3]
    assert first['best_validation_score'] == second['best_validation_score']
    checkpoint = torch.load(tmp_path / 'derivative' / 'last.pt', weights_only=True)
    scale = checkpoint['derivative_swd_scale']
    assert scale['estimate_float64'] == pytest.approx(expected, rel=1e-15)
    assert scale['used_float32'] == float(np.float32(expected))
    assert scale['train_indices_sha256'] == checkpoint['selection']['train_indices_sha256']
    assert scale['train_npz_sha256'] == checkpoint['data_hashes']['train.npz']
    assert scale['fitting_scope'] == 'actual_gradient_training_subset_only'
    assert scale == first['derivative_swd_scale'] == checkpoint['provenance']['derivative_swd']['scale']
    assert scale['condition_counts'] == {str(label): int(np.sum(selected_labels == label)) for label in np.unique(selected_labels)}
    assert scale['normalization']['fitting_scope'] == 'full_prepared_train_before_subset_selection'
    for row in json.loads((tmp_path / 'derivative' / 'history.json').read_text()):
        assert row['loss_derivative_swd'] > 0
        assert row['loss_g'] == pytest.approx(row['loss_g_adversarial'] + 10 * row['loss_swd'] + row['loss_derivative_swd'], rel=1e-6)
    assert loss_calls[:2] == loss_calls[2:]
    np.testing.assert_array_equal(generate(tmp_path / 'derivative' / 'best.pt', np.array([0, 8])),
                                  generate(tmp_path / 'replay' / 'best.pt', np.array([0, 8])))


def test_derivative_rng_does_not_consume_latent_sampler_or_profile_projection_streams(tmp_path, monkeypatch):
    import fluvius_cgan.training as training
    data = prepared(tmp_path)
    options = settings(normalization='standard', batch_sampling='balanced_conditions', conditions_per_batch=2,
                       batch_size=64, swd_weight=10., swd_projections=5)
    observed, phase = {0: {'noise': [], 'profile': []}, 1: {'noise': [], 'profile': []}}, [0]
    original_forward = Generator.forward
    original_profile = training.conditional_sliced_wasserstein_loss

    def forward(self, noise, condition):
        observed[phase[0]]['noise'].append(noise.detach().clone())
        return original_forward(self, noise, condition)

    def profile(real, generated, real_labels, generated_labels, **kwargs):
        observed[phase[0]]['profile'].append((real.detach().clone(), real_labels.clone(), kwargs['generator'].get_state().clone()))
        return original_profile(real, generated, real_labels, generated_labels, **kwargs)

    monkeypatch.setattr(Generator, 'forward', forward)
    monkeypatch.setattr(training, 'conditional_sliced_wasserstein_loss', profile)
    train(data, tmp_path / 'profile_only', options)
    phase[0] = 1
    train(data, tmp_path / 'profile_and_derivative', {**options, 'derivative_swd_weight': 1.})
    assert len(observed[0]['noise']) == len(observed[1]['noise'])
    assert all(torch.equal(a, b) for a, b in zip(observed[0]['noise'], observed[1]['noise']))
    assert len(observed[0]['profile']) == len(observed[1]['profile']) == options['epochs']
    assert all(all(torch.equal(a, b) for a, b in zip(left, right))
               for left, right in zip(observed[0]['profile'], observed[1]['profile']))


@pytest.mark.parametrize('overrides', [
    {'normalization': 'minmax', 'batch_sampling': 'balanced_conditions', 'batch_size': 64, 'conditions_per_batch': 2},
    {'normalization': 'standard', 'batch_sampling': 'shuffle'},
    {'normalization': 'standard', 'batch_sampling': 'balanced_conditions', 'batch_size': 32, 'conditions_per_batch': 2},
])
def test_derivative_training_scope_guards_precede_data_access(monkeypatch, overrides):
    import fluvius_cgan.training as training
    monkeypatch.setattr(training, '_load_prepared', lambda *a, **k: pytest.fail('invalid derivative config must fail before loading'))
    with pytest.raises(ValueError, match='standard normalization|balanced_conditions|32 samples'):
        train('unused', 'unused', settings(derivative_swd_weight=1., **overrides))


def test_zero_derivative_scale_fails_only_when_auxiliary_enabled(tmp_path):
    data = prepared(tmp_path)
    rows = np.repeat(np.arange(8, dtype=np.float32)[:, None], 24, axis=1)
    change_npz(data / 'train.npz', x=rows)
    (data / 'scaler.json').write_text(json.dumps({'minimum': 0., 'maximum': 7.}))
    options = settings(epochs=1, normalization='standard', batch_sampling='balanced_conditions',
                       batch_size=64, conditions_per_batch=2)
    train(data, tmp_path / 'zero_disabled', options)
    with pytest.raises(ValueError, match='positive'):
        train(data, tmp_path / 'zero_enabled', {**options, 'derivative_swd_weight': 1.})
    assert not (tmp_path / 'zero_enabled' / 'best.pt').exists()


def test_unconditional_derivative_matches_existing_pooled_loss_scope(tmp_path, monkeypatch):
    import fluvius_cgan.training as training
    original = training.derivative_sliced_wasserstein_loss
    observed = []

    def observe(real, generated, real_labels, generated_labels, scale, **kwargs):
        assert not real_labels.any() and not generated_labels.any()
        observed.append(len(real))
        return original(real, generated, real_labels, generated_labels, scale, **kwargs)

    monkeypatch.setattr(training, 'derivative_sliced_wasserstein_loss', observe)
    summary = train(prepared(tmp_path), tmp_path / 'run', settings(
        epochs=1, normalization='standard', unconditional=True, batch_sampling='balanced_conditions',
        batch_size=64, conditions_per_batch=2, derivative_swd_weight=1., swd_projections=4))
    assert observed == [64]
    assert summary['derivative_swd_scale']['condition_counts'] == {'0': 4, '8': 4}
    assert summary['provenance']['derivative_swd']['conditioning'] == 'pooled'
