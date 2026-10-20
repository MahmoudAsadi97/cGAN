"""Synthetic-only integration checks for the embedded TRAIN quantile decoder."""
import copy
import hashlib
import json

import numpy as np
import pytest
import torch

from fluvius_cgan import training
from test_training import prepared, settings


def decoder_settings(**changes):
    return settings(epochs=2, normalization="standard", objective="hinge_sn", adam_beta1=.5,
                    batch_sampling="balanced_conditions", batch_size=64, conditions_per_batch=2,
                    swd_weight=10., swd_projections=4, validation_selection="conditional_w1",
                    validation_samples=4, conditional_quantile_decoder=True, **changes)


@pytest.fixture(scope="module")
def decoder_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("decoder_integration")
    data, out = prepared(root), root / "trained"
    training.train(data, out, decoder_settings())
    return data, out


def test_physical_fit_is_selected_train_only_and_selector_observes_decoded_output(tmp_path, monkeypatch):
    from fluvius_cgan import quantile_decoder
    data = prepared(tmp_path)
    with np.load(data / "train.npz", allow_pickle=False) as archive:
        raw, labels = archive["x"].copy(), archive["y"].copy()
    options = decoder_settings(max_train_samples=3)
    selected = np.sort(np.random.default_rng(options["seed"]).choice(len(raw), 3, replace=False))
    original_fit, original_score = quantile_decoder.fit_quantile_decoder, training._validation_score
    original_hash = training._sha256
    fitting, scoring = [], []

    def fit(profiles, fit_labels, **kwargs):
        np.testing.assert_array_equal(profiles, raw)
        np.testing.assert_array_equal(fit_labels, labels)
        np.testing.assert_array_equal(kwargs["selected_indices"], selected)
        assert kwargs["split"] == "train"
        fitting.append(kwargs["source_binding"])
        return original_fit(profiles, fit_labels, **kwargs)

    def score(generator, normalized, val_labels, noise, condition_dim, device, batch_size, **kwargs):
        assert kwargs == {"selection": "conditional_w1"}
        c = training._conditions(val_labels, condition_dim, device)
        with torch.no_grad():
            physical = generator.decode_physical(noise, c).numpy()
            decoded = generator(noise, c).numpy()
        scaler = {"minimum": float(raw.min()), "maximum": float(raw.max()), "kind": "standard",
                  "mean": float(raw.astype(np.float64).mean()), "std": float(raw.astype(np.float64).std())}
        np.testing.assert_array_equal(decoded, training._normalized(physical, scaler))
        for label in np.unique(val_labels):
            fitted = raw[selected][labels[selected] == label]
            assert (physical[val_labels == label] >= fitted.min(axis=0)).all()
            assert (physical[val_labels == label] <= fitted.max(axis=0)).all()
        scoring.append(physical)
        return original_score(generator, normalized, val_labels, noise, condition_dim, device, batch_size, **kwargs)

    def no_test_hash(path):
        assert path.name != "test.npz"
        return original_hash(path)

    monkeypatch.setattr(quantile_decoder, "fit_quantile_decoder", fit)
    monkeypatch.setattr(training, "_validation_score", score)
    monkeypatch.setattr(training, "_sha256", no_test_hash)
    first, replay = tmp_path / "first", tmp_path / "replay"
    summary = training.train(data, first, options)
    repeated = training.train(data, replay, options)
    assert len(fitting) == 2 and len(scoring) == 4
    assert summary["best_validation_score"] == repeated["best_validation_score"]
    a, b = (torch.load(path / "last.pt", weights_only=True) for path in (first, replay))
    for name in ("generator_state_dict", "discriminator_state_dict"):
        for key in a[name]:
            if isinstance(a[name][key], torch.Tensor):
                assert torch.equal(a[name][key], b[name][key])
            else:
                assert a[name][key] == b[name][key]
    fit_record = a["quantile_decoder"]
    assert fit_record == summary["quantile_decoder"] == a["provenance"]["quantile_decoder"]
    assert fit_record["selected_profiles"] == 3
    assert fit_record["selected_profiles_sha256"] == hashlib.sha256(raw[selected].astype("<f4").tobytes()).hexdigest()
    assert fit_record["selected_labels_sha256"] == hashlib.sha256(labels[selected].astype("<i8").tobytes()).hexdigest()
    assert fitting[0]["train_indices_sha256"] == summary["selection"]["train_indices_sha256"]
    assert "test.npz" not in fitting[0]["data_hashes"]
    assert fit_record["n_quantiles"] == 1025
    assert summary["generator_updates"] == summary["discriminator_updates"] == 2
    history = json.loads((first / "history.json").read_text())
    assert all(row["loss_g"] == pytest.approx(row["loss_g_adversarial"] + 10 * row["loss_swd"])
               and row["loss_derivative_swd"] == 0 for row in history)
    assert {p.name for p in first.iterdir()} == {"best.pt", "last.pt", "history.json", "config.json", "summary.json"}


def test_saved_decoder_buffers_are_fixed_and_excluded_from_optimizers(decoder_run):
    _, out = decoder_run
    first, last = (torch.load(out / name, weights_only=True) for name in ("best.pt", "last.pt"))
    model = training.load_quantile_decoded_generator(last)
    assert all(name.startswith("base.") for name, _ in model.named_parameters())
    assert all(not buffer.requires_grad for _, buffer in model.named_buffers())
    assert len(last["optimizer_g_state_dict"]["param_groups"][0]["params"]) == len(list(model.base.parameters()))
    for name in ("decoder.quantiles", "decoder.supported", "decoder.training_counts",
                 "normalization_offset", "normalization_scale"):
        assert torch.equal(first["generator_state_dict"][name], last["generator_state_dict"][name])
    assert model.decoder.quantiles.dtype == torch.float64
    model.decoder.validate_state()
    with torch.random.fork_rng():
        torch.manual_seed(last["config"]["seed"])
        initial = training.Generator(last["config"]["latent_dim"], 64,
                                     last["config"]["hidden_dim"], last["output_dim"])
    assert any(not torch.equal(value, model.base.state_dict()[name]) for name, value in initial.state_dict().items())


def test_generation_and_diagnostics_are_same_draw_without_refit_or_prepared_reads(decoder_run, monkeypatch):
    from fluvius_cgan import quantile_decoder
    _, out = decoder_run
    fail = lambda *a, **k: pytest.fail("inference must not fit or read prepared arrays")
    monkeypatch.setattr(quantile_decoder, "fit_quantile_decoder", fail)
    monkeypatch.setattr(training, "_load_prepared", fail)
    monkeypatch.setattr(training, "_read_split", fail)
    monkeypatch.setattr(training, "_sha256", fail)
    monkeypatch.setattr(np, "load", fail)
    labels = np.array([0, 8, 8, 0, 8], dtype=np.int64)
    physical = training.generate(out / "best.pt", labels, seed=71, batch_size=2)
    rich, probabilities, fit = training.generate_quantile_decoder_diagnostics(out / "best.pt", labels, seed=71, batch_size=2)
    np.testing.assert_array_equal(physical, rich)
    saved = torch.load(out / "best.pt", weights_only=True)
    model = training.load_quantile_decoded_generator(saved).eval()
    noise = torch.randn((len(labels), saved["config"]["latent_dim"]), generator=torch.Generator().manual_seed(71))
    direct, normalized = [], []
    with torch.no_grad():
        for start in range(0, len(labels), 2):
            conditions = training._conditions(labels[start:start + 2], 64, torch.device("cpu"))
            direct.append(model.decode_physical(noise[start:start + 2], conditions).numpy())
            normalized.append(model(noise[start:start + 2], conditions).numpy())
    np.testing.assert_array_equal(physical, np.concatenate(direct))
    offset, scale = training._scaler_parameters(saved["scaler"])
    roundtrip = (np.concatenate(normalized).astype(np.float64) * scale + offset).astype(np.float32)
    assert np.any(physical != roundtrip), "fixture must exercise the avoided affine-roundtrip rounding"
    assert physical.dtype == np.float32 and probabilities.dtype == np.float64
    assert probabilities.shape == physical.shape and ((probabilities >= 0) & (probabilities <= 1)).all()
    assert fit["selected_profiles"] == 8
    empty, pempty, _ = training.generate_quantile_decoder_diagnostics(out / "best.pt", np.array([], dtype=np.int64))
    assert empty.shape == pempty.shape == (0, 24)
    with pytest.raises(ValueError, match="unsupported"):
        training.generate(out / "best.pt", np.array([63]))


@pytest.mark.parametrize("violation", [
    "fit_missing", "fit_mismatch", "provenance", "train_hash", "indices", "labels", "profiles", "counts",
    "source", "support", "size", "scaler", "offset", "scale", "table", "table_dtype", "buffer_support",
    "buffer_count", "nonfinite_weight", "weight_dtype", "normalization_dtype", "disabled_flag", "missing_state",
])
def test_decoder_checkpoint_inconsistencies_fail_closed(decoder_run, tmp_path, violation):
    _, out = decoder_run
    saved = torch.load(out / "best.pt", weights_only=True)
    state = saved["generator_state_dict"]
    if violation == "fit_missing":
        saved.pop("quantile_decoder")
    elif violation == "fit_mismatch":
        saved["quantile_decoder"] = copy.deepcopy(saved["quantile_decoder"])
        saved["quantile_decoder"]["selected_profiles"] += 1
    elif violation == "provenance":
        saved["provenance"].pop("quantile_decoder")
    elif violation == "train_hash":
        saved["data_hashes"]["train.npz"] = "0" * 64
    elif violation in ("indices", "labels", "profiles"):
        saved["selection"][f"train_{violation}_sha256"] = "0" * 64
    elif violation == "counts":
        saved["selection"]["train_profiles_per_condition"]["0"] += 1
    elif violation == "source":
        saved["provenance"]["source_files_sha256"].pop("src/fluvius_cgan/quantile_decoder.py")
    elif violation == "support":
        saved["supported_labels"].append(63)
    elif violation == "size":
        saved["metadata"]["splits"]["train"]["profiles"] += 1
    elif violation == "scaler":
        saved["scaler"]["mean"] += 1
    elif violation in ("offset", "scale"):
        state[f"normalization_{violation}"] += 1
    elif violation == "table":
        state["decoder.quantiles"][0] += .01
    elif violation == "table_dtype":
        state["decoder.quantiles"] = state["decoder.quantiles"].float()
    elif violation == "buffer_support":
        state["decoder.supported"][63] = True
    elif violation == "buffer_count":
        state["decoder.training_counts"][0] += 1
    elif violation == "nonfinite_weight":
        state["base.model.0.weight"][0, 0] = float("nan")
    elif violation == "weight_dtype":
        state["base.model.0.weight"] = state["base.model.0.weight"].double()
    elif violation == "normalization_dtype":
        state["normalization_scale"] = state["normalization_scale"].float()
    elif violation == "disabled_flag":
        saved["config"]["conditional_quantile_decoder"] = False
    else:
        state.pop("decoder.quantiles")
    altered = tmp_path / "altered.pt"
    torch.save(saved, altered)
    with pytest.raises(ValueError):
        training.generate(altered, np.array([0, 8]))


@pytest.mark.parametrize("changes", [
    {"conditional_quantile_decoder": None}, {"conditional_quantile_decoder": 1},
    {"conditional_quantile_decoder": "true"}, {"unconditional": True},
    {"normalization": "minmax"}, {"conditional_mean_residual": True}, {"derivative_swd_weight": 1.},
])
def test_decoder_invalid_configs_rejected_before_data_access(tmp_path, changes):
    with pytest.raises(ValueError):
        training.train(tmp_path / "no_data", tmp_path / "out", {**decoder_settings(), **changes})


def test_disabled_decoder_path_and_legacy_checkpoint_replay(tmp_path, monkeypatch):
    from fluvius_cgan import quantile_decoder
    monkeypatch.setattr(quantile_decoder, "fit_quantile_decoder", lambda *a, **k: pytest.fail("disabled fit"))
    data = prepared(tmp_path)
    one, two = tmp_path / "implicit", tmp_path / "explicit"
    options = settings(epochs=1, normalization="standard", objective="hinge_sn")
    training.train(data, one, options)
    training.train(data, two, {**options, "conditional_quantile_decoder": False})
    a, b = (torch.load(path / "last.pt", weights_only=True) for path in (one, two))
    for name in ("generator_state_dict", "discriminator_state_dict"):
        assert all(torch.equal(a[name][key], b[name][key]) for key in a[name])
    assert "quantile_decoder" not in a and "quantile_decoder" not in a["provenance"]
    a["config"].pop("conditional_quantile_decoder")
    legacy = tmp_path / "legacy.pt"
    torch.save(a, legacy)
    labels = np.array([0, 8])
    np.testing.assert_array_equal(training.generate(legacy, labels), training.generate(two / "last.pt", labels))
    with pytest.raises(ValueError, match="enabled decoder"):
        training.generate_quantile_decoder_diagnostics(legacy, labels)
