"""Synthetic physical-decoder contracts; no prepared data or real fits."""
import copy
import hashlib
import io

import numpy as np
import pytest
import torch

from fluvius_cgan.models import Generator, QuantileDecodedGenerator
from fluvius_cgan.quantile_decoder import ConditionalQuantileDecoder, fit_quantile_decoder


def fit(values, labels=None, **kwargs):
    values = np.asarray(values, dtype=np.float64)
    labels = np.zeros(len(values), dtype=np.int64) if labels is None else np.asarray(labels)
    return fit_quantile_decoder(values, labels, condition_dim=kwargs.pop("condition_dim", 1), **kwargs)


def test_piecewise_autograd_matches_finite_difference_and_analytic_sigmoid_slope():
    decoder = fit([[-2., 1.], [3., 7.]])
    p = torch.tensor([[.37123, .63248]], dtype=torch.float64)
    logits = torch.logit(p).requires_grad_()
    labels = torch.tensor([0])
    assert torch.autograd.gradcheck(lambda value: decoder(value, labels), (logits,), eps=1e-6, atol=1e-7)
    gradient, = torch.autograd.grad(decoder(logits, labels).sum(), logits)
    torch.testing.assert_close(gradient, torch.tensor([[5., 6.]]) * p * (1 - p), rtol=1e-10, atol=1e-12)
    assert decoder.quantiles.shape == (1, 1025, 2)
    assert not list(decoder.parameters())


@pytest.mark.parametrize("constant", [np.nextafter(.1, 1.), -np.nextafter(.1, 1.)])
def test_arbitrary_signed_constant_and_tied_plateau_are_exact_with_zero_gradients(constant):
    decoder = fit(np.column_stack(([constant] * 4, [0., 0., 0., 4.])))
    logits = torch.logit(torch.tensor([[.137, .2], [.937, .9]], dtype=torch.float64)).requires_grad_()
    output = decoder(logits, torch.tensor([0, 0]))
    assert torch.equal(output[:, 0], torch.full((2,), constant, dtype=torch.float64))
    assert output[0, 1].item() == 0
    gradient, = torch.autograd.grad(output.sum(), logits)
    assert torch.equal(gradient[:, 0], torch.zeros(2, dtype=torch.float64))
    assert gradient[0, 1].item() == 0 and gradient[1, 1].item() > 0


def test_true_endpoints_choose_interior_slope_and_saturated_logits_keep_zero_chain_gradient():
    decoder = fit([[-3.], [7.]])
    p = torch.tensor([[0.], [1.], [-.1], [1.1]], dtype=torch.float64, requires_grad=True)
    output = decoder.decode_probabilities(p, torch.zeros(4, dtype=torch.int64))
    assert torch.equal(output, torch.tensor([[-3.], [7.], [-3.], [7.]], dtype=torch.float64))
    gradient, = torch.autograd.grad(output.sum(), p)
    assert torch.equal(gradient, torch.tensor([[10.], [10.], [0.], [0.]], dtype=torch.float64))
    logits = torch.tensor([[-1000.], [1000.]], dtype=torch.float64, requires_grad=True)
    output = decoder(logits, torch.tensor([0, 0]))
    assert torch.equal(output, torch.tensor([[-3.], [7.]], dtype=torch.float64))
    gradient, = torch.autograd.grad(output.sum(), logits)
    assert torch.equal(gradient, torch.zeros_like(logits))
    tied = fit([[0.], [0.], [1.]])
    p = torch.tensor([[.5], [1.]], dtype=torch.float64, requires_grad=True)
    gradient, = torch.autograd.grad(tied.decode_probabilities(p, torch.tensor([0, 0])).sum(), p)
    assert torch.equal(gradient, torch.full_like(p, 2.))


def test_negative_boundary_cancellation_preserves_support_monotonicity_and_slope():
    # The former right anchor rounded b - (1-w)*(b-a) below a for tiny w.
    # Keep the demonstrated float64 values, not a tolerance that hides one ULP.
    a, b = -(1.5 + 2**-52), -(3 * 2**-53)
    decoder = fit([[a]] + [[b]] * 1024)
    assert decoder.quantiles[0, :2, 0].tolist() == [a, b]
    p = torch.tensor([[0.], [1e-30]], dtype=torch.float64, requires_grad=True)
    output = decoder.decode_probabilities(p, torch.tensor([0, 0]))
    assert torch.equal(output, torch.full_like(output, a))
    gradient, = torch.autograd.grad(output.sum(), p)
    assert torch.equal(gradient, torch.full_like(p, 1536.))


def test_interpolator_is_monotone_at_nextafter_knots_and_midpoints():
    decoder = fit([[1., -1e16, -2.], [1e16, -1., 5.]])
    knots = np.linspace(0., 1., 1025)
    centers = np.r_[knots, (knots[:-1] + knots[1:]) / 2]
    p = np.unique(np.clip(np.r_[centers, np.nextafter(centers, -np.inf), np.nextafter(centers, np.inf)], 0, 1))
    values = decoder.decode_probabilities(torch.from_numpy(np.repeat(p[:, None], 3, axis=1)), torch.zeros(len(p), dtype=torch.int64))
    assert (values[1:] >= values[:-1]).all()
    assert torch.equal(values[0], decoder.quantiles[0, 0])
    assert torch.equal(values[-1], decoder.quantiles[0, -1])


def test_signed_cancellation_subnormal_and_extreme_segments_stay_bounded_at_adjacent_probabilities():
    tiny = np.nextafter(0., 1.)
    large = np.finfo(np.float64).max / 4096
    intervals = np.array([
        [-(1.5 + 2**-52), -(3 * 2**-53)],
        [3 * 2**-53, 1.5 + 2**-52],
        [-1., np.nextafter(-1., 0.)],
        [np.nextafter(1., 0.), 1.],
        [0., 8 * tiny], [-8 * tiny, -tiny], [-tiny, tiny],
        [-large, -large / 2], [large / 2, large], [-large, large],
    ])
    a, b = intervals.T
    weights = np.array([0., tiny, 1e-30, .125, np.nextafter(.5, 0.),
                        .5, np.nextafter(.5, 1.), np.nextafter(1., 0.), 1.])
    for last_segment in (False, True):
        observations = np.repeat((a if last_segment else b)[None, :], 1025, axis=0)
        observations[-1 if last_segment else 0] = b if last_segment else a
        decoder = fit(observations)
        p = ((1023 + weights) if last_segment else weights) / 1024
        probabilities = torch.from_numpy(np.repeat(p[:, None], len(a), axis=1)).requires_grad_()
        result = decoder.decode_probabilities(probabilities, torch.zeros(len(p), dtype=torch.int64))
        assert (result[1:] >= result[:-1]).all()
        assert (result >= torch.from_numpy(a)).all()
        assert (result <= torch.from_numpy(b)).all()
        assert torch.equal(result[0], torch.from_numpy(a))
        assert torch.equal(result[-1], torch.from_numpy(b))
        gradient, = torch.autograd.grad(result.sum(), probabilities)
        expected = torch.from_numpy((b - a) * 1024).expand_as(gradient)
        # The first segment's exact right knot selects the following plateau;
        # the final segment instead retains the declared p=1 interior slope.
        rows = slice(None) if last_segment else slice(None, -1)
        torch.testing.assert_close(gradient[rows], expected[rows], rtol=0, atol=0)


def test_signed_large_values_have_finite_gradients_and_unrepresentable_output_is_rejected():
    decoder = fit([[-1e150], [1e150]])
    logits = torch.zeros((1, 1), dtype=torch.float64, requires_grad=True)
    gradient, = torch.autograd.grad(decoder(logits, torch.tensor([0])).sum(), logits)
    torch.testing.assert_close(gradient, torch.tensor([[5e149]], dtype=torch.float64), rtol=1e-12, atol=0)
    with pytest.raises(ValueError, match="finite.*dtype"):
        decoder(torch.ones((1, 1), dtype=torch.float32), torch.tensor([0]))
    with pytest.raises(ValueError, match="slopes"):
        fit([[-1e308], [1e308]])


def test_selected_train_only_ordered_digests_support_and_no_heldout_inspection():
    class NeverInspect:
        def __array__(self, *args, **kwargs):
            raise AssertionError("held-out values inspected")
    with pytest.raises(ValueError, match="TRAIN split"):
        fit_quantile_decoder(NeverInspect(), NeverInspect(), split="val")
    x = np.array([[10000., -10000.], [2., -7.], [4., -3.], [np.nan, np.nan]], dtype=np.float32)
    y, indices = np.array([2, 0, 0, 3]), np.array([2, 1])
    decoder = fit_quantile_decoder(x, y, selected_indices=indices, condition_dim=4)
    assert decoder.training_counts.tolist() == [2, 0, 0, 0]
    assert decoder.fit_metadata["condition_counts"] == {"0": 2}
    assert torch.equal(decoder.quantiles[0, 0], torch.tensor([2., -7.], dtype=torch.float64))
    assert torch.equal(decoder.quantiles[0, -1], torch.tensor([4., -3.], dtype=torch.float64))
    metadata = decoder.fit_metadata
    assert metadata["input_dtype"] == "<f4"
    assert metadata["selected_profiles_sha256"] == hashlib.sha256(x[indices].astype("<f4").tobytes()).hexdigest()
    assert metadata["selected_indices_sha256"] == hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest()
    metadata["selected_profiles"] = 999
    assert decoder.fit_metadata["selected_profiles"] == 2
    with pytest.raises(ValueError, match="absent"):
        decoder(torch.zeros((1, 2)), torch.tensor([2]))
    with pytest.raises(ValueError, match="source_binding"):
        fit_quantile_decoder(x, y, selected_indices=indices, condition_dim=4,
                             source_binding={"train_profiles_used": 999})


def test_condition_order_dtype_device_and_batch_invariance():
    decoder = fit([[-4., 2.], [6., 8.], [10., -7.], [12., -3.]], [0, 0, 2, 2], condition_dim=3)
    labels = torch.tensor([2, 0, 2, 0])
    logits = torch.tensor([[.1, -.2], [.5, .2], [-2., 3.], [1., -1.]], dtype=torch.float32)
    physical, probabilities = decoder.decode_with_probabilities(logits, labels)
    assert physical.dtype == torch.float32 and probabilities.dtype == torch.float64
    assert torch.equal(probabilities, torch.sigmoid(logits.double()))
    assert torch.equal(physical, torch.cat([decoder(logits[i:i+1], labels[i:i+1]) for i in range(len(labels))]))
    assert ((physical.double() >= decoder.quantiles[labels, 0]) &
            (physical.double() <= decoder.quantiles[labels, -1])).all()
    for invalid in (torch.tensor([1, 0, 2, 0]), labels.float(), labels[:2], torch.tensor([3, 0, 2, 0])):
        with pytest.raises(ValueError):
            decoder(logits, invalid)
    with pytest.raises(ValueError, match="device"):
        decoder(logits, torch.empty(4, dtype=torch.int64, device="meta"))
    with pytest.raises(ValueError, match="canonical"):
        decoder.float()(logits, labels)


def test_interpolation_never_materializes_batch_by_knots_table():
    # TorchDispatch observes actual indexing results, detecting a q[labels]
    # expansion even if it is later gathered down to the correct output shape.
    from torch.utils._python_dispatch import TorchDispatchMode
    decoder = fit([[0., -1.], [2., 3.], [10., -3.], [12., 5.]], [0, 0, 1, 1], condition_dim=2)
    class NoExpandedTable(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            result = func(*args, **(kwargs or {}))
            if func == torch.ops.aten.index.Tensor:
                assert result.ndim <= 2, "interpolation expanded condition quantile tables across batch rows"
            return result
    with NoExpandedTable():
        result = decoder(torch.zeros((512, 2)), torch.arange(512) % 2)
    assert result.shape == (512, 2)


def test_wrapper_normalizes_exact_deployed_physical_values_and_retains_legacy_parameters():
    x = np.array([[-2., 1.], [3., 7.], [10., -7.], [12., -3.]], dtype=np.float32)
    decoder = fit_quantile_decoder(x, np.array([0, 0, 2, 2]), condition_dim=3)
    base = Generator(latent_dim=2, condition_dim=3, hidden_dim=4, output_dim=2)
    old_keys, old_count = set(base.state_dict()), sum(p.numel() for p in base.parameters())
    offset, scale = float(x.astype(np.float64).mean()), float(x.astype(np.float64).std())
    wrapper = QuantileDecodedGenerator(base, decoder, offset=offset, scale=scale)
    noise = torch.tensor([[.1, -.2], [.2, .4]])
    onehot = torch.tensor([[1., 0., 0.], [0., 0., 1.]])
    physical = wrapper.decode_physical(noise, onehot)
    normalized = wrapper(noise, onehot)
    expected = ((physical.detach().numpy().astype(np.float64) - offset) / scale).astype(np.float32)
    assert torch.equal(normalized.detach(), torch.from_numpy(expected))
    normalized.square().sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in base.parameters())
    assert any(torch.count_nonzero(p.grad) for p in base.parameters())
    assert old_count == sum(p.numel() for p in wrapper.parameters())
    assert set(wrapper.base.state_dict()) == old_keys
    assert {name.removeprefix("base.") for name, _ in wrapper.named_parameters()} == set(dict(base.named_parameters()))
    # Direct decoding, not an inverse of rounded normalized values, is deployed.
    assert torch.equal(wrapper.decode_with_probabilities(noise, onehot)[0], physical)


def test_safe_serialization_reconstruction_replay_and_no_refit(monkeypatch):
    decoder = fit([[-2., 1.], [3., 7.]])
    wrapper = QuantileDecodedGenerator(Generator(2, 1, 4, 2), decoder, offset=.123456789, scale=2.987654321)
    noise, onehot = torch.randn(3, 2), torch.ones(3, 1)
    expected = wrapper.decode_physical(noise, onehot)
    stream = io.BytesIO()
    torch.save(wrapper.state_dict(), stream)
    stream.seek(0)
    state = torch.load(stream, weights_only=True)
    monkeypatch.setattr("fluvius_cgan.quantile_decoder._inverse_quantiles", lambda *args: pytest.fail("reconstruction refitted quantiles"))
    replay = QuantileDecodedGenerator(Generator(2, 1, 4, 2), ConditionalQuantileDecoder.from_state_dict(state),
                                     offset=.123456789, scale=2.987654321)
    replay.load_state_dict(state, strict=True)
    assert torch.equal(replay.decode_physical(noise, onehot), expected)
    assert torch.equal(replay(noise, onehot), wrapper(noise, onehot))
    assert replay.decoder.fit_metadata == decoder.fit_metadata


@pytest.mark.parametrize("mutation", ["table", "support", "counts", "hash", "algorithm", "arithmetic", "missing", "dtype",
                                      "diagnostics", "bool_schema", "float_count", "bool_label"])
def test_malformed_saved_decoder_rejected(mutation):
    decoder = fit([[0.], [1.]])
    state = {"decoder." + key: value for key, value in copy.deepcopy(decoder.state_dict()).items()}
    if mutation == "table": state["decoder.quantiles"][0, 400, 0] += .0001
    elif mutation == "support": state["decoder.supported"][0] = False
    elif mutation == "counts": state["decoder.training_counts"][0] += 1
    elif mutation == "hash": state["decoder._extra_state"]["fit_metadata"]["quantiles_sha256"] = "a" * 64
    elif mutation == "algorithm": state["decoder._extra_state"]["fit_metadata"]["algorithm"] = "unknown"
    elif mutation == "arithmetic": state["decoder._extra_state"]["fit_metadata"]["arithmetic"] = {}
    elif mutation == "missing": del state["decoder.quantiles"]
    elif mutation == "dtype": state["decoder.quantiles"] = state["decoder.quantiles"].float()
    elif mutation == "diagnostics": state["decoder._extra_state"]["fit_metadata"]["diagnostics"]["0"]["constant_slots"] = 1
    elif mutation == "bool_schema": state["decoder._extra_state"]["fit_metadata"]["schema_version"] = True
    elif mutation == "float_count": state["decoder._extra_state"]["fit_metadata"]["selected_profiles"] = 2.
    else: state["decoder._extra_state"]["fit_metadata"]["supported_labels"] = [False]
    with pytest.raises(ValueError):
        ConditionalQuantileDecoder.from_state_dict(state)


def test_constant_decoder_zero_scale_fallback_and_no_exact_marginal_guarantee():
    decoder = fit(np.full((3, 2), -7.))
    wrapper = QuantileDecodedGenerator(Generator(2, 1, 4, 2), decoder, offset=-7., scale=1.)
    noise, onehot = torch.zeros(2, 2), torch.ones(2, 1)
    assert torch.equal(wrapper(noise, onehot), torch.zeros(2, 2))
    wrapper(noise, onehot).sum().backward()
    assert all(torch.equal(p.grad, torch.zeros_like(p.grad)) for p in wrapper.base.parameters())
    nonconstant = fit([[-3.], [7.]])
    collapsed = nonconstant(torch.zeros(20, 1), torch.zeros(20, dtype=torch.int64))
    assert torch.equal(collapsed, torch.full((20, 1), 2.))
    assert np.var([-3., 7.]) > 0 and collapsed.var().item() == 0


@pytest.mark.parametrize("bad", ["indices", "labels", "nonfinite", "integer", "unconditional"])
def test_fit_rejects_invalid_actual_train_inputs(bad):
    x, y, indices, dim = np.array([[0.], [1.]]), np.array([0, 0]), None, 1
    if bad == "indices": indices = [0, 0]
    elif bad == "labels": y = np.array([0., 0.])
    elif bad == "nonfinite": x[0, 0] = np.nan
    elif bad == "integer": x = x.astype(np.int64)
    else: dim = 0
    with pytest.raises(ValueError):
        fit_quantile_decoder(x, y, selected_indices=indices, condition_dim=dim)
