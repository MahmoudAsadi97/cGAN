"""Fixed conditional TRAIN quantiles for differentiable physical generation.

The learned sigmoid probabilities need not be uniform: this parameterization
does not guarantee empirical marginals or a learned copula. Quantiles are fitted
once on the selected physical TRAIN rows, without a generated-CDF fit. Flat
segments retain zero gradients; signed values and observed endpoints are kept.
"""
from __future__ import annotations

import hashlib
import json
from numbers import Integral

import numpy as np
import torch
from torch import nn

from .copula import _inverse_quantiles


QUANTILE_DECODER_ALGORITHM = "conditional_physical_quantile_decoder_v1"
QUANTILE_DECODER_KNOTS = 1025
_CHUNK_ROWS = 4096
_ARITHMETIC = {
    "fit": "selected physical TRAIN, condition-wise float64 linear empirical quantiles",
    "probabilities": "float32/64 logits cast to float64 before sigmoid",
    "interpolation": "float64 same-sign left anchor, crossing-sign convex blend, exact right-endpoint segment formula",
    "boundary_gradient": "interior one-sided at zero and one; zero strictly outside",
    "physical_output": "cast once to logits dtype before standardization",
    "normalization": "physical output cast to float64, scalar subtract/divide, cast to logits dtype",
    "inference": "direct physical output; no affine roundtrip or posthoc map",
}


def _json_copy(value):
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError("decoder metadata must be finite JSON-safe values") from error


def _same_json(left, right):
    """Preserve schema types: True, 1 and 1.0 are different metadata values."""
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def _quantile_diagnostics(table):
    with np.errstate(over="ignore", invalid="ignore"):
        gaps = np.diff(table, axis=0)
        slopes = gaps * (QUANTILE_DECODER_KNOTS - 1)
    if not np.isfinite(slopes).all() or np.any(gaps < 0):
        raise ValueError("physical TRAIN quantiles have nonfinite or negative interpolation slopes")
    positive = gaps[gaps > 0]
    return {
        "uniform_grid_flat_segment_fraction": float(np.mean(gaps == 0)),
        "constant_slots": int(np.sum(table[0] == table[-1])),
        "maximum_physical_segment_slope": float(slopes.max()),
        "minimum_positive_quantile_gap": float(positive.min()) if len(positive) else None,
    }


def _digest(array, dtype):
    values = np.ascontiguousarray(array, dtype=dtype)
    return hashlib.sha256(memoryview(values).cast("B")).hexdigest()


def _selected_profiles_digest(profiles, selected, dtype):
    digest = hashlib.sha256()
    for start in range(0, len(selected), _CHUNK_ROWS):
        block = np.ascontiguousarray(profiles[selected[start:start + _CHUNK_ROWS]], dtype=dtype)
        if not np.isfinite(block).all():
            raise ValueError("selected physical TRAIN profiles must be finite")
        digest.update(memoryview(block).cast("B"))
    return digest.hexdigest()


def fit_quantile_decoder(physical_train, train_labels, *, selected_indices=None,
                         condition_dim=64, source_binding=None, split="train"):
    """Fit 1,025 physical knots condition by condition on actual selected TRAIN.

    ``selected_indices`` preserves its supplied order for source hashes. The
    fitter never creates a float64 copy of the complete training pool: only one
    condition's observations and the fixed C-by-1025-by-T table are materialized.
    Source-file identity is verified by the training caller, not inferred from
    arrays. An explicitly non-TRAIN role is rejected before array inspection.
    """
    if split != "train":
        raise ValueError("quantile decoder fitting requires the TRAIN split")
    if isinstance(condition_dim, bool) or not isinstance(condition_dim, Integral) or condition_dim < 1:
        raise ValueError("condition_dim must be a positive integer")
    condition_dim = int(condition_dim)
    x, all_labels = np.asarray(physical_train), np.asarray(train_labels)
    if x.ndim != 2 or not all(x.shape) or x.dtype.kind != "f" or x.dtype.itemsize not in (4, 8):
        raise ValueError("physical TRAIN must be a nonempty float32/64 profile matrix")
    if all_labels.shape != (len(x),) or all_labels.dtype.kind not in "iu":
        raise ValueError("TRAIN labels must be an integer vector matching profiles")
    selected = np.arange(len(x), dtype=np.int64) if selected_indices is None else np.asarray(selected_indices)
    if (selected.ndim != 1 or not len(selected) or selected.dtype.kind not in "iu"
            or np.any(selected < 0) or np.any(selected >= len(x))
            or len(np.unique(selected)) != len(selected)):
        raise ValueError("selected TRAIN indices must be distinct valid integer row indices")
    selected = selected.astype(np.int64, copy=False)
    labels = all_labels[selected]
    if np.any(labels < 0) or np.any(labels >= condition_dim):
        raise ValueError("selected TRAIN labels exceed the conditional decoder range")
    labels = labels.astype(np.int64, copy=False)
    input_dtype = "<f4" if x.dtype.itemsize == 4 else "<f8"
    profile_hash = _selected_profiles_digest(x, selected, input_dtype)
    indices_hash, labels_hash = _digest(selected, "<i8"), _digest(labels, "<i8")
    binding = {} if source_binding is None else _json_copy(source_binding)
    if not isinstance(binding, dict):
        raise ValueError("source_binding must be a JSON object")
    expected = {"train_indices_sha256": indices_hash, "train_labels_sha256": labels_hash,
                "train_profiles_available": len(x), "train_profiles_used": len(selected),
                "fitting_scope": "actual_gradient_training_subset_only"}
    for key, value in expected.items():
        if key in binding and not _same_json(binding[key], value):
            raise ValueError(f"source_binding {key} disagrees with selected TRAIN")
    grid = np.linspace(0., 1., QUANTILE_DECODER_KNOTS, dtype=np.float64)
    probabilities = np.broadcast_to(grid[:, None], (len(grid), x.shape[1]))
    tables = np.zeros((condition_dim, len(grid), x.shape[1]), dtype=np.float64)
    counts = np.bincount(labels, minlength=condition_dim).astype(np.int64)
    supported = counts > 0
    diagnostics = {}
    for label in np.flatnonzero(supported):
        # Advanced indexing selects only this condition's actual TRAIN rows.
        observations = np.asarray(x[selected[labels == label]], dtype=np.float64)
        observations.sort(axis=0)
        tables[label] = (_inverse_quantiles(probabilities, observations) if len(observations) > 1
                         else np.broadcast_to(observations, probabilities.shape))
        diagnostics[str(int(label))] = _quantile_diagnostics(tables[label])
        del observations
    metadata = {
        "schema_version": 1, "algorithm": QUANTILE_DECODER_ALGORITHM,
        "n_quantiles": len(grid), "condition_dim": condition_dim, "output_dim": x.shape[1],
        "input_dtype": input_dtype, "selected_profiles": len(selected),
        "supported_labels": np.flatnonzero(supported).tolist(),
        "condition_counts": {str(int(c)): int(counts[c]) for c in np.flatnonzero(supported)},
        "selected_indices_sha256": indices_hash, "selected_profiles_sha256": profile_hash,
        "selected_labels_sha256": labels_hash,
        "probability_grid_sha256": _digest(grid, "<f8"), "quantiles_sha256": _digest(tables, "<f8"),
        "support_sha256": _digest(supported, np.bool_), "counts_sha256": _digest(counts, "<i8"),
        "arithmetic": _ARITHMETIC, "source_binding": binding, "diagnostics": diagnostics,
    }
    return ConditionalQuantileDecoder(tables, supported, counts, metadata)


class ConditionalQuantileDecoder(nn.Module):
    """Fixed physical Q(sigmoid(logits)), with O(batch*slots) interpolation.

    Canonical knots always remain float64. Move the module with ``to(device)``;
    conversion of these buffers to another dtype is rejected. Base-generator
    logits may be float32 or float64. The returned physical values use that same
    dtype, making physical rounding explicit before training standardization.
    """

    def __init__(self, quantiles, supported, training_counts, fit_metadata):
        super().__init__()
        self.register_buffer("quantiles", torch.as_tensor(quantiles).detach().clone())
        self.register_buffer("supported", torch.as_tensor(supported).detach().clone())
        self.register_buffer("training_counts", torch.as_tensor(training_counts).detach().clone())
        self._fit_metadata = _json_copy(fit_metadata)
        self.validate_state()
        self.condition_dim, _, self.output_dim = self.quantiles.shape
        self.register_load_state_dict_post_hook(self._validate_after_load)

    @staticmethod
    def _validate_after_load(module, incompatible):
        module.validate_state()

    @property
    def fit_metadata(self):
        return _json_copy(self._fit_metadata)

    def get_extra_state(self):
        return {"schema_version": 1, "fit_metadata": self.fit_metadata}

    def set_extra_state(self, state):
        if (not isinstance(state, dict) or type(state.get("schema_version")) is not int
                or state["schema_version"] != 1 or "fit_metadata" not in state):
            raise ValueError("invalid quantile decoder extra state")
        self._fit_metadata = _json_copy(state["fit_metadata"])

    @classmethod
    def from_state_dict(cls, state, prefix="decoder."):
        """Reconstruct fixed buffers/metadata without fitting or reading TRAIN."""
        if not isinstance(state, dict):
            raise ValueError("quantile decoder state must be a dictionary")
        try:
            extra = state[prefix + "_extra_state"]
            if not isinstance(extra, dict) or type(extra.get("schema_version")) is not int or extra["schema_version"] != 1:
                raise ValueError("invalid quantile decoder extra state")
            return cls(state[prefix + "quantiles"], state[prefix + "supported"],
                       state[prefix + "training_counts"], extra["fit_metadata"])
        except KeyError as error:
            raise ValueError("incomplete saved quantile decoder state") from error

    def validate_state(self):
        q, support, counts, metadata = self.quantiles, self.supported, self.training_counts, self._fit_metadata
        if (q.dtype != torch.float64 or q.ndim != 3 or not q.shape[0] or not q.shape[2]
                or q.shape[1] != QUANTILE_DECODER_KNOTS or support.dtype != torch.bool
                or counts.dtype != torch.int64 or support.shape != (len(q),) or counts.shape != support.shape
                or q.device != support.device or q.device != counts.device or not support.any()
                or not torch.isfinite(q).all() or (q[:, 1:] < q[:, :-1]).any()
                or (counts[support] < 1).any() or (counts[~support] != 0).any()
                or (q[~support] != 0).any()):
            raise ValueError("invalid quantile decoder tables, support, counts or canonical dtype")
        if (not isinstance(metadata, dict) or metadata.get("input_dtype") not in ("<f4", "<f8")
                or not isinstance(metadata.get("source_binding"), dict)):
            raise ValueError("invalid quantile decoder metadata")
        q_numpy, support_numpy, counts_numpy = (v.detach().cpu().numpy() for v in (q, support, counts))
        supported_labels = np.flatnonzero(support_numpy).tolist()
        expected = {
            "schema_version": 1, "algorithm": QUANTILE_DECODER_ALGORITHM,
            "n_quantiles": QUANTILE_DECODER_KNOTS, "condition_dim": len(q), "output_dim": q.shape[2],
            "arithmetic": _ARITHMETIC,
            "supported_labels": supported_labels,
            "selected_profiles": int(counts_numpy.sum()),
            "condition_counts": {str(c): int(counts_numpy[c]) for c in supported_labels},
            "probability_grid_sha256": _digest(np.linspace(0., 1., QUANTILE_DECODER_KNOTS), "<f8"),
            "quantiles_sha256": _digest(q_numpy, "<f8"),
            "support_sha256": _digest(support_numpy, np.bool_), "counts_sha256": _digest(counts_numpy, "<i8"),
            "diagnostics": {str(label): _quantile_diagnostics(q_numpy[label]) for label in supported_labels},
        }
        for key, value in expected.items():
            if not _same_json(metadata.get(key), value):
                raise ValueError(f"quantile decoder {key} disagrees with saved buffers")
        for key in ("selected_indices_sha256", "selected_profiles_sha256", "selected_labels_sha256"):
            value = metadata.get(key)
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"invalid quantile decoder {key}")
        binding = metadata["source_binding"]
        for key, value in {"train_indices_sha256": metadata["selected_indices_sha256"],
                           "train_labels_sha256": metadata["selected_labels_sha256"],
                           "train_profiles_used": metadata["selected_profiles"],
                           "fitting_scope": "actual_gradient_training_subset_only"}.items():
            if key in binding and not _same_json(binding[key], value):
                raise ValueError(f"quantile decoder source_binding {key} is inconsistent")
        if "train_profiles_available" in binding:
            available = binding["train_profiles_available"]
            if type(available) is not int or available < metadata["selected_profiles"]:
                raise ValueError("invalid quantile decoder source TRAIN count")
        return self.fit_metadata

    def _labels(self, values, labels):
        if (not isinstance(values, torch.Tensor) or values.ndim != 2 or not len(values)
                or values.shape[1] != self.output_dim or values.dtype not in (torch.float32, torch.float64)
                or values.device != self.quantiles.device or self.quantiles.dtype != torch.float64
                or not torch.isfinite(values).all()):
            raise ValueError("decoder inputs must be finite float32/64 profiles on the canonical decoder device")
        if (not isinstance(labels, torch.Tensor) or labels.shape != (len(values),)
                or labels.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
                or labels.device != values.device or (labels < 0).any() or (labels >= self.condition_dim).any()):
            raise ValueError("decoder labels must be integer rows on the same device and in range")
        labels = labels.to(torch.int64)
        if not self.supported[labels].all():
            raise ValueError("decoder condition absent from actual selected TRAIN")
        return labels

    def decode_probabilities(self, probabilities, labels):
        """Decode probabilities; equality chooses interior one-sided slopes."""
        labels = self._labels(probabilities, labels)
        output_dtype = probabilities.dtype
        p = probabilities.to(torch.float64)
        bounded = torch.where(p < 0., 0., torch.where(p > 1., 1., p))
        positions = bounded * (QUANTILE_DECODER_KNOTS - 1)
        lower = positions.floor().to(torch.int64).clamp(max=QUANTILE_DECODER_KNOTS - 2)
        weight = positions - lower
        slots = torch.arange(self.output_dim, device=p.device)[None, :]
        # Gather only two scalar knots per output; never q[labels] (N,K,T).
        left = self.quantiles[labels[:, None], lower, slots]
        right = self.quantiles[labels[:, None], lower + 1, slots]
        gap = right - left
        if not torch.isfinite(gap * (QUANTILE_DECODER_KNOTS - 1)).all():
            raise ValueError("selected quantile segment has a nonfinite slope")
        same_sign = (left >= 0) | (right <= 0)
        crossing = ~same_sign
        result = torch.empty_like(weight)
        # A fixed left anchor avoids cancellation below the lower knot for
        # tiny positive weights on negative intervals. Do not switch anchors
        # at the midpoint: separately rounded formulas need not meet there.
        # Keep multiply and add separate; at w<1 the rounded product stays
        # below an upward-rounded gap. The exact right endpoint is handled
        # below, preserving its interior one-sided derivative.
        result[same_sign] = left[same_sign] + weight[same_sign] * gap[same_sign]
        result[crossing] = ((1. - weight[crossing]) * left[crossing]
                            + weight[crossing] * right[crossing])
        # Exact endpoint values with the same segment derivative; no detached
        # endpoint replacement, jitter, epsilon slope or straight-through rank.
        at_left, at_right = weight == 0., weight == 1.
        result[at_left] = left[at_left] + weight[at_left] * gap[at_left]
        result[at_right] = right[at_right] - (1. - weight[at_right]) * gap[at_right]
        result = result.to(output_dtype)
        if not torch.isfinite(result).all():
            raise ValueError("decoded physical profiles exceed finite output dtype range")
        return result

    def decode_with_probabilities(self, logits, labels):
        self._labels(logits, labels)
        probabilities = torch.sigmoid(logits.to(torch.float64))
        physical = self.decode_probabilities(probabilities, labels).to(logits.dtype)
        if not torch.isfinite(physical).all():
            raise ValueError("decoded physical profiles exceed finite logits dtype range")
        return physical, probabilities

    def forward(self, logits, labels):
        return self.decode_with_probabilities(logits, labels)[0]
