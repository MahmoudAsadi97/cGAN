"""Fixed train-only marginal maps for a frozen conditional generator.

Fit T(x) = Q_train(H_generator(x)) once using actual training rows and a separate
generated calibration reference. Applying T never ranks the application batch,
refits a marginal, reads data, or draws randomness. Thus maps are monotone and
batch-invariant. The caller must use independent calibration/production draws
and keep held-out profiles out of fitting; metadata records, but cannot prove,
that caller-side provenance.

Q_train is the same linear empirical inverse used by the copula baseline. H is
the inverse of a generated linear quantile table. Repeated source knots span a
probability interval [L, R]: at the repeated value H=(L+R)/2; between distinct
values a<x<b, interpolate from R(a) to L(b). This quantile-plateau midpoint is
explicitly NOT the sample mid-CDF (#less + 0.5*#equal)/n. Source values outside
the fitted range map to 0/1. Outputs never extrapolate past training extrema.

Uniform knot compression (1025 by default) approximates both marginal tables;
None retains every sorted observation. Source atoms cannot be split by this
deterministic map. Constant source/nonconstant target margins therefore fail.
Marginals and marginal diversity are imposed by calibration and are not
independent evidence of model success. Calibration cannot recover missing
dependence, and plateaus or saturation can weaken rank dependence.

Related principle: Janke et al., Implicit Generative Copulas, NeurIPS 2021,
https://proceedings.neurips.cc/paper/2021/file/dac4a67bdc4a800113b0f1ad67ed696f-Paper.pdf
Their generator is trained in copula space; this post-training ablation is an
adaptation of the marginal-transform idea, not an IGC replication.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from .copula import _inverse_quantiles, _labels, _positive_integer


def _profiles(values, name: str, *, empty: bool = False) -> np.ndarray:
    try:
        values = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real rank-2 array") from exc
    if (values.ndim != 2 or not values.shape[1] or (not empty and not len(values))
            or values.dtype.kind not in "iuf" or not np.isfinite(values).all()):
        raise ValueError(f"{name} must be a finite real rank-2 array with nonzero width")
    return values


def _knots(value):
    return None if value is None else _positive_integer(value, "quantile_knots", 2)


def _provenance(values):
    if values is None:
        return {}
    if not isinstance(values, dict) or any(not isinstance(key, str) for key in values):
        raise ValueError("provenance must be a JSON object with string keys")
    try:
        copied = json.loads(json.dumps(values, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("provenance must contain finite JSON-serializable values") from exc
    for name in ("checkpoint_sha256", "train_indices_sha256", "train_subset_sha256"):
        if name in copied and (not isinstance(copied[name], str)
                               or re.fullmatch(r"[0-9a-f]{64}", copied[name]) is None):
            raise ValueError(f"provenance {name} must be a SHA-256 hex digest")
    return copied


def _quantiles(sorted_values: np.ndarray, knots: int | None) -> np.ndarray:
    n = len(sorted_values)
    count = max(2, n if knots is None else min(n, knots))
    if n == 1:
        return np.repeat(sorted_values, 2, axis=0)
    if count == n:
        return sorted_values.copy()
    probabilities = np.broadcast_to(np.linspace(0, 1, count)[:, None], (count, sorted_values.shape[1]))
    return _inverse_quantiles(probabilities, sorted_values)


@dataclass(frozen=True)
class _SourceCDF:
    values: np.ndarray
    lower: np.ndarray
    upper: np.ndarray

    def evaluate(self, values: np.ndarray) -> np.ndarray:
        """Inverse of the fixed quantile table; midpoint only at actual plateaus."""
        indices = np.searchsorted(self.values, values, side="left")
        at = np.minimum(indices, len(self.values) - 1)
        exact = (indices < len(self.values)) & (self.values[at] == values)
        result = np.zeros(len(values), dtype=np.float64)
        result[indices == len(self.values)] = 1.0
        result[exact] = 0.5 * self.lower[at[exact]] + 0.5 * self.upper[at[exact]]
        between = (~exact) & (indices > 0) & (indices < len(self.values))
        right = indices[between]
        left = right - 1
        low, high, x = self.values[left], self.values[right], values[between]
        with np.errstate(over="ignore", invalid="ignore"):
            distance = high - low
            weight = (x - low) / distance
        overflow = ~np.isfinite(distance)
        if overflow.any():
            # Halving avoids overflow when a finite interval spans both extremes.
            weight[overflow] = ((0.5 * x[overflow] - 0.5 * low[overflow])
                                / (0.5 * high[overflow] - 0.5 * low[overflow]))
        result[between] = ((1.0 - weight) * self.upper[left] + weight * self.lower[right])
        return np.clip(result, 0.0, 1.0)


def _source_cdfs(quantiles: np.ndarray) -> tuple[_SourceCDF, ...]:
    probabilities = np.linspace(0.0, 1.0, len(quantiles))
    result = []
    for slot in range(quantiles.shape[1]):
        values, first, counts = np.unique(quantiles[:, slot], return_index=True, return_counts=True)
        result.append(_SourceCDF(values, probabilities[first], probabilities[first + counts - 1]))
    return tuple(result)


@dataclass(frozen=True)
class _ConditionMap:
    train_count: int
    calibration_count: int
    target_quantiles: np.ndarray
    source_quantiles: np.ndarray
    source_cdfs: tuple[_SourceCDF, ...]
    source_tie_values: tuple[np.ndarray, ...]
    source_tie_counts: tuple[np.ndarray, ...]
    source_largest_atom_counts: np.ndarray


def _condition_map(train_count, calibration_count, target, source, tie_values, tie_counts,
                   largest, width, knots, label):
    """Validate both fitted and deserialized maps without silently repairing them."""
    train_count = _positive_integer(train_count, "train_count")
    calibration_count = _positive_integer(calibration_count, "calibration_count")
    for name, quantiles, count in (("target", target, train_count), ("source", source, calibration_count)):
        expected = max(2, count if knots is None else min(count, knots))
        if (quantiles.dtype.kind not in "iuf" or quantiles.shape != (expected, width)
                or not np.isfinite(quantiles).all() or np.any(quantiles[1:] < quantiles[:-1])):
            raise ValueError(f"invalid {name} quantile table")
    source_constant, target_constant = source[0] == source[-1], target[0] == target[-1]
    impossible = source_constant & ~target_constant
    if impossible.any():
        raise ValueError(f"constant source with nonconstant target: condition {label}, slots {np.flatnonzero(impossible).tolist()}")
    if (len(tie_values) != width or len(tie_counts) != width or largest.shape != (width,)
            or largest.dtype.kind not in "iu" or np.any(largest < 1) or np.any(largest > calibration_count)):
        raise ValueError("invalid source tie metadata")
    for slot, (values, counts) in enumerate(zip(tie_values, tie_counts)):
        if (values.ndim != 1 or values.dtype.kind not in "iuf" or not np.isfinite(values).all()
                or counts.shape != values.shape or counts.dtype.kind not in "iu"
                or np.any(values[1:] <= values[:-1]) or np.any(counts < 2)
                or np.any(counts > calibration_count) or counts.sum() > calibration_count
                or np.any(values < source[0, slot]) or np.any(values > source[-1, slot])
                or largest[slot] != (int(counts.max()) if len(counts) else 1)):
            raise ValueError("invalid source tie metadata")
    target, source = np.asarray(target, dtype=np.float64).copy(), np.asarray(source, dtype=np.float64).copy()
    if not np.isfinite(target).all() or not np.isfinite(source).all():
        raise ValueError("quantiles cannot be represented as finite float64 values")
    return _ConditionMap(train_count, calibration_count, target, source, _source_cdfs(source),
                         tuple(np.asarray(x, dtype=np.float64).copy() for x in tie_values),
                         tuple(np.asarray(x, dtype=np.int64).copy() for x in tie_counts),
                         largest.astype(np.int64, copy=True))


@dataclass(frozen=True)
class ConditionalMarginalCalibrator:
    n_features: int
    quantile_knots: int | None
    models: dict[int, _ConditionMap]
    provenance: dict

    @property
    def supported_labels(self) -> np.ndarray:
        return np.asarray(sorted(self.models), dtype=np.int64)

    def _inputs(self, values, labels):
        values = _profiles(values, "new_generated", empty=True)
        if values.shape[1] != self.n_features:
            raise ValueError("new_generated width disagrees with fitted maps")
        labels = _labels(labels, 64, len(values))
        unsupported = np.setdiff1d(np.unique(labels), self.supported_labels)
        if len(unsupported):
            raise ValueError(f"unsupported conditions absent from calibration: {unsupported.tolist()}")
        values = np.asarray(values, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("new_generated cannot be represented as finite float64 values")
        return values, labels

    def _apply(self, values, labels):
        values, labels = self._inputs(values, labels)
        uniforms = np.empty(values.shape, dtype=np.float64)
        output = np.empty_like(uniforms)
        for label in np.unique(labels):
            selected = labels == label
            model = self.models[int(label)]
            local = np.column_stack([cdf.evaluate(values[selected, slot])
                                     for slot, cdf in enumerate(model.source_cdfs)])
            uniforms[selected] = local
            output[selected] = _inverse_quantiles(local, model.target_quantiles)
        return values, labels, uniforms, output

    def transform(self, new_generated, new_labels) -> np.ndarray:
        """Apply fixed fitted maps; no randomness, batch ranks, or re-fitting."""
        return self._apply(new_generated, new_labels)[3]

    def fitting_diagnostics(self) -> dict:
        """Report actual source-sample tie masses separately from knot plateaus."""
        conditions = {}
        for label, model in self.models.items():
            slots = []
            for slot, cdf in enumerate(model.source_cdfs):
                counts = model.source_tie_counts[slot]
                slots.append({
                    "slot": slot,
                    "source_constant": bool(model.source_quantiles[0, slot] == model.source_quantiles[-1, slot]),
                    "target_constant": bool(model.target_quantiles[0, slot] == model.target_quantiles[-1, slot]),
                    "source_minimum": float(model.source_quantiles[0, slot]),
                    "source_maximum": float(model.source_quantiles[-1, slot]),
                    "target_minimum": float(model.target_quantiles[0, slot]),
                    "target_maximum": float(model.target_quantiles[-1, slot]),
                    "source_tie_mass": float(counts.sum() / model.calibration_count),
                    "source_largest_atom_mass": float(model.source_largest_atom_counts[slot] / model.calibration_count),
                    "source_ties": [{"value": float(value), "count": int(count),
                                     "mass": float(count / model.calibration_count)}
                                    for value, count in zip(model.source_tie_values[slot], counts)],
                    "source_knot_plateaus": [{"value": float(value), "probability_lower": float(lower),
                                              "probability_upper": float(upper), "mass": float(upper - lower)}
                                             for value, lower, upper in zip(cdf.values, cdf.lower, cdf.upper)
                                             if upper > lower],
                })
            conditions[str(label)] = {"train_count": model.train_count,
                                      "calibration_count": model.calibration_count, "slots": slots}
        return {"quantile_knots": self.quantile_knots, "per_condition": conditions,
                "tie_convention": "Quantile-plateau midpoint; interpolation between plateau edges, not between midpoints.",
                "source_tie_mass_definition": "Fraction of original calibration observations in equal-value groups of size >=2.",
                "knot_mass_definition": "Probability span of repeated quantile knots; differs from actual source-sample atom mass."}

    def diagnostics(self, new_generated, new_labels) -> dict:
        """Describe support exceedance and saturation without changing any map."""
        values, labels, uniforms, output = self._apply(new_generated, new_labels)
        fitted = self.fitting_diagnostics()["per_condition"]
        conditions = {}
        for label in np.unique(labels):
            selected = labels == label
            count = int(selected.sum())
            model = self.models[int(label)]
            slots = fitted[str(label)]["slots"]
            for slot, result in enumerate(slots):
                events = {
                    "source_below": values[selected, slot] < model.source_quantiles[0, slot],
                    "source_above": values[selected, slot] > model.source_quantiles[-1, slot],
                    "cdf_zero": uniforms[selected, slot] == 0,
                    "cdf_one": uniforms[selected, slot] == 1,
                    "target_lower_endpoint": output[selected, slot] == model.target_quantiles[0, slot],
                    "target_upper_endpoint": output[selected, slot] == model.target_quantiles[-1, slot],
                }
                for name, event in events.items():
                    result[f"{name}_count"] = int(event.sum())
                    result[f"{name}_fraction"] = float(event.mean())
            conditions[str(label)] = {"count": count, "slots": slots}
        return {"n_profiles": len(values), "per_condition": conditions,
                "interpretation": "Source exceedance is strict; CDF saturation is H=0/1; target endpoint equality can also arise from target atoms. Constant targets count at both endpoints."}

    def save(self, path: str | Path) -> None:
        """Store fixed map arrays and caller provenance as NPZ without pickle."""
        arrays = {"metadata": np.asarray(json.dumps({
            "schema_version": 1, "n_features": self.n_features,
            "quantile_knots": self.quantile_knots, "provenance": self.provenance,
        }, allow_nan=False)), "labels": self.supported_labels}
        for label, model in self.models.items():
            arrays[f"counts_{label}"] = np.asarray([model.train_count, model.calibration_count], dtype=np.int64)
            arrays[f"target_{label}"] = model.target_quantiles
            arrays[f"source_{label}"] = model.source_quantiles
            arrays[f"largest_{label}"] = model.source_largest_atom_counts
            for slot in range(self.n_features):
                arrays[f"tie_values_{label}_{slot}"] = model.source_tie_values[slot]
                arrays[f"tie_counts_{label}_{slot}"] = model.source_tie_counts[slot]
        with Path(path).open("wb") as stream:
            np.savez_compressed(stream, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> ConditionalMarginalCalibrator:
        """Restore and validate maps without accessing training or generator data."""
        with np.load(path, allow_pickle=False) as saved:
            try:
                metadata = json.loads(str(saved["metadata"].item()))
                if metadata["schema_version"] != 1:
                    raise ValueError("unsupported calibration schema_version")
                width = _positive_integer(metadata["n_features"], "n_features")
                knots, provenance = _knots(metadata["quantile_knots"]), _provenance(metadata["provenance"])
                labels = _labels(saved["labels"], 64)
                if not len(labels) or len(labels) != len(np.unique(labels)):
                    raise ValueError("invalid saved condition support")
                models = {}
                for label in labels:
                    counts = saved[f"counts_{label}"]
                    if counts.shape != (2,) or counts.dtype.kind not in "iu":
                        raise ValueError("invalid saved training/calibration counts")
                    models[int(label)] = _condition_map(
                        int(counts[0]), int(counts[1]), saved[f"target_{label}"], saved[f"source_{label}"],
                        tuple(saved[f"tie_values_{label}_{slot}"] for slot in range(width)),
                        tuple(saved[f"tie_counts_{label}_{slot}"] for slot in range(width)),
                        saved[f"largest_{label}"], width, knots, int(label))
            except (KeyError, TypeError, OverflowError) as exc:
                raise ValueError("invalid calibration archive") from exc
        return cls(width, knots, models, provenance)


def fit(real_train, train_labels, calibration_generated, calibration_labels,
        quantile_knots=1025, *, provenance=None) -> ConditionalMarginalCalibrator:
    """Fit fixed per-condition/per-slot maps on training and calibration draws.

    Both inputs must have identical observed condition support. They need not
    have equal row counts or correspond row-by-row. Provenance may include
    checkpoint_sha256, train_indices_sha256, calibration_seed and sample counts;
    it is copied and persisted. No held-out data should be supplied to this API.
    """
    knots, provenance = _knots(quantile_knots), _provenance(provenance)
    train = _profiles(real_train, "real_train")
    source = _profiles(calibration_generated, "calibration_generated")
    if train.shape[1] != source.shape[1]:
        raise ValueError("training and calibration profile widths must match")
    train_labels, calibration_labels = (_labels(train_labels, 64, len(train)),
                                         _labels(calibration_labels, 64, len(source)))
    support = np.unique(train_labels)
    if not np.array_equal(support, np.unique(calibration_labels)):
        raise ValueError("training and calibration condition support must match exactly")
    models, width = {}, train.shape[1]
    for label in support:
        target = np.sort(np.asarray(train[train_labels == label], dtype=np.float64), axis=0)
        generated = np.sort(np.asarray(source[calibration_labels == label], dtype=np.float64), axis=0)
        if not np.isfinite(target).all() or not np.isfinite(generated).all():
            raise ValueError("fitting values cannot be represented as finite float64")
        tied_values, tied_counts, largest = [], [], []
        for slot in range(width):
            values, counts = np.unique(generated[:, slot], return_counts=True)
            tied_values.append(values[counts > 1])
            tied_counts.append(counts[counts > 1])
            largest.append(int(counts.max()))
        models[int(label)] = _condition_map(
            len(target), len(generated), _quantiles(target, knots), _quantiles(generated, knots),
            tuple(tied_values), tuple(tied_counts), np.asarray(largest, dtype=np.int64), width, knots, int(label))
    return ConditionalMarginalCalibrator(width, knots, models, provenance)


def load_verified_calibrator(path, *, checkpoint_sha256, train_indices_sha256,
                             prepared_data_hashes, generation_seed, n_features,
                             train_labels):
    """Bind a saved map to verified checkpoint/data/subset identities before use.

    Callers first verify their checkpoint and prepared arrays, then pass those
    identities here. This function reads only the calibration archive. It
    verifies provenance consistency, not the truth of an arbitrary caller's
    original calibration procedure. Production generation must use a different
    seed from the saved calibration reference.
    """
    def digest_file(filename):
        digest = hashlib.sha256()
        with Path(filename).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    expected = _provenance({"checkpoint_sha256": checkpoint_sha256,
                            "train_indices_sha256": train_indices_sha256})
    if not isinstance(prepared_data_hashes, dict) or not prepared_data_hashes:
        raise ValueError("prepared_data_hashes must be a nonempty fingerprint mapping")
    if any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
           for value in prepared_data_hashes.values()):
        raise ValueError("prepared_data_hashes must contain SHA-256 hex digests")
    generation_seed = _positive_integer(generation_seed, "generation_seed", 0)
    n_features = _positive_integer(n_features, "n_features")
    if generation_seed >= 2 ** 32:
        raise ValueError("generation_seed must be smaller than 2**32")
    train_labels = _labels(train_labels, 64)
    labels, counts = np.unique(train_labels, return_counts=True)
    path = Path(path)
    before = digest_file(path)
    model = ConditionalMarginalCalibrator.load(path)
    if digest_file(path) != before:
        raise ValueError("calibration archive changed while loading")
    provenance = model.provenance
    for name, value in expected.items():
        if provenance.get(name) != value:
            raise ValueError(f"calibration {name} does not match the verified checkpoint training subset")
    if provenance.get("prepared_data_hashes") != prepared_data_hashes:
        raise ValueError("calibration prepared_data_hashes do not match verified prepared data")
    if model.n_features != n_features:
        raise ValueError("calibration feature width disagrees with verified data")
    if not np.array_equal(model.supported_labels, labels):
        raise ValueError("calibration condition support differs from verified training subset")
    if any(model.models[int(label)].train_count != int(count) for label, count in zip(labels, counts)):
        raise ValueError("calibration training counts differ from verified training subset")
    calibration_seed = _positive_integer(provenance.get("calibration_seed"), "calibration_seed", 0)
    if calibration_seed >= 2 ** 32 or calibration_seed == generation_seed:
        raise ValueError("calibration and generation seeds must be distinct valid 32-bit seeds")
    per_condition = _positive_integer(provenance.get("calibration_samples_per_condition"),
                                      "calibration_samples_per_condition")
    total = _positive_integer(provenance.get("calibration_profiles_total"), "calibration_profiles_total")
    if (any(fitted.calibration_count != per_condition for fitted in model.models.values())
            or total != sum(fitted.calibration_count for fitted in model.models.values())):
        raise ValueError("calibration sample counts disagree with saved maps")
    for name in ("calibration_generated_profiles_sha256", "calibration_labels_sha256"):
        value = provenance.get(name)
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError(f"calibration provenance lacks a valid {name}")
    reference_labels = np.repeat(model.supported_labels, per_condition).astype("<i8")
    if hashlib.sha256(reference_labels.tobytes(order="C")).hexdigest() != provenance["calibration_labels_sha256"]:
        raise ValueError("calibration_labels_sha256 disagrees with the fixed per-condition reference labels")
    _positive_integer(provenance.get("generator_inference_batch_size"), "generator_inference_batch_size")
    if "quantile_knots" not in provenance or provenance["quantile_knots"] != model.quantile_knots:
        raise ValueError("calibration quantile_knots provenance disagrees with saved maps")
    details = {"path": str(path), "sha256": before, "quantile_knots": model.quantile_knots,
               "calibration_seed": calibration_seed, "generation_seed": generation_seed,
               "provenance": _provenance(provenance),
               "training_condition_counts": {str(label): int(count) for label, count in zip(labels, counts)}}
    return model, details
