"""A fixed finite-projection diagnostic for whole-profile joint distributions.

This is a secondary empirical diagnostic, independent of the training projection
stream. Directions, count and seed are fixed before curated endpoint evaluation.
No learned transform, fitting, clipping or current-batch ranking is applied.
"""
from __future__ import annotations

import hashlib

import numpy as np
from scipy.stats import wasserstein_distance

from .evaluation import _labels, _matching_width, _profiles


_PROJECTION_COUNT = 128
_PROJECTION_SEED = 9182


def conditional_projected_wasserstein(real, real_labels, generated, generated_labels):
    """Average exact empirical 1D W1 over fixed directions and then conditions.

    Inputs are physical-unit daily profiles, with no row pairing required.
    The same 128 unit Gaussian directions are used for every condition and every
    method at a given profile width. Independent sorted empirical CDFs allow
    unequal real/generated sample counts. Each observed condition receives equal
    macro weight. Both label supports must match exactly.

    The finite-direction mean is not the integral over the sphere, a full joint
    distribution equality certificate, a confidence interval or a new hypothesis
    test. Its magnitude depends on profile width and energy resolution. Compare
    methods only within the same resolution using the same frozen directions.
    """
    real = _profiles(real, "real")
    generated = _profiles(generated, "generated")
    _matching_width(real, generated)
    real_labels = _labels(real_labels, "real_labels", len(real))
    generated_labels = _labels(generated_labels, "generated_labels", len(generated))
    real_support, generated_support = np.unique(real_labels), np.unique(generated_labels)
    if not np.array_equal(real_support, generated_support):
        raise ValueError(
            "condition support differs; missing generated conditions: "
            f"{np.setdiff1d(real_support, generated_support).tolist()}; missing real conditions: "
            f"{np.setdiff1d(generated_support, real_support).tolist()}"
        )
    directions = np.random.default_rng(_PROJECTION_SEED).normal(size=(_PROJECTION_COUNT, real.shape[1]))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    try:
        with np.errstate(over="raise", invalid="raise"):
            real_projected = real @ directions.T
            generated_projected = generated @ directions.T
    except FloatingPointError as error:
        raise ValueError("profiles exceed finite projected-value range") from error
    if not np.isfinite(real_projected).all() or not np.isfinite(generated_projected).all():
        raise ValueError("profiles exceed finite projected-value range")
    per_condition = {}
    for label in real_support:
        real_part = real_projected[real_labels == label]
        generated_part = generated_projected[generated_labels == label]
        distances = [wasserstein_distance(real_part[:, column], generated_part[:, column])
                     for column in range(_PROJECTION_COUNT)]
        mean = float(np.mean(distances))
        if not np.isfinite(mean):
            raise ValueError("projected Wasserstein distance exceeds finite numeric range")
        per_condition[str(label)] = {
            "mean_projected_wasserstein_kwh": mean,
            "n_real": len(real_part), "n_generated": len(generated_part),
        }
    return {
        "metric": "conditional_finite_projected_wasserstein_1",
        "role": "secondary_whole_profile_joint_diagnostic",
        "macro": float(np.mean([row["mean_projected_wasserstein_kwh"] for row in per_condition.values()])),
        "per_condition": per_condition,
        "n_conditions": len(per_condition),
        "condition_weighting": "equal_over_observed_support",
        "units": "kWh/projection; unit-norm weighted interval energies",
        "directions": {
            "count": _PROJECTION_COUNT, "seed": _PROJECTION_SEED, "n_features": real.shape[1],
            "distribution": "IID standard Gaussian coordinates, each direction normalized to Euclidean norm one",
            "reuse": "Same directions across all conditions and methods at this profile width",
            "sha256": hashlib.sha256(np.asarray(directions, dtype="<f8").tobytes(order="C")).hexdigest(),
            "hash_encoding": "Direction matrix (count, n_features), little-endian float64 C-order bytes",
            "numpy_version": str(np.__version__),
        },
        "comparability": "Within-resolution comparisons only; magnitude changes with profile width and energy aggregation.",
        "interpretation": (
            "Exact empirical 1D W1 on each of 128 fixed whole-profile projections, averaged over projections "
            "then conditions. Finite directions do not certify equality of full joint distributions. "
            "This secondary descriptive diagnostic has no bootstrap interval or significance interpretation; "
            "marginal and dependence differences both affect it."
        ),
    }
