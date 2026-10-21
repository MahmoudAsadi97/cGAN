"""Finite-IID-ensemble-corrected energy and variogram score functions.

The corrections are unbiased over IID ensemble generation for a fixed fitted
predictive distribution and fixed observation, under the required moments.
Conditioning on one saved ensemble instead defines fixed score functions with
model-specific random offsets. These routines perform no inference and cannot
resolve checkpoint selection, reused validation data, or dependent ensemble
members. All physical values, including signed net energy, remain unmodified.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
from scipy.spatial.distance import cdist


SCHEMA = 1


def _profiles(values, minimum_rows):
    values = np.asarray(values)
    if (values.ndim != 2 or len(values) < minimum_rows or values.shape[1] < 2
            or values.dtype.kind not in 'fiu' or not np.isfinite(values).all()):
        raise ValueError(f'Expected finite real profiles with at least {minimum_rows} rows and 2 coordinates')
    with np.errstate(over='raise', invalid='raise'):
        result = np.asarray(values, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError('Profiles exceed finite float64 precision')
    return result


def _finite(values, calculation):
    if not np.isfinite(values).all():
        raise FloatingPointError(f'Nonfinite {calculation}; score calculation exceeded float64 precision')
    return values


def _distance(first, second, scale):
    # Direct coordinate differences inside cdist avoid cancellation from the
    # squared-norm identity at large shared offsets. Check its C-level overflow.
    return _finite(cdist(first, second, metric='euclidean') / scale, 'Euclidean distances')


def _variogram_features(rows, first, second):
    return _finite(np.sqrt(np.abs(rows[:, first] - rows[:, second])), 'variogram features')


def _feature_moments(ensemble, first, second, block_size):
    """Merge centered block moments; never subtract two large raw moments."""
    means = np.empty(len(first), dtype=np.float64)
    variances = np.empty_like(means)
    for start in range(0, len(first), block_size):
        stop = min(start + block_size, len(first))
        count, mean, centered_sum = 0, np.zeros(stop - start), np.zeros(stop - start)
        for row_start in range(0, len(ensemble), block_size):
            features = _variogram_features(ensemble[row_start:row_start + block_size],
                                           first[start:stop], second[start:stop])
            block_count = len(features)
            # Subtract a shared reference before the mean, so even large
            # constant features retain exactly zero within-block variance.
            reference = features[0]
            shifted = features - reference
            shifted_mean = shifted.mean(axis=0)
            block_mean = reference + shifted_mean
            block_centered_sum = ((shifted - shifted_mean) ** 2).sum(axis=0)
            if count == 0:
                mean, centered_sum = block_mean, block_centered_sum
            else:
                delta = block_mean - mean
                total = count + block_count
                centered_sum = centered_sum + block_centered_sum + delta ** 2 * (count * block_count / total)
                mean = mean + delta * (block_count / total)
            count += block_count
        means[start:stop] = mean
        variances[start:stop] = centered_sum / (count - 1)
    return _finite(means, 'variogram means'), _finite(variances, 'variogram sample variances')


def _readonly(values):
    result = np.array(values, copy=True, order='C')
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class FittedEnsembleScores:
    """Frozen ensemble and score offsets; use ``fit_ensemble_scores`` to construct.

    Pair-distance and variogram-feature work arrays have both axes bounded by
    ``block_size``. Persistent storage is O(M*D + D**2), with O(N) returned
    scores; no N*M*D or N*M work array is required. Small problems may fit in a
    single tile. Score arrays and metadata returned to callers are fresh copies.
    """

    _ensemble: np.ndarray
    _first: np.ndarray
    _second: np.ndarray
    _weights: np.ndarray
    _feature_mean: np.ndarray
    _feature_variance: np.ndarray
    _pair_mean: float
    _variogram_offset: float
    _ensemble_sha256: str
    block_size: int

    @property
    def ensemble_size(self):
        return len(self._ensemble)

    @property
    def profile_dim(self):
        return self._ensemble.shape[1]

    def score(self, observations):
        """Return per-observation energy/variogram scores; lower is better.

        Negative corrected variogram estimates are retained. No result clipping
        is performed. A nonfinite derived quantity fails explicitly, including
        arithmetic overflow despite finite input values.
        """
        observations = _profiles(observations, minimum_rows=1)
        if observations.shape[1] != self.profile_dim:
            raise ValueError('Observation profile dimension differs from ensemble')
        energy = np.empty(len(observations), dtype=np.float64)
        variogram = np.empty_like(energy)
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            scale = np.sqrt(self.profile_dim)
            for start in range(0, len(observations), self.block_size):
                stop = min(start + self.block_size, len(observations))
                query = observations[start:stop]
                first_term, second_score = np.zeros(len(query)), np.zeros(len(query))
                for member_start in range(0, self.ensemble_size, self.block_size):
                    distance = _distance(query, self._ensemble[member_start:member_start + self.block_size], scale)
                    first_term += (distance / self.ensemble_size).sum(axis=1)
                for pair_start in range(0, len(self._first), self.block_size):
                    pair_stop = min(pair_start + self.block_size, len(self._first))
                    features = _variogram_features(query, self._first[pair_start:pair_stop],
                                                   self._second[pair_start:pair_stop])
                    difference = features - self._feature_mean[pair_start:pair_stop]
                    second_score += (difference ** 2) @ self._weights[pair_start:pair_stop]
                energy[start:stop] = first_term - .5 * self._pair_mean
                variogram[start:stop] = second_score - self._variogram_offset
        return {'energy_score': _finite(energy, 'energy scores'),
                'variogram_score': _finite(variogram, 'variogram scores')}

    def metadata(self):
        """Describe the fixed finite-ensemble score functions and random offsets."""
        return {'schema_version': SCHEMA, 'ensemble_size': self.ensemble_size,
            'profile_dim': self.profile_dim, 'ensemble_sha256': self._ensemble_sha256,
            'ensemble_hash_encoding': 'row-major little-endian float64 bytes',
            'block_size': self.block_size, 'distance': 'Euclidean / sqrt(profile_dim)',
            'energy_estimator': 'mean_i d(z_i,y) - sum_{i!=j} d(z_i,z_j)/(2*M*(M-1))',
            'energy_off_diagonal_pair_mean': self._pair_mean,
            'energy_empirical_distribution_offset': self._pair_mean / (2 * self.ensemble_size),
            'variogram_power': .5, 'variogram_pairs': 'all h<k within the profile; no cyclic wrap',
            'variogram_pair_count': len(self._first),
            'variogram_weights': '(1/(k-h)) / sum_{a<b}(1/(b-a))',
            'variogram_estimator': 'sum_{h<k} w_hk*((abs(y_h-y_k)^0.5 - mean_i A_i,hk)^2 - s_hk^2/M)',
            'variogram_sample_variance_ddof': 1,
            'variogram_empirical_distribution_offset': self._variogram_offset,
            'offset_convention': 'corrected score = empirical-distribution score - stated offset',
            'negative_variogram_estimates_retained': True, 'score_units': 'same as physical profile values',
            'interpretation': 'Unbiasedness requires IID ensemble members and finite moments for a fixed '
                'generator and observation. Conditional on the ensemble these are fixed score functions; '
                'no selection adjustment, independence validation, confidence interval or coverage claim is provided.'}


def fit_ensemble_scores(ensemble, block_size=256):
    """Fit score offsets from M>=2 profiles of dimension D>=2, without real data."""
    if type(block_size) is not int or block_size < 1:
        raise ValueError('block_size must be a positive integer')
    ensemble = _readonly(_profiles(ensemble, minimum_rows=2))
    count, dimension = ensemble.shape
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        scale, pair_mean = np.sqrt(dimension), 0.
        for first_start in range(0, count, block_size):
            for second_start in range(first_start, count, block_size):
                distance = _distance(ensemble[first_start:first_start + block_size],
                                     ensemble[second_start:second_start + block_size], scale)
                if first_start == second_start:
                    np.fill_diagonal(distance, 0.)
                    multiplicity = 1
                else:
                    multiplicity = 2  # Account for both ordered cross-block pairs.
                pair_mean += multiplicity * float((distance / (count * (count - 1))).sum())
        _finite(pair_mean, 'energy off-diagonal pair mean')
        first, second = np.triu_indices(dimension, k=1)
        weights = 1. / (second - first)
        weights /= weights.sum()
        means, variances = _feature_moments(ensemble, first, second, block_size)
        variogram_offset = float(np.dot(weights, variances / count))
        _finite(variogram_offset, 'variogram finite-ensemble offset')
    digest = hashlib.sha256(np.ascontiguousarray(ensemble, dtype='<f8').tobytes()).hexdigest()
    return FittedEnsembleScores(ensemble, _readonly(first), _readonly(second), _readonly(weights),
        _readonly(means), _readonly(variances), pair_mean, variogram_offset, digest, block_size)
