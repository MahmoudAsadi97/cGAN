"""Frozen conditional empirical marginals for dependence-model comparators.

Fitting provenance is enforced by the experiment runner. This numerical module
must receive only fitting-partition rows; it does not identify partition roles.
It preserves measured atoms and signed hourly support with empirical step
quantiles, whose tails stop at the observed conditional TRAIN extrema.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from types import MappingProxyType
import zipfile

import numpy as np


SCHEMA = 1
METHOD = 'conditional_empirical_marginals'


def _profiles(values):
    values = np.asarray(values)
    if (values.ndim != 2 or not all(values.shape) or values.dtype.kind not in 'fiu'
            or not np.isfinite(values).all()):
        raise ValueError('Profiles must be a nonempty finite real matrix')
    result = np.asarray(values, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError('Profiles exceed finite float64 precision')
    return result


def _labels(values, length):
    values = np.asarray(values)
    if (values.ndim != 1 or len(values) != length or values.dtype.kind not in 'fiu'
            or not np.isfinite(values).all() or np.any(values != np.floor(values))
            or np.any((values < 0) | (values >= 64))):
        raise ValueError('Condition labels must be integer values in [0,64) matching profiles')
    return values.astype(np.int64)


@dataclass(frozen=True)
class EmpiricalMarginals:
    """Owned read-only sorted TRAIN samples, separately for each condition/hour."""

    profile_dim: int
    _targets: dict

    def __post_init__(self):
        if type(self.profile_dim) is not int or self.profile_dim < 1 or not self._targets:
            raise ValueError('Marginals require a positive profile dimension and conditions')
        conditions = list(self._targets)
        _labels(conditions, len(conditions))
        if any(type(label) is not int for label in conditions):
            raise ValueError('Fitted condition keys must be integers')
        groups = {}
        for label in sorted(conditions):
            target = _profiles(self._targets[label])
            if target.shape[1] != self.profile_dim:
                raise ValueError('Marginal profile dimensions differ')
            if np.any(target[1:] < target[:-1]):
                raise ValueError('Fitted marginal columns must be sorted')
            target = target.copy()
            target.flags.writeable = False
            groups[label] = target
        object.__setattr__(self, '_targets', MappingProxyType(groups))

    @property
    def conditions(self):
        return tuple(self._targets)

    def _query(self, values, labels):
        values = _profiles(values)
        labels = _labels(labels, len(values))
        if values.shape[1] != self.profile_dim:
            raise ValueError('Application profile dimension differs from fitted marginals')
        missing = set(labels.tolist()) - self._targets.keys()
        if missing:
            raise ValueError(f'Unsupported condition labels: {sorted(missing)}')
        return values, labels

    def intervals(self, values, labels):
        """Return F_train(x-) and F_train(x) without fitting on query values.

        For uniformly sampled TRAIN rows and fresh independent coordinatewise
        Uniform(0,1) jitter V, lower + V*(upper-lower) has uniform marginal
        probabilities in the ideal continuous-RNG model. Ties share an interval
        whose width is their empirical mass; a constant coordinate has [0,1].
        Unseen query values in gaps have zero-width intervals. Applying this
        frozen transform to validation data does not ensure uniform marginals.
        """
        values, labels = self._query(values, labels)
        lower, upper = np.empty_like(values), np.empty_like(values)
        for label in np.unique(labels):
            positions = np.flatnonzero(labels == label)
            query, target = values[positions], self._targets[int(label)]
            for hour in range(self.profile_dim):
                lower[positions, hour] = np.searchsorted(target[:, hour], query[:, hour], side='left') / len(target)
                upper[positions, hour] = np.searchsorted(target[:, hour], query[:, hour], side='right') / len(target)
        return lower, upper

    def inverse(self, probabilities, labels):
        """Decode with sorted TRAIN[clip(ceil(n*u)-1, 0, n-1)].

        Probabilities must already be finite and in [0,1]; no out-of-range
        probabilities are clipped. Index bounding defines u=0 at the minimum
        and u=1 at the maximum. Exact positive lower atom boundaries belong to
        the preceding atom. Consequently randomized TRAIN-interval inversion
        recovers the original row almost surely for continuous jitter, but a
        finite RNG returning exactly zero can hit these lower boundaries.
        No target interpolation, extrapolation or physical-value clipping is used.
        """
        probabilities, labels = self._query(probabilities, labels)
        if np.any((probabilities < 0) | (probabilities > 1)):
            raise ValueError('Marginal probabilities must lie in [0,1]')
        result = np.empty_like(probabilities)
        for label in np.unique(labels):
            positions = np.flatnonzero(labels == label)
            target = self._targets[int(label)]
            ranks = np.clip(np.ceil(len(target) * probabilities[positions]).astype(np.int64) - 1,
                            0, len(target) - 1)
            result[positions] = target[ranks, np.arange(self.profile_dim)]
        return result

    def summary(self):
        """Return fresh JSON-safe fit counts; duplicate counts are n minus unique."""
        groups = {}
        for label, target in self._targets.items():
            groups[str(label)] = {'training_profiles': len(target),
                'constant_hours': np.flatnonzero(target[0] == target[-1]).tolist(),
                'duplicate_counts_by_hour': (target[1:] == target[:-1]).sum(axis=0).tolist(),
                'zero_counts': (target == 0).sum(axis=0).tolist(),
                'negative_counts': (target < 0).sum(axis=0).tolist(),
                'min_counts': (target == target[0]).sum(axis=0).tolist(),
                'max_counts': (target == target[-1]).sum(axis=0).tolist(),
                'minimum': target[0].tolist(), 'maximum': target[-1].tolist()}
        return {'schema_version': SCHEMA, 'method': METHOD, 'profile_dim': self.profile_dim,
                'conditions': list(self.conditions), 'per_condition': groups}

    def save(self, path):
        """Write a numeric NPZ exclusively; the runner records provenance and hash."""
        metadata = {'schema_version': SCHEMA, 'method': METHOD, 'profile_dim': self.profile_dim,
                    'conditions': list(self.conditions)}
        arrays = {'metadata': np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False)),
                  **{f'target_{label}': target for label, target in self._targets.items()}}
        with Path(path).open('xb') as handle:
            np.savez_compressed(handle, **arrays)


def fit_marginals(training, labels):
    """Fit conditional empirical targets from fitting rows supplied by the runner."""
    training = _profiles(training)
    labels = _labels(labels, len(training))
    targets = {int(label): np.sort(training[labels == label], axis=0) for label in np.unique(labels)}
    return EmpiricalMarginals(training.shape[1], targets)


def load_marginals(path, expected_sha256):
    """Load a hash-bound schema-checked numeric marginal artifact without pickle."""
    if not isinstance(expected_sha256, str) or not re.fullmatch('[0-9a-f]{64}', expected_sha256):
        raise ValueError('A lowercase hexadecimal expected SHA-256 digest is required')
    with Path(path).open('rb') as handle:
        digest = hashlib.sha256()
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
        if digest.hexdigest() != expected_sha256:
            raise ValueError('Marginal artifact SHA-256 mismatch')
        handle.seek(0)
        try:
            with np.load(handle, allow_pickle=False) as stored:
                raw = stored['metadata']
                if raw.shape != () or raw.dtype.kind != 'U':
                    raise ValueError('Invalid marginal metadata encoding')
                metadata = json.loads(raw.item())
                if (not isinstance(metadata, dict)
                        or set(metadata) != {'schema_version', 'method', 'profile_dim', 'conditions'}
                        or type(metadata['schema_version']) is not int or metadata['schema_version'] != SCHEMA
                        or metadata['method'] != METHOD or type(metadata['profile_dim']) is not int
                        or metadata['profile_dim'] < 1 or not isinstance(metadata['conditions'], list)):
                    raise ValueError('Unsupported marginal metadata schema')
                labels = metadata['conditions']
                if (not labels or any(type(label) is not int for label in labels)
                        or labels != sorted(set(labels))):
                    raise ValueError('Invalid saved condition labels')
                _labels(labels, len(labels))
                expected = {'metadata'} | {f'target_{label}' for label in labels}
                if set(stored.files) != expected or len(stored.files) != len(expected):
                    raise ValueError('Marginal array keys differ from metadata')
                targets = {}
                for label in labels:
                    target = stored[f'target_{label}']
                    if target.dtype != np.float64:
                        raise ValueError('Saved marginal arrays must be float64')
                    targets[label] = target
                return EmpiricalMarginals(metadata['profile_dim'], targets)
        except (ValueError, TypeError, KeyError, OSError, zipfile.BadZipFile) as error:
            raise ValueError(f'Invalid marginal artifact: {error}') from error
