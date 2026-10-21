"""Frozen conditional empirical marginal calibration.

For each condition/hour, map x to Q_train(F_generated(x)), using the midpoint
empirical CDF of an independent fixed generator sample. Q is the empirical
step inverse used by the Gaussian-copula baseline. This classical operation
does not fit on, rerank, or jitter an evaluation batch. It cannot repair a
collapsed generated coordinate or establish realistic joint-profile support.
Empirical target marginals truncate tails at the observed TRAIN extrema.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
from types import MappingProxyType
import zipfile

import numpy as np


SCHEMA = 1
METHOD = 'conditional_mid_ecdf_step_inverse'
DECILE_EDGES = np.linspace(0., 1., 11)
COUNT_FIELDS = ('below_source_min', 'above_source_max', 'equal_source_min',
                'equal_source_max', 'source_tie_matches', 'source_value_matches',
                'probability_zero', 'probability_one', 'mapped_target_min',
                'mapped_target_max', 'mapped_target_zero', 'mapped_target_negative')


def _profiles(values):
    values = np.asarray(values)
    if (values.ndim != 2 or not all(values.shape) or values.dtype.kind not in 'fiu'
            or not np.isfinite(values).all()):
        raise ValueError('Profiles must be a nonempty finite real matrix')
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError('Profiles exceed finite float64 precision')
    return values


def _labels(values, length):
    values = np.asarray(values)
    if (values.ndim != 1 or len(values) != length or values.dtype.kind not in 'fiu'
            or not np.isfinite(values).all() or np.any(values != np.floor(values))
            or np.any((values < 0) | (values >= 64))):
        raise ValueError('Condition labels must be integer values in [0,64) matching profiles')
    return values.astype(np.int64)


def _group(source, target, dimension, condition):
    source, target = _profiles(source), _profiles(target)
    if source.shape[1] != dimension or target.shape[1] != dimension:
        raise ValueError('Source and target profile dimensions differ')
    if np.any(source[1:] < source[:-1]) or np.any(target[1:] < target[:-1]):
        raise ValueError('Calibration source and target columns must be sorted')
    collapsed = (source[0] == source[-1]) & (target[0] != target[-1])
    if np.any(collapsed):
        raise ValueError(f'A constant generated source cannot calibrate a varying target: '
                         f'condition {condition}, hours {np.flatnonzero(collapsed).tolist()}')
    source, target = source.copy(), target.copy()
    source.flags.writeable = target.flags.writeable = False
    return source, target


@dataclass(frozen=True)
class ConditionalMarginalCalibration:
    """A fixed deterministic map; source/target arrays are owned read-only copies.

    ``apply`` reports counts per evaluated condition/hour. Duplicate fit counts
    are n minus the number of unique values. ``source_tie_matches`` counts query
    values exactly equal to a source value occurring more than once; the wider
    ``source_value_matches`` includes singleton source values. Min/max counts
    overlap for constant columns. ``training_*_counts`` describe the target
    fitting rows, for comparison with the mapped evaluation counts; their
    denominator is ``training_profiles``, not ``profiles``.
    Probability histograms use NumPy intervals
    [edge_i, edge_i+1), with the final interval including one. They describe
    finite-source calibration on independent draws, without uniformity p-values.
    """

    profile_dim: int
    _groups: dict
    _training_counts: dict = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if type(self.profile_dim) is not int or self.profile_dim < 1 or not self._groups:
            raise ValueError('A fitted map requires a positive profile dimension and conditions')
        conditions = list(self._groups)
        _labels(conditions, len(conditions))
        if any(type(label) is not int for label in conditions):
            raise ValueError('Fitted condition keys must be integers')
        groups = {label: _group(*self._groups[label], self.profile_dim, label)
                  for label in sorted(conditions)}
        object.__setattr__(self, '_groups', MappingProxyType(groups))
        counts = {}
        for label, (_, target) in groups.items():
            indicators = {'training_zero_counts': target == 0,
                          'training_negative_counts': target < 0,
                          'training_min_counts': target == target[0],
                          'training_max_counts': target == target[-1]}
            counts[label] = MappingProxyType({key: tuple(value.sum(axis=0).tolist())
                                             for key, value in indicators.items()})
        object.__setattr__(self, '_training_counts', MappingProxyType(counts))

    @property
    def conditions(self):
        return tuple(self._groups)

    def apply(self, values, labels):
        """Return calibrated float64 profiles and JSON-safe descriptive diagnostics."""
        values = _profiles(values)
        labels = _labels(labels, len(values))
        if values.shape[1] != self.profile_dim:
            raise ValueError('Application profile dimension differs from fitted map')
        missing = set(labels.tolist()) - self._groups.keys()
        if missing:
            raise ValueError(f'Unsupported condition labels: {sorted(missing)}')
        result = np.empty_like(values)
        diagnostics = {'schema_version': SCHEMA, 'method': METHOD, 'profiles': len(values),
            'profile_dim': self.profile_dim, 'probability_decile_edges': DECILE_EDGES.tolist(),
            'probability_histogram_orientation': '[hour][bin]; final bin includes 1',
            'probability_definition': '(count(source<x)+count(source<=x))/(2*source_rows)',
            'target_inverse': 'sorted TRAIN[clip(ceil(TRAIN_rows*u)-1, 0, TRAIN_rows-1)]',
            'interpretation': 'Frozen empirical marginals; no evaluation-batch fitting or random tie breaking. '
                'Output ties can change rank statistics. Conditional hourly support does not imply joint support. '
                'Empirical targets truncate unseen tails; finite source CDFs do not guarantee marginal frequencies.',
            'per_condition': {}}
        for label in np.unique(labels):
            positions = np.flatnonzero(labels == label)
            query = values[positions]
            source, target = self._groups[int(label)]
            left = np.empty(query.shape, dtype=np.int64)
            right = np.empty(query.shape, dtype=np.int64)
            for hour in range(self.profile_dim):
                left[:, hour] = np.searchsorted(source[:, hour], query[:, hour], side='left')
                right[:, hour] = np.searchsorted(source[:, hour], query[:, hour], side='right')
            probability = (left + right) / (2. * len(source))
            ranks = np.clip(np.ceil(len(target) * probability).astype(np.int64) - 1,
                            0, len(target) - 1)
            mapped = target[ranks, np.arange(self.profile_dim)]
            result[positions] = mapped
            indicators = {'below_source_min': query < source[0],
                'above_source_max': query > source[-1], 'equal_source_min': query == source[0],
                'equal_source_max': query == source[-1], 'source_tie_matches': right - left > 1,
                'source_value_matches': right > left, 'probability_zero': probability == 0,
                'probability_one': probability == 1, 'mapped_target_min': mapped == target[0],
                'mapped_target_max': mapped == target[-1], 'mapped_target_zero': mapped == 0,
                'mapped_target_negative': mapped < 0}
            diagnostics['per_condition'][str(int(label))] = {
                'profiles': len(query), 'training_profiles': len(target),
                'calibration_profiles': len(source),
                'source_constant_hours': np.flatnonzero(source[0] == source[-1]).tolist(),
                'target_constant_hours': np.flatnonzero(target[0] == target[-1]).tolist(),
                'source_duplicate_counts_by_hour': (source[1:] == source[:-1]).sum(axis=0).tolist(),
                'target_duplicate_counts_by_hour': (target[1:] == target[:-1]).sum(axis=0).tolist(),
                **{name: list(counts) for name, counts in self._training_counts[int(label)].items()},
                **{name: indicator.sum(axis=0).tolist() for name, indicator in indicators.items()},
                'probability_decile_counts': [np.histogram(probability[:, hour], DECILE_EDGES)[0].tolist()
                                              for hour in range(self.profile_dim)],
                'probability_mean': probability.mean(axis=0).tolist()}
        diagnostics['total_coordinate_counts'] = {name: sum(sum(group[name])
            for group in diagnostics['per_condition'].values()) for name in COUNT_FIELDS}
        return result, diagnostics

    def save(self, path):
        """Write a numeric NPZ map exclusively; provenance belongs to its run manifest."""
        metadata = {'schema_version': SCHEMA, 'method': METHOD,
                    'profile_dim': self.profile_dim, 'conditions': list(self.conditions)}
        arrays = {'metadata': np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False))}
        for label, (source, target) in self._groups.items():
            arrays[f'source_{label}'], arrays[f'target_{label}'] = source, target
        with Path(path).open('xb') as handle:
            np.savez_compressed(handle, **arrays)


def fit_calibration(training, training_labels, generated, generated_labels):
    """Fit source and target marginals from caller-supplied TRAIN/calibration data.

    This numerical API cannot identify data provenance. The experiment runner
    must bind fitting-partition hashes, checkpoint identity and disjoint seeds.
    Generated fitting rows must not be the later evaluation samples.
    """
    training, generated = _profiles(training), _profiles(generated)
    training_labels = _labels(training_labels, len(training))
    generated_labels = _labels(generated_labels, len(generated))
    if training.shape[1] != generated.shape[1]:
        raise ValueError('Source and target profile dimensions differ')
    conditions = np.unique(training_labels)
    if not np.array_equal(conditions, np.unique(generated_labels)):
        raise ValueError('Source and target condition coverage must match')
    groups = {int(label): (np.sort(generated[generated_labels == label], axis=0),
                           np.sort(training[training_labels == label], axis=0))
              for label in conditions}
    return ConditionalMarginalCalibration(training.shape[1], groups)


def load_calibration(path, expected_sha256):
    """Load a hash-bound, schema-checked numeric map without pickle support."""
    if not isinstance(expected_sha256, str) or not re.fullmatch('[0-9a-f]{64}', expected_sha256):
        raise ValueError('A lowercase hexadecimal expected SHA-256 digest is required')
    with Path(path).open('rb') as handle:
        digest = hashlib.sha256()
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
        if digest.hexdigest() != expected_sha256:
            raise ValueError('Calibration artifact SHA-256 mismatch')
        handle.seek(0)
        try:
            with np.load(handle, allow_pickle=False) as stored:
                raw = stored['metadata']
                if raw.shape != () or raw.dtype.kind != 'U':
                    raise ValueError('Invalid calibration metadata encoding')
                metadata = json.loads(raw.item())
                if (not isinstance(metadata, dict)
                        or set(metadata) != {'schema_version', 'method', 'profile_dim', 'conditions'}
                        or type(metadata['schema_version']) is not int or metadata['schema_version'] != SCHEMA
                        or metadata['method'] != METHOD or type(metadata['profile_dim']) is not int
                        or metadata['profile_dim'] < 1 or not isinstance(metadata['conditions'], list)):
                    raise ValueError('Unsupported calibration metadata schema')
                labels = metadata['conditions']
                if (not labels or any(type(label) is not int for label in labels)
                        or labels != sorted(set(labels))):
                    raise ValueError('Invalid saved condition labels')
                _labels(labels, len(labels))
                expected = {'metadata'} | {f'{kind}_{label}' for label in labels for kind in ('source', 'target')}
                if set(stored.files) != expected or len(stored.files) != len(expected):
                    raise ValueError('Calibration array keys differ from metadata')
                groups = {}
                for label in labels:
                    source, target = stored[f'source_{label}'], stored[f'target_{label}']
                    if source.dtype != np.float64 or target.dtype != np.float64:
                        raise ValueError('Saved calibration arrays must be float64')
                    groups[label] = source, target
                return ConditionalMarginalCalibration(metadata['profile_dim'], groups)
        except (ValueError, TypeError, KeyError, OSError, zipfile.BadZipFile) as error:
            raise ValueError(f'Invalid calibration artifact: {error}') from error
