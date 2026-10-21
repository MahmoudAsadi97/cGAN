"""Conditional Gaussian mixtures with analytic marginal PIT and TRAIN quantiles.

Each condition is fitted in normal scores from independently randomized full
column ranks. A Gaussian mixture models these scores; generated scores pass
through the *whole mixture's* univariate CDFs before empirical TRAIN decoding.
This is a normal-score mixture estimator, not copula maximum likelihood.

The mixture-to-copula simulation construction is documented by Bilgrau et al.,
GMCM, Journal of Statistical Software 70(2), DOI 10.18637/jss.v070.i02, and
https://raw.githubusercontent.com/AEBilgrau/GMCM/master/R/SimulateGMCMData.R
(reviewed source SHA-256
25b0e3c60468f37ecdde125795dd6664bb474f235c3a90ac0078ea0dc5352704).
The estimator here is independent and deliberately does not claim their
likelihood optimization or exact recovery of an unidentified latent copula.

Empirical hourly support and frequencies hold in law. This does not establish
joint support, privacy, novelty, causal appliance labels or unseen tails.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
from numbers import Integral, Real
from pathlib import Path
import platform
import re
from types import MappingProxyType
import warnings
import zipfile

import numpy as np
from scipy.special import ndtr, ndtri
from sklearn.mixture import GaussianMixture
from threadpoolctl import threadpool_limits

from .marginals import EmpiricalMarginals
from .strong_baselines import _digest, _labels, _profiles, _readonly, _seed


SCHEMA = 1
METHOD = 'conditional_normal_score_gaussian_mixture_pit'
_RULES = MappingProxyType({
    'source_array_encoding': 'C-order little-endian float64 profiles and int64 labels',
    'copula_maximum_likelihood_fit': False,
    'estimator': 'randomized_full_rank_normal_scores_gaussian_mixture',
    'rank_rule': 'independent permutation then stable value sort per condition and column; (rank-.5)/n',
    'rank_rng': 'numpy default_rng(SeedSequence([fit_seed, 0, condition]))',
    'initialization_rng': 'numpy default_rng(SeedSequence([fit_seed, 1, condition])); unique uint32 seeds',
    'selection_rule': 'maximum final TRAIN average log likelihood; earliest exact tie',
    'covariance_type': 'full', 'covariance_regularization': 'reg_covar added to diagonal',
    'init_params': 'k-means++', 'sklearn_n_init_per_attempt': 1, 'fit_threads': 1,
    'target_inverse': 'conditional empirical step ceil(n*u)-1, bounded to [0,n-1]',
    'probability_transform': 'weighted CDF sum over all components using sqrt(covariance diagonal)',
    'probability_roundoff': 'CDF results clipped to [0,1]; no physical clipping',
    'sampling_rng': 'local numpy default_rng(seed); sorted conditions; preserve requested row order',
    'constant_rule': 'omit constant physical coordinates from EM and fix their empirical value',
})


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return int(value)


def _positive_real(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f'{name} must be a positive finite number')
    try:
        value = float(value)
    except (ValueError, OverflowError) as error:
        raise ValueError(f'{name} must be a positive finite number') from error
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be a positive finite number')
    return value


def _configuration(n_components, fit_seed, n_init, max_iter, tol, reg_covar):
    return {'n_components': _positive_integer(n_components, 'n_components'),
            'fit_seed': _seed(fit_seed), 'n_init': _positive_integer(n_init, 'n_init'),
            'max_iter': _positive_integer(max_iter, 'max_iter'),
            'tol': _positive_real(tol, 'tol'), 'reg_covar': _positive_real(reg_covar, 'reg_covar')}


def _mixture_marginal_cdf(values, weights, means, covariances):
    """Apply each marginal CDF using all components and covariance diagonals."""
    probability = np.zeros_like(values, dtype=np.float64)
    for weight, mean, covariance in zip(weights, means, covariances, strict=True):
        probability += weight * ndtr((values - mean) / np.sqrt(np.diag(covariance)))
    # Only numerical CDF roundoff is clipped; physical outputs are never clipped.
    return np.clip(probability, 0., 1.)


@dataclass(frozen=True)
class _Group:
    target: np.ndarray
    active: tuple[int, ...]
    weights: np.ndarray
    means: np.ndarray
    covariances: np.ndarray
    factors: np.ndarray


def _group(target, active, weights, means, covariances):
    factors = (np.linalg.cholesky(covariances) if len(active)
               else np.empty((0, 0, 0), dtype=np.float64))
    return _Group(_readonly(target, np.float64), tuple(active), _readonly(weights, np.float64),
                  _readonly(means, np.float64), _readonly(covariances, np.float64),
                  _readonly(factors, np.float64))


def _array_hashes(group):
    return {name + '_sha256': _digest(getattr(group, name), '<f8')
            for name in ('target', 'weights', 'means', 'covariances')}


@dataclass(frozen=True)
class MixtureConditionalBaseline:
    """Immutable fitted numeric sampler; construct through the fitting function."""

    profile_dim: int
    _groups: dict
    _metadata_json: str

    @property
    def conditions(self):
        return tuple(self._groups)

    def metadata(self):
        """Return independent JSON-safe provenance, rules and every EM attempt."""
        return json.loads(self._metadata_json)

    def sample(self, labels, seed):
        """Draw physical float64 profiles with a local RNG in requested order."""
        labels, seed = _labels(labels), _seed(seed)
        missing = set(labels.tolist()) - self._groups.keys()
        if missing:
            raise ValueError(f'Unsupported condition labels: {sorted(missing)}')
        output = np.empty((len(labels), self.profile_dim), dtype=np.float64)
        rng = np.random.default_rng(seed)
        for label in np.unique(labels):
            positions = np.flatnonzero(labels == label)
            group = self._groups[int(label)]
            output[positions] = group.target[0]
            if not group.active:
                continue
            selected = rng.choice(len(group.weights), size=len(positions), p=group.weights)
            normal = rng.standard_normal((len(positions), len(group.active)))
            latent = np.empty_like(normal)
            for component in range(len(group.weights)):
                rows = selected == component
                latent[rows] = normal[rows] @ group.factors[component].T + group.means[component]
            probability = _mixture_marginal_cdf(latent, group.weights, group.means, group.covariances)
            indices = np.clip(np.ceil(len(group.target) * probability).astype(np.int64) - 1,
                              0, len(group.target) - 1)
            output[np.ix_(positions, group.active)] = group.target[indices, np.asarray(group.active)]
        return output

    def save(self, path):
        """Exclusively write a numeric NPZ; callers preserve its SHA-256 receipt."""
        arrays = {'metadata': np.asarray(self._metadata_json)}
        for label, group in self._groups.items():
            for name in ('target', 'weights', 'means', 'covariances'):
                arrays[f'{name}_{label}'] = getattr(group, name)
        with Path(path).open('xb') as handle:
            np.savez_compressed(handle, **arrays)


def _ranks(values, fit_seed, label):
    rng = np.random.default_rng(np.random.SeedSequence([fit_seed, 0, label]))
    ranks = np.empty(values.shape, dtype=np.int64)
    for hour in range(values.shape[1]):
        shuffled = rng.permutation(len(values))
        order = shuffled[np.argsort(values[shuffled, hour], kind='stable')]
        ranks[order, hour] = np.arange(1, len(values) + 1)
    return ranks


def _initialization_seeds(fit_seed, label, count):
    rng = np.random.default_rng(np.random.SeedSequence([fit_seed, 1, label]))
    seeds = []
    while len(seeds) < count:
        value = int(rng.integers(0, 2**32, dtype=np.uint64))
        if value not in seeds:
            seeds.append(value)
    return seeds


def fit_mixture_baseline(x, labels, n_components, fit_seed=20261027, n_init=2,
                         max_iter=100, tol=1e-3, reg_covar=1e-3):
    """Fit all supplied TRAIN rows, preserving every attempted initialization.

    Independent full rank permutations break ties before ``ndtri((rank-.5)/n)``.
    Constant physical coordinates are removed from EM and fixed at decoding.
    Each initialization is a separate full-covariance scikit-learn fit, with
    diagonal covariance regularization and k-means++ initialization. The largest
    final TRAIN average log likelihood wins; exact ties keep the earliest fit.
    Nonconvergence is recorded and retained without retries or validation use.

    Rank and initialization streams are condition-specific and independent of K.
    Fitting temporarily limits BLAS threads to one; local RNGs preserve global
    NumPy state. This numerical API cannot establish the caller's TRAIN provenance.
    """
    config = _configuration(n_components, fit_seed, n_init, max_iter, tol, reg_covar)
    x = _profiles(x)
    labels = _labels(labels, len(x))
    groups, records = {}, {}
    for label in np.unique(labels):
        label = int(label)
        values = x[labels == label]
        target = np.sort(values, axis=0)
        active = tuple(np.flatnonzero(target[0] != target[-1]).tolist())
        ranks = _ranks(values, config['fit_seed'], label)
        scores = ndtri((ranks[:, active].astype(np.float64) - .5) / len(values))
        record = {'training_profiles': len(values), 'active_hours': list(active),
                  'constant_hours': [hour for hour in range(x.shape[1]) if hour not in active],
                  'ranks_sha256': _digest(ranks, '<i8'),
                  'normal_scores_sha256': _digest(scores, '<f8'),
                  'initializations': [], 'selected_init': None}
        if not active:
            group = _group(target, active, np.empty(0), np.empty((0, 0)), np.empty((0, 0, 0)))
            record.update(status='constant', weights=[], effective_occupancies=[],
                          hard_occupancies=[], covariance_min_eigenvalues=[],
                          covariance_max_eigenvalues=[], converged=None)
        else:
            if config['n_components'] > len(values):
                raise ValueError(f'n_components exceeds nonconstant condition {label} training rows')
            selected_model, selected_score, selected_index = None, -np.inf, None
            seeds = _initialization_seeds(config['fit_seed'], label, config['n_init'])
            with threadpool_limits(limits=1):
                for index, seed in enumerate(seeds):
                    model = GaussianMixture(n_components=config['n_components'], covariance_type='full',
                                            tol=config['tol'], reg_covar=config['reg_covar'],
                                            max_iter=config['max_iter'], n_init=1,
                                            init_params='k-means++', random_state=seed)
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter('always')
                        model.fit(scores)
                    score = float(model.score(scores))
                    if not np.isfinite(score) or not np.isfinite(model.lower_bound_):
                        raise ValueError(f'Nonfinite mixture fit for condition {label}, initialization {index}')
                    record['initializations'].append({
                        'index': index, 'seed': seed, 'final_score': score,
                        'lower_bound': float(model.lower_bound_), 'converged': bool(model.converged_),
                        'n_iter': int(model.n_iter_),
                        'warnings': [{'category': warning.category.__name__, 'message': str(warning.message)}
                                     for warning in caught]})
                    if score > selected_score:
                        selected_model, selected_score, selected_index = model, score, index
                responsibility = selected_model.predict_proba(scores)
                eigenvalues = np.linalg.eigvalsh(selected_model.covariances_)
                group = _group(target, active, selected_model.weights_, selected_model.means_,
                               selected_model.covariances_)
            record.update(status='fitted', selected_init=selected_index,
                          weights=group.weights.tolist(),
                          effective_occupancies=responsibility.sum(axis=0).tolist(),
                          hard_occupancies=np.bincount(responsibility.argmax(axis=1),
                                                      minlength=config['n_components']).tolist(),
                          covariance_min_eigenvalues=eigenvalues[:, 0].tolist(),
                          covariance_max_eigenvalues=eigenvalues[:, -1].tolist(),
                          converged=bool(selected_model.converged_))
        record.update(_array_hashes(group))
        groups[label], records[str(label)] = group, record
    metadata = {
        'schema_version': SCHEMA, 'method': METHOD, 'config': config,
        'n_components': config['n_components'], 'profile_dim': x.shape[1],
        'training_profiles': len(x), 'conditions': list(groups),
        'source_profile_bytes_sha256': _digest(x, '<f8'),
        'source_label_bytes_sha256': _digest(labels, '<i8'),
        **_RULES,
        'marginals': EmpiricalMarginals(x.shape[1], {label: group.target for label, group in groups.items()}).summary(),
        'per_condition': records,
        'library_versions': {'python': platform.python_version(), **{
            name: importlib.metadata.version(name)
            for name in ('numpy', 'scipy', 'scikit-learn', 'joblib', 'threadpoolctl')}},
    }
    return MixtureConditionalBaseline(x.shape[1], MappingProxyType(groups),
                                      json.dumps(metadata, sort_keys=True, allow_nan=False))


def _validate_metadata(metadata):
    if (not isinstance(metadata, dict) or type(metadata.get('schema_version')) is not int
            or metadata['schema_version'] != SCHEMA or metadata.get('method') != METHOD
            or type(metadata.get('profile_dim')) is not int or metadata['profile_dim'] < 1
            or type(metadata.get('training_profiles')) is not int or metadata['training_profiles'] < 1
            or not isinstance(metadata.get('config'), dict)):
        raise ValueError('Unsupported mixture metadata schema')
    config = metadata['config']
    if set(config) != {'n_components', 'fit_seed', 'n_init', 'max_iter', 'tol', 'reg_covar'}:
        raise ValueError('Invalid saved mixture configuration')
    checked = _configuration(**config)
    if checked != config or metadata.get('n_components') != config['n_components']:
        raise ValueError('Invalid saved mixture configuration')
    labels = metadata.get('conditions')
    if (not isinstance(labels, list) or not labels or any(type(label) is not int for label in labels)
            or labels != sorted(set(labels))):
        raise ValueError('Invalid saved condition labels')
    _labels(labels)
    if not isinstance(metadata.get('per_condition'), dict) or set(metadata['per_condition']) != set(map(str, labels)):
        raise ValueError('Invalid saved condition records')
    for key in ('source_profile_bytes_sha256', 'source_label_bytes_sha256'):
        if not isinstance(metadata.get(key), str) or not re.fullmatch('[0-9a-f]{64}', metadata[key]):
            raise ValueError('Invalid saved input digest')
    if any(type(metadata.get(key)) is not type(value) or metadata[key] != value
           for key, value in _RULES.items()):
        raise ValueError('Invalid mixture estimator declaration')
    return config, labels


def _validate_saved_group(label, arrays, metadata, config):
    record = metadata['per_condition'][str(label)]
    if not isinstance(record, dict):
        raise ValueError('Invalid saved condition record')
    target, weights, means, covariances = [arrays[f'{name}_{label}']
                                         for name in ('target', 'weights', 'means', 'covariances')]
    for array in (target, weights, means, covariances):
        if array.dtype != np.float64 or not np.isfinite(array).all():
            raise ValueError('Mixture arrays must be finite float64')
    if (target.ndim != 2 or not len(target) or target.shape[1] != metadata['profile_dim']
            or np.any(target[1:] < target[:-1]) or record.get('training_profiles') != len(target)):
        raise ValueError('Invalid saved empirical target')
    active = tuple(np.flatnonzero(target[0] != target[-1]).tolist())
    constants = [hour for hour in range(metadata['profile_dim']) if hour not in active]
    if record.get('active_hours') != list(active) or record.get('constant_hours') != constants:
        raise ValueError('Saved active coordinates differ from target')
    k, dimension = (config['n_components'] if active else 0), len(active)
    if weights.shape != (k,) or means.shape != (k, dimension) or covariances.shape != (k, dimension, dimension):
        raise ValueError('Saved mixture parameter shapes differ from configuration')
    if active:
        if (k > len(target) or np.any(weights <= 0) or not np.isclose(weights.sum(), 1., rtol=0, atol=1e-12)
                or not np.allclose(covariances, covariances.transpose(0, 2, 1), rtol=0, atol=1e-12)):
            raise ValueError('Invalid saved mixture probabilities or covariances')
        eigenvalues = np.linalg.eigvalsh(covariances)
        if np.any(eigenvalues <= 0):
            raise ValueError('Saved covariance is not positive definite')
        attempts = record.get('initializations')
        if not isinstance(attempts, list) or len(attempts) != config['n_init']:
            raise ValueError('Missing saved initialization records')
        expected_seeds = _initialization_seeds(config['fit_seed'], label, config['n_init'])
        scores = []
        for index, attempt in enumerate(attempts):
            if (not isinstance(attempt, dict) or attempt.get('index') != index
                    or attempt.get('seed') != expected_seeds[index]
                    or type(attempt.get('converged')) is not bool
                    or type(attempt.get('n_iter')) is not int or not 1 <= attempt['n_iter'] <= config['max_iter']
                    or not isinstance(attempt.get('warnings'), list)):
                raise ValueError('Invalid saved initialization record')
            for name in ('final_score', 'lower_bound'):
                if type(attempt.get(name)) not in (int, float) or not np.isfinite(attempt[name]):
                    raise ValueError('Invalid saved fit score')
            if any(not isinstance(warning, dict) or set(warning) != {'category', 'message'}
                   or not all(isinstance(value, str) for value in warning.values())
                   for warning in attempt['warnings']):
                raise ValueError('Invalid saved fit warnings')
            scores.append(attempt['final_score'])
        selected = int(np.argmax(scores))
        if (type(record.get('selected_init')) is not int or record['selected_init'] != selected
                or record.get('converged') is not attempts[selected]['converged']
                or record.get('status') != 'fitted'):
            raise ValueError('Saved selection differs from declared TRAIN rule')
        for name, expected in [('weights', weights), ('covariance_min_eigenvalues', eigenvalues[:, 0]),
                               ('covariance_max_eigenvalues', eigenvalues[:, -1])]:
            observed = np.asarray(record.get(name), dtype=np.float64)
            if observed.shape != expected.shape or not np.allclose(observed, expected, rtol=1e-12, atol=1e-12):
                raise ValueError('Saved mixture diagnostics differ from fitted arrays')
        for name in ('effective_occupancies', 'hard_occupancies'):
            counts = np.asarray(record.get(name), dtype=np.float64)
            if (counts.shape != (k,) or not np.isfinite(counts).all() or np.any(counts < 0)
                    or not np.isclose(counts.sum(), len(target), rtol=1e-10, atol=1e-8)
                    or (name == 'hard_occupancies' and np.any(counts != np.floor(counts)))):
                raise ValueError('Invalid saved component occupancies')
    elif (record.get('status') != 'constant' or record.get('initializations') != []
          or record.get('selected_init') is not None or record.get('converged') is not None):
        raise ValueError('Invalid constant condition record')
    group = _group(target, active, weights, means, covariances)
    if any(record.get(name) != digest for name, digest in _array_hashes(group).items()):
        raise ValueError('Saved parameter digest mismatch')
    for name in ('ranks_sha256', 'normal_scores_sha256'):
        if not isinstance(record.get(name), str) or not re.fullmatch('[0-9a-f]{64}', record[name]):
            raise ValueError('Invalid saved rank or score digest')
    return group


def load_mixture_baseline(path, expected_sha256):
    """Load only a hash-bound, schema-checked numeric artifact, never pickle."""
    if not isinstance(expected_sha256, str) or not re.fullmatch('[0-9a-f]{64}', expected_sha256):
        raise ValueError('A lowercase hexadecimal expected SHA-256 digest is required')
    with Path(path).open('rb') as handle:
        digest = hashlib.sha256()
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
        if digest.hexdigest() != expected_sha256:
            raise ValueError('Mixture artifact SHA-256 mismatch')
        handle.seek(0)
        try:
            with np.load(handle, allow_pickle=False) as arrays:
                raw = arrays['metadata']
                if raw.shape != () or raw.dtype.kind != 'U':
                    raise ValueError('Invalid mixture metadata encoding')
                metadata = json.loads(raw.item())
                config, labels = _validate_metadata(metadata)
                expected = {'metadata'} | {f'{name}_{label}' for label in labels
                                          for name in ('target', 'weights', 'means', 'covariances')}
                if set(arrays.files) != expected or len(arrays.files) != len(expected):
                    raise ValueError('Mixture array keys differ from metadata')
                groups = {label: _validate_saved_group(label, arrays, metadata, config) for label in labels}
                if sum(len(group.target) for group in groups.values()) != metadata['training_profiles']:
                    raise ValueError('Saved training counts differ from parameters')
                summary = EmpiricalMarginals(metadata['profile_dim'],
                                             {label: group.target for label, group in groups.items()}).summary()
                if metadata.get('marginals') != summary:
                    raise ValueError('Saved marginal summary differs from target arrays')
                return MixtureConditionalBaseline(metadata['profile_dim'], MappingProxyType(groups),
                                                  json.dumps(metadata, sort_keys=True, allow_nan=False))
        except (ValueError, TypeError, KeyError, OSError, zipfile.BadZipFile, np.linalg.LinAlgError) as error:
            raise ValueError(f'Invalid mixture artifact: {error}') from error
