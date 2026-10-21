"""Additional conditional copulas fitted exclusively on caller-supplied TRAIN.

The empirical beta copula samples a common rank row, then independent
Beta(rank, n + 1 - rank) coordinates. See Segers, Sibuya and Tsukahara (2017),
The Empirical Beta Copula, Journal of Multivariate Analysis 155, 35--51,
https://arxiv.org/abs/1607.04430; the reference package documentation is
https://raw.githubusercontent.com/cran/copula/master/man/empCopula.Rd.
The continuous-data construction assumes no ties. Here independently seeded
random tie breaking gives full column rank permutations, retaining exactly
uniform univariate beta-mixture marginals even when the observed data have atoms.

Student-t controls use fixed degrees of freedom and average-rank t-score
Pearson correlations, a moment estimator rather than maximum likelihood.
Sampling follows the shared chi-square scale construction documented in SciPy's
multivariate_t and the multivariate-t/coordinatewise-t-CDF construction in
https://raw.githubusercontent.com/cran/copula/master/R/tCopula.R.

Both controls decode through fixed conditional empirical TRAIN quantiles.
Hourly marginal support and frequencies hold in law, not exactly per draw.
They do not establish joint-profile support, novelty, privacy or unseen tails.
The beta control can remain close to TRAIN profiles; nearest-neighbor checks
are therefore needed in addition to exact-match diagnostics.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from numbers import Integral, Real
from types import MappingProxyType

import numpy as np
from scipy.special import stdtr, stdtrit
from scipy.stats import rankdata

from .marginals import EmpiricalMarginals, fit_marginals


IDENTITY_SHRINKAGE = .01
EIGENVALUE_FLOOR = 1e-8


def _profiles(values):
    values = np.asarray(values)
    if (values.ndim != 2 or not all(values.shape) or values.dtype.kind not in 'fiu'
            or not np.isfinite(values).all()):
        raise ValueError('Profiles must be a nonempty finite real matrix')
    with np.errstate(over='ignore', invalid='ignore'):
        values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError('Profiles exceed finite float64 precision')
    return values


def _labels(values, length=None):
    values = np.asarray(values)
    if (values.ndim != 1 or (length is not None and len(values) != length)
            or values.dtype.kind not in 'fiu' or not np.isfinite(values).all()
            or np.any(values != np.floor(values)) or np.any((values < 0) | (values >= 64))):
        raise ValueError('condition labels must be integer values in [0,64) matching profiles')
    return values.astype(np.int64)


def _seed(value):
    if isinstance(value, bool) or not isinstance(value, Integral) or not 0 <= value < 2**32:
        raise ValueError('Seeds must be unsigned 32-bit integers')
    return int(value)


def _digest(values, dtype):
    return hashlib.sha256(np.asarray(values, dtype=dtype).tobytes(order='C')).hexdigest()


def _readonly(values, dtype):
    """Own an immutable bytes-backed array; writeability cannot be re-enabled."""
    values = np.asarray(values, dtype=dtype)
    return np.frombuffer(values.tobytes(order='C'), dtype=dtype).reshape(values.shape)


@dataclass(frozen=True)
class _Group:
    rows: int
    active: tuple[int, ...]
    ranks: np.ndarray | None = None
    factor: np.ndarray | None = None


@dataclass(frozen=True)
class StrongConditionalBaseline:
    """A fitted conditional sampler with immutable copula arrays and metadata.

    Construct with :func:`fit_strong_baseline`. Sampling uses a local RNG and
    visits conditions in sorted order; fitting never inspects evaluation rows.
    """

    kind: str
    profile_dim: int
    df: float | None
    _marginals: EmpiricalMarginals
    _groups: dict
    _metadata_json: str

    @property
    def conditions(self):
        return tuple(self._groups)

    def metadata(self):
        """Return independent JSON-safe rules, source hashes and fit diagnostics."""
        return json.loads(self._metadata_json)

    def sample(self, labels, seed):
        """Return float64 profiles in requested label order from a local RNG."""
        labels, seed = _labels(labels), _seed(seed)
        missing = set(labels.tolist()) - self._groups.keys()
        if missing:
            raise ValueError(f'Unsupported condition labels: {sorted(missing)}')
        if not len(labels):
            return np.empty((0, self.profile_dim), dtype=np.float64)
        rng = np.random.default_rng(seed)
        probability = np.full((len(labels), self.profile_dim), .5, dtype=np.float64)
        for label in np.unique(labels):
            positions = np.flatnonzero(labels == label)
            group = self._groups[int(label)]
            active = group.active
            if not active:
                continue
            if self.kind == 'beta':
                row = rng.integers(group.rows, size=len(positions))
                ranks = group.ranks[np.ix_(row, active)].astype(np.float64)
                draws = rng.beta(ranks, group.rows + 1. - ranks)
            else:
                normal = rng.standard_normal((len(positions), len(active))) @ group.factor.T
                # One radial scale per day induces t tail dependence even at R=I.
                scale = np.sqrt(rng.chisquare(self.df, size=len(positions)) / self.df)
                draws = stdtr(self.df, normal / scale[:, None])
            probability[np.ix_(positions, active)] = draws
        return self._marginals.inverse(probability, labels)


def fit_strong_baseline(kind, x, labels, fit_seed=20261026, df=None):
    """Fit an empirical beta or fixed-df Student-t conditional copula.

    ``beta`` requires ``df=None``. Its fit seed randomizes ties independently
    across columns using a permutation followed by a stable value sort.
    ``student_t`` requires finite numeric ``df>2``; its fit is deterministic
    and does not use the fit seed. Average ranks produce finite probabilities
    ``(rank-.5)/n``. The fitted correlation has 0.01 identity shrinkage, a 1e-8
    eigenvalue floor, and unit-diagonal renormalization. With tied observations
    these latent correlations are approximations, not identified latent truth.

    This numerical API cannot establish input provenance. Runners must bind
    their TRAIN partition and evaluation selection to preserved source hashes.
    """
    if not isinstance(kind, str) or kind not in {'beta', 'student_t'}:
        raise ValueError('kind must be beta or student_t')
    fit_seed = _seed(fit_seed)
    if kind == 'beta':
        if df is not None:
            raise ValueError('The beta copula requires df=None')
    else:
        if isinstance(df, bool) or not isinstance(df, Real):
            raise ValueError('Student-t df must be a finite numeric value greater than 2')
        try:
            df = float(df)
        except (OverflowError, ValueError) as error:
            raise ValueError('Student-t df must fit finite float64 precision') from error
        if not np.isfinite(df) or df <= 2:
            raise ValueError('Student-t df must be a finite numeric value greater than 2')
    x = _profiles(x)
    labels = _labels(labels, len(x))
    marginals = fit_marginals(x, labels)
    rng = np.random.default_rng(fit_seed)
    groups, records = {}, {}
    for label in np.unique(labels):
        values = x[labels == label]
        n, dimension = values.shape
        active = tuple(np.flatnonzero(np.any(values != values[0], axis=0)).tolist())
        record = {'training_profiles': n, 'active_hours': list(active),
                  'constant_hours': sorted(set(range(dimension)) - set(active)),
                  'source_profile_bytes_sha256': _digest(values, '<f8')}
        if kind == 'beta':
            rank_dtype = '<u4' if n <= np.iinfo(np.uint32).max else '<u8'
            ranks = np.empty(values.shape, dtype=rank_dtype)
            for hour in range(dimension):
                permutation = rng.permutation(n)
                order = permutation[np.argsort(values[permutation, hour], kind='stable')]
                ranks[order, hour] = np.arange(1, n + 1, dtype=rank_dtype)
            groups[int(label)] = _Group(n, active, ranks=_readonly(ranks, rank_dtype))
            record.update(ranks_sha256=_digest(ranks, rank_dtype),
                          ranks_dtype=rank_dtype, ranks_shape=list(ranks.shape))
        else:
            if active:
                ranks = rankdata(values[:, active], axis=0, method='average')
                scores = stdtrit(df, (ranks - .5) / n)
                if not np.isfinite(scores).all():
                    raise ValueError('Student-t rank scores are nonfinite at the requested df')
                correlation = np.atleast_2d(np.corrcoef(scores, rowvar=False))
                correlation = ((1. - IDENTITY_SHRINKAGE) * correlation
                               + IDENTITY_SHRINKAGE * np.eye(len(active)))
                eigenvalues, eigenvectors = np.linalg.eigh(correlation)
                correlation = ((eigenvectors * np.maximum(eigenvalues, EIGENVALUE_FLOOR))
                               @ eigenvectors.T)
                diagonal = np.sqrt(np.diag(correlation))
                correlation /= diagonal[:, None] * diagonal[None, :]
                factor = np.linalg.cholesky(correlation)
            else:
                factor = np.empty((0, 0), dtype=np.float64)
            groups[int(label)] = _Group(n, active, factor=_readonly(factor, '<f8'))
            record.update(correlation_factor_sha256=_digest(factor, '<f8'),
                          correlation_factor_dtype='<f8', correlation_factor_shape=list(factor.shape))
        records[str(int(label))] = record
    metadata = {'schema_version': 1, 'kind': kind,
        'method': ('conditional_empirical_beta_copula' if kind == 'beta'
                   else 'conditional_student_t_copula'),
        'profile_dim': x.shape[1], 'training_profiles': len(x),
        'conditions': list(marginals.conditions), 'df': df,
        'fit_seed': fit_seed if kind == 'beta' else None,
        'fit_seed_role': 'independent column tie breaking' if kind == 'beta' else 'unused; deterministic fit',
        'source_profile_bytes_sha256': _digest(x, '<f8'),
        'source_label_bytes_sha256': _digest(labels, '<i8'),
        'source_array_encoding': 'C order; profiles little-endian float64; labels little-endian int64',
        'target_inverse': 'empirical_step',
        'target_inverse_rule': 'sorted TRAIN[clip(ceil(n*u)-1,0,n-1)]',
        'constant_hours': 'fixed at observed TRAIN value',
        'sampling_rng': 'numpy.default_rng(seed); sorted conditions; local state',
        'marginals': marginals.summary(), 'per_condition': records,
        'interpretation': 'Exact empirical hourly marginals in law, with finite-draw sampling noise. '
            'Empirical support truncates unseen tails and does not establish joint-profile support, '
            'novelty or privacy. Beta smoothing can stay close to resampled TRAIN rows.'}
    if kind == 'beta':
        metadata.update(estimator='empirical_beta_full_rank_permutations',
            ties='independent column permutations followed by stable value sort',
            sampling='one uniform TRAIN row per day; independent Beta(r,n+1-r) active coordinates',
            rank_columns='each is a permutation of integers 1 through conditional TRAIN count',
            maximum_likelihood_fit=False)
    else:
        metadata.update(estimator='average_rank_t_score_pearson_moment',
            ties='average ranks', rank_probability='(average_rank-0.5)/conditional_TRAIN_count',
            sampling='normal @ Cholesky(R).T; shared chi-square(df)/df per day; coordinatewise t CDF',
            identity_shrinkage=IDENTITY_SHRINKAGE, eigenvalue_floor=EIGENVALUE_FLOOR,
            correlation_renormalization='unit diagonal after eigenvalue floor',
            maximum_likelihood_fit=False)
    return StrongConditionalBaseline(kind, x.shape[1], df, marginals,
        MappingProxyType(groups), json.dumps(metadata, sort_keys=True, allow_nan=False))
