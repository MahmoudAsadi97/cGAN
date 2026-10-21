"""Calendar-balanced meter summaries and paired stratified studentization.

This numerical core does not validate independent evaluation meters. On reused
validation data its intervals are postselection descriptive intervals ignoring
selection, conditional on fixed fitted models and generated score functions.
They do not include generation uncertainty or establish new-meter coverage.
"""
from __future__ import annotations

import hashlib

import numpy as np


SCHEMA = 1
SCOPE = ('Conditional fixed score functions; reused-validation postselection descriptive '
         'intervals ignoring selection. No nominal new-meter coverage or generation-uncertainty claim.')


def _matrix(values):
    values = np.asarray(values)
    vector = values.ndim == 1
    if vector:
        values = values[:, None]
    if (values.ndim != 2 or not all(values.shape) or values.dtype.kind not in 'fiu'
            or not np.isfinite(values).all()):
        raise ValueError('Contributions must be a nonempty finite real matrix or vector')
    with np.errstate(over='raise', invalid='raise'):
        result = np.asarray(values, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError('Contributions exceed finite float64 precision')
    return result, vector


def _labels(values, length, limit):
    values = np.asarray(values)
    if (values.ndim != 1 or len(values) != length or values.dtype.kind not in 'fiu'
            or not np.isfinite(values).all() or np.any(values != np.floor(values))
            or np.any((values < 0) | (values >= limit))):
        raise ValueError(f'Labels must be integer values in [0,{limit}) matching contributions')
    return values.astype(np.int64)


def _finite(values, calculation):
    if not np.isfinite(values).all():
        raise FloatingPointError(f'Nonfinite {calculation}; calculation exceeded float64 precision')
    return values


def _readonly(values):
    result = np.array(values, copy=True)
    result.flags.writeable = False
    return result


def _meter_key(value):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError('Meter IDs cannot be boolean or missing')
    if isinstance(value, str) and value:
        return ('string', str(value)), str(value)
    if isinstance(value, (int, np.integer)):
        return ('numeric', int(value)), int(value)
    if isinstance(value, (float, np.floating)) and np.isfinite(value):
        converted = float(value)
        if np.isfinite(converted) and converted == value:
            return ('numeric', converted), converted
    raise ValueError('Meter IDs must be nonempty strings or finite real numbers')


def meter_contributions(profile_differences, meter_ids, conditions):
    """Aggregate paired profile columns with equal weight for each calendar bin.

    A one-dimensional input is one comparison column; ``values`` always has
    shape [meters, columns]. Meter IDs retain first-appearance order. Numeric
    1 and 1.0 identify the same meter; the string "1" is distinct. Each meter
    must have one equipment stratum condition//8 and all eight condition%8
    bins. No missing-bin renormalization or pooled-day weighting is allowed.

    Returned arrays are owned read-only copies. Bin sums/counts retain the
    aggregation audit trail; means use centered arithmetic to reduce loss at
    large shared offsets. Floating-point reconstruction may differ by rounding.
    """
    differences, vector = _matrix(profile_differences)
    conditions = _labels(conditions, len(differences), 64)
    identifiers = np.asarray(meter_ids, dtype=object)
    if identifiers.ndim != 1 or len(identifiers) != len(differences):
        raise ValueError('Meter IDs must be a vector matching contributions')
    lookup, stable_ids, strata = {}, [], []
    meter_index = np.empty(len(differences), dtype=np.int64)
    for row, (identifier, condition) in enumerate(zip(identifiers, conditions)):
        key, display = _meter_key(identifier)
        group = int(condition // 8)
        if key not in lookup:
            lookup[key] = len(stable_ids)
            stable_ids.append(display)
            strata.append(group)
        index = lookup[key]
        if strata[index] != group:
            raise ValueError(f'Inconsistent equipment stratum for meter {display!r}')
        meter_index[row] = index
    bin_index = meter_index * 8 + conditions % 8
    counts = np.bincount(bin_index, minlength=len(stable_ids) * 8).reshape(-1, 8)
    if np.any(counts == 0):
        raise ValueError('Every meter must contain all eight calendar bins')
    sums = np.zeros((len(stable_ids) * 8, differences.shape[1]), dtype=np.float64)
    means = np.empty_like(sums)
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        np.add.at(sums, bin_index, differences)
        ordered = np.argsort(bin_index, kind='stable')
        cursor = 0
        for index, count in enumerate(counts.ravel()):
            rows = differences[ordered[cursor:cursor + count]]
            reference = rows[0]
            means[index] = reference + (rows - reference).mean(axis=0)
            cursor += count
        means = means.reshape(len(stable_ids), 8, differences.shape[1])
        reference = means[:, :1]
        values = reference[:, 0] + (means - reference).mean(axis=1)
    _finite(sums, 'calendar-bin sums')
    _finite(values, 'meter contributions')
    return {'meter_ids': tuple(stable_ids), 'strata': _readonly(np.asarray(strata, dtype=np.int64)),
            'values': _readonly(values), 'bin_counts': _readonly(counts),
            'bin_sums': _readonly(sums.reshape(len(stable_ids), 8, differences.shape[1])),
            'input_was_vector': vector}


def _variance_guard(variance, scale):
    """Only roundoff-size negative variances may be set to zero."""
    _finite(variance, 'variance estimates')
    tolerance = 64 * np.finfo(np.float64).eps * scale
    if np.any(variance < -tolerance):
        raise FloatingPointError('A negative variance exceeds the declared roundoff tolerance')
    negative = variance < 0
    return np.where(negative, 0., variance), negative


def _weighted_moments(centered, multiplicities):
    """Centered weighted-mean/M2 merges without expanded meter or daily samples.

    Both the replicate and comparison-column axes are bounded by the caller.
    Exact constant resamples retain zero M2, including duplicate meter values.
    """
    rows, columns = len(multiplicities), centered.shape[1]
    mean, m2 = np.zeros((rows, columns)), np.zeros((rows, columns))
    counts = np.zeros(rows, dtype=np.int64)
    for value, weight in zip(centered, multiplicities.T):
        active = weight > 0
        if not np.any(active):
            continue
        previous = counts[active]
        total = previous + weight[active]
        difference = value - mean[active]
        coefficient = previous * (weight[active] / total)
        m2[active] += difference ** 2 * coefficient[:, None]
        mean[active] += difference * (weight[active] / total)[:, None]
        counts[active] = total
    return _finite(mean, 'bootstrap centered means'), _finite(m2, 'bootstrap centered second moments')


def stratified_studentized_bootstrap(values, strata, seed, replicates=1999, level=.95, block_size=128):
    """Paired meter bootstrap with equal weights across all eight strata.

    Input rows are meters, columns are fixed paired comparison/score functions;
    a vector is treated as one column. Every stratum 0..7 requires >=2 rows.
    One SeedSequence child stream per stratum produces IID multinomial meter
    multiplicities, shared across all columns. Draws and their hash do not
    depend on block size. Working multiplicities have at most block_size rows;
    weighted-moment arrays also tile columns. Studentized statistics are stored
    as [replicates, columns] to compute the fixed NumPy linear quantiles.

    Observed zero SE has no interval. If any bootstrap replicate has zero SE
    for a column, that column has no interval either; resamples are never
    discarded and other columns continue. All returned metadata are JSON-safe.
    """
    values, vector = _matrix(values)
    strata = _labels(strata, len(values), 8)
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('seed must be an unsigned 32-bit integer')
    if type(replicates) is not int or replicates < 2:
        raise ValueError('replicates must be an integer of at least two')
    if (isinstance(level, (bool, np.bool_)) or not isinstance(level, (int, float, np.integer, np.floating))
            or not np.isfinite(level) or not 0 < level < 1):
        raise ValueError('level must be a finite probability strictly between zero and one')
    if type(block_size) is not int or block_size < 1:
        raise ValueError('block_size must be a positive integer')
    positions = [np.flatnonzero(strata == group) for group in range(8)]
    sizes = np.asarray([len(index) for index in positions])
    if np.any(sizes < 2):
        raise ValueError('All eight equipment strata require at least two meters')
    columns = values.shape[1]
    centers, centered, shifts, variances, scales = [], [], [], [], []
    roundoff_clips = np.zeros(columns, dtype=np.int64)
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        for index in positions:
            group = values[index]
            center = group[0]
            relative = group - center
            shift = relative.mean(axis=0)
            variance = ((relative - shift) ** 2).sum(axis=0) / (len(group) - 1)
            scale = np.max(np.abs(relative), axis=0) ** 2
            variance, clipped = _variance_guard(variance, scale)
            roundoff_clips += clipped.astype(np.int64)
            centers.append(center)
            centered.append(relative)
            shifts.append(shift)
            variances.append(variance)
            scales.append(scale)
        centers, shifts = np.asarray(centers), np.asarray(shifts)
        estimate = centers[0] + ((centers - centers[0]) + shifts).mean(axis=0)
        standard_error = np.sqrt((np.asarray(variances) / sizes[:, None]).sum(axis=0) / 64.)
        _finite(estimate, 'observed estimates')
        _finite(standard_error, 'observed standard errors')
        streams = [np.random.default_rng(child) for child in np.random.SeedSequence(seed).spawn(8)]
        statistics = np.zeros((replicates, columns), dtype=np.float64)
        zero_counts = np.zeros(columns, dtype=np.int64)
        digest = hashlib.sha256()
        for start in range(0, replicates, block_size):
            stop = min(start + block_size, replicates)
            weights = np.zeros((stop - start, len(values)), dtype=np.int64)
            mean_difference, squared_se = np.zeros((stop - start, columns)), np.zeros((stop - start, columns))
            for group, index in enumerate(positions):
                count = len(index)
                weights[:, index] = streams[group].multinomial(count, np.full(count, 1. / count), size=stop - start)
                for column in range(0, columns, block_size):
                    end = min(column + block_size, columns)
                    mean, m2 = _weighted_moments(centered[group][:, column:end], weights[:, index])
                    variance, clipped = _variance_guard(m2 / (count - 1), scales[group][column:end])
                    roundoff_clips[column:end] += clipped.sum(axis=0)
                    # Differences of centered means avoid subtracting two huge
                    # absolute bootstrap/observed estimates during studentization.
                    mean_difference[:, column:end] += (mean - shifts[group, column:end]) / 8.
                    squared_se[:, column:end] += variance / (64. * count)
            digest.update(weights.astype('<i8', copy=False).tobytes(order='C'))
            bootstrap_se = np.sqrt(squared_se)
            _finite(bootstrap_se, 'bootstrap standard errors')
            zero_counts += (bootstrap_se == 0).sum(axis=0)
            valid = (bootstrap_se > 0) & (standard_error[None] > 0)
            current = statistics[start:stop]
            current[valid] = mean_difference[valid] / bootstrap_se[valid]
            _finite(current, 'studentized bootstrap statistics')
        status, intervals, quantiles = [], [], []
        alpha = (1. - float(level)) / 2.
        for column in range(columns):
            if standard_error[column] == 0:
                status.append('degenerate_observed_variance')
                intervals.append(None)
                quantiles.append(None)
            elif zero_counts[column] > 0:
                status.append('failed_studentization')
                intervals.append(None)
                quantiles.append(None)
            else:
                lower, upper = np.quantile(statistics[:, column], [alpha, 1. - alpha], method='linear')
                interval = np.array([estimate[column] - upper * standard_error[column],
                                     estimate[column] - lower * standard_error[column]])
                _finite(interval, 'studentized interval endpoints')
                status.append('ok')
                intervals.append(interval.tolist())
                quantiles.append([float(lower), float(upper)])
    return {'schema_version': SCHEMA, 'scope': SCOPE, 'columns': columns, 'input_was_vector': vector,
            'estimate': estimate.tolist(), 'standard_error': standard_error.tolist(),
            'status': status, 'interval': intervals, 'studentized_quantiles': quantiles,
            'bootstrap_zero_standard_error_counts': zero_counts.tolist(),
            'variance_roundoff_clip_counts': roundoff_clips.tolist(),
            'stratum_sizes': sizes.tolist(), 'seed': seed, 'replicates': replicates,
            'level': float(level), 'quantile_method': 'linear', 'block_size': block_size,
            'multiplicity_sha256': digest.hexdigest(),
            'multiplicity_hash_encoding': 'replicate-major little-endian int64 counts, meters in original input row order',
            'random_streams': 'numpy SeedSequence(seed).spawn(8), default_rng per stratum in order 0..7',
            'estimate_rule': 'equal mean of eight stratum means',
            'standard_error_rule': 'sqrt(sum_g(sample_variance_g / meter_count_g) / 64)',
            'studentization_numerator': 'sum_g(bootstrap centered mean_g - observed centered mean_g) / 8',
            'zero_standard_error_rule': 'Observed zero SE: no interval. Any bootstrap zero SE: no interval for that column; no resamples dropped.',
            'variance_rule': 'centered weighted mean/M2 merges; negative variance clipped only within 64*eps*max(abs(centered_group))^2, otherwise failure',
            'interval_scope': 'marginal per column; no simultaneous or familywise coverage claim'}
