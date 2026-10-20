"""Clustered uncertainty primitives for repeated daily meter observations."""

from __future__ import annotations

import numpy as np


def _metadata(values, name):
    array = np.asarray(values, dtype=object)
    if array.ndim != 1 or len(array) == 0:
        raise ValueError(f'{name} must be a nonempty one-dimensional array')
    for value in array:
        if not isinstance(value, (str, int, float, bool, np.integer, np.floating, np.bool_)):
            raise ValueError(f'{name} contains missing or unsupported metadata')
        if isinstance(value, str):
            if not value.strip():
                raise ValueError(f'{name} contains missing metadata')
        elif not np.isfinite(value):
            raise ValueError(f'{name} contains nonfinite metadata')
    return array


def cluster_bootstrap_indices(meter_ids, technology, rng):
    """Draw complete meter clusters with replacement within technology strata.

    Each stratum draws its original number of distinct meters. Every selected
    meter contributes all its row indices in original row order, including on
    repeated draws. The resulting row count may differ from the input when
    meters have different numbers of daily observations. Strata and meters
    follow first-appearance order, so draws are stable for the same metadata
    and supplied NumPy Generator state. A meter spanning technologies fails.
    """
    meter_ids = _metadata(meter_ids, 'meter_ids')
    technology = _metadata(technology, 'technology')
    if len(meter_ids) != len(technology):
        raise ValueError('meter_ids and technology must have matching lengths')
    if not isinstance(rng, np.random.Generator):
        raise ValueError('rng must be a numpy.random.Generator')
    meter_rows, meter_technology, strata = {}, {}, {}
    for index, (meter, stratum) in enumerate(zip(meter_ids, technology)):
        if meter not in meter_rows:
            meter_rows[meter] = []
            meter_technology[meter] = stratum
            strata.setdefault(stratum, []).append(meter)
        elif meter_technology[meter] != stratum:
            raise ValueError('each meter must have one stable technology')
        meter_rows[meter].append(index)
    result = []
    for meters in strata.values():
        for draw in rng.integers(0, len(meters), size=len(meters)):
            result.extend(meter_rows[meters[draw]])
    return np.asarray(result, dtype=np.int64)
