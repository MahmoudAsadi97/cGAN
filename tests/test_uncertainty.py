"""A meter's full rows must travel together through a bootstrap draw."""

import numpy as np
import pytest


def bootstrap(meter_ids, technology, rng):
    from fluvius_cgan.uncertainty import cluster_bootstrap_indices
    return cluster_bootstrap_indices(meter_ids, technology, rng)


def test_bootstrap_resamples_whole_clusters_within_technology():
    meter_ids = np.array(['a', 'b', 'a', 'c', 'b', 'd', 'd', 'e', 'e', 'e'])
    technology = np.array(['pv', 'pv', 'pv', 'pv', 'pv', 'ev', 'ev', 'ev', 'ev', 'ev'])
    indices = bootstrap(meter_ids, technology, np.random.default_rng(3))
    repeated = bootstrap(meter_ids, technology, np.random.default_rng(3))
    np.testing.assert_array_equal(indices, repeated)
    assert indices.dtype.kind in 'iu'
    assert np.all((0 <= indices) & (indices < len(meter_ids)))
    # Three PV meter draws and two EV meter draws; every selected meter carries
    # all rows, in original within-meter order, once per draw.
    selected = indices.tolist()
    draws = []
    while selected:
        meter = meter_ids[selected[0]]
        rows = np.flatnonzero(meter_ids == meter).tolist()
        assert selected[:len(rows)] == rows
        selected = selected[len(rows):]
        draws.append(meter)
    assert sum(meter in ['a', 'b', 'c'] for meter in draws) == 3
    assert sum(meter in ['d', 'e'] for meter in draws) == 2
    assert len(set(draws)) < len(draws)  # seed draws with replacement


def test_bootstrap_single_meter_keeps_every_row():
    np.testing.assert_array_equal(bootstrap(['m'] * 3, ['pv'] * 3, np.random.default_rng(1)), [0, 1, 2])


@pytest.mark.parametrize('meter_ids,technology', [
    ([], []), (['a', 'b'], ['pv']), (['a', 'a'], ['pv', 'ev']),
    (['a', None], ['pv', 'pv']), (['a', 'b'], ['pv', np.nan]),
    ([['a'], ['b']], ['pv', 'pv']), (['a', 'b'], ['', 'pv']),
])
def test_bootstrap_rejects_missing_or_inconsistent_cluster_metadata(meter_ids, technology):
    with pytest.raises(ValueError):
        bootstrap(meter_ids, technology, np.random.default_rng(1))
