"""Scientific regression checks for fixed-draw, meter-cluster comparisons."""
import hashlib

import numpy as np
import pytest
from scipy.stats import wasserstein_distance


def test_cached_weighted_w1_matches_scipy_unequal_repeated_support():
    from fluvius_cgan.inference import _PairedW1Cache
    real = np.array([-4., -1., -1., 0., 3., 9.])
    a = np.array([-3., -3., 1., 8.])
    b = np.array([-5., 0., 0., 0., 4., 6., 12.])
    weights = np.array([[1, 1, 1, 1, 1, 1], [0, 2, 2, 0, 3, 3], [5, 0, 0, 1, 0, 0]])
    cache = _PairedW1Cache(real, a, b)
    actual_a, actual_b = cache.distances(weights)
    for i, weight in enumerate(weights):
        assert actual_a[i] == pytest.approx(wasserstein_distance(real, a, u_weights=weight))
        assert actual_b[i] == pytest.approx(wasserstein_distance(real, b, u_weights=weight))


def fixture():
    # Unequal histories/condition counts, duplicated values and two technology strata.
    real = np.array([[-3., 1.], [0., 0.], [2., 4.], [1., -2.], [1., -2.],
                     [-1., 4.], [5., 0.], [-2., -1.], [4., 2.], [0., 1.]])
    labels = np.array(['x', 'y', 'x', 'y', 'x', 'y', 'x', 'y', 'x', 'x'])
    meters = np.array(['a', 'a', 'b', 'b', 'b', 'b', 'c', 'c', 'd', 'd'])
    tech = np.array(['pv'] * 6 + ['ev'] * 4)
    return real, real + .25, real * .7 - .2, labels, meters, tech


def test_paired_cluster_comparison_matches_direct_whole_meter_reference():
    from fluvius_cgan.evaluation import projection_directions
    from fluvius_cgan.inference import paired_cluster_comparison
    from fluvius_cgan.uncertainty import cluster_bootstrap_indices
    real, a, b, labels, meters, tech = fixture()
    directions = projection_directions(2, count=5)
    result = paired_cluster_comparison(real, a, b, labels, meters, tech,
                                      directions=directions, n_bootstrap=29, seed=13)
    def direct(indices):
        values = {'daily_energy_w1_kwh': [], 'profile_sw1_kwh': []}
        for label in ['x', 'y']:
            rows = indices[labels[indices] == label]
            mask = labels == label
            values['daily_energy_w1_kwh'].append(
                wasserstein_distance(real[rows].sum(1), a[mask].sum(1))
                - wasserstein_distance(real[rows].sum(1), b[mask].sum(1)))
            values['profile_sw1_kwh'].append(np.mean([
                wasserstein_distance(real[rows] @ d, a[mask] @ d)
                - wasserstein_distance(real[rows] @ d, b[mask] @ d) for d in directions]))
        return {name: np.mean(value) for name, value in values.items()}
    observed = direct(np.arange(len(real)))
    rng = np.random.default_rng(13)
    draws = [direct(cluster_bootstrap_indices(meters, tech, rng)) for _ in range(29)]
    for metric in observed:
        assert result['observed_difference'][metric] == pytest.approx(observed[metric], abs=1e-12)
        np.testing.assert_allclose(result['ci95'][metric], np.percentile([d[metric] for d in draws], [2.5, 97.5]), atol=1e-12)
    metadata = result['metadata']
    assert metadata['projection_sha256'] == hashlib.sha256(np.asarray(directions, dtype='<f8').tobytes()).hexdigest()
    assert metadata['n_bootstrap'] == 29
    assert metadata['seed'] == 13
    assert metadata['positive_difference'] == 'A worse than B'
    assert 'fixed generation draws' in metadata['uncertainty_scope']
    assert '2024' in metadata['uncertainty_scope']


def test_all_zero_single_slot_is_stable_and_default_projections_recorded():
    from fluvius_cgan.inference import paired_cluster_comparison
    result = paired_cluster_comparison(np.zeros((4, 1)), np.zeros((4, 1)), np.zeros((4, 1)),
                                      ['x'] * 4, ['a', 'a', 'b', 'b'], ['pv'] * 4, n_bootstrap=9)
    assert result['observed_difference'] == {'daily_energy_w1_kwh': 0., 'profile_sw1_kwh': 0.}
    assert all(value == [0., 0.] for value in result['ci95'].values())
    assert result['metadata']['projection_count'] == 128
    assert result['metadata']['projection_seed'] == 2026


def test_missing_bootstrap_condition_explicitly_withholds_interval():
    from fluvius_cgan.inference import paired_cluster_comparison
    result = paired_cluster_comparison([[1.], [2.]], [[1.], [2.]], [[0.], [0.]],
                                      ['x', 'y'], ['a', 'b'], ['pv', 'pv'], n_bootstrap=30)
    assert result['metadata']['missing_condition_replicates'] > 0
    assert all(ci is None for ci in result['ci95'].values())
    assert 'missing' in result['metadata']['ci_unavailable_reason']


@pytest.mark.parametrize('change', ['length', 'complex', 'direction', 'meter', 'empty', 'bootstrap'])
def test_invalid_paired_inputs_rejected(change):
    from fluvius_cgan.inference import paired_cluster_comparison
    real, a, b, labels, meters, tech = fixture()
    kwargs = {'n_bootstrap': 3}
    if change == 'length':
        a = a[:-1]
    elif change == 'complex':
        b = b.astype(complex) + 1j
    elif change == 'direction':
        kwargs['directions'] = [[0., 0.]]
    elif change == 'meter':
        tech[0] = 'ev'
    elif change == 'empty':
        labels[0] = ''
    else:
        kwargs['n_bootstrap'] = True
    with pytest.raises(ValueError):
        paired_cluster_comparison(real, a, b, labels, meters, tech, **kwargs)


def test_nearest_neighbor_detects_complete_day_resampling_without_clipping():
    from fluvius_cgan.inference import nearest_neighbor_diagnostic
    train = np.array([[-5., 1.], [2., 3.], [-1., -4.], [7., -2.]])
    labels = ['x', 'x', 'y', 'y']
    heldout = train + 1.
    result = nearest_neighbor_diagnostic(train, labels, heldout, labels,
                                        train[[1, 0, 3, 2]], labels)
    for entry in result['per_condition'].values():
        generated = entry['generated_to_train']
        reference = entry['heldout_to_train']
        assert generated['median_rmse_kwh'] == 0.
        assert generated['exact_match_rate'] == 1.
        assert generated['tolerance_match_rate'] == 1.
        assert reference['median_rmse_kwh'] == pytest.approx(1.)
        assert reference['exact_match_rate'] == 0.
    assert 'no privacy guarantee' in result['metadata']['interpretation']


def test_nearest_neighbor_conditioning_and_query_cap_are_reproducible():
    from fluvius_cgan.inference import nearest_neighbor_diagnostic
    train = [[0.], [100.]]
    labels = ['x', 'y']
    query = np.arange(20.).reshape(-1, 1)
    query_labels = ['x'] * 10 + ['y'] * 10
    result = nearest_neighbor_diagnostic(train, labels, query, query_labels,
                                        query, query_labels, max_queries_per_condition=3, seed=14)
    assert result == nearest_neighbor_diagnostic(train, labels, query, query_labels,
                                               query, query_labels, max_queries_per_condition=3, seed=14)
    assert result['per_condition']['y']['generated_to_train']['median_rmse_kwh'] > 80
    assert result['per_condition']['x']['generated_to_train']['n_queries'] == 3
    assert result['per_condition']['x']['generated_to_train']['n_available'] == 10
    assert result['metadata']['max_queries_per_condition'] == 3


@pytest.mark.parametrize('labels', [['x', 'z'], ['x', None]])
def test_nearest_neighbor_requires_matching_condition_support(labels):
    from fluvius_cgan.inference import nearest_neighbor_diagnostic
    with pytest.raises(ValueError):
        nearest_neighbor_diagnostic([[0.], [1.]], ['x', 'y'], [[0.], [1.]], labels,
                                    [[0.], [1.]], ['x', 'y'])
