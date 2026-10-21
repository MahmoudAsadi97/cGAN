"""Synthetic checks only: these fixtures never use Fluvius profile values."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def module():
    path = Path(__file__).parents[1] / 'scripts' / 'cohort_reference.py'
    assert path.exists(), 'Standalone reference generator must exist'
    spec = importlib.util.spec_from_file_location('cohort_reference', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def fixture(tmp_path, constant=False, uneven=False):
    ids, labels, profiles = [], [], []
    rng = np.random.default_rng(10)
    counts = {str(g): 1 + (g % 2 if uneven else 0) for g in range(8)}
    for group in range(8):
        for meter in range(6):
            for condition in range(8 * group, 8 * group + 8):
                for day in range(3):
                    ids.append(f'g{group}m{meter}')
                    labels.append(condition)
                    profiles.append(np.zeros(24) if constant else rng.normal(size=24))
    path = tmp_path / 'train.npz'
    np.savez(path, x=np.array(profiles), y=np.array(labels), meter_id=np.array(ids))
    metadata = {'resolution': 24, 'condition_dim': 64, 'splits': {
        'train': {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'meters': 48},
        'val': {'meters': sum(counts.values()), 'technology_meter_counts': counts}}}
    meta = tmp_path / 'metadata.json'
    meta.write_text(json.dumps(metadata))
    # Deliberately unreadable as NPZ: opening either is a bug.
    (tmp_path / 'val.npz').write_bytes(b'NOT VALIDATION VALUES')
    (tmp_path / 'test.npz').write_bytes(b'NOT TEST VALUES')
    return path, meta, counts


def test_stratified_disjoint_reproducible_and_only_training_values(tmp_path, monkeypatch):
    api = module()
    train, meta, counts = fixture(tmp_path, uneven=True)
    original = api.np.load
    original_open = Path.open
    reads = []
    def guarded(path, *args, **kwargs):
        reads.append(Path(path).name)
        assert Path(path) == train
        return original(path, *args, **kwargs)
    monkeypatch.setattr(api.np, 'load', guarded)
    def guarded_open(path, *args, **kwargs):
        assert path.name not in ('val.npz', 'test.npz'), 'Held-out profile bytes must not be read'
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded_open)
    first = api.build_reference(train, meta, tmp_path / 'out', pairs=3, per_condition=2)
    second = api.build_reference(train, meta, tmp_path / 'again', pairs=3, per_condition=2)
    assert reads == ['train.npz', 'train.npz']
    assert first['summary'] == second['summary']
    assert len(first['pairs']) == 3
    for left, right in zip(first['pairs'], second['pairs']):
        assert left == right
        a, b = left['cohorts']['a'], left['cohorts']['b']
        assert not set(a['meter_ids']) & set(b['meter_ids'])
        assert a['technology_meter_counts'] == counts == b['technology_meter_counts']
        assert all(v == 2 for v in a['sampled_condition_counts'].values())
        assert len(left['metrics']['per_condition']) == 64
        assert len(left['groups']) == 8
    assert set(first['summary']['macro']['profile_sw1_kwh']['quantiles']) == {'0.1', '0.5', '0.9', '0.95'}


def test_constant_profiles_are_zero_without_nan_and_preserve_all_pairs(tmp_path):
    api = module()
    train, meta, _ = fixture(tmp_path, constant=True)
    result = api.build_reference(train, meta, tmp_path / 'out', pairs=2, per_condition=3)
    assert result['normalization']['constant_training_range'] is True
    assert result['summary']['macro']['profile_sw1_kwh']['quantiles']['0.9'] == 0
    assert result['summary']['macro']['absolute_log_diversity_ratio']['quantiles']['0.9'] == 0
    assert all(p['metrics']['macro']['diversity_ratio'] == 1 for p in result['pairs'])
    json.dumps(result, allow_nan=False)


def test_existing_output_is_never_overwritten(tmp_path):
    api = module()
    train, meta, _ = fixture(tmp_path)
    output = tmp_path / 'out'
    output.mkdir()
    with pytest.raises(FileExistsError):
        api.build_reference(train, meta, output, pairs=1)
    assert not list(output.iterdir())


def test_insufficient_meters_and_stale_hash_fail_before_output(tmp_path):
    api = module()
    train, meta, _ = fixture(tmp_path)
    data = json.loads(meta.read_text())
    data['splits']['val']['technology_meter_counts']['0'] = 4
    data['splits']['val']['meters'] += 3
    meta.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='enough|Insufficient'):
        api.build_reference(train, meta, tmp_path / 'out', pairs=1)
    assert not (tmp_path / 'out').exists()
    data['splits']['train']['sha256'] = 'bad'
    meta.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='hash'):
        api.build_reference(train, meta, tmp_path / 'out', pairs=1)


def test_reject_non_training_input_and_invalid_counts(tmp_path):
    api = module()
    train, meta, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match='train.npz'):
        api.build_reference(tmp_path / 'val.npz', meta, tmp_path / 'out', pairs=1)
    for value in (0, -1, True):
        with pytest.raises(ValueError):
            api.build_reference(train, meta, tmp_path / 'out', pairs=value)


def test_one_sided_zero_variance_is_undefined_not_infinite():
    api = module()
    assert api.absolute_log_ratio(1.) == 0.
    assert api.absolute_log_ratio(.25) == pytest.approx(api.absolute_log_ratio(4.))
    assert api.absolute_log_ratio(0.) is None
    assert api.absolute_log_ratio(None) is None


def test_physical_endpoints_group_means_and_symmetric_variance():
    api = module()
    rng = np.random.default_rng(77)
    labels = np.repeat(np.arange(64), 4)
    real = rng.normal(size=(len(labels), 24))
    other = 2 * real
    forward, groups = api.score_pair(real, other, labels, 10.)
    reverse, _ = api.score_pair(other, real, labels, 10.)
    for group in range(8):
        for name, value in groups[str(group)]['macro'].items():
            expected = np.mean([forward['per_condition'][str(c)]['metrics'][name]
                                for c in range(group * 8, group * 8 + 8)])
            assert value == pytest.approx(expected)
    assert forward['macro']['absolute_log_diversity_ratio'] == pytest.approx(np.log(4))
    assert reverse['macro']['absolute_log_diversity_ratio'] == pytest.approx(np.log(4))
    assert forward['macro']['normalized_profile_sw1'] == pytest.approx(
        forward['macro']['profile_sw1_kwh'] / 10)
    summary = api.quantile_summary([None, 0., 10., 20.])
    assert summary['n_valid'] == 3 and summary['n_undefined'] == 1
    assert summary['quantiles']['0.5'] == 10.
