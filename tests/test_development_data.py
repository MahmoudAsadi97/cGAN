"""Inner development validation uses only original TRAIN meters."""
import builtins
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from fluvius_cgan.data import load_split
from fluvius_cgan.development_data import prepare_development
from fluvius_cgan.training import _read_data


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_fixture(tmp_path, meters=5):
    source = tmp_path / 'original'
    source.mkdir()
    ids, labels, dates, profiles = [], [], [], []
    for technology in range(8):
        for meter in range(meters):
            for calendar in range(8):
                ids.append(f'g{technology}-m{meter}')
                labels.append(technology * 8 + calendar)
                dates.append(f'2024-01-{calendar + 1:02d}')
                profiles.append(np.arange(24) + technology + meter + calendar - 8.)
    # An arbitrary source order makes row-preservation assertions meaningful.
    order = np.random.default_rng(19).permutation(len(ids))
    arrays = {'x': np.asarray(profiles, dtype=np.float32)[order],
              'y': np.asarray(labels, dtype=np.int64)[order],
              'meter_id': np.asarray(ids)[order], 'date': np.asarray(dates)[order]}
    np.savez_compressed(source / 'train.npz', **arrays)
    # Deliberately invalid siblings must never be parsed or copied.
    for name in ('val.npz', 'test.npz', 'metadata.json', 'scaler.json'):
        (source / name).write_bytes(b'not input')
    return source / 'train.npz', arrays


def test_split_disjoint_stratified_complete_and_preserves_rows(tmp_path):
    source, original = source_fixture(tmp_path)
    output = tmp_path / 'development'
    report = prepare_development(source, output)
    train, val, scaler, metadata, _ = _read_data(output)
    assert report == metadata
    assert set(train['meter_id']).isdisjoint(val['meter_id'])
    assert set(train['meter_id']) | set(val['meter_id']) == set(original['meter_id'])
    assert set(train['y']) == set(val['y']) == set(range(64))
    for name, arrays, count in [('train', train, 4), ('val', val, 1)]:
        mask = np.isin(original['meter_id'], arrays['meter_id'])
        for key in original:
            np.testing.assert_array_equal(arrays[key], original[key][mask])
        assert metadata['splits'][name]['technology_meter_counts'] == {str(i): count for i in range(8)}
        assert metadata['splits'][name]['sha256'] == digest(output / f'{name}.npz')
        assert metadata['splits'][name]['meter_ids'] == sorted(set(arrays['meter_id']))
    assert scaler['minimum'] == float(train['x'].min())
    assert scaler['maximum'] == float(train['x'].max())
    assert metadata['sampling']['mode'] == 'full'
    development = metadata['development']
    assert development['source_partition_role'] == 'original_train'
    assert development['parent_train_sha256'] == digest(source)
    assert development['is_new_untouched_holdout'] is False
    assert 'exposure' in development['prior_exposure_caveat'].lower()
    assert development['meter_assignments'] == {
        name: sorted(set(arrays['meter_id'])) for name, arrays in [('train', train), ('val', val)]}
    assert set(path.name for path in output.iterdir()) == {'train.npz', 'val.npz', 'scaler.json', 'metadata.json'}


def test_scaler_uses_inner_train_only_despite_validation_outlier(tmp_path):
    source, original = source_fixture(tmp_path)
    initial = tmp_path / 'initial'
    prepare_development(source, initial)
    validation_ids = load_split(initial / 'val.npz')['meter_id']
    original['x'][np.isin(original['meter_id'], validation_ids)] = 1_000_000.
    np.savez_compressed(source, **original)
    output = tmp_path / 'outlier'
    prepare_development(source, output)
    train, val, scaler, _, _ = _read_data(output)
    assert val['x'].max() == 1_000_000.
    assert scaler['maximum'] == float(train['x'].max()) < 1_000_000.


def test_deterministic_bytes_and_no_original_val_or_test_reads(tmp_path, monkeypatch):
    source, _ = source_fixture(tmp_path)
    original_open, original_builtin_open = Path.open, builtins.open
    reads = []

    def audit(path, mode):
        if isinstance(path, (str, Path)) and Path(path).parent == source.parent:
            assert Path(path) == source, f'Unexpected original-data access: {path}'
            assert mode in ('rb', 'r')
            reads.append(Path(path).name)

    def audited_open(path, mode='r', *args, **kwargs):
        audit(path, mode)
        return original_open(path, mode, *args, **kwargs)

    def audited_builtin_open(path, mode='r', *args, **kwargs):
        audit(path, mode)
        return original_builtin_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', audited_open)
    monkeypatch.setattr(builtins, 'open', audited_builtin_open)
    first, second = tmp_path / 'first', tmp_path / 'second'
    prepare_development(source, first)
    prepare_development(source, second)
    assert reads
    assert {p.name: p.read_bytes() for p in first.iterdir()} == {p.name: p.read_bytes() for p in second.iterdir()}


def test_readonly_source_and_no_overwrite_even_empty_output(tmp_path):
    source, _ = source_fixture(tmp_path)
    hashes = {p.name: digest(p) for p in source.parent.iterdir()}
    source.chmod(0o400)
    source.parent.chmod(0o500)
    output = tmp_path / 'output'
    try:
        prepare_development(source, output)
        before = {p.name: p.read_bytes() for p in output.iterdir()}
        with pytest.raises(FileExistsError):
            prepare_development(source, output)
        assert before == {p.name: p.read_bytes() for p in output.iterdir()}
        empty = tmp_path / 'empty'
        empty.mkdir()
        with pytest.raises(FileExistsError):
            prepare_development(source, empty)
        assert hashes == {p.name: digest(p) for p in source.parent.iterdir()}
    finally:
        source.parent.chmod(0o700)
        source.chmod(0o600)


@pytest.mark.parametrize('change', ['nonfinite', 'resolution', 'technology', 'missing_condition', 'alignment'])
def test_invalid_sources_fail_before_output(tmp_path, change):
    source, arrays = source_fixture(tmp_path)
    if change == 'nonfinite':
        arrays['x'][0, 0] = np.nan
    elif change == 'resolution':
        arrays['x'] = np.repeat(arrays['x'], 4, axis=1)
    elif change == 'technology':
        arrays['y'][0] = (arrays['y'][0] + 8) % 64
    elif change == 'missing_condition':
        arrays['y'][arrays['y'] == 63] = 62
    else:
        arrays['meter_id'] = arrays['meter_id'][:-1]
    np.savez_compressed(source, **arrays)
    with pytest.raises(ValueError):
        prepare_development(source, tmp_path / 'invalid')
    assert not (tmp_path / 'invalid').exists()


@pytest.mark.parametrize('fraction', [0, 1, np.nan, True, '0.2'])
def test_invalid_validation_fraction(tmp_path, fraction):
    source, _ = source_fixture(tmp_path)
    with pytest.raises(ValueError):
        prepare_development(source, tmp_path / 'invalid', validation_fraction=fraction)


def test_rounded_per_group_size_and_insufficient_meters(tmp_path):
    source, _ = source_fixture(tmp_path, meters=8)
    report = prepare_development(source, tmp_path / 'rounded')
    assert report['splits']['val']['meters'] == 16
    other = tmp_path / 'other'
    other.mkdir()
    small, _ = source_fixture(other, meters=1)
    with pytest.raises(ValueError, match='meters'):
        prepare_development(small, tmp_path / 'too_small')


def test_missing_inner_condition_rejected_even_when_parent_has_all_conditions(tmp_path):
    source, arrays = source_fixture(tmp_path)
    prepare_development(source, tmp_path / 'membership')
    validation = load_split(tmp_path / 'membership/val.npz')
    keep = ~(np.isin(arrays['meter_id'], validation['meter_id']) & (arrays['y'] == 63))
    np.savez_compressed(source, **{key: value[keep] for key, value in arrays.items()})
    with pytest.raises(ValueError, match='val: all 64'):
        prepare_development(source, tmp_path / 'unsupported')
    assert not (tmp_path / 'unsupported').exists()


def test_failed_write_leaves_no_output_or_lock(tmp_path, monkeypatch):
    source, _ = source_fixture(tmp_path)
    original_save = np.savez_compressed
    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError('synthetic write failure')
        return original_save(*args, **kwargs)

    monkeypatch.setattr('fluvius_cgan.development_data.np.savez_compressed', fail_second)
    with pytest.raises(OSError, match='synthetic write failure'):
        prepare_development(source, tmp_path / 'failed')
    assert not (tmp_path / 'failed').exists()
    assert not list(tmp_path.glob('.failed*'))


def test_rejects_original_val_test_and_nested_output(tmp_path):
    source, _ = source_fixture(tmp_path)
    for name in ('val.npz', 'test.npz'):
        with pytest.raises(ValueError, match='original TRAIN'):
            prepare_development(source.parent / name, tmp_path / name)
    with pytest.raises(ValueError, match='outside'):
        prepare_development(source, source.parent / 'nested')
