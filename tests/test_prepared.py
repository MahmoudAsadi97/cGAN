"""Quarter-hour identity curation precedes physical hourly aggregation."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from fluvius_cgan.curation import curate_prepared
from fluvius_cgan.data import load_split
from fluvius_cgan.prepared import aggregate_hourly


DATES = np.array(['2024-01-01', '2024-01-02', '2024-01-03'])


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_fixture(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    # Two distinct complete traces collapse to identical hourly totals; hourly
    # aggregation must retain the canonical quarter-hour membership unchanged.
    first = np.tile(np.array([-3., 1., 2., 4.], dtype=np.float32), (3, 24))
    second = first.copy()
    second[:, :2] += np.array([1., -1.], dtype=np.float32)
    heldout = np.full((3, 96), 1_000_000., dtype=np.float32)
    members = {'train': [('train_a', first), ('train_z', first)],
               'val': [('val_alias', first), ('val_unique', second)],
               'test': [('test_alias', first), ('test_unique', heldout)]}
    metadata = {'resolution': 96, 'interval_hours': .25, 'target': 'net',
                'units': 'kWh/interval', 'split_seed': 42, 'source_sha256': 'a' * 64,
                'counts': {'meters_kept': 6, 'profiles_kept': 18}, 'splits': {}}
    for split, records in members.items():
        arrays = {'x': np.concatenate([x for _, x in records]),
                  'y': np.zeros(len(records) * 3, dtype=np.int64),
                  'meter_id': np.repeat([name for name, _ in records], 3),
                  'date': np.tile(DATES, len(records))}
        np.savez_compressed(source / f'{split}.npz', **arrays)
        metadata['splits'][split] = {'profiles': len(arrays['x']), 'meters': len(records),
                                    'meter_ids': [name for name, _ in records],
                                    'sha256': digest(source / f'{split}.npz')}
    (source / 'metadata.json').write_text(json.dumps(metadata))
    (source / 'scaler.json').write_text(json.dumps({'minimum': -3., 'maximum': 4.}))
    curated = tmp_path / 'curated'
    curate_prepared(source, curated, expected_dates=DATES)
    return curated


def test_hourly_energy_membership_scaler_and_fresh_manifests(tmp_path):
    source = source_fixture(tmp_path)
    original_hashes = {path.name: digest(path) for path in source.iterdir()}
    original_meta = json.loads((source / 'metadata.json').read_text())
    output = tmp_path / 'hourly'
    report = aggregate_hourly(source, output)
    for split in ('train', 'val', 'test'):
        before, after = load_split(source / f'{split}.npz'), load_split(output / f'{split}.npz')
        assert after['x'].dtype == np.float32
        np.testing.assert_array_equal(after['x'], before['x'].reshape(-1, 24, 4).sum(axis=2, dtype=np.float32))
        np.testing.assert_allclose(after['x'].sum(axis=1), before['x'].sum(axis=1))
        for key in ('y', 'meter_id', 'date'):
            np.testing.assert_array_equal(after[key], before[key])
        assert report['splits'][split]['sha256'] == digest(output / f'{split}.npz')
        assert report['splits'][split]['meter_ids'] == original_meta['splits'][split]['meter_ids']
    # Distinct quarter-hour traces remain in their original splits after collapse.
    np.testing.assert_array_equal(load_split(output / 'train.npz')['x'], load_split(output / 'val.npz')['x'])
    assert report['resolution'] == 24 and report['interval_hours'] == 1.
    assert report['source_sha256'] == original_meta['source_sha256']
    assert report['curation']['holdout'] == original_meta['curation']['holdout']
    assert report['curation']['test_access_scope'] == original_meta['curation']['test_access_scope']
    assert report['aggregation']['identity_fingerprints_resolution'] == 96
    assert report['aggregation']['canonical_membership_preserved']
    scaler = json.loads((output / 'scaler.json').read_text())
    assert scaler['minimum'] == scaler['maximum'] == 4.
    assert scaler['constant_range'] and scaler['fitted_on'] == 'train'
    assert load_split(output / 'test.npz')['x'].max() == 4_000_000.
    for filename in ('curation_manifest.json', 'aggregation_manifest.json'):
        manifest = json.loads((output / filename).read_text())
        for name, value in manifest['output_files_sha256'].items():
            assert value == digest(output / name)
    provenance = json.loads((output / 'curation_manifest.json').read_text())
    assert provenance['identity_fingerprints_resolution'] == 96
    assert provenance['output_resolution'] == 24
    assert {path.name: digest(path) for path in source.iterdir()} == original_hashes


def test_negative_exports_and_float32_group_summation(tmp_path):
    source = source_fixture(tmp_path)
    # Build a second curated source whose training trace has a negative hour.
    original = tmp_path / 'source'
    for split in ('train', 'val', 'test'):
        arrays = load_split(original / f'{split}.npz')
        arrays['x'][:, 4:8] = np.array([-8., -4., -2., -1.], dtype=np.float32)
        np.savez_compressed(original / f'{split}.npz', **arrays)
    meta = json.loads((original / 'metadata.json').read_text())
    for split in ('train', 'val', 'test'):
        meta['splits'][split]['sha256'] = digest(original / f'{split}.npz')
    (original / 'metadata.json').write_text(json.dumps(meta))
    (original / 'scaler.json').write_text(json.dumps({'minimum': -8., 'maximum': 4.}))
    curated = tmp_path / 'negative'
    curate_prepared(original, curated, expected_dates=DATES)
    aggregate_hourly(curated, tmp_path / 'negative_hourly')
    x = load_split(tmp_path / 'negative_hourly/train.npz')['x']
    assert np.all(x[:, 1] == -15.)
    scaler = json.loads((tmp_path / 'negative_hourly/scaler.json').read_text())
    assert scaler['minimum'] == -15. and scaler['maximum'] == 4.


@pytest.mark.parametrize('change', ['uncurated', 'resolution', 'split_hash', 'curation_manifest', 'scaler', 'membership'])
def test_incompatible_or_stale_source_fails_without_output(tmp_path, change):
    source = source_fixture(tmp_path)
    meta_path = source / 'metadata.json'
    meta = json.loads(meta_path.read_text())
    if change == 'uncurated':
        meta.pop('curation')
    elif change == 'resolution':
        meta['resolution'] = 24
    elif change == 'split_hash':
        meta['splits']['test']['sha256'] = '0' * 64
    elif change == 'curation_manifest':
        manifest_path = source / 'curation_manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest['output_files_sha256']['test.npz'] = '0' * 64
        manifest_path.write_text(json.dumps(manifest))
    elif change == 'scaler':
        (source / 'scaler.json').write_text(json.dumps({'minimum': -100., 'maximum': 100.}))
    elif change == 'membership':
        meta['curation']['holdout']['retained_original_test_meter_ids'] = ['invented']
    meta_path.write_text(json.dumps(meta))
    # Keep the outer metadata hash consistent to exercise the semantic checks.
    manifest_path = source / 'curation_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['output_files_sha256']['metadata.json'] = digest(meta_path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        aggregate_hourly(source, tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


def test_existing_nested_output_and_write_failure_are_atomic(tmp_path, monkeypatch):
    source = source_fixture(tmp_path)
    output = tmp_path / 'out'
    output.mkdir()
    (output / 'keep').write_text('keep')
    with pytest.raises(FileExistsError):
        aggregate_hourly(source, output)
    with pytest.raises(ValueError, match='outside'):
        aggregate_hourly(source, source / 'nested')
    original_save = np.savez_compressed
    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError('synthetic write failure')
        return original_save(*args, **kwargs)

    monkeypatch.setattr('fluvius_cgan.prepared.np.savez_compressed', fail_second)
    with pytest.raises(OSError, match='synthetic write failure'):
        aggregate_hourly(source, tmp_path / 'failed')
    assert not (tmp_path / 'failed').exists()
    assert not list(tmp_path.glob('.failed*'))
    assert (output / 'keep').read_text() == 'keep'
