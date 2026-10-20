"""Derive hourly physical energy after complete quarter-hour trace curation.

Identity decisions refer to the canonical 96-slot source. Hourly sums may make
previously distinct traces equal; membership is never recalculated here.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from .curation import (
    INPUT_FILES, SPLIT_PRIORITY, _calendar, _collect_traces, _identity_bytes,
    _identity_digest, _load_inputs, _sha256, _write_json,
)


def _curated_inputs(source):
    """Require an intact quarter-hour curation artifact and its identity record."""
    files = (*INPUT_FILES, 'curation_manifest.json')
    hashes = {name: _sha256(source / name) for name in files}
    arrays, metadata, scaler, _ = _load_inputs(source)
    if metadata['resolution'] != 96 or metadata.get('interval_hours') != .25:
        raise ValueError('Hourly aggregation requires 96-slot quarter-hour source data')
    curation = metadata.get('curation')
    if not isinstance(curation, dict) or curation.get('fingerprint_version') != 2:
        raise ValueError('Hourly aggregation requires complete-trace identity curation version 2')
    if (curation.get('split_priority') != list(SPLIT_PRIORITY)
            or curation.get('original_split_assignments_preserved') is not True
            or curation.get('development_traces_transferred_to_test') is not False
            or curation.get('is_new_random_split') is not False):
        raise ValueError('Curation must preserve original split assignments and held-out membership')
    manifest = json.loads((source / 'curation_manifest.json').read_text())
    if not isinstance(manifest, dict):
        raise ValueError('Curation manifest must be a JSON object')
    if manifest.get('output_files_sha256') != {name: hashes[name] for name in INPUT_FILES}:
        raise ValueError('Curation manifest output hashes disagree with source files')
    if (manifest.get('fingerprint_version') != 2
            or manifest.get('holdout') != curation.get('holdout')
            or manifest.get('test_access_scope') != curation.get('test_access_scope')):
        raise ValueError('Curation manifest provenance disagrees with metadata')
    for split in SPLIT_PRIORITY:
        if metadata['splits'][split].get('sha256') != hashes[f'{split}.npz']:
            raise ValueError(f'{split}: missing or stale split hash')
        ids = sorted(set(arrays[split]['meter_id'].tolist()))
        if (metadata['splits'][split].get('meter_ids') != ids
                or curation.get('after', {}).get(split) != {
                    key: value for key, value in metadata['splits'][split].items() if key != 'sha256'
                }):
            raise ValueError(f'{split}: canonical curation membership disagrees with current split')
    if curation.get('holdout', {}).get('retained_original_test_meter_ids') != metadata['splits']['test']['meter_ids']:
        raise ValueError('Original held-out provenance disagrees with retained TEST membership')
    records, groups = _collect_traces(arrays, _calendar(curation.get('expected_dates')))
    if any(len(group) != 1 for group in groups.values()):
        raise ValueError('Source still contains duplicate complete traces')
    expected = {}
    for group in curation.get('identity_groups', []):
        canonical = group.get('canonical', {})
        key = (canonical.get('split'), canonical.get('meter_id'))
        if key in expected:
            raise ValueError('Repeated canonical member in curation identity groups')
        expected[key] = group.get('fingerprint_sha256')
    actual = {(split, trace.meter_id): _identity_digest(_identity_bytes(arrays[split], trace.rows))
              for split, traces in records.items() for trace in traces}
    if expected != actual:
        raise ValueError('Canonical quarter-hour fingerprints disagree with source values')
    if hashes != {name: _sha256(source / name) for name in files}:
        raise ValueError('Curated source files changed while loading')
    return arrays, metadata, scaler, manifest, hashes


def aggregate_hourly(source_dir, output_dir):
    """Atomically sum adjacent float32 quarter-hours without changing membership.

    Input must be a verified 96-slot output of ``curate_prepared``. Profiles stay
    in kWh/interval; only retained TRAIN fits the new scalar min/max transform.
    Original identity fingerprints and TEST access history remain explicitly
    scoped to the quarter-hour source. Both output manifests bind fresh bytes.
    """
    source, output = Path(source_dir).resolve(), Path(output_dir).resolve()
    if output == source or source in output.parents:
        raise ValueError('output must be outside the curated source directory')
    if output.exists():
        raise FileExistsError('Hourly output must be a new directory')
    arrays, metadata, scaler, curation_manifest, hashes = _curated_inputs(source)
    metadata = copy.deepcopy(metadata)
    hourly = {}
    for split, data in arrays.items():
        with np.errstate(over='ignore', invalid='ignore'):
            x = data['x'].reshape(-1, 24, 4).sum(axis=2, dtype=np.float32)
        if not np.isfinite(x).all():
            raise ValueError(f'{split}: hourly energy exceeds finite float32 storage range')
        hourly[split] = {**data, 'x': x}
    minimum, maximum = float(hourly['train']['x'].min()), float(hourly['train']['x'].max())
    scaler = {**scaler, 'minimum': minimum, 'maximum': maximum, 'fitted_on': 'train',
              'range': [0, 1], 'constant_range': minimum == maximum,
              'transform': '(x - minimum) / (maximum - minimum); use denominator 1 if constant'}
    metadata.update(resolution=24, interval_hours=1.)
    metadata['aggregation'] = {
        'schema_version': 1, 'source_resolution': 96, 'output_resolution': 24,
        'operation': 'Sum each adjacent group of four float32 kWh quarter-hours with float32 accumulation',
        'input_files_sha256': hashes,
        'implementation_files_sha256': {'prepared.py': _sha256(Path(__file__))},
        'canonical_membership_preserved': True, 'original_split_assignments_preserved': True,
        'identity_fingerprints_resolution': 96,
        'identity_scope': 'curation identity groups and fingerprints describe the quarter-hour source; no hourly deduplication is performed',
        'train_statistics_scope': 'Retained TRAIN hourly global minimum and maximum only',
        'test_access_scope': 'Read retained original TEST rows only for structural/identity validation and deterministic adjacent-quarter-hour summation; no performance or energy-distribution metrics',
        'requires_new_training_artifacts': True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.parent / f'.{output.name}.aggregation.lock'
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise FileExistsError('Another aggregation already reserved this output') from error
    os.close(descriptor)
    try:
        if output.exists():
            raise FileExistsError('Hourly output must remain a new directory')
        with tempfile.TemporaryDirectory(prefix=f'.{output.name}-', dir=output.parent) as staging:
            stage = Path(staging)
            for split, data in hourly.items():
                with (stage / f'{split}.npz').open('wb') as handle:
                    np.savez_compressed(handle, **data)
                    handle.flush()
                    os.fsync(handle.fileno())
                metadata['splits'][split]['sha256'] = _sha256(stage / f'{split}.npz')
            _write_json(stage / 'scaler.json', scaler)
            _write_json(stage / 'metadata.json', metadata)
            fresh_hashes = {name: _sha256(stage / name) for name in INPUT_FILES}
            # The historical input and identity records remain unchanged. Output
            # hashes are replaced, never copied from the quarter-hour artifact.
            _write_json(stage / 'curation_manifest.json', {
                **curation_manifest, 'output_files_sha256': fresh_hashes,
                'identity_fingerprints_resolution': 96, 'output_resolution': 24,
                'identity_scope': metadata['aggregation']['identity_scope'],
                'quarter_hour_curation_manifest_sha256': hashes['curation_manifest.json'],
                'quarter_hour_output_files_sha256': curation_manifest['output_files_sha256'],
                'aggregation_input_files_sha256': hashes,
            })
            _write_json(stage / 'aggregation_manifest.json', {
                **metadata['aggregation'],
                'output_files_sha256': {**fresh_hashes, 'curation_manifest.json': _sha256(stage / 'curation_manifest.json')},
                'holdout': metadata['curation']['holdout'],
                'original_curation_test_access_scope': metadata['curation']['test_access_scope'],
            })
            if hashes != {name: _sha256(source / name) for name in hashes}:
                raise ValueError('Curated source files changed during aggregation')
            if output.exists():
                raise FileExistsError('Hourly output appeared during aggregation')
            os.rename(stage, output)
    finally:
        lock.unlink(missing_ok=True)
    return metadata
