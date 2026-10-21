"""Create an explicit development split from original TRAIN only.

The original benchmark VAL/TEST and its preparation metadata are never opened.
The inner validation meters may have prior model exposure as original TRAIN;
this artifact therefore cannot establish a new untouched holdout.
"""
from __future__ import annotations

from collections import Counter
import json
import math
from numbers import Real
import os
from pathlib import Path
import tempfile

import numpy as np

from .data import _sha256, load_split


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def prepare_development(source_train_npz, output, seed=20261021, validation_fraction=.2):
    """Split sorted original-TRAIN meters within equipment strata, retaining rows.

    A seeded NumPy permutation chooses rounded validation counts per technology,
    bounded to leave at least one meter in each split. Both outputs must support
    all 64 conditions. Profiles remain physical hourly kWh; only inner TRAIN
    fits the fresh scalar transform. Output must be a new directory outside the
    source directory. No original validation/test files or sidecars are read.
    """
    if type(seed) is not int or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    if (isinstance(validation_fraction, bool) or not isinstance(validation_fraction, Real)
            or not math.isfinite(validation_fraction) or not 0 < validation_fraction < 1):
        raise ValueError('validation_fraction must be a finite number between zero and one')
    validation_fraction = float(validation_fraction)
    source, output = Path(source_train_npz).resolve(), Path(output).resolve()
    if source.name != 'train.npz':
        raise ValueError('Source must be original TRAIN, named train.npz')
    if output == source.parent or source.parent in output.parents:
        raise ValueError('output must be outside the original source directory')
    if output.exists():
        raise FileExistsError('Development output must be a new directory')
    parent_hash = _sha256(source)
    arrays = load_split(source)
    if arrays['x'].shape[1] != 24:
        raise ValueError('Development data requires physical hourly [N,24] profiles')
    ids, inverse = np.unique(arrays['meter_id'], return_inverse=True)
    technology_min, technology_max = np.full(len(ids), 8), np.full(len(ids), -1)
    np.minimum.at(technology_min, inverse, arrays['y'] // 8)
    np.maximum.at(technology_max, inverse, arrays['y'] // 8)
    if not np.array_equal(technology_min, technology_max):
        raise ValueError('Each meter must have constant equipment technology y//8')
    rng = np.random.default_rng(seed)
    validation_ids = []
    for technology in range(8):
        group = ids[technology_min == technology]
        if len(group) < 2:
            raise ValueError(f'Technology {technology} requires at least two original TRAIN meters')
        count = min(len(group) - 1, max(1, round(len(group) * validation_fraction)))
        validation_ids.extend(group[rng.permutation(len(group))[:count]].tolist())
    validation_mask = np.isin(arrays['meter_id'], validation_ids)
    masks = {'train': ~validation_mask, 'val': validation_mask}
    splits = {name: {key: value[mask] for key, value in arrays.items()}
              for name, mask in masks.items()}
    for name, data in splits.items():
        if not np.array_equal(np.unique(data['y']), np.arange(64)):
            raise ValueError(f'{name}: all 64 conditions must be represented')
    if _sha256(source) != parent_hash:
        raise ValueError('Original TRAIN changed while loading')
    assignments = {name: sorted(set(data['meter_id'].tolist())) for name, data in splits.items()}
    minimum, maximum = float(splits['train']['x'].min()), float(splits['train']['x'].max())
    scaler = {'minimum': minimum, 'maximum': maximum, 'fitted_on': 'train',
              'range': [0, 1], 'constant_range': minimum == maximum,
              'transform': '(x - minimum) / (maximum - minimum); use denominator 1 if constant'}
    metadata = {
        'schema_version': 1, 'resolution': 24, 'condition_dim': 64,
        'interval_hours': 1., 'units': 'kWh/interval', 'target': 'net',
        'label_formula': '((HP*4 + EV*2 + PV)*2 + weekend)*4 + season',
        'split_seed': seed,
        'split_fractions_requested': {'train': 1 - validation_fraction, 'val': validation_fraction},
        'split_method': 'NumPy default_rng seeded permutation of sorted meter IDs per technology; validation uses round(n*fraction), bounded to [1,n-1]; source row order preserved',
        'sampling': {'mode': 'full', 'max_meters_per_group': None,
                     'selection': 'All rows and meters from original TRAIN only; explicit development subset of the original benchmark'},
        'development': {
            'schema_version': 1, 'source_train_npz': str(source),
            'parent_train_sha256': parent_hash, 'source_partition_role': 'original_train',
            'source_partitions_read': ['original_train'],
            'original_validation_read': False, 'original_test_read': False,
            'meter_assignments': assignments, 'is_new_untouched_holdout': False,
            'prior_exposure_caveat': 'Inner validation is drawn from original TRAIN and may have prior model-training exposure. It supports development selection only, and is not a new untouched holdout. Original benchmark results remain frozen.',
            'scaler_fit_scope': 'Complete inner TRAIN physical hourly extrema only',
            'source_rows': len(arrays['x']), 'source_meters': len(ids),
        },
        'splits': {},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.parent / f'.{output.name}.development.lock'
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise FileExistsError('Another preparation already reserved this output') from error
    os.close(descriptor)
    try:
        if output.exists():
            raise FileExistsError('Development output must remain a new directory')
        with tempfile.TemporaryDirectory(prefix=f'.{output.name}-', dir=output.parent) as staging:
            stage = Path(staging)
            for name, data in splits.items():
                # NumPy's NPZ writer uses fixed ZIP member timestamps; identical
                # arrays and key order therefore produce identical archive bytes.
                np.savez_compressed(stage / f'{name}.npz', **data)
                technology_by_id = {str(meter): int(technology) for meter, technology in zip(ids, technology_min)}
                metadata['splits'][name] = {
                    'sha256': _sha256(stage / f'{name}.npz'),
                    'profiles': len(data['x']), 'meters': len(assignments[name]),
                    'meter_ids': assignments[name],
                    'technology_meter_counts': dict(sorted(Counter(str(technology_by_id[meter]) for meter in assignments[name]).items())),
                    'condition_counts': dict(sorted(Counter(str(int(label)) for label in data['y']).items(), key=lambda pair: int(pair[0]))),
                    'date_min': str(data['date'].astype('datetime64[D]').min()),
                    'date_max': str(data['date'].astype('datetime64[D]').max()),
                }
            _write_json(stage / 'scaler.json', scaler)
            _write_json(stage / 'metadata.json', metadata)
            if _sha256(source) != parent_hash:
                raise ValueError('Original TRAIN changed during preparation')
            if output.exists():
                raise FileExistsError('Development output appeared during preparation')
            os.rename(stage, output)
    finally:
        lock.unlink(missing_ok=True)
    return metadata
