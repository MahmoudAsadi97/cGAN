import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from fluvius_cgan.development import train_probe


def runner():
    path = Path(__file__).parents[1] / 'scripts' / 'evaluate_development_baselines.py'
    spec = importlib.util.spec_from_file_location('development_baselines', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_comparators_match_training_selection_and_preserve_existing_results(tmp_path):
    data = tmp_path / 'data'
    data.mkdir()
    rng = np.random.default_rng(45)
    for split in ('train', 'val'):
        x = rng.normal(2., .3, (24, 24)).astype(np.float32)
        np.savez(data / f'{split}.npz', x=x, y=np.arange(24, dtype=np.int64) % 2,
                 meter_id=np.array([f'{split}{i}' for i in range(24)]), date=np.full(24, '2024-01-01'))
    with np.load(data / 'train.npz') as values:
        scaler = {'minimum': float(values['x'].min()), 'maximum': float(values['x'].max()), 'fitted_on': 'train'}
    (data / 'scaler.json').write_text(json.dumps(scaler))
    (data / 'metadata.json').write_text(json.dumps({'resolution': 24, 'condition_dim': 64,
        'development': {'source_partition_role': 'original_train', 'is_new_untouched_holdout': False}}))
    (data / 'test.npz').write_bytes(b'FORBIDDEN')
    probe = train_probe(data, tmp_path / 'probe', {'steps': 1, 'validation_per_condition': 5})
    module = runner()
    output = tmp_path / 'baseline'
    result = module.evaluate(data, output, seeds=[20261022, 20261023], per_condition=5)
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['selection']['indices'] == probe['selection']['indices']
    assert set(manifest['data_hashes']) == {'train.npz', 'val.npz', 'scaler.json', 'metadata.json'}
    assert manifest['status'] == 'completed'
    assert set(result['methods']) == {'mean', 'resample', 'copula'}
    assert result['methods']['mean']['normalized_profile_sw1']['sampling_sd'] == 0.
    original = (output / 'summary.json').read_bytes()
    with pytest.raises(FileExistsError):
        module.evaluate(data, output, seeds=[20261022], per_condition=5)
    assert (output / 'summary.json').read_bytes() == original
    with pytest.raises(ValueError, match='distinct'):
        module.evaluate(data, tmp_path / 'bad', seeds=[1, 1], per_condition=5)
