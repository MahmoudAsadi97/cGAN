import json
import zipfile
import pytest
from fluvius_cgan.cli import main


def test_audit_cli_writes_schema(tmp_path):
    archive = tmp_path / 'sample.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('data.csv', 'meter,time\n1,2024-01-01\n')
    output = tmp_path / 'audit.json'
    main(['audit', '--archive', str(archive), '--output', str(output)])
    assert json.loads(output.read_text())['members'][0]['columns'] == ['meter', 'time']


def test_evaluate_rejects_train_as_holdout():
    with pytest.raises(SystemExit):
        main(['evaluate', '--prepared', 'x', '--checkpoint', 'x', '--output', 'x', '--split', 'train'])


def test_generate_rejects_nonpositive_count(tmp_path):
    with pytest.raises(ValueError, match='positive'):
        main(['generate', '--checkpoint', 'missing.pt', '--labels', '0',
              '--samples-per-label', '0', '--output', str(tmp_path/'out.npz')])


def _prepared_for_cli(tmp_path):
    import numpy as np
    prepared = tmp_path / 'prepared'
    prepared.mkdir()
    counts = {}
    for split in ['train', 'val', 'test']:
        values = np.arange(96, dtype=np.float32).reshape(4, 24) / 100
        labels = np.array([0, 0, 8, 8], dtype=np.int64)
        ids = np.array([split+'0', split+'0', split+'1', split+'1'])
        np.savez(prepared / f'{split}.npz', x=values, y=labels,
                 meter_id=ids, date=np.array(['2024-01-01', '2024-01-02']*2))
        counts[split] = {'profiles': 4, 'meters': 2, 'meter_ids': sorted(set(ids))}
    (prepared / 'scaler.json').write_text(json.dumps({'minimum': 0., 'maximum': float(values.max())}))
    (prepared / 'metadata.json').write_text(json.dumps({
        'resolution': 24, 'target': 'net', 'units': 'kWh/interval',
        'source_sha256': 'a'*64, 'split_seed': 42, 'splits': counts}))
    return prepared


def test_train_generate_evaluate_cli_and_fingerprint_guard(tmp_path):
    import numpy as np
    prepared = _prepared_for_cli(tmp_path)
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'epochs': 1, 'batch_size': 4, 'hidden_dim': 4,
                                 'latent_dim': 4, 'validation_samples': 4}))
    run = tmp_path / 'run'
    main(['train', '--prepared', str(prepared), '--output', str(run), '--config', str(config)])
    generated = tmp_path / 'generated.npz'
    main(['generate', '--checkpoint', str(run/'best.pt'), '--labels', '0', '8',
          '--samples-per-label', '2', '--output', str(generated)])
    with np.load(generated) as data:
        assert data['x'].shape == (4, 24)
        assert data['y'].tolist() == [0, 0, 8, 8]
    report = tmp_path / 'report.json'
    main(['evaluate', '--prepared', str(prepared), '--checkpoint', str(run/'best.pt'),
          '--output', str(report), '--split', 'test'])
    result = json.loads(report.read_text())
    assert result['split'] == 'test'
    assert set(result['methods']) == {'cgan', 'conditional_mean', 'conditional_resampling'}
    assert result['profile_count'] == 4
    metadata = json.loads((prepared/'metadata.json').read_text())
    metadata['target'] = 'import'
    (prepared/'metadata.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='fingerprint'):
        main(['evaluate', '--prepared', str(prepared), '--checkpoint', str(run/'best.pt'),
              '--output', str(tmp_path/'invalid.json')])
