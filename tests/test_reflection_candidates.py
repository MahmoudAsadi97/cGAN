"""Fresh reflection comparisons must preserve decoder and control provenance."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from fluvius_cgan import flow, reflection


def runner():
    path = Path(__file__).parents[1] / 'scripts' / 'evaluate_development_candidates.py'
    spec = importlib.util.spec_from_file_location('reflection_candidate_evaluation', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope='module')
def study(tmp_path_factory):
    root = tmp_path_factory.mktemp('reflection-candidates')
    data = root / 'data'
    data.mkdir()
    rng = np.random.default_rng(98)
    labels = np.repeat(np.array([0, 8], dtype=np.int64), 12)
    for name in ('train', 'val'):
        x = rng.normal(.2, .5, (24, 24)).astype(np.float32)
        x[:, 0] = 0
        np.savez(data / f'{name}.npz', x=x, y=labels,
                 meter_id=np.array([f'{name}{i}' for i in range(24)]),
                 date=np.full(24, '2024-01-01'))
    with np.load(data / 'train.npz') as values:
        scaler = {'minimum': float(values['x'].min()), 'maximum': float(values['x'].max()), 'fitted_on': 'train'}
    (data / 'scaler.json').write_text(json.dumps(scaler))
    (data / 'metadata.json').write_text(json.dumps({'resolution': 24, 'condition_dim': 64,
        'development': {'source_partition_role': 'original_train'}}))
    (data / 'test.npz').write_bytes(b'FORBIDDEN TEST')
    config = {'steps': 1, 'batch_size': 7, 'validation_per_condition': 4, 'solver_steps': 1}
    reflection.train_probe(data, root / 'first', config)
    reflection.train_probe(data, root / 'second', config | {'seed': 202, 'condition_mode': 'pooled'})
    flow.train_probe(data, root / 'flow', config)
    return root


def test_mixed_fresh_evaluation_preserves_reflection_samples_and_deduplicates_marginals(study, tmp_path):
    evaluate = runner()
    output = tmp_path / 'fresh'
    summary = evaluate.evaluate(study / 'data', [study / 'first', study / 'second', study / 'flow'], output)
    assert set(summary['methods']) == {'candidate_000', 'candidate_001', 'candidate_002',
        'source_000', 'independent_000', 'baseline_mean', 'baseline_resample', 'baseline_copula'}
    manifest = read(output / 'manifest.json')
    assert manifest['status'] == 'completed' and manifest['completed_draws'] == 16
    assert manifest['test_performance_access'] is False
    with np.load(study / 'data' / 'val.npz') as values:
        labels = values['y'][manifest['selection']['indices']]
    for index, name in enumerate(('first', 'second')):
        report = read(output / f'candidate_{index:03d}.json')
        assert report['method'] == 'conditional_reflection_copula'
        assert report['marginals'] == read(study / name / 'manifest.json')['marginals']
        for draw in report['draws']:
            expected = reflection.generate(study / name / 'best.pt', labels, draw['generation_seed'])
            assert draw['generated_profile_bytes_sha256'] == evaluate._digest(expected, '<f8')
    control = read(output / 'independent_000.json')
    assert control['method'] == 'independent_marginals'
    assert len(control['originating_candidates']) == 2
    _, _, marginals = reflection._load(study / 'first' / 'best.pt')
    for draw in control['draws']:
        expected = reflection.independent_marginals(marginals, labels, draw['generation_seed'])
        assert draw['generated_profile_bytes_sha256'] == evaluate._digest(expected, '<f8')
    record, _, _ = evaluate._candidate(study / 'flow', 0)
    old_source = evaluate._source_controls([(record, None, None)])[0]
    assert old_source['method_id'] == 'source_000'
    assert read(output / 'source_000.json')['source_distribution_sha256'] == old_source['source_distribution_sha256']
    with pytest.raises(ValueError, match='selection'):
        evaluate.evaluate(study / 'data', [study / 'first'], tmp_path / 'bad_seed', seeds=(20261022,))


@pytest.mark.parametrize('artifact', ['marginals.npz', 'independent_marginals.json', 'best_probability_diagnostics.json'])
def test_modified_reflection_artifacts_fail_before_evaluation(study, tmp_path, artifact):
    run = tmp_path / 'run'
    shutil.copytree(study / 'first', run)
    with (run / artifact).open('ab') as handle:
        handle.write(b'tamper')
    with pytest.raises(ValueError, match='Artifact hash mismatch'):
        runner()._candidate(run, 0)


@pytest.mark.parametrize('mutation', ['source_training_hash', 'source_marginals', 'source_selection',
                                     'probability_steps', 'sensitivity_checkpoint'])
def test_rehashed_reflection_reports_must_match_checkpoint_provenance(study, tmp_path, mutation):
    assert runner()._candidate(study / 'first', 0)[0]['method'] == 'conditional_reflection_copula'
    run = tmp_path / 'run'
    shutil.copytree(study / 'first', run)
    name = 'independent_marginals.json'
    if mutation == 'probability_steps':
        name = 'best_probability_diagnostics.json'
    elif mutation == 'sensitivity_checkpoint':
        name = 'solver_sensitivity.json'
    report = read(run / name)
    if mutation == 'source_training_hash':
        report['data_hashes']['train.npz'] = '0' * 64
    elif mutation == 'source_marginals':
        report['marginals']['sha256'] = '0' * 64
    elif mutation == 'source_selection':
        report['generation_seed'] += 1
    elif mutation == 'probability_steps':
        report['solver']['steps'] += 1
    else:
        report['checkpoint_sha256'] = '0' * 64
    (run / name).write_text(json.dumps(report))
    manifest = read(run / 'manifest.json')
    manifest['artifact_hashes'][name] = digest(run / name)
    (run / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='reflection|Reflection|marginal'):
        runner()._candidate(run, 0)


@pytest.mark.parametrize('receipt_hash_failure', [False, True])
def test_failed_reflection_draw_preserves_partial_hash_inventory_and_original_error(
        study, tmp_path, monkeypatch, receipt_hash_failure):
    evaluate = runner()
    output = tmp_path / 'failed'
    generate = reflection.generate
    def fail_second(checkpoint, labels, seed, solver_steps=None):
        if seed == 20261024:
            raise FloatingPointError('synthetic second reflection draw failure')
        return generate(checkpoint, labels, seed, solver_steps)
    monkeypatch.setattr(reflection, 'generate', fail_second)
    if receipt_hash_failure:
        original_hash = evaluate._sha256
        def fail_receipt_hash(path):
            if Path(path).parent == output:
                raise OSError('synthetic receipt read failure')
            return original_hash(path)
        monkeypatch.setattr(evaluate, '_sha256', fail_receipt_hash)
    with pytest.raises(FloatingPointError, match='synthetic second reflection draw failure'):
        evaluate.evaluate(study / 'data', [study / 'first'], output)
    manifest = read(output / 'manifest.json')
    assert manifest['status'] == 'failed' and manifest['completed_draws'] == 1
    assert manifest['error'] == 'FloatingPointError: synthetic second reflection draw failure'
    partial = read(output / 'candidate_000.json')
    assert len(partial['draws']) == 1 and partial['draws'][0]['generation_seed'] == 20261023
    if receipt_hash_failure:
        assert manifest['artifact_hashes'] == {}
        assert 'synthetic receipt read failure' in manifest['artifact_hash_errors']['candidate_000.json']
    else:
        assert manifest['artifact_hashes'] == {'candidate_000.json': digest(output / 'candidate_000.json')}
