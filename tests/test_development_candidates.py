"""Fresh-draw diagnostics must preserve units, selection and model provenance."""
import importlib.util
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from fluvius_cgan import development, flow


def module(name):
    path = Path(__file__).parents[1] / 'scripts' / f'{name}.py'
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_copy_precision_and_signed_export_diagnostics():
    runner = module('evaluate_development_candidates')
    real = np.stack([np.full(24, -1.), np.full(24, 2.)])
    fake = real.copy()
    labels = np.array([0, 8])  # Source PV flag 0 then 1.
    index = runner.CopyIndex(real)
    assert index.matches(fake).tolist() == [True, True]
    assert index.matches(fake + 1e-10).tolist() == [True, True]
    assert index.matches(fake + .01).tolist() == [False, False]
    scores = runner.tail_diagnostics(real, fake, labels, index.matches(fake))
    assert scores['macro']['pointwise_quantile_mae_kwh'] == 0.
    assert scores['macro']['covariance_rmse_kwh2'] == 0.
    assert scores['copying']['quantized_exact_match_fraction'] == 1.
    assert scores['source_pv0']['real']['profiles'] == 1
    assert scores['source_pv0']['real']['export_cell_fraction'] == 1.
    assert scores['source_pv0']['real']['negative_daily_energy_fraction'] == 1.
    assert scores['source_pv0']['real']['mean_export_magnitude_kwh_per_day'] == 24.
    assert scores['per_condition']['8']['real']['daily_energy_quantiles_kwh'] == [48.] * 5
    assert scores['per_condition']['0']['errors']['covariance_relative_frobenius'] is None
    constant = np.full((3, 24), .1)
    constant_scores = runner.tail_diagnostics(constant, constant, np.zeros(3, dtype=np.int64),
                                              np.ones(3, dtype=bool))
    assert constant_scores['per_condition']['0']['errors']['covariance_relative_frobenius'] is None


def inputs(root):
    root.mkdir()
    rng = np.random.default_rng(77)
    labels = np.repeat(np.array([0, 8], dtype=np.int64), 12)
    for split in ('train', 'val'):
        x = rng.normal(0., .5, (24, 24)).astype(np.float32)
        np.savez(root / f'{split}.npz', x=x, y=labels,
                 meter_id=np.array([f'{split}{i}' for i in range(24)]),
                 date=np.full(24, '2024-01-01'))
    with np.load(root / 'train.npz') as values:
        scaler = {'minimum': float(values['x'].min()), 'maximum': float(values['x'].max()), 'fitted_on': 'train'}
    (root / 'scaler.json').write_text(json.dumps(scaler))
    (root / 'metadata.json').write_text(json.dumps({'resolution': 24, 'condition_dim': 64,
        'development': {'source_partition_role': 'original_train'}}))
    (root / 'test.npz').write_bytes(b'FORBIDDEN TEST FILE')


def test_fresh_candidate_draws_match_baselines_and_preserve_artifacts(tmp_path, monkeypatch):
    data = tmp_path / 'data'
    inputs(data)
    common = {'steps': 1, 'batch_size': 7, 'validation_per_condition': 4}
    cgan, cflow = tmp_path / 'cgan', tmp_path / 'flow'
    second_flow, correlated, second_correlated = [tmp_path / name for name in ('flow_second', 'correlated', 'correlated_second')]
    development.train_probe(data, cgan, common | {'normalization': 'standard'})
    flow.train_probe(data, cflow, common | {'solver_steps': 1})
    flow.train_probe(data, second_flow, common | {'solver_steps': 1, 'seed': 202})
    flow.train_probe(data, correlated, common | {'solver_steps': 1, 'source_prior': 'training_correlation'})
    flow.train_probe(data, second_correlated, common | {'solver_steps': 1, 'source_prior': 'training_correlation', 'seed': 202})
    baseline = tmp_path / 'baselines'
    module('evaluate_development_baselines').evaluate(
        data, baseline, seeds=(20261023, 20261024), per_condition=4)
    runner = module('evaluate_development_candidates')
    output = tmp_path / 'diagnostics'
    result = runner.evaluate(data, [cgan, cflow, second_flow, correlated, second_correlated], output, baselines=baseline)
    assert set(result['methods']) == {'candidate_000', 'candidate_001', 'candidate_002', 'candidate_003', 'candidate_004',
                                     'source_000', 'source_001',
                                     'baseline_mean', 'baseline_resample', 'baseline_copula'}
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['status'] == 'completed'
    assert manifest['generation_seeds'] == [20261023, 20261024]
    assert manifest['selection']['indices'] == json.loads((cgan / 'manifest.json').read_text())['selection']['indices']
    assert set(manifest['data_hashes']) == {'train.npz', 'val.npz', 'scaler.json', 'metadata.json'}
    assert manifest['test_performance_access'] is False
    resampling = json.loads((output / 'baseline_resample.json').read_text())
    assert all(draw['diagnostics']['copying']['quantized_exact_match_fraction'] == 1.
               for draw in resampling['draws'])
    assert all(draw['existing_baseline_report_match'] for draw in resampling['draws'])
    assert json.loads((output / 'candidate_001.json').read_text())['method'] == 'conditional_flow_matching'
    with np.load(data / 'val.npz') as values:
        labels = values['y'][manifest['selection']['indices']]
    source_profiles = []
    for source_id, run, expected_kind in [('source_000', cflow, 'isotropic'),
                                           ('source_001', correlated, 'training_correlation')]:
        source_report = json.loads((output / f'{source_id}.json').read_text())
        assert source_report['source_prior_kind'] == expected_kind
        assert len(source_report['originating_candidates']) == 2
        selected = torch.load(run / 'best.pt', weights_only=True)
        expected = flow.source_samples(selected['representation'], labels, 20261023, selected['source_prior'])
        expected_hash = hashlib.sha256(expected.astype('<f8').tobytes()).hexdigest()
        assert source_report['draws'][0]['generated_profile_bytes_sha256'] == expected_hash
        source_profiles.append(expected)
    assert not np.array_equal(*source_profiles)
    original = (output / 'summary.json').read_bytes()
    with pytest.raises(FileExistsError):
        runner.evaluate(data, [cgan, cflow], output)
    assert (output / 'summary.json').read_bytes() == original
    with pytest.raises(ValueError, match='selection'):
        runner.evaluate(data, [cgan], tmp_path / 'bad_seed', seeds=(20261022,))
    threads, rng = torch.get_num_threads(), torch.get_rng_state().clone()
    def fail_generation(*args, **kwargs):
        raise FloatingPointError('synthetic failed generation')
    with monkeypatch.context() as patch:
        patch.setattr(development, 'generate', fail_generation)
        with pytest.raises(FloatingPointError, match='synthetic failed generation'):
            runner.evaluate(data, [cgan], tmp_path / 'failed')
    failed = json.loads((tmp_path / 'failed' / 'manifest.json').read_text())
    assert failed['status'] == 'failed' and failed['completed_draws'] == 0
    assert torch.get_num_threads() == threads and torch.equal(torch.get_rng_state(), rng)
    with (cgan / 'best.pt').open('ab') as handle:
        handle.write(b'tamper')
    with pytest.raises(ValueError, match='hash'):
        runner.evaluate(data, [cgan], tmp_path / 'tampered')
    assert not (tmp_path / 'tampered').exists()


def test_selected_correlated_prior_must_match_manifest_and_training_hash(tmp_path):
    data = tmp_path / 'data'
    inputs(data)
    run = tmp_path / 'correlated'
    flow.train_probe(data, run, {'steps': 1, 'validation_per_condition': 2,
                               'solver_steps': 1, 'source_prior': 'training_correlation'})
    runner = module('evaluate_development_candidates')
    record, manifest, _ = runner._candidate(run, 0)
    assert record['source_prior'] == manifest['source_prior']
    original_prior = manifest['source_prior']
    manifest['source_prior'] = flow.fit_source_prior(np.zeros((2, 24)), 'isotropic')
    (run / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='source prior'):
        runner._candidate(run, 0)
    manifest['source_prior'] = original_prior
    checkpoint = torch.load(run / 'best.pt', weights_only=True)
    checkpoint['source_prior']['fit']['training_npz_sha256'] = '0' * 64
    manifest['source_prior'] = checkpoint['source_prior']
    torch.save(checkpoint, run / 'best.pt')
    manifest['artifact_hashes']['best.pt'] = hashlib.sha256((run / 'best.pt').read_bytes()).hexdigest()
    (run / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='training hash'):
        runner._candidate(run, 0)


def test_legacy_isotropic_checkpoint_without_prior_record_is_accepted(tmp_path):
    data = tmp_path / 'data'
    inputs(data)
    run = tmp_path / 'legacy'
    flow.train_probe(data, run, {'steps': 1, 'validation_per_condition': 2, 'solver_steps': 1})
    runner = module('evaluate_development_candidates')
    modern_candidate = runner._candidate(run, 1)
    manifest = json.loads((run / 'manifest.json').read_text())
    checkpoint = torch.load(run / 'best.pt', weights_only=True)
    manifest.pop('source_prior')
    manifest['config'].pop('source_prior')
    manifest['artifact_hashes'].pop('prior_baseline.json')
    (run / 'prior_baseline.json').unlink()
    checkpoint.pop('source_prior')
    checkpoint['config'].pop('source_prior')
    torch.save(checkpoint, run / 'best.pt')
    manifest['artifact_hashes']['best.pt'] = hashlib.sha256((run / 'best.pt').read_bytes()).hexdigest()
    (run / 'manifest.json').write_text(json.dumps(manifest))
    legacy_candidate = runner._candidate(run, 0)
    record = legacy_candidate[0]
    assert record['source_prior'] is None
    controls = runner._source_controls([legacy_candidate, modern_candidate])
    assert len(controls) == 1
    assert controls[0]['source_prior_kind'] == 'isotropic'
    assert len(controls[0]['originating_candidates']) == 2
    assert (controls[0]['originating_candidates'][0]['source_prior_sha256'] !=
            controls[0]['originating_candidates'][1]['source_prior_sha256'])
    labels = np.array([0, 8])
    np.testing.assert_array_equal(
        flow.source_samples(record['representation'], labels, 81, record['source_prior']),
        flow.source_samples(modern_candidate[0]['representation'], labels, 81,
                            modern_candidate[0]['source_prior']))
    different_representation = dict(record['representation'],
                                    scale=(np.asarray(record['representation']['scale']) * 2).tolist())
    different_candidate = (dict(record, representation=different_representation),
                           legacy_candidate[1], legacy_candidate[2])
    assert len(runner._source_controls([legacy_candidate, different_candidate])) == 2
