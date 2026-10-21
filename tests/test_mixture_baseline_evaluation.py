"""Mixture screens reuse verified reports and fit only the declared TRAIN arms."""
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest
import torch

from test_strong_baseline_evaluation import study as previous_study, module, read, digest


ROOT = Path(__file__).resolve().parents[1]


def runner():
    path = ROOT / 'scripts/evaluate_mixture_baselines.py'
    assert path.exists(), 'The plan-bound mixture evaluator is not implemented'
    return module('evaluate_mixture_baselines')


@pytest.fixture(scope='module')
def study(previous_study):
    root, previous_plan = previous_study
    reference = root / 'strong_reference'
    module('evaluate_strong_baselines').evaluate(previous_plan, reference)
    prior = read(previous_plan)
    plan = {'schema_version': 1, 'phase': 1, 'data': prior['data'],
        'data_hashes': prior['data_hashes'], 'generation_seeds': prior['generation_seeds'],
        'reference': {'directory': str(reference), 'manifest_sha256': digest(reference / 'manifest.json')},
        'methods': [{'method_id': f'gmm_k{k}', 'n_components': k, 'fit_seed': 20261027,
                     'n_init': 1, 'max_iter': 3, 'tol': .001, 'reg_covar': .001} for k in (1, 2)]}
    path = root / 'mixture_plan.json'
    path.write_text(json.dumps(plan))
    return root, path


def test_fitted_models_reload_and_historical_reports_are_reused_without_generation(study, tmp_path, monkeypatch):
    root, plan = study
    evaluate = runner()
    from fluvius_cgan import development, flow
    def forbidden(*args, **kwargs):
        raise AssertionError('Historical generators and calibrations must not be refitted or regenerated')
    monkeypatch.setattr(development, 'generate', forbidden)
    monkeypatch.setattr(flow, 'generate', forbidden)
    monkeypatch.setattr(evaluate.strong, 'fit_baseline', forbidden)
    monkeypatch.setattr(evaluate.strong, 'fit_strong_baseline', forbidden)
    output = tmp_path / 'result'
    rng, threads = torch.get_rng_state().clone(), torch.get_num_threads()
    numpy_rng = np.random.get_state()
    summary = evaluate.evaluate(plan, output, workers=1)
    manifest = read(output / 'manifest.json')
    reference = root / 'strong_reference'
    previous = read(reference / 'summary.json')
    assert set(summary['methods']) == set(previous['methods']) | {'gmm_k1', 'gmm_k2'}
    assert manifest['status'] == 'completed' and manifest['completed_draws'] == 4
    assert manifest['reused_draws'] == 22
    assert manifest['test_performance_access'] is False and manifest['historical_models_regenerated'] is False
    assert manifest['physical_clipping'] is False
    assert manifest['probability_roundoff_clamp']['interval'] == [0., 1.]
    assert manifest['plan_sha256'] == digest(plan)
    assert manifest['artifact_hashes'] == {p.name: digest(p) for p in output.iterdir()
                                          if p.is_file() and p.name != 'manifest.json'}
    for name in previous['methods']:
        assert (output / f'{name}.json').read_bytes() == (reference / f'{name}.json').read_bytes()
        assert summary['methods'][name] == previous['methods'][name]
    with np.load(root / 'data/train.npz') as archive:
        training, train_labels = archive['x'], archive['y']
    with np.load(root / 'data/val.npz') as archive:
        labels = archive['y'][manifest['selection']['indices']]
    for name in ('gmm_k1', 'gmm_k2'):
        report = read(output / f'{name}.json')
        assert report['status'] == 'completed'
        assert report['data_hashes'] == read(plan)['data_hashes']
        assert report['model_sha256'] == digest(output / report['model_artifact'])
        fitted = evaluate.load_mixture_baseline(output / report['model_artifact'], report['model_sha256'])
        assert fitted.metadata() == report['fit_metadata']
        for draw in report['draws']:
            fake = fitted.sample(labels, draw['generation_seed'])
            assert draw['generated_profile_bytes_sha256'] == evaluate.diagnostics._digest(fake, '<f8')
            for condition in np.unique(labels):
                for hour in range(24):
                    assert np.isin(fake[labels == condition, hour], training[train_labels == condition, hour]).all()
            assert draw['proximity']['macro']['mean_rms_kwh'] >= 0
            for condition in np.unique(labels):
                support = draw['marginal_support']['per_condition'][str(int(condition))]
                values, target = fake[labels == condition], training[train_labels == condition]
                assert support['profiles'] == len(values) and support['training_profiles'] == len(target)
                assert support['zero_counts'] == (values == 0).sum(axis=0).tolist()
                assert support['negative_counts'] == (values < 0).sum(axis=0).tolist()
                assert support['at_training_min_counts'] == (values == target.min(axis=0)).sum(axis=0).tolist()
                assert support['at_training_max_counts'] == (values == target.max(axis=0)).sum(axis=0).tolist()
                assert support['outside_training_support_counts'] == [0] * 24
    assert torch.equal(rng, torch.get_rng_state()) and torch.get_num_threads() == threads
    after = np.random.get_state()
    assert numpy_rng[0] == after[0] and np.array_equal(numpy_rng[1], after[1]) and numpy_rng[2:] == after[2:]
    with pytest.raises(FileExistsError):
        evaluate.evaluate(plan, output, workers=1)


def test_process_workers_retain_the_same_draws_as_single_worker(study, tmp_path):
    _, plan = study
    evaluate = runner()
    one, parallel = tmp_path / 'one', tmp_path / 'parallel'
    first = evaluate.evaluate(plan, one, workers=1)
    second = evaluate.evaluate(plan, parallel, workers=2)
    assert first['methods'] == second['methods']
    for name in ('gmm_k1', 'gmm_k2'):
        left, right = read(one / f'{name}.json'), read(parallel / f'{name}.json')
        assert left['fit_metadata'] == right['fit_metadata']
        assert [d['generated_profile_bytes_sha256'] for d in left['draws']] == [d['generated_profile_bytes_sha256'] for d in right['draws']]


def test_process_failure_does_not_cancel_another_declared_arm(study, tmp_path):
    _, original = study
    plan = read(original)
    plan['methods'][1]['n_components'] = 1000
    path = tmp_path / 'oversized.json'
    path.write_text(json.dumps(plan))
    output = tmp_path / 'parallel_failure'
    with pytest.raises(RuntimeError, match='failed'):
        runner().evaluate(path, output, workers=2)
    assert read(output / 'gmm_k1.json')['status'] == 'completed'
    assert read(output / 'gmm_k2.json')['status'] == 'failed'
    assert read(output / 'gmm_k1_worker.json')['returncode'] == 0
    assert read(output / 'gmm_k2_worker.json')['returncode'] != 0
    manifest = read(output / 'manifest.json')
    assert manifest['completed_draws'] == 2
    assert manifest['artifact_hashes'] == {p.name: digest(p) for p in output.iterdir()
        if p.is_file() and p.name != 'manifest.json'}


@pytest.mark.parametrize('mutation,match', [
    (lambda p: p['data_hashes'].update({'train.npz': '0' * 64}), 'data hash'),
    (lambda p: p['reference'].update(manifest_sha256='0' * 64), 'reference manifest'),
    (lambda p: p.update(generation_seeds=[20261022]), 'seed|draw'),
    (lambda p: p['methods'][0].update(fit_seed=20261023), 'fit seed'),
    (lambda p: p['methods'][0].update(n_components=0), 'n_components'),
    (lambda p: p['methods'][0].update(method_id='../gmm'), 'method_id'),
    (lambda p: p['methods'].append(p['methods'][0].copy()), 'distinct'),
])
def test_bad_plan_is_rejected_before_output(study, tmp_path, mutation, match):
    _, original = study
    plan = read(original)
    mutation(plan)
    path = tmp_path / 'changed.json'
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match=match):
        runner().evaluate(path, tmp_path / 'rejected')
    assert not (tmp_path / 'rejected').exists()


def test_reference_corruption_and_rehashed_draw_identity_are_rejected(study, tmp_path):
    root, original = study
    reference = tmp_path / 'reference'
    shutil.copytree(root / 'strong_reference', reference)
    report = read(reference / 'baseline_mean.json')
    report['method_id'] = 'baseline_resample'
    (reference / 'baseline_mean.json').write_text(json.dumps(report))
    manifest = read(reference / 'manifest.json')
    manifest['artifact_hashes']['baseline_mean.json'] = digest(reference / 'baseline_mean.json')
    (reference / 'manifest.json').write_text(json.dumps(manifest))
    plan = read(original)
    plan['reference'] = {'directory': str(reference), 'manifest_sha256': digest(reference / 'manifest.json')}
    path = tmp_path / 'changed.json'
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match='identity'):
        runner().evaluate(path, tmp_path / 'rejected')
    assert not (tmp_path / 'rejected').exists()


def test_failed_arm_retains_receipts_and_later_declared_arm_finishes(study, tmp_path, monkeypatch):
    _, plan = study
    evaluate = runner()
    original = evaluate.load_mixture_baseline
    def corrupt_first(path, expected_sha256):
        if Path(path).name.startswith('gmm_k1_'):
            with Path(path).open('ab') as handle:
                handle.write(b'corruption')
        return original(path, expected_sha256)
    monkeypatch.setattr(evaluate, 'load_mixture_baseline', corrupt_first)
    output = tmp_path / 'failed'
    with pytest.raises(RuntimeError, match='failed'):
        evaluate.evaluate(plan, output, workers=1)
    manifest = read(output / 'manifest.json')
    assert manifest['status'] == 'failed'
    assert read(output / 'gmm_k1.json')['status'] == 'failed'
    assert read(output / 'gmm_k2.json')['status'] == 'completed'
    assert read(output / 'summary.json')['methods']['gmm_k1']['status'] == 'failed'
    assert manifest['artifact_hashes'] == {p.name: digest(p) for p in output.iterdir()
        if p.is_file() and p.name != 'manifest.json'}


def test_receipt_directory_error_does_not_mask_the_original_arm_failure(study, tmp_path, monkeypatch):
    _, original = study
    plan = read(original)
    plan['methods'][1]['n_components'] = 1000
    path = tmp_path / 'oversized.json'
    path.write_text(json.dumps(plan))
    output = tmp_path / 'failed_receipt'
    iterdir = Path.iterdir
    def denied(directory):
        if directory == output:
            raise PermissionError('synthetic receipt listing error')
        return iterdir(directory)
    monkeypatch.setattr(Path, 'iterdir', denied)
    with pytest.raises(RuntimeError, match='Declared mixture arms failed'):
        runner().evaluate(path, output, workers=1)
    manifest = read(output / 'manifest.json')
    assert manifest['status'] == 'failed' and 'Declared mixture arms failed' in manifest['error']
    assert 'synthetic receipt listing error' in manifest['artifact_hash_errors']['<directory>']


def test_worker_launch_error_still_collects_completed_sibling(study, tmp_path, monkeypatch):
    _, plan = study
    evaluate = runner()
    run = evaluate.subprocess.run
    def fail_first(command, **kwargs):
        if '--worker' in command and str(command[-1]).endswith('gmm_k1_job.json'):
            raise OSError('synthetic worker launch failure')
        return run(command, **kwargs)
    monkeypatch.setattr(evaluate.subprocess, 'run', fail_first)
    output = tmp_path / 'launch_failure'
    with pytest.raises(RuntimeError, match='Declared mixture arms failed'):
        evaluate.evaluate(plan, output, workers=2)
    manifest = read(output / 'manifest.json')
    assert manifest['completed_draws'] == 2
    assert read(output / 'gmm_k2.json')['status'] == 'completed'
    summary = read(output / 'summary.json')
    assert 'synthetic worker launch failure' in summary['methods']['gmm_k1']['error']
    assert 'metrics' in summary['methods']['gmm_k2']
