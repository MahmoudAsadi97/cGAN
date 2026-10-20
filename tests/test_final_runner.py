"""Final-study launcher preflight and sequencing, using synthetic provenance only."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import threading

import pytest
import torch

from fluvius_cgan.study import DATA_FILES, GENERATION_SEEDS, TRAINING_SEEDS, sha256


def runner_module():
    path = Path(__file__).parents[1] / 'scripts' / 'evaluate_phase1.py'
    spec = importlib.util.spec_from_file_location('phase1_final_runner', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def prepared(tmp_path):
    data, runs = tmp_path / 'data', tmp_path / 'runs'
    data.mkdir()
    for name in DATA_FILES:
        (data / name).write_text('synthetic fixture; not actual research data: ' + name)
    hashes = {name: sha256(data / name) for name in DATA_FILES}
    base = {'epochs': 1, 'num_threads': 1}
    protocol = {'schema_version': 1, 'frozen': True, 'phase': 1, 'training_seeds': TRAINING_SEEDS,
                'generation_seeds': GENERATION_SEEDS, 'methods': ['conditional', 'unconditional', 'mean', 'resample', 'copula'],
                'projection_count': 128, 'projection_seed': 2026, 'training_config': base, 'data_hashes': hashes,
                'uncertainty': {'bootstrap_replicates': 499, 'bootstrap_seed': 20261020,
                                'exemplar_training_seed': 11, 'exemplar_generation_seed': 31001,
                                'comparators': ['resample', 'copula'],
                                'metrics': ['normalized_profile_sw1', 'daily_energy_w1_kwh'],
                                'nearest_neighbor_query_cap_per_condition': 256, 'nearest_neighbor_seed': 41001}}
    frozen = tmp_path / 'protocol.json'
    frozen.write_text(json.dumps(protocol))
    training_hashes = {name: hashes[name] for name in ['train.npz', 'val.npz', 'scaler.json', 'metadata.json']}
    for method in ['conditional', 'unconditional']:
        for seed in TRAINING_SEEDS:
            directory = runs / method / f'seed_{seed}'
            directory.mkdir(parents=True)
            config = {**base, 'seed': seed, 'unconditional': method == 'unconditional'}
            payload = {'config': config, 'epoch': 1, 'data_hashes': training_hashes, 'selection': {}}
            torch.save(payload, directory / 'best.pt')
            torch.save(payload, directory / 'last.pt')
            (directory / 'history.json').write_text(json.dumps([{'epoch': 1, 'validation_sw1': .1}]))
            manifest = {'status': 'completed', 'config': config, 'completed_epochs': 1,
                        'best_epoch': 1, 'data_hashes': training_hashes,
                        'artifact_hashes': {name: sha256(directory / name) for name in ['best.pt', 'last.pt', 'history.json']}}
            (directory / 'manifest.json').write_text(json.dumps(manifest))
    return data, frozen, runs, tmp_path / 'output'


@pytest.mark.parametrize('defect', ['failed_run', 'missing_run', 'data_hash', 'artifact', 'protocol'])
def test_preflight_rejects_before_subprocesses_or_output_creation(prepared, monkeypatch, defect):
    module = runner_module()
    data, protocol, runs, output = prepared
    path = runs / 'conditional' / 'seed_11' / 'manifest.json'
    if defect == 'failed_run':
        manifest = json.loads(path.read_text())
        manifest['status'] = 'failed'
        path.write_text(json.dumps(manifest))
    elif defect == 'missing_run':
        path.unlink()
    elif defect == 'data_hash':
        (data / 'test.npz').write_text('changed synthetic fixture')
    elif defect == 'artifact':
        (path.parent / 'best.pt').write_bytes(b'changed checkpoint')
    else:
        value = json.loads(protocol.read_text())
        value['uncertainty']['bootstrap_replicates'] = 7
        protocol.write_text(json.dumps(value))
    def unexpected(*args, **kwargs):
        pytest.fail('preflight must not launch evaluation')
    monkeypatch.setattr(module.subprocess, 'run', unexpected)
    assert module.main(['--data', str(data), '--protocol', str(protocol), '--runs', str(runs), '--output', str(output)]) == 2
    assert not output.exists()


def test_nonempty_output_is_preserved(prepared, monkeypatch):
    module = runner_module()
    data, protocol, runs, output = prepared
    output.mkdir()
    sentinel = output / 'keep.txt'
    sentinel.write_text('prior results')
    assert module.main(['--data', str(data), '--protocol', str(protocol), '--runs', str(runs), '--output', str(output)]) == 2
    assert sentinel.read_text() == 'prior results'


def test_orchestrates_thirteen_evaluations_then_summary_then_single_inference(prepared, monkeypatch):
    module = runner_module()
    data, protocol, runs, output = prepared
    calls, lock = [], threading.Lock()
    def fake_run(command, *, env, stdout, stderr, **kwargs):
        for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
            assert env[name] == '1'
        with lock:
            calls.append(command)
        if 'evaluate' in command:
            report = Path(command[command.index('--output') + 1])
            report.write_text('{}')
            assert command[command.index('--split') + 1] == 'test'
            assert command[command.index('--seeds') + 1:command.index('--output')] == ['31001', '31002', '31003']
        elif 'summarize' in command:
            assert len([c for c in calls if 'evaluate' in c]) == 13
            Path(command[command.index('--output') + 1]).write_text('{}')
        else:
            assert '--inference-only' in command
            assert (output / 'summary.json').exists()
            (output / 'inference.json').write_text('{}')
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(module.subprocess, 'run', fake_run)
    assert module.main(['--data', str(data), '--protocol', str(protocol), '--runs', str(runs), '--output', str(output)]) == 0
    assert len(calls) == 15
    assert [c for c in calls if 'summarize' in c] == [calls[-2]]
    assert '--inference-only' in calls[-1]
    record = json.loads((output / 'evaluation_execution.json').read_text())
    assert len(record['jobs']) == 13
    assert record['workers'] == 3
    assert all(job['returncode'] == 0 for job in record['jobs'])
    assert len(list((output / 'logs').glob('*.stdout.log'))) == 15


def test_failed_evaluation_retains_all_job_records_and_skips_inference(prepared, monkeypatch):
    module = runner_module()
    data, protocol, runs, output = prepared
    calls = []
    def fake_run(command, **kwargs):
        calls.append(command)
        method = command[command.index('--method') + 1]
        code = 1 if method == 'copula' else 0
        if not code:
            Path(command[command.index('--output') + 1]).write_text('{}')
        return subprocess.CompletedProcess(command, code)
    monkeypatch.setattr(module.subprocess, 'run', fake_run)
    assert module.main(['--data', str(data), '--protocol', str(protocol), '--runs', str(runs), '--output', str(output), '--workers', '2']) == 2
    assert len(calls) == 13
    records = json.loads((output / 'evaluation_execution.json').read_text())
    assert len(records['jobs']) == 13
    assert sum(job['returncode'] != 0 for job in records['jobs']) == 1
    assert not (output / 'summary.json').exists()
    assert not (output / 'inference.json').exists()


def test_training_seed_differences_are_paired_and_use_sample_sd():
    module = runner_module()
    reports = []
    for method, offset in [('conditional', 0), ('unconditional', 2)]:
        for index, seed in enumerate(TRAINING_SEEDS):
            reports.append({'method': method, 'model': {'config': {'seed': seed}},
                            'macro_summary': {'normalized_profile_sw1': {'mean': index + offset},
                                              'daily_energy_w1_kwh': {'mean': 2 * index + offset}}})
    for method in ['mean', 'resample', 'copula']:
        reports.append({'method': method, 'model': None,
                        'macro_summary': {'normalized_profile_sw1': {'mean': 0}, 'daily_energy_w1_kwh': {'mean': 0}}})
    result = module.training_seed_differences(reports, TRAINING_SEEDS)
    assert result['unconditional']['normalized_profile_sw1']['mean_difference'] == -2
    assert result['unconditional']['normalized_profile_sw1']['training_seed_sd'] == 0
    assert result['resample']['normalized_profile_sw1']['mean_difference'] == 2
    assert result['resample']['normalized_profile_sw1']['training_seed_sd'] == pytest.approx(2.5 ** .5)
    assert [row['training_seed'] for row in result['resample']['normalized_profile_sw1']['per_training_seed']] == TRAINING_SEEDS


def inference_fixture(module, prepared):
    """Saved synthetic reports and the provenance checkpoint expected by inference."""
    data, protocol, runs, output = prepared
    output.mkdir()
    (output / 'reports').mkdir()
    frozen, provenance = module.preflight(data, protocol, runs)
    (output / 'execution_plan.json').write_text(json.dumps({'provenance': provenance}))
    (output / 'summary.json').write_text('{}')
    records = []
    for name, path in module._report_paths(output).items():
        checkpoint = provenance['checkpoints'].get(name)
        method = name.split('_seed_')[0]
        seed = int(name.split('_seed_')[1]) if checkpoint else None
        report = {'method': method, 'model': {'config': {'seed': seed}} if checkpoint else None,
                  'split': 'test', 'protocol_sha256': sha256(protocol),
                  'checkpoint_sha256': checkpoint['sha256'] if checkpoint else None,
                  'generation_seeds': GENERATION_SEEDS,
                  'data_hashes': {name: frozen['data_hashes'][name] for name in ['train.npz', 'test.npz', 'scaler.json', 'metadata.json']},
                  'macro_summary': {metric: {'mean': .1} for metric in module.METRICS}}
        path.write_text(json.dumps(report))
        records.append({'name': name, 'returncode': 0, 'report_sha256': sha256(path)})
    (output / 'evaluation_execution.json').write_text(json.dumps({'jobs': records}))
    return frozen, provenance


def test_report_provenance_requires_complete_dataset_hashes(prepared):
    module = runner_module()
    _, _, _, output = prepared
    frozen, provenance = inference_fixture(module, prepared)
    path = output / 'reports' / 'mean.json'
    report = json.loads(path.read_text())
    report['data_hashes'] = {}
    path.write_text(json.dumps(report))
    execution = json.loads((output / 'evaluation_execution.json').read_text())
    for record in execution['jobs']:
        if record['name'] == 'mean':
            record['report_sha256'] = sha256(path)
    (output / 'evaluation_execution.json').write_text(json.dumps(execution))
    with pytest.raises(ValueError, match='dataset'):
        module._load_reports(output, frozen, provenance)


def test_inference_loads_data_once_normalizes_cluster_intervals_and_audits(prepared, monkeypatch):
    import argparse
    import numpy as np
    import fluvius_cgan.study as study
    import fluvius_cgan.training as training
    module = runner_module()
    data, protocol, runs, output = prepared
    frozen, _ = inference_fixture(module, prepared)
    train = {'x': np.tile(np.array([[-2.], [-1.], [1.], [2.]], dtype=np.float32), (1, 24)),
             'y': np.array([0, 1, 0, 1]), 'meter_id': np.array(['tr-a', 'tr-a', 'tr-b', 'tr-b'])}
    real = {'x': train['x'] + .25, 'y': train['y'],
            'meter_id': np.array(['te-a', 'te-a', 'te-b', 'te-b'])}
    loads, generations = [], []
    def verified(root, split):
        loads.append((root, split))
        return train, real, {'source_sha256': 'synthetic-source'}, {'minimum': -2., 'maximum': 2.}, {
            name: frozen['data_hashes'][name] for name in ['train.npz', 'test.npz', 'scaler.json', 'metadata.json']}
    def generated(checkpoint, labels, seed):
        generations.append((checkpoint, seed))
        return real['x'] + .5
    monkeypatch.setattr(study, '_verified_data', verified)
    monkeypatch.setattr(training, 'generate', generated)
    args = argparse.Namespace(data=data, protocol=protocol, runs=runs, output=output)
    module.run_inference(args)
    assert loads == [(data, 'test')]
    assert len(generations) == 2
    assert all(seed == 31001 for _, seed in generations)
    report = json.loads((output / 'inference.json').read_text())
    assert set(report['nearest_neighbor_diagnostics']) == set(module.METHODS)
    assert report['provenance']['protocol']['sha256'] == sha256(protocol)
    for result in report['paired_cluster_comparisons'].values():
        assert result['observed_difference']['normalized_profile_sw1'] == pytest.approx(
            result['observed_difference']['profile_sw1_kwh'] / 4.)
        assert result['ci95']['normalized_profile_sw1'] == pytest.approx(
            np.asarray(result['ci95']['profile_sw1_kwh']) / 4.)
        assert result['metadata']['n_bootstrap'] == 499
        assert result['metadata']['seed'] == 20261020
    events = [json.loads(line) for line in (output / 'test_access.jsonl').read_text().splitlines()]
    assert [event['event'] for event in events] == ['test_inference_started', 'test_inference_completed']
    assert all(event['protocol_sha256'] == sha256(protocol) for event in events)


def test_failed_inference_is_logged_without_replacing_reports(prepared, monkeypatch):
    import argparse
    import fluvius_cgan.study as study
    module = runner_module()
    data, protocol, runs, output = prepared
    inference_fixture(module, prepared)
    def failure(*args):
        raise ValueError('synthetic load failure')
    monkeypatch.setattr(study, '_verified_data', failure)
    with pytest.raises(ValueError, match='synthetic load failure'):
        module.run_inference(argparse.Namespace(data=data, protocol=protocol, runs=runs, output=output))
    events = [json.loads(line) for line in (output / 'test_access.jsonl').read_text().splitlines()]
    assert [event['event'] for event in events] == ['test_inference_started', 'test_inference_failed']
    assert 'synthetic load failure' in events[-1]['message']
    assert not (output / 'inference.json').exists()
    assert (output / 'summary.json').read_text() == '{}'
