import csv
import importlib.util
import json
from pathlib import Path

import pytest

from fluvius_cgan.training import _sha256


def module():
    path = Path(__file__).parents[1] / 'scripts' / 'summarize_development.py'
    spec = importlib.util.spec_from_file_location('development_summary', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def write(path, value):
    path.write_text(json.dumps(value))


def test_flow_history_keeps_its_objective_separate(study):
    run, baseline, output = study
    history = json.loads((run / 'history.json').read_text())
    for row in history:
        row.pop('loss_g')
        row.pop('loss_d')
        row['loss_velocity_mse'] = .25
    write(run / 'history.json', history)
    manifest = json.loads((run / 'manifest.json').read_text())
    manifest['method'] = 'conditional_flow_matching'
    manifest['artifact_hashes']['history.json'] = _sha256(run / 'history.json')
    write(run / 'manifest.json', manifest)
    result = module().summarize([run], baseline, output)
    assert result['runs'][0]['method'] == 'conditional_flow_matching'
    with (output / 'learning_curves.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    assert all(row['loss_velocity_mse'] == '0.25' and row['loss_g'] == '' for row in rows)


@pytest.fixture
def study(tmp_path):
    run, baseline = tmp_path / 'run', tmp_path / 'baseline'
    run.mkdir()
    baseline.mkdir()
    metrics = {'normalized_profile_sw1': .2, 'daily_energy_w1_kwh': 2.,
               'diversity_ratio': .8, 'mean_profile_rmse_kwh': .3, 'profile_sw1_kwh': .4}
    diagnostics = {'condition_mean_error_rmse_kwh': .35, 'common_hour_error_fraction': .7}
    scores = {'macro': metrics, 'diagnostics': diagnostics, 'per_condition': {
        str(label): {'n_real': 2, 'n_fake': 2, 'metrics': metrics} for label in range(64)}}
    history = [{'step': step, 'row_presentations': step * 512, 'passes_started': 1,
        'elapsed_seconds': float(step), 'validation': dict(metrics, normalized_profile_sw1=score),
        'diagnostics': diagnostics, 'loss_g': .7, 'loss_d': .6}
        for step, score in [(1, .4), (2, .2), (3, .3)]]
    write(run / 'history.json', history)
    write(run / 'best_validation.json', scores)
    (run / 'best.pt').write_bytes(b'fake checkpoint never deserialized')
    (run / 'last.pt').write_bytes(b'fake checkpoint never deserialized')
    selection = {'indices': list(range(128)), 'conditions': list(range(64)),
        'index_seed': 20261021, 'generation_seed': 20261022, 'directions_seed': 2026, 'directions_count': 128}
    common = {'status': 'completed', 'data_hashes': {name: name for name in
        ('train.npz', 'val.npz', 'scaler.json', 'metadata.json')},
        'metric_normalization': {'minimum': -1., 'maximum': 1., 'fitted_on': 'train'},
        'selection': selection}
    write(run / 'manifest.json', dict(common, config={'seed': 101, 'normalization': 'standard',
        'steps': 3, 'batch_size': 512}, best_step=2, best_validation_sw1=.2,
        completed_steps=3, row_presentations=1536, elapsed_seconds=3., available_training_rows=10000,
        artifact_hashes={name: _sha256(run / name) for name in
            ('history.json', 'best_validation.json', 'best.pt', 'last.pt')}))
    for kind in ('mean', 'resample', 'copula'):
        write(baseline / f'{kind}.json', {'method': kind, 'draws': [
            {'generation_seed': 20261022, 'metrics': scores}]})
    write(baseline / 'summary.json', {})
    write(baseline / 'manifest.json', dict(common,
        artifact_hashes={name: _sha256(baseline / name) for name in
            ('mean.json', 'resample.json', 'copula.json', 'summary.json')}))
    return run, baseline, tmp_path / 'out'


def test_summary_keeps_initial_best_last_compute_and_conditions(study):
    run, baseline, output = study
    result = module().summarize([run], baseline, output)
    entry = result['runs'][0]
    assert [entry[stage]['step'] for stage in ('initial', 'best', 'last')] == [1, 2, 3]
    assert entry['best_at_budget_boundary'] is False
    assert entry['row_presentations'] == 1536
    assert entry['training_pass_equivalents'] == .1536
    with (output / 'conditions.csv').open() as handle:
        assert len(list(csv.DictReader(handle))) == 4 * 64 * 5
    assert (output / 'learning_curves.png').stat().st_size > 1000
    with pytest.raises(FileExistsError):
        module().summarize([run], baseline, output)


@pytest.mark.parametrize('defect', ['artifact', 'selection', 'data', 'status'])
def test_rejects_incomparable_or_unverified_inputs_before_output(study, defect):
    run, baseline, output = study
    if defect == 'artifact':
        (run / 'best.pt').write_bytes(b'tampered')
    else:
        path = run / 'manifest.json'
        value = json.loads(path.read_text())
        if defect == 'selection':
            value['selection']['indices'][0] = 900
        elif defect == 'data':
            value['data_hashes']['train.npz'] = 'different'
        else:
            value['status'] = 'running'
        write(path, value)
    with pytest.raises(ValueError):
        module().summarize([run], baseline, output)
    assert not output.exists()
