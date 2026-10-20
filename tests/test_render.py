"""Synthetic JSON-only smoke fixtures for the Phase 1 figure renderer."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


SPEC = importlib.util.spec_from_file_location('render_phase1', Path(__file__).parents[1] / 'scripts/render_phase1.py')
renderer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(renderer)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(tmp_path, inference=True):
    runs, evaluations = tmp_path / 'runs', tmp_path / 'evaluations'
    keys = [(method, seed) for method in ('conditional', 'unconditional') for seed in (11, 22, 33, 44, 55)]
    keys += [(method, None) for method in ('mean', 'resample', 'copula')]
    for method, seed in keys:
        method_offset = {'conditional': .2, 'unconditional': .25, 'mean': .3, 'resample': .1, 'copula': .12}[method]
        base = method_offset + (seed or 0) / 1000
        replicates = []
        for generation_seed, offset in zip((31001, 31002, 31003), (0., .001, .002)):
            conditions, curves = {}, {}
            for label in range(64):
                score = base + offset + label / 10000
                conditions[str(label)] = {'n_real': 100 + label, 'n_fake': 100 + label,
                    'metrics': {'profile_sw1_kwh': score * 10, 'daily_energy_w1_kwh': score * 20,
                                'diversity_ratio': None if method == 'mean' else 1.}}
                real = 1 + .2 * np.cos(np.arange(24) * np.pi / 12) + label / 64
                fake = real + method_offset / 3
                probabilities = [.01, .05, .5, .95, .99]
                curves[str(label)] = {'quantile_probabilities': probabilities,
                    'real_mean_kwh': real.tolist(), 'fake_mean_kwh': fake.tolist(),
                    'real_quantiles_kwh': [(real + p - .5).tolist() for p in probabilities],
                    'fake_quantiles_kwh': [(fake + p - .5).tolist() for p in probabilities],
                    'slot_w1_kwh': [method_offset / 3] * 24}
            macro = {'normalized_profile_sw1': base + offset + .00315,
                     'profile_sw1_kwh': (base + offset + .00315) * 10,
                     'daily_energy_w1_kwh': (base + offset + .00315) * 20,
                     'diversity_ratio': None if method == 'mean' else 1.}
            replicates.append({'generation_seed': generation_seed,
                'metrics': {'per_condition': conditions, 'macro': macro, 'weighted': macro}, 'curves': curves})
        summary = {}
        for metric in replicates[0]['metrics']['macro']:
            values = [rep['metrics']['macro'][metric] for rep in replicates]
            available = [v for v in values if v is not None]
            summary[metric] = {'mean': float(np.mean(available)) if available else None,
                              'sampling_sd': float(np.std(available, ddof=1)) if len(available) > 1 else 0.,
                              'valid_replicates': len(available)}
        name = f'{method}_seed_{seed}' if seed else method
        report = {'schema_version': 1, 'method': method, 'split': 'test',
            'model': {'config': {'seed': seed}} if seed else None,
            'generation_seeds': [31001, 31002, 31003], 'replicates': replicates,
            'macro_summary': summary, 'normalization': {'minimum': -2., 'maximum': 8., 'fitted_on': 'train'},
            'data_hashes': {'test.npz': 'synthetic'}, 'directions_sha256': 'synthetic',
            'protocol_sha256': 'synthetic'}
        write(evaluations / 'reports' / f'{name}.json', report)
        if seed:
            history = [{'epoch': 1, 'validation_sw1': .5}, {'epoch': 2, 'validation_sw1': None},
                       {'epoch': 3, 'validation_sw1': .4}]
            path = runs / method / f'seed_{seed}'
            write(path / 'history.json', history)
            write(path / 'manifest.json', {'status': 'completed', 'config': {'epochs': 3},
                  'completed_epochs': 3, 'best_epoch': 3,
                  'artifact_hashes': {'history.json': hashlib.sha256((path / 'history.json').read_bytes()).hexdigest()}})
    if inference:
        comparison = {'observed_difference': {'normalized_profile_sw1': .1, 'daily_energy_w1_kwh': 2.},
                      'ci95': {'normalized_profile_sw1': [.08, .12], 'daily_energy_w1_kwh': [1., 3.]},
                      'metadata': {'uncertainty_scope': 'fixed fitted models and draws', 'n_bootstrap': 499}}
        proximity = {'n_available': 100, 'n_queries': 100, 'median_rmse_kwh': .2,
                     'quantiles_rmse_kwh': {'0.05': .1, '0.25': .15, '0.50': .2, '0.75': .3, '0.95': .5},
                     'exact_match_rate': .01, 'tolerance_match_rate': .02}
        training_difference = {'per_training_seed': [
            {'training_seed': seed, 'a': .2, 'b': .1, 'difference': .1} for seed in (11, 22, 33, 44, 55)],
            'mean_difference': .1, 'training_seed_sd': 0., 'n_training_seeds': 5}
        write(evaluations / 'inference.json', {'provenance': {'protocol': {'sha256': 'synthetic'}},
              'evaluation_report_hashes': {path.stem: hashlib.sha256(path.read_bytes()).hexdigest()
                                           for path in (evaluations / 'reports').glob('*.json')},
              'exemplar': {'training_seed': 11, 'generation_seed': 31001},
              'paired_cluster_comparisons': {'resample': comparison, 'copula': comparison},
              'paired_training_seed_differences': {'resample': {'normalized_profile_sw1': training_difference}},
              'nearest_neighbor_diagnostics': {'conditional': {'per_condition': {'0': {
                  'n_train': 200, 'generated_to_train': proximity, 'heldout_to_train': proximity}},
                  'metadata': {'interpretation': 'training proximity diagnostic; no privacy guarantee'}}}})
    return runs, evaluations


def test_json_only_render_preserves_values_units_and_uncertainty_types(tmp_path, monkeypatch):
    runs, evaluations = fixture(tmp_path)
    monkeypatch.setattr(np, 'load', lambda *a, **kw: pytest.fail('renderer must never load arrays/checkpoints'))
    output = tmp_path / 'figures'
    assert renderer.main(['--runs', str(runs), '--evaluations', str(evaluations), '--output', str(output)]) == 0
    for name in ('learning_curves', 'primary_sw1', 'technology_errors', 'representative_profiles', 'paired_cluster_differences'):
        assert (output / f'{name}.png').stat().st_size > 1000
        assert (output / f'{name}.pdf').read_bytes().startswith(b'%PDF')
    with (output / 'primary_summary.csv').open() as handle:
        summary = {row['method']: row for row in csv.DictReader(handle)}
    assert float(summary['conditional']['mean_normalized_profile_sw1']) == pytest.approx(.23715)
    assert float(summary['conditional']['training_seed_sd']) == pytest.approx(np.std([.211, .222, .233, .244, .255], ddof=1))
    assert summary['resample']['training_seed_sd'] == ''
    assert float(summary['resample']['sampling_sd']) == pytest.approx(.001)
    with (output / 'condition_metrics.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 13 * 64 and {int(row['condition']) for row in rows} == set(range(64))
    first = next(row for row in rows if row['method'] == 'conditional' and row['training_seed'] == '11' and row['condition'] == '0')
    assert float(first['normalized_profile_sw1']) == pytest.approx(.212)
    manifest = json.loads((output / 'render_manifest.json').read_text())
    assert manifest['representative_conditions'] == list(range(0, 64, 8))
    assert manifest['profile_generation_seed'] == 31001
    assert manifest['source_access'] == 'saved JSON reports only; no profile arrays or checkpoints'
    assert len(manifest['evaluation_reports']) == 13
    with (output / 'nearest_neighbor_diagnostics.csv').open() as handle:
        proximity = list(csv.DictReader(handle))
    assert len(proximity) == 2
    assert {r['sample'] for r in proximity} == {'generated_to_train', 'heldout_to_train'}
    assert float(proximity[0]['rmse_kwh_quantile_0.95']) == .5
    assert float(proximity[0]['exact_match_rate']) == .01
    with (output / 'paired_training_seed_differences.csv').open() as handle:
        assert len(list(csv.DictReader(handle))) == 5


def test_missing_condition_fails_clearly_before_output(tmp_path):
    runs, evaluations = fixture(tmp_path)
    report = evaluations / 'reports/conditional_seed_11.json'
    payload = json.loads(report.read_text())
    del payload['replicates'][0]['metrics']['per_condition']['63']
    write(report, payload)
    output = tmp_path / 'bad'
    with pytest.raises(ValueError, match='64|condition'):
        renderer.render(runs, evaluations, output)
    assert not output.exists()


def test_existing_output_refused_and_inference_is_optional(tmp_path):
    runs, evaluations = fixture(tmp_path, inference=False)
    output = tmp_path / 'existing'
    output.mkdir()
    with pytest.raises(FileExistsError):
        renderer.render(runs, evaluations, output)


def test_unrelated_inference_protocol_is_rejected_before_output(tmp_path):
    runs, evaluations = fixture(tmp_path)
    path = evaluations / 'inference.json'
    inference = json.loads(path.read_text())
    inference['provenance']['protocol']['sha256'] = 'different experiment'
    write(path, inference)
    output = tmp_path / 'bad'
    with pytest.raises(ValueError, match='protocol'):
        renderer.render(runs, evaluations, output)
    assert not output.exists()
