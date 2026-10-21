"""Extract reflection reports from completed JSON artifacts; never load data/models."""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[3]
SCREEN = ROOT / 'runs/phase1-development/reflection_screen'
FRESH = ROOT / 'artifacts/phase1-development/reflection_diagnostics'
SCREEN_OUT = ROOT / 'reports/phase1-development/reflection_screen'
FRESH_OUT = ROOT / 'reports/phase1-development/reflection_diagnostics'
METRICS = ('normalized_profile_sw1', 'daily_energy_w1_kwh', 'peak_w1_kw',
           'ramp_w1_kw', 'acf_lag1_w1', 'acf_lag6_w1', 'diversity_ratio')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path, inputs, expected=None):
    digest = sha(path)
    if expected is not None and digest != expected:
        raise ValueError(f'Artifact hash mismatch: {path}')
    inputs[str(path.relative_to(ROOT))] = digest
    return json.loads(path.read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def write_csv(path, rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    path.write_bytes(stream.getvalue().encode())


def probability_summary(diagnostic):
    probability = diagnostic['probabilities']
    groups = list(probability['per_condition'].values())
    coordinates = sum(g['profiles'] * len(g['mean']) for g in groups)
    result = {
        'probability_macro_decile_l1': probability['macro_decile_l1_from_uniform'],
        'probability_macro_mean_absolute_error_from_half': probability['macro_mean_absolute_error_from_half'],
        'probability_mean_coordinate_variance': statistics.mean(v for g in groups for v in g['population_variance']),
        'probability_exact_zero_count': sum(sum(g['zero_counts']) for g in groups),
        'probability_exact_one_count': sum(sum(g['one_counts']) for g in groups),
        'coordinate_count': coordinates,
        'maximum_step_reflection_coordinate_fraction': max(diagnostic['solver']['reflection_coordinate_fractions']),
    }
    for b in range(10):
        result[f'pooled_probability_decile_{b}_fraction'] = sum(
            counts[b] for g in groups for counts in g['decile_counts']) / coordinates
    return result


def table(rows, fields):
    return '\n'.join(['| ' + ' | '.join(label for _, label in fields) + ' |',
                      '| ' + ' | '.join('---' for _ in fields) + ' |'] + [
        '| ' + ' | '.join(str(row[key]) if isinstance(row[key], str)
                          else f'{row[key]:.6g}' for key, _ in fields) + ' |'
        for row in rows])


def main():
    SCREEN_OUT.mkdir(parents=True, exist_ok=True)
    FRESH_OUT.mkdir(parents=True, exist_ok=True)
    screen_inputs, fresh_inputs = {}, {}
    plan = read(SCREEN / 'plan.json', screen_inputs)
    execution = read(SCREEN / 'execution.json', screen_inputs)
    manifest = read(FRESH / 'manifest.json', fresh_inputs)
    if execution['status'] != 'completed' or manifest['status'] != 'completed':
        raise ValueError('Only completed experiments may be summarized')
    if manifest['completed_draws'] != 22:
        raise ValueError('Expected all 22 declared draws')
    summary = read(FRESH / 'summary.json', fresh_inputs, manifest['artifact_hashes']['summary.json'])
    fresh_reports = {name: read(FRESH / f'{name}.json', fresh_inputs,
                               manifest['artifact_hashes'][f'{name}.json'])
                     for name in summary['methods']}
    solver_rows, curve_rows, selection_rows, screen_methods = [], [], [], {}
    for name in plan['jobs']:
        directory = SCREEN / name
        run = read(directory / 'manifest.json', screen_inputs)
        if run['status'] != 'completed' or run['completed_steps'] != 6000:
            raise ValueError(f'Incomplete declared run: {name}')
        payloads = {filename: read(directory / filename, screen_inputs, run['artifact_hashes'][filename])
                    for filename in ('solver_sensitivity.json', 'history.json',
                                     'best_probability_diagnostics.json', 'best_validation.json',
                                     'independent_marginals.json')}
        sensitivity = payloads['solver_sensitivity.json']
        selected = sensitivity['selection_solver']['metrics']['macro']
        refined = sensitivity['sensitivity_solver']['metrics']['macro']
        if sensitivity['selection_solver']['diagnostics'] != payloads['best_probability_diagnostics.json']:
            raise ValueError('Solver diagnostic does not match selected checkpoint')
        method_id = next(key for key, value in fresh_reports.items()
                         if value.get('run_directory') == str(directory))
        fresh = summary['methods'][method_id]
        selection_rows.append({'run': name, 'method_id': method_id,
            'training_seed': run['config']['seed'], 'learning_rate': run['config']['learning_rate'],
            'selected_step': run['best_step'], 'selection_sw1': run['best_validation_sw1'],
            'paired_sampling_mean_sw1': fresh['metrics']['normalized_profile_sw1']['mean'],
            'solver_128_sw1': refined['normalized_profile_sw1'],
            'solver_128_relative_sw1_change': refined['normalized_profile_sw1'] / selected['normalized_profile_sw1'] - 1})
        for solver_name in ('selection_solver', 'sensitivity_solver'):
            value = sensitivity[solver_name]
            solver_rows.append({'run': name, 'solver_steps': value['steps'],
                **{k: value['metrics']['macro'][k] for k in METRICS},
                **probability_summary(value['diagnostics'])})
        for entry in payloads['history.json']:
            curve_rows.append({'run': name, 'step': entry['step'],
                'row_presentations': entry['row_presentations'],
                'loss_velocity_mse': entry['loss_velocity_mse'],
                'probability_macro_decile_l1': entry['probability_macro_decile_l1'],
                **{k: entry['validation'][k] for k in METRICS}})
        screen_methods[name] = {'method_id': method_id, 'config': run['config'],
            'best_step': run['best_step'], 'completed_steps': run['completed_steps'],
            'row_presentations': run['row_presentations'], 'architecture': run['architecture'],
            'solver': run['solver'], 'training_git_head': run['git_head'],
            'checkpoint_sha256': run['artifact_hashes']['best.pt'],
            'selected_metrics': selected, 'refined_solver_metrics': refined,
            'paired_generated_profile_change': sensitivity['paired_generated_profile_change'],
            'last_loss_by_time_bin': payloads['history.json'][-1]['loss_by_time_bin'],
            'independent_marginals_selection_metrics': payloads['independent_marginals.json']['metrics']['macro']}
    write_json(SCREEN_OUT / 'summary.json', {'schema_version': 1,
        'role': 'Post-test development; extraction from saved reflection screen diagnostics',
        'selection_generation_seed': 20261022, 'paired_sampling_generation_seeds': [20261023, 20261024],
        'selection_rows_already_used': True, 'solver_sensitivity_used_for_selection': False,
        'methods': screen_methods})
    write_csv(SCREEN_OUT / 'selection_and_sampling.csv', selection_rows)
    write_csv(SCREEN_OUT / 'solver_diagnostics.csv', solver_rows)
    write_csv(SCREEN_OUT / 'training_curves.csv', curve_rows)
    copied_screen = {}
    for name in ('plan.json', 'execution.json'):
        (SCREEN_OUT / name).write_bytes((SCREEN / name).read_bytes())
        copied_screen[name] = sha(SCREEN_OUT / name)
    copied_fresh = {}
    for name in ('manifest.json', 'summary.json', 'metrics.csv', 'condition_diagnostics.csv'):
        source = FRESH / name
        expected = manifest['artifact_hashes'].get(name)
        if expected is not None and sha(source) != expected:
            raise ValueError(f'Artifact hash mismatch: {source}')
        fresh_inputs[str(source.relative_to(ROOT))] = sha(source)
        (FRESH_OUT / name).write_bytes(source.read_bytes())
        copied_fresh[name] = sha(FRESH_OUT / name)
    pv_rows, joint_rows = [], []
    labels = {f'candidate_{i:03d}': name.replace('reflection_', 'Reflection ').replace('_', ' ')
              for i, name in enumerate(plan['jobs'])}
    labels.update({'candidate_004': 'Basic flow 6k seed101', 'candidate_005': 'Basic flow 6k seed202',
                   'source_000': 'Isotropic source', 'independent_000': 'Independent marginals',
                   'baseline_mean': 'Conditional mean', 'baseline_resample': 'Day resampling',
                   'baseline_copula': 'Gaussian copula'})
    for method_id, report in fresh_reports.items():
        for draw in report['draws']:
            pv = draw['diagnostics']['source_pv0']
            pv_rows.append({'method_id': method_id, 'generation_seed': draw['generation_seed'],
                **{f'{side}_{key}': value for side in ('real', 'fake') for key, value in pv[side].items()}})
        entry = summary['methods'][method_id]
        joint_rows.append({'method_id': method_id, 'method': labels[method_id],
            **{k: entry['metrics'][k]['mean'] for k in METRICS},
            **{k: v['mean'] for k, v in entry['tail_errors'].items()},
            'pv0_negative_day_percent': 100 * statistics.mean(
                d['diagnostics']['source_pv0']['fake']['export_day_fraction'] for d in report['draws']),
            'quantized_exact_training_match_fraction': entry['quantized_exact_training_match_fraction']['mean']})
    write_csv(FRESH_OUT / 'pv0_diagnostics.csv', pv_rows)
    write_csv(FRESH_OUT / 'joint_summary.csv', joint_rows)
    screen_readme = '''# Conditional reflection screen: all four declared arms

The direct conditional reflection adapter did not improve the existing strong
baselines. All four 6,000-update arms and selected checkpoints are retained.
The larger learning rate produced better physical distributions than the smaller rate,
but its two seeds differ substantially. This is a small conditional adaptation
of the inspected construction, not a reproduction or refutation of the author
model. See [design](../../../docs/DEPENDENCE_COMPARISON_PLAN.md).

These are previously used inner-TRAIN/inner-VAL development data. Selection uses
generation seed 20261022; paired sampling diagnostics use 20261023/20261024 on
the same 16,384 real rows. No new meters, independent confirmation or original
TEST performance are involved. The 128-step solver was not used for selection.

''' + table(selection_rows, [('run', 'Run'), ('selected_step', 'Selected update'),
    ('selection_sw1', 'Selection SW1'), ('paired_sampling_mean_sw1', 'Paired-draw SW1'),
    ('solver_128_sw1', '128-step selection-draw SW1')]) + '''

Doubling reflected Euler from 64 to 128 steps worsens SW1 in every arm by
8.8–15.9%, increases diversity and worsens peaks/ramps. Common-noise probability
RMSE changes are only 0.00048–0.00196, while decoded physical RMSE is
0.217–0.339 kWh and maximum coordinate changes are 11.19–13.81 kWh. The physical
sampler cannot be treated as numerically converged. A finer solver is not a
validated improvement in distribution fidelity here.

At the selected 64-step higher-rate checkpoints, pooled upper probability
deciles contain only 5.50%, against 10% under uniform marginals; middle bins
contain about 12–13%. The lower-rate arms have less conspicuous decile errors
but much worse physical peaks and spread. Coarse deciles do not diagnose tiny
tail intervals. Exact zero/one probabilities are mostly absent, yet up to
3.2–6.9% of coordinate proposals require reflection at a solver step; similar
fractions persist at 128 steps. These counts concern proposals, not unique
profiles or wall-crossing multiplicity. They suggest checking near-wall mass
and decoder tail amplification, without identifying the cause by themselves.

Loss falls toward 0.95 (higher rate) or 0.985 (lower rate); this MSE includes
irreducible velocity variation and is not a fidelity score. Selected steps are
4k/6k at the higher rate and 5k/6k at the lower rate. The learning curves and
solver diagnostics do not justify extending only the favorable seed.

The next defensible diagnostic is near-wall probability/decoded-endpoint
behavior at fixed checkpoints. Any calibration combination or zero-normal-
velocity boundary constraint must be a separately declared comparison, retain
these direct outputs, and include both seeds. It cannot retroactively establish
solver convergence. No result here establishes publication readiness.

`summary.json`, `selection_and_sampling.csv`, `solver_diagnostics.csv` and
`training_curves.csv` are deterministic extracts. `plan.json` and
`execution.json` are byte-identical copies. `DERIVATION.json` binds all inputs
and generated files. Run `python reports/phase1-development/reflection_screen/derive_reports.py`
from the repository to reproduce both reflection report directories without
loading datasets, checkpoints or generating samples.
'''
    (SCREEN_OUT / 'README.md').write_text(screen_readme)
    fresh_readme = '''# Reflection comparator: paired generation diagnostics

All 22 declared draws completed: four reflection arms, two matched 6k basic
flows, an isotropic source control, independent empirical marginals, and three
classical baselines, each with generation seeds 20261023 and 20261024. These
reuse the existing 16,384 inner-validation rows and previously used generation
seeds. They are paired development diagnostics, not unseen validation or
independent training replications. All reported means average two sampling
draws; the saved sampling SD is not a confidence interval.

Energy W1 is in kWh; peak and ramp W1 are in kW. SW1 is divided by the fixed
TRAIN range, 33.619001388549805 kWh; ACF errors and diversity are dimensionless.

''' + table(joint_rows, [('method', 'Method'), ('normalized_profile_sw1', 'SW1'),
    ('daily_energy_w1_kwh', 'Energy W1'), ('peak_w1_kw', 'Peak W1'),
    ('ramp_w1_kw', 'Ramp W1'), ('acf_lag1_w1', 'Lag-1 ACF W1'),
    ('diversity_ratio', 'Diversity')]) + '''

The best reflection arm has SW1 0.007024, versus 0.005816–0.005855 for the
matched-budget basic flows, 0.005509 for Gaussian copula and 0.005272 for
day resampling. The two lower-rate reflection arms (about 0.0141) are also worse
than independent marginals (0.008477). The higher-rate reflection models improve
ramps, ACF and energy relative to independent marginals, showing some learned
dependence; this does not make the complete generator competitive.

Reflection's average ramp W1 at the higher rate (0.180–0.192) is slightly below
basic flow (0.194–0.198), but absolute-ramp quantile errors are 0.666–0.747,
versus 0.194–0.206 for basic flow and 0.126 for resampling. Peak quantile errors
are also much worse. The limited aggregate ramp advantage therefore does not
establish good ramp tails. Reflection lag-6 ACF is comparatively close to the
strong baselines at the higher rate; lag-1 ACF and covariance remain worse.

Basic flow still has excess PV0 negative-day frequencies (84.1–88.0%). Reflection
reduces this to 0.0183–0.0732%, compared with 0.00610% for independent marginals,
Gaussian copula and resampling. Rare negative TRAIN values are retained by the
decoder; neither zero errors nor exact rare-event frequencies are established.
The selected real PV0 sample has no negative hours. Reflection's empirical
hourly support therefore improves this defect while leaving joint and marginal
frequency errors. Physical support alone is not marginal calibration.

Reflection diversity ratios are 1.80/2.16 (higher rate) and 3.43/3.57 (lower
rate), far above the empirical controls' 1.25–1.28. Those controls themselves
exceed one on this cohort; one is not an automatic failure threshold. Here
reflection's excess spread is accompanied by much worse peaks and covariance.

The [selection and solver report](../reflection_screen/README.md) records the
64-to-128-step sensitivity and probability-marginal defects. The present screen
supports targeted numerical/marginal diagnosis before broad longer training.
It does not establish that reflection methods generally fail, nor superiority,
privacy, appliance disaggregation or publication readiness. Zero exact TRAIN
row matches for the learned models are only a precision-specific copy check;
resampling has exact match fraction one by construction.

`manifest.json`, `summary.json`, `metrics.csv` and `condition_diagnostics.csv`
are byte-identical copies of completed evaluator outputs. `joint_summary.csv`
and `pv0_diagnostics.csv` are deterministic extracts. `DERIVATION.json` records
their source hashes and the shared derivation script. No dataset/model access
or new evaluation is needed to reproduce these reports.
'''
    (FRESH_OUT / 'README.md').write_text(fresh_readme)
    for output, inputs, copied, role in [(SCREEN_OUT, screen_inputs, copied_screen, 'screen'),
                                        (FRESH_OUT, fresh_inputs, copied_fresh, 'paired diagnostics')]:
        receipt = {'schema_version': 1,
            'role': f'Deterministic extraction of reflection {role}; no dataset/model access or new experiments',
            'input_sha256': dict(sorted({**screen_inputs, **fresh_inputs}.items())),
            'copied_byte_identical': copied,
            'derivation_script': str(Path(__file__).resolve().relative_to(ROOT)),
            'derivation_script_sha256': sha(Path(__file__)),
            'output_sha256': {p.name: sha(p) for p in sorted(output.iterdir())
                              if p.is_file() and p.name not in ('DERIVATION.json', 'derive_reports.py')},
            'aggregation': 'Means over two paired sampling draws within a training seed; no population uncertainty claim.',
            'probability_diagnostics_scope': 'Selected checkpoint and selection seed 20261022 at 64/128 steps; not fresh-draw probability diagnostics.'}
        write_json(output / 'DERIVATION.json', receipt)
    print('Wrote both reflection reports from verified completed JSON artifacts.')


if __name__ == '__main__':
    main()
