"""Extract the fixed calibration result without loading data or fitted models."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np


REPORT = Path(__file__).resolve().parent
ROOT = REPORT.parents[2]
SOURCE = ROOT / 'artifacts/phase1-development/marginal_calibration'
COPIED = ('manifest.json', 'summary.json', 'metrics.csv', 'condition_diagnostics.csv')


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text())


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    manifest = read(SOURCE / 'manifest.json')
    assert manifest['status'] == 'completed' and manifest['completed_draws'] == 22
    assert manifest['calibration']['seed'] == 20261025
    assert manifest['calibration']['profiles_per_condition'] == 16384
    assert manifest['generation_seeds'] == [20261023, 20261024]
    assert manifest['test_performance_access'] is False
    for name, expected in manifest['artifact_hashes'].items():
        assert Path(name).name == name and digest(SOURCE / name) == expected, name
    summary = read(SOURCE / 'summary.json')
    assert len(summary['methods']) == 11 and summary['selection_profiles'] == 16384
    for name in COPIED:
        shutil.copyfile(SOURCE / name, REPORT / name)
        assert digest(SOURCE / name) == digest(REPORT / name)

    fit_rows, mapping_rows = [], []
    for index in range(4):
        prefix = f'candidate_{index:03d}'
        fit = read(SOURCE / f'{prefix}_fit.json')
        fit_rows.append({'method_id': prefix, 'method': fit['candidate']['method'],
            'training_seed': fit['candidate']['training_seed'],
            'selected_step': fit['candidate']['selected_step'],
            **{key: fit[key] for key in ('source_rows', 'target_rows', 'source_generation_seconds',
                'fit_seconds', 'save_and_verified_reload_seconds', 'calibration_sha256')},
            'fit_record_sha256': manifest['artifact_hashes'][f'{prefix}_fit.json']})
        raw = read(SOURCE / f'{prefix}_raw.json')
        calibrated = read(SOURCE / f'{prefix}_calibrated.json')
        assert all(draw['reference_draw_match'] for draw in raw['draws'])
        for draw in calibrated['draws']:
            diagnostic = draw['calibration_diagnostics']
            cells = diagnostic['profiles'] * diagnostic['profile_dim']
            groups = list(diagnostic['per_condition'].values())
            row = {'method_id': calibrated['method_id'], 'method': calibrated['method'],
                'training_seed': calibrated['training_seed'], 'generation_seed': draw['generation_seed'],
                **{f'{key}_fraction': count / cells for key, count in
                   diagnostic['total_coordinate_counts'].items()}}
            for atom in ('zero', 'negative', 'min', 'max'):
                differences = [np.abs(np.asarray(group[f'mapped_target_{atom}']) / group['profiles']
                    - np.asarray(group[f'training_{atom}_counts']) / group['training_profiles'])
                    for group in groups]
                row[f'{atom}_frequency_macro_absolute_error_vs_train'] = float(np.mean(differences))
            # Equal-weight condition/hour diagnostics; source atoms need not be uniform.
            row['probability_decile_macro_l1_from_uniform'] = float(np.mean([
                np.abs(np.asarray(group['probability_decile_counts']) / group['profiles'] - .1).sum(axis=1)
                for group in groups]))
            row['source_constant_condition_hours'] = sum(len(g['source_constant_hours']) for g in groups)
            row['target_constant_condition_hours'] = sum(len(g['target_constant_hours']) for g in groups)
            row['source_report_sha256'] = manifest['artifact_hashes'][f'{prefix}_calibrated.json']
            mapping_rows.append(row)
    assert len(mapping_rows) == 8
    write_csv(REPORT / 'calibration_diagnostics.csv', mapping_rows)
    write_csv(REPORT / 'calibration_fits.csv', fit_rows)
    receipt = {'schema_version': 1,
        'role': 'Extraction from completed calibration artifacts; no evaluation or dataset/model access',
        'source_manifest': str((SOURCE / 'manifest.json').relative_to(ROOT)),
        'source_manifest_sha256': digest(SOURCE / 'manifest.json'),
        'source_evaluation_git_head': manifest['git_head'],
        'copied_byte_identical': {name: digest(REPORT / name) for name in COPIED},
        'input_artifact_sha256': manifest['artifact_hashes'],
        'derivation_script_sha256': digest(Path(__file__)),
        'derived_csv_sha256': {name: digest(REPORT / name) for name in
                                ('calibration_diagnostics.csv', 'calibration_fits.csv')},
        'frequency_definition': 'Strict zero or negative and equality to fitted hourly endpoints. Values are fractions, not percentages; min/max counts overlap for constant targets.',
        'mapping_error_definition': 'Mean absolute difference between generated and fitted TRAIN frequencies over all 64 conditions and 24 hours.',
        'probability_histogram_definition': 'Mean over condition/hour of ten-bin L1 distance from .1. Descriptive only: flat bins are not expected for source atoms, and finite draw noise is not subtracted.',
        'aggregation': 'Report means average two paired generation draws within each training seed; these draws are not independent model replications.'}
    (REPORT / 'DERIVATION.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print('Verified 22 draws; copied four artifacts and derived eight mapping diagnostics and four fit records.')


if __name__ == '__main__':
    main()
