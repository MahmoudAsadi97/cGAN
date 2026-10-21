"""Package existing diagnostic artifacts; no data/model loading or evaluation."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil


REPORT = Path(__file__).resolve().parent
REPOSITORY = REPORT.parents[2]
SOURCE = REPOSITORY / 'artifacts/phase1-development/fresh_diagnostics'
COPIED = ('metrics.csv', 'summary.json', 'manifest.json')
MEASURES = ('profiles', 'export_cell_fraction', 'export_day_fraction',
            'negative_daily_energy_fraction', 'mean_export_magnitude_kwh_per_day')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def variant(record):
    config = record.get('config', {})
    if record['method'] == 'conditional_cgan':
        return (f"cGAN beta1={config['beta1']:g}, dropout={config['discriminator_dropout']:g}, "
                f"energy_weight={config['energy_weight']:g}")
    if record['method'] == 'conditional_flow_matching':
        return f"flow source_prior={config['source_prior']}"
    return record['method']


def main():
    manifest = json.loads((SOURCE / 'manifest.json').read_text())
    summary = json.loads((SOURCE / 'summary.json').read_text())
    assert manifest['status'] == 'completed'
    assert manifest['completed_draws'] == 34
    assert summary['generation_seeds'] == [20261023, 20261024]
    assert len(summary['methods']) == 17
    for name in COPIED:
        if name != 'manifest.json':
            assert digest(SOURCE / name) == manifest['artifact_hashes'][name]
        shutil.copyfile(SOURCE / name, REPORT / name)
        assert (SOURCE / name).read_bytes() == (REPORT / name).read_bytes()

    inputs, rows = {}, []
    for method_id in summary['methods']:
        source = SOURCE / f'{method_id}.json'
        source_hash = digest(source)
        assert source_hash == manifest['artifact_hashes'][source.name]
        inputs[str(source.relative_to(REPOSITORY))] = source_hash
        record = json.loads(source.read_text())
        assert record['method_id'] == method_id
        assert [draw['generation_seed'] for draw in record['draws']] == summary['generation_seeds']
        for draw in record['draws']:
            pv0 = draw['diagnostics']['source_pv0']
            assert pv0['real']['profiles'] == pv0['fake']['profiles'] == 8192
            rows.append({'method_id': method_id, 'method': record['method'],
                         'variant': variant(record), 'training_seed': record['training_seed'],
                         'completed_steps': record.get('completed_steps'),
                         'selected_step': record.get('selected_step'),
                         'generation_seed': draw['generation_seed'],
                         **{f'{side}_{measure}': pv0[side][measure]
                            for side in ('real', 'fake') for measure in MEASURES},
                         'source_artifact': str(source.relative_to(REPOSITORY)),
                         'source_sha256': source_hash})
    assert len(rows) == manifest['completed_draws']
    assert len({(row['method_id'], row['generation_seed']) for row in rows}) == len(rows)
    output = REPORT / 'pv0_diagnostics.csv'
    with output.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    provenance = {
        'schema_version': 1,
        'role': 'Extraction from completed diagnostics; no new evaluation or dataset/model access',
        'source_manifest': str((SOURCE / 'manifest.json').relative_to(REPOSITORY)),
        'source_manifest_sha256': digest(SOURCE / 'manifest.json'),
        'source_evaluation_git_head': manifest['git_head'],
        'byte_identical_copies': {name: digest(REPORT / name) for name in COPIED},
        'input_artifact_sha256': inputs,
        'derivation_script': 'derive_pv0.py',
        'derivation_script_sha256': digest(Path(__file__).resolve()),
        'derived_csv': {'path': output.name, 'sha256': digest(output), 'rows': len(rows),
            'grain': 'one declared method/training-seed checkpoint or control per fresh generation draw',
            'identity_source': 'top-level method metadata in the named input JSON',
            'measure_source': 'draws[].diagnostics.source_pv0.{real,fake}',
            'proportions': 'fractions in [0,1], not percentages',
            'empty_training_fields': 'not applicable to source-only controls or directly fitted baselines',
            'profile_population': '8192 profiles per side from the 32 source-PV=0 conditions',
            'export_cell_fraction': 'fraction of hourly net-import values strictly below zero',
            'export_day_fraction': 'fraction of profiles with at least one hourly value strictly below zero',
            'negative_daily_energy_fraction': 'fraction of profiles whose 24-hour net-import sum is below zero',
            'mean_export_magnitude_kwh_per_day': 'mean across all profiles of sum_h max(-net_import_h,0); not restricted to exporting days'},
        'readme_aggregation': 'Displayed method metrics are arithmetic means over two generation draws; PV0 values are means of derived per-draw fields. Budgets/selected steps come from input JSON metadata. No refitting, reselection, or clipping.'}
    (REPORT / 'DERIVATION.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(f'Copied {len(COPIED)} byte-identical artifacts; extracted {len(rows)} PV0 rows.')


if __name__ == '__main__':
    main()
