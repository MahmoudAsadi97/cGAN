"""Plot every original-study seed's paired ramp interval from verified JSON."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEEDS = [11, 22, 33, 44, 55]
COMPARATORS = [('gaussian_copula', 'Gaussian copula', '#386FA4'),
               ('conditional_resampling', 'Conditional resampling', '#C76C2F')]
METRIC = 'ramp_wasserstein_kw'


def extract(report):
    if report.get('split') != 'val' or report.get('test_used') is not False:
        raise ValueError('only the completed validation report is accepted')
    protocol = report['paired_physical_interval_protocol']
    if (protocol['comparison'] != 'generated_minus_comparator' or
            protocol['negative_difference_favors'] != 'generated' or
            protocol['n_bootstrap'] != 500 or protocol['confidence'] != .95 or
            protocol['interval_status'] != 'complete' or protocol['n_meters'] != 353 or
            protocol['n_conditions'] != 64 or protocol['all_three_references_profiles_per_condition'] != 100):
        raise ValueError('paired primary validation protocol differs')
    runs = sorted(report['runs'], key=lambda row: row['training_seed'])
    if [row['training_seed'] for row in runs] != SEEDS:
        raise ValueError('all five distinct fixed training seeds are required')
    column = protocol['metric_order'].index(METRIC)
    physical_column = report['physical']['metric_order'].index(METRIC)
    panels = {}
    for comparator, _, _ in COMPARATORS:
        points = []
        control = report['physical']['identical_controls'][comparator][physical_column]
        for row in runs:
            item = row['paired_physical_intervals'][comparator]
            if item['valid_replicates'] != 500 or item['invalid_replicates'] or item['missing_condition_draws']:
                raise ValueError('incomplete paired interval')
            estimate, lower, upper = item['intervals'][column]
            if not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
                       for x in (estimate, lower, upper)) or lower > upper:
                raise ValueError('invalid interval values')
            expected = row['physical_neural_macros']['calibrated_cgan'][physical_column] - control
            if not math.isclose(estimate, expected, rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError('paired estimate differs from the physical macro difference')
            points.append({'seed': row['training_seed'], 'estimate': estimate, 'lower': lower, 'upper': upper,
                           'source_report_sha256': item['sha256']})
        panels[comparator] = points
    return panels


def plot(report_path, output_dir, raster_dir):
    raw = Path(report_path).read_bytes()
    report = json.loads(raw)
    panels = extract(report)
    output_dir, raster_dir = Path(output_dir), Path(raster_dir)
    svg = output_dir / 'curated_primary_ramp_uncertainty.svg'
    png = raster_dir / 'curated_primary_ramp_uncertainty.png'
    receipt = output_dir / 'curated_primary_ramp_uncertainty.provenance.json'
    for path in (svg, png, receipt):
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'svg.fonttype': 'none', 'svg.hashsalt': 'curated-paired-ramps-v1',
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.edgecolor': '#A3A8AF', 'figure.facecolor': 'white'})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.8), sharex=True, sharey=True)
    fig.subplots_adjust(left=.09, right=.97, bottom=.27, top=.72, wspace=.12)
    for ax, (key, title, color) in zip(axes, COMPARATORS):
        ax.axvline(0, color='#7C858E', linestyle=(0, (4, 3)), linewidth=1)
        for y, point in enumerate(panels[key]):
            ax.hlines(y, point['lower'], point['upper'], color=color, linewidth=2)
            ax.plot([point['lower'], point['upper']], [y, y], '|', color=color, markersize=8)
            ax.plot(point['estimate'], y, 'o', color=color, markersize=6)
        ax.set_title('Calibrated cGAN − ' + title, fontsize=11.5, loc='left', pad=12)
        ax.set_xlim(-.08, .09)
        ax.set_xticks([-.08, -.04, 0, .04, .08])
        ax.set_ylim(4.6, -.6)
        ax.set_yticks(range(5), [str(seed) for seed in SEEDS])
        ax.set_xlabel('Difference in condition-macro ramp W₁ (kW)', fontsize=9.5, labelpad=10)
        ax.grid(axis='x', color='#E8EBEE', linewidth=.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel('Training seed', labelpad=10)
    fig.suptitle('Primary validation: paired ramp uncertainty', x=.09, y=.96,
                 ha='left', fontsize=17, fontweight='semibold', color='#20262D')
    fig.text(.09, .887, 'Original SWD10 model · 64 conditions · 100 profiles per condition · 353 validation meters',
             fontsize=10.2, color='#515963')
    fig.text(.09, .827, 'Points: paired estimates. Lines: separate 95% descriptive meter-bootstrap intervals (500 draws).',
             fontsize=10, color='#343A40')
    fig.text(.09, .165, 'Negative differences favor the calibrated cGAN; positive differences favor the comparator.',
             fontsize=10, color='#343A40')
    fig.text(.09, .108, 'Models, maps and generated references are held fixed. Intervals exclude training and selection uncertainty.',
             fontsize=9.3, color='#515963')
    fig.text(.09, .055, 'No intervals are pooled or averaged. Validation remains exploratory; this figure does not establish publication readiness.',
             fontsize=9.1, color='#515963')
    digest = hashlib.sha256(raw).hexdigest()
    fig.savefig(svg, metadata={'Date': None, 'Title': 'Paired primary ramp validation',
                              'Description': 'Input SHA256 ' + digest})
    fig.savefig(png, dpi=150)
    plt.close(fig)
    provenance = {'input': str(report_path), 'input_sha256': digest, 'plotted_values': panels,
                  'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'python': platform.python_version(), 'matplotlib': matplotlib.__version__,
                  'outputs_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (svg, png)},
                  'scope': 'Original five-seed primary validation only; separate conditional intervals, no pooling.'}
    with receipt.open('x') as handle:
        json.dump(provenance, handle, indent=2)
        handle.write('\n')
    print(json.dumps({'svg': str(svg), 'png': str(png), 'provenance': str(receipt)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--raster-dir', required=True)
    args = parser.parse_args()
    plot(args.report, args.output_dir, args.raster_dir)
