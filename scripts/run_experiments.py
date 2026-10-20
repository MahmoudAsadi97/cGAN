"""Run declared seeds with independent output directories and validation reports."""
import argparse
import json
from pathlib import Path
import sys

from fluvius_cgan.cli import evaluate_checkpoint, write_json
from fluvius_cgan.training import train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seeds', nargs='+', type=int, default=[11, 22, 33, 44, 55])
    parser.add_argument('--samples-per-condition', type=int, default=200)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error('Seeds must be unique')
    config = json.loads(Path(args.config).read_text())
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        parser.error('Output directory must be empty; do not mix experiment batches')
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'experiment.json', {'config': config, 'seeds': args.seeds,
               'selection_split': 'val', 'prepared': str(Path(args.prepared).resolve()),
               'argv': sys.argv})
    results = []
    for seed in args.seeds:
        run = output / f'seed_{seed}'
        result = train(args.prepared, run, {**config, 'seed': seed})
        report = evaluate_checkpoint(args.prepared, run / 'best.pt', split='val',
                                    samples_per_condition=args.samples_per_condition, seed=2026)
        write_json(run / 'validation_report.json', report)
        results.append({'seed': seed, 'training': result, 'metrics': report['methods']['cgan']['macro']})
        write_json(output / 'summary.json', results)
        print(f'Completed seed {seed}', flush=True)


if __name__ == '__main__':
    main()
