"""Command line entry points for source audit, preparation and experiments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from .release import audit_archive, download_release, verify_archive, sha256_file


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    download = commands.add_parser('download', help='Download and SHA-256 verify pinned raw.zip')
    download.add_argument('--output', default='data/raw.zip')
    audit = commands.add_parser('audit', help='Inspect archive members and headers; no extraction')
    audit.add_argument('--archive', required=True)
    audit.add_argument('--output', default='reports/archive_audit.json')
    prepare = commands.add_parser('prepare', help='Prepare meter-disjoint daily profiles')
    prepare.add_argument('--archive', required=True)
    prepare.add_argument('--output', default='data/prepared96')
    prepare.add_argument('--resolution', type=int, choices=[24, 96], default=96)
    prepare.add_argument('--target', choices=['net', 'import', 'injection'], default='net')
    prepare.add_argument('--seed', type=int, default=42)
    prepare.add_argument('--chunksize', type=int, default=250000)
    prepare.add_argument('--max-meters-per-group', type=int)
    train = commands.add_parser('train', help='Train using train and validation only')
    train.add_argument('--prepared', required=True)
    train.add_argument('--output', required=True)
    train.add_argument('--config', required=True, help='JSON training configuration')
    train.add_argument('--seed', type=int)
    generate = commands.add_parser('generate', help='Generate profiles in kWh per interval')
    generate.add_argument('--checkpoint', required=True)
    generate.add_argument('--labels', nargs='+', type=int, required=True)
    generate.add_argument('--samples-per-label', type=int, default=100)
    generate.add_argument('--seed', type=int, default=42)
    generate.add_argument('--output', required=True)
    evaluate = commands.add_parser('evaluate', help='Compare model and training-only baselines')
    evaluate.add_argument('--prepared', required=True)
    evaluate.add_argument('--checkpoint', required=True)
    evaluate.add_argument('--output', required=True)
    evaluate.add_argument('--split', choices=['val', 'test'], default='val')
    evaluate.add_argument('--samples-per-condition', type=int, default=200)
    evaluate.add_argument('--seed', type=int, default=2026)
    return parser


def evaluate_checkpoint(prepared, checkpoint, *, split='val', samples_per_condition=200, seed=2026):
    from .training import generate, _read_split
    from .evaluation import (evaluate_conditional, resample_baseline, mean_baseline,
                             nearest_neighbor_diagnostics)
    import torch
    if split not in ('val', 'test'):
        raise ValueError('Evaluation split must be val or test')
    if samples_per_condition < 1:
        raise ValueError('samples_per_condition must be positive')
    prepared = Path(prepared)
    metadata = json.loads((prepared / 'metadata.json').read_text())
    # Bind evaluation to the exact training dataset recorded in the checkpoint.
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    train_data, digest = _read_split(prepared / 'train.npz')
    heldout_data, heldout_digest = _read_split(prepared / f'{split}.npz')
    train_x, train_y, train_ids = (train_data[k] for k in ('x', 'y', 'meter_id'))
    real_x, real_y, real_ids = (heldout_data[k] for k in ('x', 'y', 'meter_id'))
    if set(train_ids) & set(real_ids):
        raise ValueError('Training and evaluation meters overlap')
    hashes = state.get('data_hashes', {})
    for filename in ('metadata.json', 'scaler.json', 'train.npz'):
        actual = digest if filename == 'train.npz' else sha256_file(prepared / filename)
        if hashes.get(filename) != actual:
            raise ValueError(f'Checkpoint data fingerprint does not match prepared {filename}')
    if split == 'val' and hashes.get('val.npz') != heldout_digest:
        raise ValueError('Checkpoint validation data fingerprint does not match')
    recorded = metadata['splits'][split]
    if recorded['profiles'] != len(real_x) or recorded['meters'] != len(np.unique(real_ids)):
        raise ValueError('Evaluation counts disagree with prepared metadata')
    if 'meter_ids' in recorded and set(recorded['meter_ids']) != set(real_ids):
        raise ValueError('Evaluation meter IDs disagree with prepared metadata')
    if real_x.shape[1] != metadata['resolution']:
        raise ValueError('Evaluation resolution disagrees with metadata')
    rng = np.random.default_rng(seed)
    selected = np.concatenate([
        rng.choice(np.flatnonzero(real_y == label),
                   min(samples_per_condition, int(np.sum(real_y == label))), replace=False)
        for label in np.unique(real_y)
    ])
    real_x, real_y, real_ids = real_x[selected], real_y[selected], real_ids[selected]
    fake = generate(checkpoint, real_y, seed=seed)
    alternatives = {
        'cgan': fake,
        'conditional_resampling': resample_baseline(train_x, train_y, real_y, seed=seed),
        'conditional_mean': mean_baseline(train_x, train_y, real_y),
    }
    interval = 24 / metadata['resolution']
    report = {
        'split': split, 'seed': seed, 'samples_per_condition_cap': samples_per_condition,
        'units': metadata['units'], 'target': metadata['target'],
        'checkpoint_sha256': sha256_file(checkpoint), 'train_sha256': digest,
        'evaluation_sha256': heldout_digest,
        'profile_count': len(real_x), 'meter_count': len(np.unique(real_ids)),
        'sampling': 'Uniform without replacement within each observed condition; macro averages weight conditions equally.',
        'methods': {name: evaluate_conditional(real_x, real_y, values, real_y, interval_hours=interval)
                    for name, values in alternatives.items()},
        'nearest_neighbors': nearest_neighbor_diagnostics(train_x, fake, real_x, seed=seed),
        'raw_output': {
            'minimum_kwh': float(fake.min()), 'maximum_kwh': float(fake.max()),
            'negative_fraction': float(np.mean(fake < 0)),
            'negative_values_valid_for_target': metadata['target'] == 'net',
            'clipping_applied': False,
        },
        'interpretation': 'Descriptive held-out diagnostics; no privacy guarantee or significance claim. '
                          'Repeated test use invalidates a final untouched-test claim.',
    }
    return report


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.command == 'download':
        print(download_release(args.output))
    elif args.command == 'audit':
        result = audit_archive(args.archive)
        write_json(args.output, result)
        print(f"Audited {len(result['members'])} members; SHA-256 {result['source_sha256']}")
    elif args.command == 'prepare':
        from .data import prepare_archive
        verify_archive(args.archive)
        result = prepare_archive(args.archive, args.output, resolution=args.resolution,
                                 target=args.target, seed=args.seed, chunksize=args.chunksize,
                                 max_meters_per_group=args.max_meters_per_group)
        print(json.dumps(result['splits'], indent=2))
    elif args.command == 'train':
        from .training import train
        config = json.loads(Path(args.config).read_text())
        if args.seed is not None:
            config['seed'] = args.seed
        print(json.dumps(train(args.prepared, args.output, config), indent=2))
    elif args.command == 'generate':
        if args.samples_per_label < 1:
            raise ValueError('samples-per-label must be positive')
        from .training import generate
        destination = Path(args.output)
        if destination.exists():
            raise FileExistsError(destination)
        labels = np.repeat(np.asarray(args.labels, dtype=np.int64), args.samples_per_label)
        profiles = generate(args.checkpoint, labels, seed=args.seed)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('xb') as stream:
            np.savez_compressed(stream, x=profiles, y=labels)
        write_json(destination.with_suffix('.json'), {
            'checkpoint_sha256': sha256_file(args.checkpoint), 'seed': args.seed,
            'units': 'kWh/interval', 'resolution': profiles.shape[1], 'labels': args.labels,
            'profiles': len(profiles), 'postprocessing': 'none; raw model output'})
        print(destination)
    elif args.command == 'evaluate':
        if Path(args.output).exists():
            raise FileExistsError(args.output)
        result = evaluate_checkpoint(args.prepared, args.checkpoint, split=args.split,
                                     samples_per_condition=args.samples_per_condition, seed=args.seed)
        write_json(args.output, result)
        print(args.output)


if __name__ == '__main__':
    main()
