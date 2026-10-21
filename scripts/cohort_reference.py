#!/usr/bin/env python3
"""TRAIN-only, meter-disjoint real/real cohort variability reference.

This is descriptive calibration, not a confidence interval or publication gate.
Only explicit train.npz and metadata.json inputs are read; no VAL/TEST array
reader, model fitting, checkpoint selection, or learned representation is used.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np

from fluvius_cgan.development import physical_metrics
import fluvius_cgan.development as development
import fluvius_cgan.evaluation as evaluation


DEFAULT_SEED = 20261031
QUANTILES = (.1, .5, .9, .95)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def positive_integer(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f'{name} must be a positive integer')
    return value


def absolute_log_ratio(ratio):
    """Both-zero variance has package ratio 1; one-sided zeros stay undefined."""
    if ratio is None or ratio <= 0:
        return None
    return float(abs(np.log(ratio)))


def load_inputs(train_path, metadata_path):
    train_path, metadata_path = Path(train_path).resolve(), Path(metadata_path).resolve()
    if train_path.name != 'train.npz' or metadata_path.name != 'metadata.json':
        raise ValueError('Inputs must be explicit train.npz and metadata.json files')
    hashes = {'train.npz': sha256(train_path), 'metadata.json': sha256(metadata_path)}
    metadata = json.loads(metadata_path.read_text())
    if metadata.get('resolution') != 24 or metadata.get('condition_dim') != 64:
        raise ValueError('Metadata must specify 24 slots and 64 conditions')
    if metadata['splits']['train'].get('sha256') != hashes['train.npz']:
        raise ValueError('TRAIN hash disagrees with metadata')
    counts = metadata['splits']['val']['technology_meter_counts']
    if set(counts) != {str(g) for g in range(8)}:
        raise ValueError('Validation metadata must contain all eight equipment groups')
    counts = {str(g): positive_integer(counts[str(g)], 'equipment count') for g in range(8)}
    if sum(counts.values()) != metadata['splits']['val']['meters']:
        raise ValueError('Validation metadata equipment counts disagree with meter count')
    with np.load(train_path, allow_pickle=False) as source:
        x, labels, meters = (source[name].copy() for name in ('x', 'y', 'meter_id'))
    if (x.ndim != 2 or x.shape[1] != 24 or not len(x) or not np.isfinite(x).all()
            or labels.shape != (len(x),) or meters.shape != (len(x),)
            or labels.dtype.kind not in 'iu' or meters.dtype.kind not in 'US'
            or not np.array_equal(np.unique(labels), np.arange(64))):
        raise ValueError('TRAIN must contain finite hourly profiles, meter IDs and all 64 integer labels')
    meters = meters.astype(str)
    if any(not meter or meter.strip() != meter for meter in np.unique(meters)):
        raise ValueError('Invalid meter ID')
    identities, inverse = np.unique(meters, return_inverse=True)
    if len(identities) != metadata['splits']['train']['meters']:
        raise ValueError('TRAIN meter count disagrees with metadata')
    equipment = labels // 8
    minimum = np.full(len(identities), 8)
    maximum = np.full(len(identities), -1)
    np.minimum.at(minimum, inverse, equipment)
    np.maximum.at(maximum, inverse, equipment)
    if not np.array_equal(minimum, maximum):
        raise ValueError('Each meter must have one stable equipment group')
    groups = {str(g): identities[minimum == g] for g in range(8)}
    if any(len(groups[g]) < 2 * counts[g] for g in counts):
        raise ValueError('Insufficient TRAIN meters for two disjoint matched cohorts')
    declared = metadata['splits']['train'].get('technology_meter_counts')
    if declared is not None and declared != {g: len(ids) for g, ids in groups.items()}:
        raise ValueError('TRAIN equipment counts disagree with metadata')
    condition_rows = {c: np.flatnonzero(labels == c) for c in range(64)}
    return x, labels, meters, groups, counts, condition_rows, hashes


def aggregate(entries, names):
    macro, valid_counts = {}, {}
    for name in names:
        values = [entry['metrics'][name] for entry in entries if entry['metrics'][name] is not None]
        valid_counts[name] = len(values)
        macro[name] = float(np.mean(values)) if values else None
    return {'macro': macro, 'metric_valid_conditions': valid_counts}


def score_pair(real, other, labels, training_range):
    result = physical_metrics(real, other, labels, training_range)
    for entry in result['per_condition'].values():
        metrics = entry['metrics']
        metrics['normalized_profile_sw1'] = metrics['profile_sw1_kwh'] / (training_range or 1.)
        metrics['absolute_log_diversity_ratio'] = absolute_log_ratio(metrics['diversity_ratio'])
    names = list(result['macro']) + ['absolute_log_diversity_ratio']
    aggregated = aggregate(result['per_condition'].values(), names)
    result['macro'] = aggregated['macro']
    result['metric_valid_conditions'] = aggregated['metric_valid_conditions']
    for name in names:
        valid = [e for e in result['per_condition'].values() if e['metrics'][name] is not None]
        result['weighted'][name] = (float(np.average(
            [e['metrics'][name] for e in valid], weights=[e['n_real'] for e in valid])) if valid else None)
    groups = {str(g): aggregate([result['per_condition'][str(c)]
        for c in range(8 * g, 8 * g + 8)], names) for g in range(8)}
    return result, groups


def quantile_summary(values):
    finite = [value for value in values if value is not None]
    return {'n_total': len(values), 'n_valid': len(finite), 'n_undefined': len(values) - len(finite),
            'quantiles': {str(q): float(np.quantile(finite, q)) if finite else None for q in QUANTILES}}


def summarize(pairs):
    names = list(pairs[0]['metrics']['macro'])
    return {
        'macro': {name: quantile_summary([p['metrics']['macro'][name] for p in pairs]) for name in names},
        'groups': {str(g): {name: quantile_summary([p['groups'][str(g)]['macro'][name]
            for p in pairs]) for name in names} for g in range(8)},
        'per_condition': {str(c): {name: quantile_summary([
            p['metrics']['per_condition'][str(c)]['metrics'][name] for p in pairs])
            for name in names} for c in range(64)},
    }


def build_reference(train_path, metadata_path, output, *, pairs=100, per_condition=256,
                    seed=DEFAULT_SEED, progress=False):
    positive_integer(pairs, 'pairs')
    positive_integer(per_condition, 'per_condition')
    if type(seed) is not int or not 0 <= seed < 2 ** 32:
        raise ValueError('seed must be an integer in [0,2**32)')
    if per_condition > 256:
        raise ValueError('per_condition may not exceed the declared 256-profile cap')
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    x, labels, meters, groups, counts, condition_rows, hashes = load_inputs(train_path, metadata_path)
    minimum, maximum = float(x.min()), float(x.max())
    training_range = maximum - minimum
    # Each condition must have enough profiles in every selected meter. Reject
    # unsupported cohorts instead of selectively skipping unfavorable pairs.
    support = set(zip(meters, labels.tolist()))
    for g, identities in groups.items():
        if any((meter, c) not in support for meter in identities
               for c in range(8 * int(g), 8 * int(g) + 8)):
            raise ValueError('Every fitting meter must support its eight calendar conditions')
    output.mkdir(parents=True, exist_ok=False)
    plan = {'schema_version': 1, 'role': 'TRAIN-only descriptive cohort reference',
        'config': {'pairs': pairs, 'per_condition': per_condition, 'seed': seed,
                   'directions_count': 128, 'directions_seed': 2026,
                   'quantiles': list(QUANTILES), 'target_split_metadata': 'val'},
        'input_sha256': hashes, 'cohort_equipment_counts': counts,
        'source_sha256': {str(Path(p).resolve()): sha256(p)
                         for p in (__file__, development.__file__, evaluation.__file__)}}
    (output / 'plan.json').write_text(json.dumps(plan, indent=2, allow_nan=False) + '\n')
    started = time.perf_counter()
    records = []
    for pair in range(pairs):
        cohort_rng = np.random.default_rng(np.random.SeedSequence([seed, pair, 0]))
        selected = {'a': [], 'b': []}
        for group in range(8):
            key, k = str(group), counts[str(group)]
            order = cohort_rng.permutation(groups[key])
            selected['a'].extend(order[:k].tolist())
            selected['b'].extend(order[k:2 * k].tolist())
        member_rows = {side: np.isin(meters, ids) for side, ids in selected.items()}
        indices = {'a': [], 'b': []}
        rngs = {side: np.random.default_rng(np.random.SeedSequence([seed, pair, 1, index]))
                for index, side in enumerate(('a', 'b'))}
        available = {side: {c: condition_rows[c][member_rows[side][condition_rows[c]]]
                           for c in range(64)} for side in ('a', 'b')}
        sample_counts = {str(c): min(per_condition, len(available['a'][c]), len(available['b'][c]))
                         for c in range(64)}
        if not all(sample_counts.values()):
            raise ValueError('Cohort pair lost calendar support; no pairs are silently discarded')
        for side in ('a', 'b'):
            indices[side] = np.concatenate([rngs[side].choice(available[side][c],
                sample_counts[str(c)], replace=False) for c in range(64)])
        ia, ib = indices['a'], indices['b']
        assert np.array_equal(labels[ia], labels[ib])
        metrics, group_metrics = score_pair(x[ia], x[ib], labels[ia], training_range)
        cohorts = {}
        for side in ('a', 'b'):
            ids = sorted(selected[side])
            cohorts[side] = {'meter_ids': ids, 'meter_ids_sha256': digest_json(ids),
                'technology_meter_counts': counts, 'sampled_condition_counts': sample_counts,
                'sampled_row_indices_sha256': hashlib.sha256(indices[side].astype('<i8').tobytes()).hexdigest()}
        record = {'pair_index': pair, 'cohorts': cohorts, 'metrics': metrics, 'groups': group_metrics}
        records.append(record)
        # Keep every completed outcome even if a later pair fails.
        (output / f'pair_{pair:03d}.json').write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
        if progress:
            print(f'Pair {pair + 1}/{pairs}: SW1={metrics["macro"]["normalized_profile_sw1"]:.7f}', flush=True)
    if hashes != {'train.npz': sha256(train_path), 'metadata.json': sha256(metadata_path)}:
        raise ValueError('Consumed inputs changed during reference generation')
    result = {'schema_version': 1, 'role': 'TRAIN-only descriptive real-real cohort variability',
        'publication_gate': False, 'confidence_interval': False,
        'validation_profile_values_read': False, 'test_profile_values_read': False,
        'config': plan['config'], 'plan_sha256': sha256(output / 'plan.json'),
        'inputs': {'train': str(Path(train_path).resolve()), 'metadata': str(Path(metadata_path).resolve()),
                   'sha256': hashes},
        'source_sha256': plan['source_sha256'],
        'runtime': {'python': platform.python_version(), 'numpy': np.__version__,
                    'torch': development.torch.__version__},
        'normalization': {'minimum': minimum, 'maximum': maximum,
            'denominator': training_range or 1., 'constant_training_range': training_range == 0,
            'fitted_on': 'consumed inner TRAIN values only'},
        'cohort_equipment_counts': counts,
        'sampling': 'Disjoint meter cohorts within each pair; all calendar histories eligible; '
                    'independent without-replacement day samples per side/condition; equal counts',
        'limitations': ['Pairs may overlap across repetitions; they are not independent experiments.',
            'Shared 2024 calendar, balanced release, prior development exposure and sampled meter dependence limit scope.',
            'Synthetic independent days do not reproduce the real cohort sampling mechanism.',
            '90th percentile is a proposed contextual tolerance, not accuracy, confidence coverage or a publication gate.',
            'Undefined variance ratios are retained with valid counts; no infinity or arbitrary epsilon is substituted.'],
        'pairs': records, 'summary': summarize(records),
        'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'elapsed_seconds': time.perf_counter() - started}
    (output / 'reference.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', required=True, type=Path)
    parser.add_argument('--metadata', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--pairs', type=int, default=100)
    parser.add_argument('--per-condition', type=int, default=256)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    build_reference(args.train, args.metadata, args.output, pairs=args.pairs,
                    per_condition=args.per_condition, seed=args.seed, progress=True)


if __name__ == '__main__':
    main()
