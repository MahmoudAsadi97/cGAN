"""Controlled TRAIN/inner-validation probes; distinct from the frozen benchmark.

All representations invert to physical kWh before scoring. No TEST reader or
test-based selection is provided. Networks and optimizer defaults follow the
reference; this module isolates explicitly recorded development changes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch
from torch import nn

from .evaluation import conditional_metrics
from .models import Discriminator, Generator
from .training import _json, _read_data, _save, _sha256


DEFAULTS = {'steps': 6000, 'batch_size': 512, 'seed': 101, 'num_threads': 1,
            'normalization': 'minmax', 'validation_every': 1000,
            'validation_per_condition': 256}
KINDS = ('minmax', 'standard', 'hourly_standard')
SCHEMA = 'phase1-development-1'


def _array(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 24 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Expected nonempty finite physical [N,24] profiles')
    return x


def fit_transform(training, kind):
    x = _array(training)
    if kind not in KINDS:
        raise ValueError(f'Unknown normalization: {kind}')
    if kind == 'minmax':
        offset, scale = np.full(24, x.min()), np.full(24, x.max() - x.min())
    elif kind == 'standard':
        offset, scale = np.full(24, x.mean()), np.full(24, x.std())
    else:
        offset, scale = x.mean(axis=0), x.std(axis=0)
    constant = scale == 0
    scale[constant] = 1.
    return {'kind': kind, 'offset': offset.tolist(), 'scale': scale.tolist(),
            'constant_coordinates': np.flatnonzero(constant).tolist(),
            'fitted_on': 'train', 'clipping': False, 'std_ddof': 0}


def transform(x, spec, inverse=False):
    x = _array(x)
    offset, scale = np.asarray(spec['offset']), np.asarray(spec['scale'])
    if (offset.shape != (24,) or scale.shape != (24,) or not np.isfinite(offset).all()
            or not np.isfinite(scale).all() or np.any(scale <= 0)):
        raise ValueError('Invalid affine representation')
    return x * scale + offset if inverse else (x - offset) / scale


def physical_metrics(real, fake, labels, training_range):
    if not np.isfinite(training_range) or training_range < 0:
        raise ValueError('Invalid training range')
    result = conditional_metrics(real, fake, labels, labels)
    denominator = training_range or 1.
    for name in ('macro', 'weighted'):
        result[name]['normalized_profile_sw1'] = result[name]['profile_sw1_kwh'] / denominator
    errors = []
    for condition in np.unique(labels):
        mask = labels == condition
        errors.append(fake[mask].mean(axis=0) - real[mask].mean(axis=0))
    errors = np.asarray(errors)
    common = errors.mean(axis=0)
    total = float(np.mean(errors ** 2))
    result['diagnostics'] = {
        'common_hour_error_kwh': common.tolist(),
        'condition_mean_error_rmse_kwh': float(np.sqrt(total)),
        'common_hour_error_fraction': float(np.mean(common ** 2) / total) if total else 0.,
    }
    return result


def _samples(model, representation, labels, seed, batch_size=2048):
    noise = np.random.default_rng(seed)
    outputs = []
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(labels), batch_size):
                batch = torch.from_numpy(labels[start:start + batch_size].astype(np.int64))
                z = torch.from_numpy(noise.standard_normal((len(batch), 24), dtype=np.float32))
                outputs.append(transform(model(z, batch).numpy(), representation, inverse=True))
    finally:
        model.train(was_training)
    return np.concatenate(outputs) if outputs else np.empty((0, 24))


def generate(checkpoint, labels, seed):
    labels = np.asarray(labels)
    if labels.ndim != 1 or labels.dtype.kind not in 'iu' or np.any((labels < 0) | (labels >= 64)):
        raise ValueError('Labels must be integers in [0,64)')
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('Invalid generation seed')
    state = torch.load(checkpoint, weights_only=True, map_location='cpu')
    if state.get('schema_version') != SCHEMA:
        raise ValueError('Expected a development checkpoint')
    with torch.random.fork_rng(devices=[]):
        model = Generator()
        model.load_state_dict(state['generator'])
        return _samples(model, state['representation'], labels, seed)


def _config(values):
    if not isinstance(values, dict) or set(values) - set(DEFAULTS):
        raise ValueError('Unknown development config keys')
    c = DEFAULTS | values
    for key in ('steps', 'batch_size', 'num_threads', 'validation_every', 'validation_per_condition'):
        if type(c[key]) is not int or c[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if type(c['seed']) is not int or not 0 <= c['seed'] < 2**32:
        raise ValueError('Invalid training seed')
    if c['normalization'] not in KINDS or c['validation_per_condition'] > 1024:
        raise ValueError('Invalid normalization or validation size')
    return c


def train_probe(data_dir, run_dir, config):
    c = _config(config)
    data, output = Path(data_dir), Path(run_dir)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(output)
    training, validation, scaler, metadata, hashes = _read_data(data)
    representation = fit_transform(training['x'], c['normalization'])
    rng = np.random.default_rng(20261021)
    indices = np.concatenate([rng.choice(np.flatnonzero(validation['y'] == label),
        min(c['validation_per_condition'], int(np.sum(validation['y'] == label))), replace=False)
        for label in np.unique(training['y'])])
    real, labels = validation['x'][indices].astype(np.float64), validation['y'][indices]
    selected = {'index_seed': 20261021, 'generation_seed': 20261022,
                'indices': indices.tolist(), 'conditions': np.unique(labels).tolist(),
                'directions_seed': 2026, 'directions_count': 128}
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(c['num_threads'])
    source = {p.name: _sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    git = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).parents[2],
                         capture_output=True, text=True, check=True).stdout.strip()
    manifest = {'schema_version': SCHEMA, 'phase': 1, 'role': 'post-test development probe',
        'test_performance_access': False, 'config': c, 'data_hashes': hashes,
        'representation': representation, 'selection': selected,
        'metric_normalization': {'fitted_on': 'train', 'minimum': scaler['minimum'], 'maximum': scaler['maximum']},
        'optimizer': {'generator_lr': 1e-4, 'discriminator_lr': 2e-4, 'betas': [.9, .999], 'g_d_updates': [1, 1]},
        'sampling': 'shuffle all inner-training rows per pass; include final partial batch',
        'available_training_rows': len(training['x']), 'source_sha256': source, 'git_head': git,
        'runtime': {'python': platform.python_version(), 'numpy': np.__version__,
                    'torch': str(torch.__version__), 'device': 'cpu', 'threads': c['num_threads']},
        'started_at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'running',
        'prior_exposure': 'Development within original TRAIN; neither inner split is a newly untouched confirmatory holdout.'}
    _json(output / 'manifest.json', manifest)
    history, best, best_step, presentations, epoch, cursor = [], float('inf'), None, 0, 0, len(training['x'])
    started = time.perf_counter()
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(c['seed'])
            generator, discriminator = Generator(), Discriminator()
            optimizer_g = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(.9, .999))
            optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(.9, .999))
            sampler = torch.Generator().manual_seed(c['seed'] + 1)
            noise = torch.Generator().manual_seed(c['seed'] + 2)
            x = torch.from_numpy(transform(training['x'], representation).astype(np.float32))
            y = torch.from_numpy(training['y'])
            loss_fn = nn.BCEWithLogitsLoss()
            sums = np.zeros(2)
            window = 0
            for step in range(1, c['steps'] + 1):
                if cursor >= len(x):
                    order = torch.randperm(len(x), generator=sampler)
                    cursor = 0
                    epoch += 1
                batch = order[cursor:cursor + c['batch_size']]
                cursor += len(batch)
                presentations += len(batch)
                real_batch, conditions = x[batch], y[batch]
                ones = torch.ones((len(batch), 1))
                optimizer_d.zero_grad(set_to_none=True)
                z = torch.randn(len(batch), 24, generator=noise)
                fake_logits = discriminator(generator(z, conditions).detach(), conditions)
                real_logits = discriminator(real_batch, conditions)
                d_loss = (loss_fn(real_logits, ones) + loss_fn(fake_logits, torch.zeros_like(ones))) / 2
                if not torch.isfinite(d_loss):
                    raise FloatingPointError('Nonfinite discriminator loss')
                d_loss.backward()
                optimizer_d.step()
                optimizer_g.zero_grad(set_to_none=True)
                discriminator.requires_grad_(False)
                try:
                    z = torch.randn(len(batch), 24, generator=noise)
                    g_loss = loss_fn(discriminator(generator(z, conditions), conditions), ones)
                    if not torch.isfinite(g_loss):
                        raise FloatingPointError('Nonfinite generator loss')
                    g_loss.backward()
                    optimizer_g.step()
                finally:
                    discriminator.requires_grad_(True)
                sums += [g_loss.item(), d_loss.item()]
                window += 1
                if step != 1 and step % c['validation_every'] and step != c['steps']:
                    continue
                fake = _samples(generator, representation, labels, selected['generation_seed'])
                metrics = physical_metrics(real, fake, labels, scaler['maximum'] - scaler['minimum'])
                score = metrics['macro']['normalized_profile_sw1']
                improved = score < best
                if improved:
                    best, best_step = score, step
                row = {'step': step, 'passes_started': epoch, 'row_presentations': presentations,
                    'loss_g': float(sums[0] / window), 'loss_d': float(sums[1] / window),
                    'last_batch_real_logit_mean': float(real_logits.detach().mean()),
                    'last_batch_fake_logit_mean': float(fake_logits.detach().mean()),
                    'validation': metrics['macro'], 'diagnostics': metrics['diagnostics'],
                    'elapsed_seconds': time.perf_counter() - started}
                history.append(row)
                state = {'schema_version': SCHEMA, 'step': step, 'config': c,
                    'generator': generator.state_dict(), 'discriminator': discriminator.state_dict(),
                    'optimizer_g': optimizer_g.state_dict(), 'optimizer_d': optimizer_d.state_dict(),
                    'representation': representation, 'data_hashes': hashes, 'selection': selected,
                    'best_step': best_step, 'best_score': best, 'history': history,
                    'model_rng_state': torch.get_rng_state(), 'noise_rng_state': noise.get_state(),
                    'sampler_rng_state': sampler.get_state(), 'sampler_order': order,
                    'sampler_cursor': cursor, 'passes_started': epoch, 'row_presentations': presentations}
                _save(output / 'last.pt', state)
                if improved:
                    _save(output / 'best.pt', state)
                    _json(output / 'best_validation.json', metrics)
                _json(output / 'history.json', history)
                manifest.update(completed_steps=step, row_presentations=presentations,
                                best_step=best_step, best_validation_sw1=best)
                _json(output / 'manifest.json', manifest)
                print(f"step={step} normalization={c['normalization']} seed={c['seed']} "
                      f"val_SW1={score:.6f} mean_RMSE={metrics['macro']['mean_profile_rmse_kwh']:.4f} "
                      f"diversity={metrics['macro']['diversity_ratio']:.4f}", flush=True)
                sums[:] = 0
                window = 0
        manifest.update(status='completed', finished_at_utc=datetime.now(timezone.utc).isoformat(),
            elapsed_seconds=time.perf_counter() - started,
            artifact_hashes={name: _sha256(output / name) for name in
                            ('best.pt', 'last.pt', 'history.json', 'best_validation.json')})
        _json(output / 'manifest.json', manifest)
        return manifest
    except Exception as error:
        manifest.update(status='failed', error=f'{type(error).__name__}: {error}')
        _json(output / 'manifest.json', manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    train_probe(args.data, args.run, json.loads(args.config.read_text()))


if __name__ == '__main__':
    main()
