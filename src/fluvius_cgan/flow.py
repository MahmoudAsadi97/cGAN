"""A direct conditional flow-matching comparator for Phase 1 development.

Independent Gaussian/data pairs define x_t=(1-t)z+t*x and target velocity x-z.
The regression objective follows Flow Matching for Generative Modeling (Lipman
et al., ICLR 2023) and the basic rectified-flow construction (Liu et al., ICLR
2023). This implementation does not copy either authors' library code.

This is whole-house profile generation, not appliance disaggregation. Fitting
and selection use the fixed inner TRAIN/VAL data; no TEST reader is provided.
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
from torch.nn import functional as F

from fluvius_cgan import development
from fluvius_cgan.development import fit_transform, physical_metrics, transform
from fluvius_cgan.training import _json, _read_data, _save, _sha256


SCHEMA = 'phase1-flow-1'
DEFAULTS = {'steps': 6000, 'batch_size': 512, 'seed': 101, 'num_threads': 1,
            'device': 'cpu', 'learning_rate': 1e-3, 'validation_every': 1000,
            'validation_per_condition': 256, 'solver_steps': 32}
ROOT = Path(development.__file__).resolve().parents[2]


class Velocity(nn.Module):
    """23,896 parameters; scalar time and 64 observed conditions, no dropout."""

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(24 + 1 + 64, 64), nn.SiLU(),
            nn.Linear(64, 128), nn.SiLU(),
            nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 24),
        )

    def forward(self, x, t, labels):
        if x.ndim != 2 or x.shape[1] != 24 or t.shape != (len(x), 1):
            raise ValueError('Expected x[N,24] and time[N,1]')
        if labels.ndim != 1 or len(labels) != len(x) or labels.dtype != torch.int64:
            raise ValueError('Expected aligned int64 condition labels')
        if torch.any((labels < 0) | (labels >= 64)):
            raise ValueError('Condition labels must be in [0,64)')
        condition = F.one_hot(labels, 64).to(dtype=x.dtype, device=x.device)
        return self.model(torch.cat((x, t.to(x), condition), dim=1))


def linear_bridge(x, z, t):
    if x.shape != z.shape or x.ndim != 2 or x.shape[1] != 24 or t.shape != (len(x), 1):
        raise ValueError('Bridge requires aligned [N,24] endpoints and [N,1] times')
    return (1 - t) * z + t * x, x - z


@torch.no_grad()
def heun(velocity, initial, labels, steps):
    """Fixed-step explicit trapezoidal integration on [0,1], exactly 2*steps NFE."""
    if type(steps) is not int or steps < 1:
        raise ValueError('Solver steps must be a positive integer')
    if initial.ndim != 2 or initial.shape[1] != 24 or not initial.dtype.is_floating_point:
        raise ValueError('Initial profiles must be floating [N,24]')
    if labels.ndim != 1 or len(labels) != len(initial):
        raise ValueError('Labels must align with initial profiles')
    if not torch.isfinite(initial).all():
        raise ValueError('Initial profiles must be finite')
    x = initial.clone()
    h = 1. / steps
    for index in range(steps):
        t = torch.full((len(x), 1), index / steps, dtype=x.dtype, device=x.device)
        next_t = torch.full_like(t, (index + 1) / steps)
        first = velocity(x, t, labels)
        second = velocity(x + h * first, next_t, labels)
        x = x + (.5 * h) * (first + second)
    if not torch.isfinite(x).all():
        raise FloatingPointError('Nonfinite flow integration output')
    return x


def _labels(labels):
    labels = np.asarray(labels)
    if labels.ndim != 1 or labels.dtype.kind not in 'iu' or np.any((labels < 0) | (labels >= 64)):
        raise ValueError('Labels must be an integer vector in [0,64)')
    return labels.astype(np.int64, copy=False)


def _seed(seed):
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('Seed must be an unsigned 32-bit integer')


def _samples(model, representation, labels, seed, solver_steps, batch_size=2048):
    labels = _labels(labels)
    _seed(seed)
    noise = np.random.default_rng(seed)
    outputs = []
    was_training = model.training
    model.eval()
    try:
        for start in range(0, len(labels), batch_size):
            condition = torch.from_numpy(labels[start:start + batch_size])
            z = torch.from_numpy(noise.standard_normal((len(condition), 24), dtype=np.float32))
            standardized = heun(model, z, condition, solver_steps).numpy()
            outputs.append(transform(standardized, representation, inverse=True))
    finally:
        model.train(was_training)
    return np.concatenate(outputs) if outputs else np.empty((0, 24), dtype=np.float64)


def generate(checkpoint, labels, seed, solver_steps=None):
    labels = _labels(labels)
    _seed(seed)
    state = torch.load(checkpoint, weights_only=True, map_location='cpu')
    if state.get('schema_version') != SCHEMA:
        raise ValueError('Expected a Phase 1 flow checkpoint')
    config = _config(state['config'])
    steps = config['solver_steps'] if solver_steps is None else solver_steps
    if type(steps) is not int or steps < 1:
        raise ValueError('Solver steps must be a positive integer')
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            model = Velocity()
            model.load_state_dict(state['velocity_model'])
            return _samples(model, state['representation'], labels, seed, steps)
    finally:
        torch.set_num_threads(previous_threads)


def _config(values):
    if not isinstance(values, dict) or set(values) - set(DEFAULTS):
        raise ValueError('Unknown flow config keys')
    c = DEFAULTS | values
    for key in ('steps', 'batch_size', 'validation_every', 'validation_per_condition', 'solver_steps'):
        if type(c[key]) is not int or c[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    _seed(c['seed'])
    if c['device'] != 'cpu' or type(c['num_threads']) is not int or c['num_threads'] != 1:
        raise ValueError('This comparator requires one CPU thread')
    if c['validation_per_condition'] > 1024:
        raise ValueError('Validation sample size exceeds 1024 per condition')
    if (type(c['learning_rate']) not in (int, float) or not np.isfinite(c['learning_rate'])
            or c['learning_rate'] <= 0):
        raise ValueError('learning_rate must be finite and positive')
    return c


def _selection(training, validation, per_condition):
    rng = np.random.default_rng(20261021)
    indices = np.concatenate([rng.choice(np.flatnonzero(validation['y'] == label),
        min(per_condition, int(np.sum(validation['y'] == label))), replace=False)
        for label in np.unique(training['y'])])
    real, labels = validation['x'][indices].astype(np.float64), validation['y'][indices]
    selected = {'index_seed': 20261021, 'generation_seed': 20261022,
                'indices': indices.tolist(), 'conditions': np.unique(labels).tolist(),
                'directions_seed': 2026, 'directions_count': 128}
    return real, labels, selected


def _sensitivity(checkpoint, real, labels, selected, training_range, solver_steps):
    started = time.perf_counter()
    first = generate(checkpoint, labels, selected['generation_seed'], solver_steps)
    first_time = time.perf_counter() - started
    started = time.perf_counter()
    second = generate(checkpoint, labels, selected['generation_seed'], 2 * solver_steps)
    second_time = time.perf_counter() - started
    state = torch.load(checkpoint, weights_only=True, map_location='cpu')
    return {'schema_version': 1, 'selected_checkpoint_step': state['step'],
        'checkpoint_sha256': _sha256(checkpoint), 'generation_seed': selected['generation_seed'],
        'common_initial_noise': True, 'used_for_selection': False,
        'selection_solver': {'method': 'Heun', 'steps': solver_steps, 'nfe': 2 * solver_steps,
            'generation_seconds_including_checkpoint_load': first_time,
            'metrics': physical_metrics(real, first, labels, training_range)},
        'sensitivity_solver': {'method': 'Heun', 'steps': 2 * solver_steps, 'nfe': 4 * solver_steps,
            'generation_seconds_including_checkpoint_load': second_time,
            'metrics': physical_metrics(real, second, labels, training_range)},
        'paired_generated_profile_change': {
            'rmse_kwh': float(np.sqrt(np.mean((first - second) ** 2))),
            'mean_absolute_kwh': float(np.mean(np.abs(first - second))),
            'maximum_absolute_kwh': float(np.max(np.abs(first - second)))},
        'interpretation': 'Numerical sensitivity at the selected checkpoint; no solver or checkpoint reselection.'}


def train_probe(data_dir, run_dir, config):
    c = _config(config)
    data, output = Path(data_dir), Path(run_dir)
    if output.exists():
        raise FileExistsError(output)
    training, validation, scaler, metadata, hashes = _read_data(data)
    if metadata.get('development', {}).get('source_partition_role') != 'original_train':
        raise ValueError('Requires inner development data sourced from original TRAIN')
    representation = fit_transform(training['x'], 'standard')
    real, labels, selected = _selection(training, validation, c['validation_per_condition'])
    training_range = scaler['maximum'] - scaler['minimum']
    package = Path(development.__file__).parent
    source = {p.name: _sha256(p) for p in sorted(package.glob('*.py'))}
    source['flow.py'] = _sha256(Path(__file__))
    git = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                         check=True, capture_output=True, text=True).stdout.strip()
    manifest = {'schema_version': SCHEMA, 'phase': 1, 'method': 'conditional_flow_matching',
        'role': 'post-test development comparator; whole-house profiles', 'status': 'running',
        'test_performance_access': False, 'config': c, 'data_hashes': hashes,
        'representation': representation, 'selection': selected,
        'metric_normalization': {'fitted_on': 'train', 'minimum': scaler['minimum'], 'maximum': scaler['maximum']},
        'objective': {'path': 'independent Gaussian/data linear bridge', 'time_distribution': 'Uniform(0,1)',
            'target_velocity': 'data - noise', 'loss': 'MSE averaged over examples and 24 coordinates'},
        'optimizer': {'name': 'Adam', 'learning_rate': c['learning_rate'], 'betas': [.9, .999],
                      'updates_per_batch': 1},
        'architecture': {'input': {'profile': 24, 'time': 1, 'one_hot_conditions': 64},
            'hidden_widths': [64, 128, 64], 'activation': 'SiLU', 'output': 24, 'dropout': 0.,
            'trainable_parameters': 23896},
        'solver': {'method': 'Heun', 'steps': c['solver_steps'], 'nfe': 2 * c['solver_steps'],
            'sensitivity_steps': 2 * c['solver_steps'], 'sensitivity_used_for_selection': False},
        'sampling': 'shuffle all inner-training rows per pass; include final partial batch',
        'available_training_rows': len(training['x']), 'source_sha256': source, 'git_head': git,
        'runtime': {'python': platform.python_version(), 'numpy': np.__version__,
                    'torch': str(torch.__version__), 'device': 'cpu', 'threads': 1},
        'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'prior_exposure': 'Development within previously used original TRAIN; no newly untouched confirmatory cohort.'}
    output.mkdir(parents=True, exist_ok=False)
    _json(output / 'manifest.json', manifest)
    history, best, best_step = [], float('inf'), None
    presentations, epoch, cursor, training_seconds = 0, 0, len(training['x']), 0.
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    started = time.perf_counter()
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(c['seed'])
            model = Velocity()
            assert sum(p.numel() for p in model.parameters() if p.requires_grad) == manifest['architecture']['trainable_parameters']
            optimizer = torch.optim.Adam(model.parameters(), lr=c['learning_rate'], betas=(.9, .999))
            sampler = torch.Generator().manual_seed(c['seed'] + 1)
            noise = torch.Generator().manual_seed(c['seed'] + 2)
            times = torch.Generator().manual_seed(c['seed'] + 3)
            x = torch.from_numpy(transform(training['x'], representation).astype(np.float32))
            y = torch.from_numpy(training['y'])
            loss_sum, window = 0., 0
            for step in range(1, c['steps'] + 1):
                tick = time.perf_counter()
                if cursor >= len(x):
                    order = torch.randperm(len(x), generator=sampler)
                    cursor, epoch = 0, epoch + 1
                batch = order[cursor:cursor + c['batch_size']]
                cursor += len(batch)
                presentations += len(batch)
                endpoint, conditions = x[batch], y[batch]
                z = torch.randn(len(batch), 24, generator=noise)
                t = torch.rand(len(batch), 1, generator=times)
                position, target = linear_bridge(endpoint, z, t)
                optimizer.zero_grad(set_to_none=True)
                loss = F.mse_loss(model(position, t, conditions), target)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite flow regression loss')
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()
                window += 1
                training_seconds += time.perf_counter() - tick
                if step != 1 and step % c['validation_every'] and step != c['steps']:
                    continue
                tick = time.perf_counter()
                fake = _samples(model, representation, labels, selected['generation_seed'], c['solver_steps'])
                generation_seconds = time.perf_counter() - tick
                tick = time.perf_counter()
                metrics = physical_metrics(real, fake, labels, training_range)
                metric_seconds = time.perf_counter() - tick
                score = metrics['macro']['normalized_profile_sw1']
                improved = score < best
                if improved:
                    best, best_step = score, step
                history.append({'step': step, 'passes_started': epoch, 'row_presentations': presentations,
                    'loss_velocity_mse': loss_sum / window, 'validation': metrics['macro'],
                    'diagnostics': metrics['diagnostics'], 'training_update_seconds': training_seconds,
                    'validation_generation_seconds': generation_seconds, 'validation_metric_seconds': metric_seconds,
                    'elapsed_seconds': time.perf_counter() - started})
                state = {'schema_version': SCHEMA, 'step': step, 'config': c,
                    'velocity_model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'representation': representation, 'data_hashes': hashes, 'selection': selected,
                    'best_step': best_step, 'best_score': best, 'history': history,
                    'model_rng_state': torch.get_rng_state(), 'noise_rng_state': noise.get_state(),
                    'time_rng_state': times.get_state(), 'sampler_rng_state': sampler.get_state(),
                    'sampler_order': order, 'sampler_cursor': cursor, 'passes_started': epoch,
                    'row_presentations': presentations}
                _save(output / 'last.pt', state)
                if improved:
                    _save(output / 'best.pt', state)
                    _json(output / 'best_validation.json', metrics)
                _json(output / 'history.json', history)
                manifest.update(completed_steps=step, row_presentations=presentations,
                                best_step=best_step, best_validation_sw1=best)
                _json(output / 'manifest.json', manifest)
                print(f"step={step} flow_seed={c['seed']} val_SW1={score:.6f} "
                      f"diversity={metrics['macro']['diversity_ratio']:.4f}", flush=True)
                loss_sum, window = 0., 0
        sensitivity = _sensitivity(output / 'best.pt', real, labels, selected, training_range, c['solver_steps'])
        _json(output / 'solver_sensitivity.json', sensitivity)
        manifest.update(status='completed', finished_at_utc=datetime.now(timezone.utc).isoformat(),
            elapsed_seconds=time.perf_counter() - started, training_update_seconds=training_seconds,
            artifact_hashes={name: _sha256(output / name) for name in
                ('best.pt', 'last.pt', 'history.json', 'best_validation.json', 'solver_sensitivity.json')})
        _json(output / 'manifest.json', manifest)
        return manifest
    except Exception as error:
        manifest.update(status='failed', error=f'{type(error).__name__}: {error}',
                        elapsed_seconds=time.perf_counter() - started)
        _json(output / 'manifest.json', manifest)
        raise
    finally:
        torch.set_num_threads(previous_threads)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    train_probe(args.data, args.run, json.loads(args.config.read_text()))


if __name__ == '__main__':
    main()
