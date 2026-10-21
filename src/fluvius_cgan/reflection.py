"""Conditional reflected-flow copula comparator for Phase 1 development.

The reflected Gaussian trajectories and velocity-regression construction follow
Huk and Damoulas, ``Diffusion and Flow-based Copulas: Forgetting and Remembering
Dependencies'', arXiv:2509.19707. Author code was inspected at commit
5014c0f35179b38803f3634c0b0acd4a6f6e2e57; the full manuscript was unavailable.
This independent implementation adds observed conditioning and TRAIN-only mixed
empirical marginals. Its small network and budget are adaptations, not a
reproduction of the authors' larger unconditional experiment.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch

from fluvius_cgan import flow
from fluvius_cgan.flow import Velocity
from fluvius_cgan.development import physical_metrics
from fluvius_cgan.marginals import fit_marginals, load_marginals
from fluvius_cgan.training import _json, _read_data, _save, _sha256


SCHEMA = 'phase1-reflection-1'
DEFAULTS = {'steps': 6000, 'batch_size': 512, 'seed': 101, 'num_threads': 1,
    'device': 'cpu', 'learning_rate': 1e-3, 'validation_every': 1000,
    'validation_per_condition': 256, 'solver_steps': 64, 'max_time': 1.5,
    'gradient_clip_norm': 1., 'condition_mode': 'conditional'}
ROOT = Path(__file__).resolve().parents[2]
DECILES = np.linspace(0., 1., 11)


def _floating(values, name):
    if (not torch.is_tensor(values) or not values.dtype.is_floating_point
            or not torch.isfinite(values).all()):
        raise ValueError(f'{name} must be a finite floating tensor')


def reflect(values):
    """Fold any finite coordinate into [0,1], including multiple crossings.

    Equivalent to 1-abs(1-(x mod 2)); the piecewise form leaves in-range values
    bit-identical, avoiding cancellation for the zero-velocity control.
    """
    _floating(values, 'Reflection input')
    remainder = torch.remainder(values, 2.)
    return torch.where(remainder <= 1., remainder, 2. - remainder)


def reflected_bridge(initial, velocity, times):
    """Reflected position and forward-right time derivative of initial+t*v."""
    for value, name in ((initial, 'Initial coordinates'), (velocity, 'Velocity'), (times, 'Time')):
        _floating(value, name)
    if (initial.ndim != 2 or initial.shape != velocity.shape or times.shape != (len(initial), 1)
            or initial.device != velocity.device or initial.device != times.device):
        raise ValueError('Reflected bridge requires aligned [N,D] coordinates/velocity and [N,1] time')
    unfolded = initial + times * velocity
    position = reflect(unfolded)
    slope = torch.where(torch.remainder(unfolded, 2.) < 1., velocity, -velocity)
    # At collisions, the right derivative in forward time points into the cube.
    slope = torch.where(position == 0., velocity.abs(), slope)
    slope = torch.where(position == 1., -velocity.abs(), slope)
    return position, slope


def _conditioning(labels, mode):
    if mode not in ('conditional', 'pooled'):
        raise ValueError('condition_mode must be conditional or pooled')
    return labels if mode == 'conditional' else torch.zeros_like(labels)


@torch.no_grad()
def reflected_euler(velocity, initial, labels, steps, max_time=1.5, condition_mode='conditional'):
    """Integrate the forward velocity over decreasing time, reflecting each step.

    One interval is one velocity evaluation. Reflection counts are coordinates
    whose unreflected Euler proposal leaves [0,1], not numbers of wall crossings.
    """
    _floating(initial, 'Initial probabilities')
    if (initial.ndim != 2 or not all(initial.shape) or torch.any((initial < 0) | (initial > 1))):
        raise ValueError('Initial probabilities must be nonempty [N,D] values in [0,1]')
    if (labels.ndim != 1 or len(labels) != len(initial) or labels.dtype != torch.int64
            or labels.device != initial.device or torch.any((labels < 0) | (labels >= 64))):
        raise ValueError('Labels must be aligned int64 conditions in [0,64)')
    if type(steps) is not int or steps < 1:
        raise ValueError('Solver steps must be a positive integer')
    if type(max_time) not in (int, float) or not np.isfinite(max_time) or max_time <= 0:
        raise ValueError('max_time must be finite and positive')
    conditions = _conditioning(labels, condition_mode)
    x, counts = initial.clone(), []
    grid = np.linspace(float(max_time), 0., steps + 1)
    for first, second in zip(grid[:-1], grid[1:]):
        t = torch.full((len(x), 1), float(first), dtype=x.dtype, device=x.device)
        field = velocity(x, t, conditions)
        if field.shape != x.shape or not torch.isfinite(field).all():
            raise FloatingPointError('Nonfinite or misaligned reflection velocity field')
        proposed = x + float(second - first) * field
        if not torch.isfinite(proposed).all():
            raise FloatingPointError('Nonfinite reflection integration proposal')
        counts.append(int(torch.sum((proposed < 0) | (proposed > 1))))
        x = reflect(proposed)
    return x, {'method': 'reflected Euler', 'steps': steps, 'nfe': steps,
        'max_time': float(max_time), 'time_grid': grid.tolist(),
        'reflection_coordinate_counts': counts,
        'reflection_coordinate_fractions': [count / initial.numel() for count in counts],
        'coordinate_denominator_per_step': initial.numel(),
        'interpretation': 'Counts indicate out-of-cube proposals before reflection, not wall-crossing multiplicity.'}


def probability_diagnostics(values, labels):
    values = np.asarray(values, dtype=np.float64)
    labels = flow._labels(labels)
    if (values.ndim != 2 or values.shape != (len(labels), 24) or not np.isfinite(values).all()
            or np.any((values < 0) | (values > 1))):
        raise ValueError('Expected aligned finite [N,24] probabilities in [0,1]')
    groups = {}
    for label in np.unique(labels):
        group = values[labels == label]
        histogram = np.stack([np.histogram(group[:, hour], DECILES)[0] for hour in range(24)])
        groups[str(int(label))] = {'profiles': len(group), 'mean': group.mean(axis=0).tolist(),
            'population_variance': group.var(axis=0).tolist(),
            'decile_counts': histogram.tolist(), 'zero_counts': (group == 0).sum(axis=0).tolist(),
            'one_counts': (group == 1).sum(axis=0).tolist(),
            'decile_l1_from_uniform': np.abs(histogram / len(group) - .1).sum(axis=1).tolist()}
    return {'schema_version': 1, 'profiles': len(values), 'decile_edges': DECILES.tolist(),
        'histogram_orientation': '[hour][bin]; final bin includes one', 'per_condition': groups,
        'macro_mean_absolute_error_from_half': float(np.mean([
            np.abs(np.asarray(group['mean']) - .5) for group in groups.values()])) if groups else None,
        'macro_decile_l1_from_uniform': float(np.mean([
            group['decile_l1_from_uniform'] for group in groups.values()])) if groups else None,
        'interpretation': 'Descriptive conditional uniformity diagnostic with finite-sample error; not a calibrated test.'}


def _config(values):
    if not isinstance(values, dict) or set(values) - set(DEFAULTS):
        raise ValueError('Unknown reflection config keys')
    config = DEFAULTS | values
    for key in ('steps', 'batch_size', 'validation_every', 'validation_per_condition', 'solver_steps'):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    flow._seed(config['seed'])
    if config['device'] != 'cpu' or type(config['num_threads']) is not int or config['num_threads'] != 1:
        raise ValueError('This comparator requires one CPU thread')
    if config['validation_per_condition'] > 1024:
        raise ValueError('Validation sample size exceeds 1024 per condition')
    for key in ('learning_rate', 'max_time', 'gradient_clip_norm'):
        if type(config[key]) not in (int, float) or not np.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f'{key} must be finite and positive')
    if config['condition_mode'] not in ('conditional', 'pooled'):
        raise ValueError('condition_mode must be conditional or pooled')
    return config


def _sample_probabilities(model, labels, seed, config, solver_steps=None, batch_size=2048):
    labels = flow._labels(labels)
    flow._seed(seed)
    steps = config['solver_steps'] if solver_steps is None else solver_steps
    if type(steps) is not int or steps < 1:
        raise ValueError('Solver steps must be a positive integer')
    rng = np.random.default_rng(seed)
    output, counts, solver = [], np.zeros(steps, dtype=np.int64), None
    was_training = model.training
    model.eval()
    try:
        for start in range(0, len(labels), batch_size):
            condition = torch.from_numpy(labels[start:start + batch_size])
            initial = torch.from_numpy(rng.random((len(condition), 24), dtype=np.float32))
            result, solver = reflected_euler(model, initial, condition, steps,
                                             config['max_time'], config['condition_mode'])
            output.append(result.numpy())
            counts += np.asarray(solver['reflection_coordinate_counts'])
    finally:
        model.train(was_training)
    probabilities = np.concatenate(output) if output else np.empty((0, 24), dtype=np.float32)
    if solver is None:
        solver = {'method': 'reflected Euler', 'steps': steps, 'nfe': steps,
                  'max_time': float(config['max_time']),
                  'time_grid': np.linspace(config['max_time'], 0., steps + 1).tolist()}
    solver.update(reflection_coordinate_counts=counts.tolist(),
        reflection_coordinate_fractions=(counts / probabilities.size).tolist() if probabilities.size else [0.] * steps,
        coordinate_denominator_per_step=probabilities.size)
    return probabilities, {'solver': solver, 'probabilities': probability_diagnostics(probabilities, labels)}


def independent_marginals(marginals, labels, seed):
    labels = flow._labels(labels)
    flow._seed(seed)
    if not len(labels):
        return np.empty((0, 24), dtype=np.float64)
    probabilities = np.random.default_rng(seed).random((len(labels), 24), dtype=np.float32)
    return marginals.inverse(probabilities, labels)


def _load(checkpoint):
    path = Path(checkpoint)
    state = torch.load(path, weights_only=True, map_location='cpu')
    if state.get('schema_version') != SCHEMA:
        raise ValueError('Expected a Phase 1 reflection checkpoint')
    config = _config(state['config'])
    record = state['marginals']
    if (not isinstance(record.get('artifact'), str) or Path(record['artifact']).name != record['artifact']
            or record['artifact'] in ('', '.', '..') or record.get('fitted_on') != 'inner train'
            or record.get('training_npz_sha256') != state['data_hashes']['train.npz']):
        raise ValueError('Invalid checkpoint marginal provenance')
    fitted = load_marginals(path.parent / record['artifact'], record['sha256'])
    if (fitted.profile_dim != 24 or record.get('summary') != fitted.summary()
            or set(state['selection']['conditions']) - set(fitted.conditions)):
        raise ValueError('Checkpoint marginal schema, coverage or summary differs')
    return state, config, fitted


def _generated(checkpoint, labels, seed, solver_steps, physical):
    labels = flow._labels(labels)
    flow._seed(seed)
    state, config, fitted = _load(checkpoint)
    if set(labels.tolist()) - set(fitted.conditions):
        raise ValueError('Unsupported requested condition for fitted marginals')
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            model = Velocity()
            model.load_state_dict(state['velocity_model'])
            probability, diagnostic = _sample_probabilities(model, labels, seed, config, solver_steps)
            if physical:
                return fitted.inverse(probability, labels) if len(labels) else np.empty((0, 24), dtype=np.float64)
            return probability, diagnostic
    finally:
        torch.set_num_threads(previous_threads)


def generate(checkpoint, labels, seed, solver_steps=None):
    """Generate physical profiles through the checkpoint-bound empirical inverse."""
    return _generated(checkpoint, labels, seed, solver_steps, True)


def generate_probabilities(checkpoint, labels, seed, solver_steps=None):
    """Return latent probabilities and descriptive solver/uniformity diagnostics."""
    return _generated(checkpoint, labels, seed, solver_steps, False)


def _sensitivity(checkpoint, real, labels, selected, training_range, steps):
    reports, samples, probabilities = [], [], []
    _, _, marginals = _load(checkpoint)
    for count in (steps, 2 * steps):
        started = time.perf_counter()
        probability, diagnostic = generate_probabilities(checkpoint, labels, selected['generation_seed'], count)
        fake = marginals.inverse(probability, labels)
        reports.append({'method': 'reflected Euler', 'steps': count, 'nfe': count,
            'generation_seconds_including_checkpoint_load_and_decode': time.perf_counter() - started,
            'metrics': physical_metrics(real, fake, labels, training_range), 'diagnostics': diagnostic})
        samples.append(fake)
        probabilities.append(probability.astype(np.float64))
    state = torch.load(checkpoint, weights_only=True, map_location='cpu')
    difference = samples[0] - samples[1]
    return {'schema_version': 1, 'selected_checkpoint_step': state['step'],
        'checkpoint_sha256': _sha256(checkpoint), 'generation_seed': selected['generation_seed'],
        'common_initial_uniforms': True, 'used_for_selection': False,
        'selection_solver': reports[0], 'sensitivity_solver': reports[1],
        'paired_generated_profile_change': {'rmse_kwh': float(np.sqrt(np.mean(difference ** 2))),
            'mean_absolute_kwh': float(np.mean(np.abs(difference))),
            'maximum_absolute_kwh': float(np.max(np.abs(difference))),
            'probability_rmse': float(np.sqrt(np.mean((probabilities[0] - probabilities[1]) ** 2)))},
        'interpretation': 'Fixed doubled-integration sensitivity at the selected checkpoint; no reselection.'}


def _artifact_hashes(output):
    return {path.name: _sha256(path) for path in sorted(output.iterdir())
            if path.is_file() and path.name != 'manifest.json'}


def train_probe(data_dir, run_dir, config):
    config = _config(config)
    data, output = Path(data_dir), Path(run_dir)
    if output.exists():
        raise FileExistsError(output)
    training, validation, scaler, metadata, hashes = _read_data(data)
    if metadata.get('development', {}).get('source_partition_role') != 'original_train':
        raise ValueError('Requires inner development data sourced from original TRAIN')
    real, labels, selected = flow._selection(training, validation, config['validation_per_condition'])
    training_range = scaler['maximum'] - scaler['minimum']
    bin_edges = np.asarray([0., 1 / 150, 1 / 15, 1 / 3, 2 / 3, 1.]) * config['max_time']
    manifest = {'schema_version': SCHEMA, 'phase': 1, 'method': 'conditional_reflection_copula',
        'role': 'post-test development conditional dependence comparator', 'status': 'running',
        'test_performance_access': False, 'config': config, 'data_hashes': hashes, 'selection': selected,
        'metric_normalization': {'fitted_on': 'train', 'minimum': scaler['minimum'], 'maximum': scaler['maximum']},
        'objective': {'path': 'coordinatewise reflection of u0+t*z with iid standard Gaussian z',
            'time_distribution': 'max_time * Uniform(0,1)^4', 'target': 'forward-right reflected velocity',
            'loss': 'mean squared error across rows and coordinates', 'loss_time_bin_edges': bin_edges.tolist()},
        'marginal_transform': {'fitted_on': 'inner train',
            'forward': '(count(TRAIN<x)+Uniform(0,1)*count(TRAIN=x))/condition_rows',
            'randomization': 'fresh independent coordinate jitter on every TRAIN presentation',
            'arithmetic': 'float64 interval endpoints and jitter; cast randomized probabilities to float32 before the reflected trajectory',
            'finite_precision': 'Finite RNG and float32 rounding can hit jump boundaries; inverse recovery is an almost-sure ideal-law statement, not a bitwise guarantee.',
            'inverse': 'empirical step inverse ceil(n*u)-1 with endpoint index handling',
            'limitation': 'Chooses a continuous extension for tied data; population copula is not uniquely identified. Empirical inverse truncates unseen hourly tails.'},
        'optimizer': {'name': 'Adam', 'learning_rate': config['learning_rate'], 'betas': [.9, .999],
            'gradient_clip_norm': config['gradient_clip_norm']},
        'architecture': {'input': {'profile': 24, 'time': 1, 'one_hot_conditions': 64},
            'hidden_widths': [64, 128, 64], 'activation': 'SiLU', 'output': 24,
            'trainable_parameters': 23896, 'condition_mode': config['condition_mode']},
        'solver': {'method': 'reflected Euler', 'steps': config['solver_steps'], 'nfe': config['solver_steps'],
            'sensitivity_steps': 2 * config['solver_steps'], 'sensitivity_used_for_selection': False,
            'initial_distribution': 'independent Uniform(0,1) coordinates', 'batch_size': 2048},
        'sampling': 'shuffle all inner-TRAIN rows per pass; include final partial batch',
        'available_training_rows': len(training['x']),
        'source_sha256': {p.name: _sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
        'git_head': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, check=True,
            capture_output=True, text=True).stdout.strip(),
        'runtime': {'python': platform.python_version(), 'numpy': np.__version__,
                    'torch': str(torch.__version__), 'device': 'cpu', 'threads': 1},
        'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'prior_exposure': 'Previously used original TRAIN development split; not untouched confirmation.'}
    output.mkdir(parents=True, exist_ok=False)
    _json(output / 'manifest.json', manifest)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    started = time.perf_counter()
    history, best, best_step = [], float('inf'), None
    training_seconds, presentations, epoch, cursor = 0., 0, 0, len(training['x'])
    try:
        tick = time.perf_counter()
        marginals = fit_marginals(training['x'], training['y'])
        lower, upper = marginals.intervals(training['x'], training['y'])
        fit_seconds = time.perf_counter() - tick
        tick = time.perf_counter()
        marginals.save(output / 'marginals.npz')
        marginal_record = {'artifact': 'marginals.npz', 'sha256': _sha256(output / 'marginals.npz'),
            'training_npz_sha256': hashes['train.npz'], 'fitted_on': 'inner train', 'summary': marginals.summary()}
        manifest.update(marginals=marginal_record, marginal_fit_seconds=fit_seconds,
                        marginal_save_seconds=time.perf_counter() - tick)
        _json(output / 'manifest.json', manifest)
        lower, widths = torch.from_numpy(lower), torch.from_numpy(upper - lower)
        del upper
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config['seed'])
            model = Velocity()
            assert sum(p.numel() for p in model.parameters()) == 23896
            optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], betas=(.9, .999))
            sampler = torch.Generator().manual_seed(config['seed'] + 1)
            noise = torch.Generator().manual_seed(config['seed'] + 2)
            times = torch.Generator().manual_seed(config['seed'] + 3)
            jitter = torch.Generator().manual_seed(config['seed'] + 4)
            y = torch.from_numpy(training['y'])
            loss_sum, window = 0., 0
            bin_sums, bin_counts = np.zeros(5), np.zeros(5, dtype=np.int64)
            for step in range(1, config['steps'] + 1):
                tick = time.perf_counter()
                if cursor >= len(lower):
                    order = torch.randperm(len(lower), generator=sampler)
                    cursor, epoch = 0, epoch + 1
                batch = order[cursor:cursor + config['batch_size']]
                cursor += len(batch)
                presentations += len(batch)
                u0 = (lower[batch] + torch.rand((len(batch), 24), generator=jitter, dtype=torch.float64)
                      * widths[batch]).to(torch.float32)
                velocity = torch.randn((len(batch), 24), generator=noise)
                t = config['max_time'] * torch.rand((len(batch), 1), generator=times) ** 4
                position, target = reflected_bridge(u0, velocity, t)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(position, t, _conditioning(y[batch], config['condition_mode']))
                squared_error = (prediction - target).square()
                loss = squared_error.mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite reflection regression loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip_norm'], error_if_nonfinite=True)
                optimizer.step()
                row_errors = squared_error.detach().mean(dim=1).numpy()
                bins = np.minimum(np.searchsorted(bin_edges, t.numpy().ravel(), side='right') - 1, 4)
                bin_sums += np.bincount(bins, weights=row_errors, minlength=5)
                bin_counts += np.bincount(bins, minlength=5)
                loss_sum += loss.item()
                window += 1
                training_seconds += time.perf_counter() - tick
                if step != 1 and step % config['validation_every'] and step != config['steps']:
                    continue
                tick = time.perf_counter()
                probability, diagnostics = _sample_probabilities(model, labels, selected['generation_seed'], config)
                generation_seconds = time.perf_counter() - tick
                tick = time.perf_counter()
                fake = marginals.inverse(probability, labels)
                decoding_seconds = time.perf_counter() - tick
                tick = time.perf_counter()
                metrics = physical_metrics(real, fake, labels, training_range)
                metric_seconds = time.perf_counter() - tick
                score = metrics['macro']['normalized_profile_sw1']
                improved = score < best
                if improved:
                    best, best_step = score, step
                history.append({'step': step, 'passes_started': epoch, 'row_presentations': presentations,
                    'loss_velocity_mse': loss_sum / window,
                    'loss_by_time_bin': [{'lower': float(bin_edges[i]), 'upper': float(bin_edges[i + 1]),
                        'rows': int(bin_counts[i]), 'mean_squared_error': float(bin_sums[i] / bin_counts[i])
                        if bin_counts[i] else None} for i in range(5)],
                    'validation': metrics['macro'], 'diagnostics': metrics['diagnostics'],
                    'probability_macro_decile_l1': diagnostics['probabilities']['macro_decile_l1_from_uniform'],
                    'probability_macro_mean_error': diagnostics['probabilities']['macro_mean_absolute_error_from_half'],
                    'solver_reflection_coordinate_fractions': diagnostics['solver']['reflection_coordinate_fractions'],
                    'training_update_seconds': training_seconds, 'validation_generation_seconds': generation_seconds,
                    'validation_decoding_seconds': decoding_seconds, 'validation_metric_seconds': metric_seconds,
                    'elapsed_seconds': time.perf_counter() - started})
                state = {'schema_version': SCHEMA, 'step': step, 'config': config,
                    'velocity_model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'marginals': marginal_record, 'data_hashes': hashes, 'selection': selected,
                    'best_step': best_step, 'best_score': best, 'history': history,
                    'model_rng_state': torch.get_rng_state(), 'noise_rng_state': noise.get_state(),
                    'time_rng_state': times.get_state(), 'jitter_rng_state': jitter.get_state(),
                    'sampler_rng_state': sampler.get_state(), 'sampler_order': order, 'sampler_cursor': cursor,
                    'passes_started': epoch, 'row_presentations': presentations}
                _save(output / 'last.pt', state)
                if improved:
                    _save(output / 'best.pt', state)
                    _json(output / 'best_validation.json', metrics)
                    _json(output / 'best_probability_diagnostics.json', diagnostics)
                _json(output / 'history.json', history)
                manifest.update(completed_steps=step, row_presentations=presentations,
                                best_step=best_step, best_validation_sw1=best)
                _json(output / 'manifest.json', manifest)
                print(f"step={step} reflection_seed={config['seed']} val_SW1={score:.6f} "
                      f"diversity={metrics['macro']['diversity_ratio']:.4f}", flush=True)
                loss_sum, window = 0., 0
                bin_sums, bin_counts = np.zeros(5), np.zeros(5, dtype=np.int64)
        tick = time.perf_counter()
        independent = independent_marginals(marginals, labels, selected['generation_seed'])
        _json(output / 'independent_marginals.json', {'schema_version': 1, 'method': 'independent_marginals',
            'generation_seed': selected['generation_seed'], 'conditions': selected['conditions'],
            'selection_rows': len(labels), 'data_hashes': hashes, 'marginals': marginal_record,
            'used_for_selection': False, 'generation_seconds': time.perf_counter() - tick,
            'generated_profile_bytes_sha256': hashlib.sha256(independent.astype('<f8').tobytes()).hexdigest(),
            'metrics': physical_metrics(real, independent, labels, training_range)})
        _json(output / 'solver_sensitivity.json', _sensitivity(output / 'best.pt', real, labels, selected,
                                                               training_range, config['solver_steps']))
        manifest.update(status='completed', finished_at_utc=datetime.now(timezone.utc).isoformat(),
            elapsed_seconds=time.perf_counter() - started, training_update_seconds=training_seconds,
            artifact_hashes=_artifact_hashes(output))
        _json(output / 'manifest.json', manifest)
        return manifest
    except Exception as error:
        manifest.update(status='failed', error=f'{type(error).__name__}: {error}',
            elapsed_seconds=time.perf_counter() - started, finished_at_utc=datetime.now(timezone.utc).isoformat(),
            artifact_hashes={})
        try:
            for path in sorted(output.iterdir()):
                if path.is_file() and path.name != 'manifest.json':
                    try:
                        manifest['artifact_hashes'][path.name] = _sha256(path)
                    except Exception as receipt_error:
                        manifest.setdefault('artifact_hash_errors', {})[path.name] = str(receipt_error)
            _json(output / 'manifest.json', manifest)
        except Exception as receipt_error:
            error.add_note(f'Could not finish failure receipt: {receipt_error}')
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
