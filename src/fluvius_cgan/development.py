"""Controlled TRAIN/inner-validation probes; distinct from the frozen benchmark.

All representations invert to physical kWh before scoring. No TEST reader or
test-based selection is provided. Networks and optimizer defaults follow the
reference; this module isolates explicitly recorded development changes.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shutil
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
            'validation_per_condition': 256, 'device': 'cpu', 'beta1': .9,
            'discriminator_dropout': .4, 'energy_weight': 0.}
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


def _conditional_energy(real, fake, labels):
    """Equal-condition Euclidean energy-distance U-statistic for training.

    Real/fake rows share one label vector. Conditions with n < 2 contribute
    nothing. For each eligible condition, use 2*RF/n² - (RR+FF)/(n*(n-1)),
    excluding RR/FF diagonals but including every RF pair. This is unbiased
    conditional on counts when real/fake profiles are independent draws from
    their condition distributions. Minibatches of dependent meter profiles do
    not provide that independence guarantee or any inferential calibration.

    Finite estimates can be negative and are retained. This statistical energy
    distance is unrelated to physical electricity energy or Energy Matching
    potentials. The real argument is detached; all-singleton batches yield a
    differentiable zero. No random draws or data-dependent clipping are used.
    """
    if (real.ndim != 2 or fake.shape != real.shape or not len(real)
            or not real.shape[1] or labels.shape != (len(real),)
            or real.dtype != fake.dtype or not fake.is_floating_point()
            or labels.dtype != torch.int64
            or real.device != fake.device or labels.device != fake.device):
        raise ValueError('Expected aligned floating [N,D] real/fake and int64 [N] labels')
    real = real.detach()
    _, inverse, group_counts = torch.unique(labels, return_inverse=True, return_counts=True)
    counts = group_counts[inverse].to(fake.dtype)
    eligible = counts >= 2
    groups = (group_counts >= 2).sum().clamp_min(1).to(fake.dtype)
    same = (labels[:, None] == labels[None, :]) & eligible[:, None]
    cross_weight = same.to(fake.dtype) / (groups * counts.square())[:, None]
    within = same & ~torch.eye(len(real), device=fake.device, dtype=torch.bool)
    within_weight = within.to(fake.dtype) / (groups * counts * (counts - 1).clamp_min(1))[:, None]
    # Direct subtraction avoids the squared-norm cancellation of the GEMM path
    # for nearby profiles. Strict CUDA deterministic mode may reject its backward
    # kernel; that explicit error is propagated, without changing devices/methods.
    mode = 'donot_use_mm_for_euclid_dist'
    cross = torch.cdist(real, fake, p=2, compute_mode=mode)
    real_pairs = torch.cdist(real, real, p=2, compute_mode=mode)
    fake_pairs = torch.cdist(fake, fake, p=2, compute_mode=mode)
    return (2 * (cross * cross_weight).sum()
            - (real_pairs * within_weight).sum() - (fake_pairs * within_weight).sum())


def _samples(model, representation, labels, seed, batch_size=2048):
    noise = np.random.default_rng(seed)
    outputs = []
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    try:
        with torch.no_grad():
            for start in range(0, len(labels), batch_size):
                batch = torch.from_numpy(labels[start:start + batch_size].astype(np.int64))
                z = torch.from_numpy(noise.standard_normal((len(batch), 24), dtype=np.float32))
                outputs.append(transform(model(z.to(device), batch.to(device)).cpu().numpy(),
                                         representation, inverse=True))
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
    if c['device'] not in ('cpu', 'cuda'):
        raise ValueError('device must be cpu or cuda')
    if type(c['beta1']) not in (int, float) or not 0 <= c['beta1'] < 1:
        raise ValueError('beta1 must be a number in [0,1)')
    if (type(c['discriminator_dropout']) not in (int, float)
            or not 0 <= c['discriminator_dropout'] < 1):
        raise ValueError('discriminator_dropout must be a number in [0,1)')
    if (type(c['energy_weight']) not in (int, float)
            or not np.isfinite(c['energy_weight']) or c['energy_weight'] < 0):
        raise ValueError('energy_weight must be a finite nonnegative number')
    return c


def _device(name):
    if name == 'cuda':
        # This must precede CUDA context creation, including device metadata reads.
        workspace = os.environ.get('CUBLAS_WORKSPACE_CONFIG')
        if workspace not in (None, ':4096:8', ':16:8'):
            raise RuntimeError('CUDA requires deterministic CUBLAS_WORKSPACE_CONFIG')
        if workspace is None and torch.cuda.is_initialized():
            raise RuntimeError('Set CUBLAS_WORKSPACE_CONFIG before initializing CUDA')
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested but is not available; no CPU fallback')
    return torch.device(name)


@contextmanager
def _execution(device):
    if device.type == 'cpu':
        yield
        return
    previous = (torch.are_deterministic_algorithms_enabled(),
                torch.is_deterministic_algorithms_warn_only_enabled(),
                torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
                torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    try:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        yield
    finally:
        torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = previous[2:4]
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous[4:6]


def _runtime(device, threads):
    result = {'python': platform.python_version(), 'numpy': np.__version__,
              'torch': str(torch.__version__), 'device': device.type, 'threads': threads,
              'deterministic_algorithms': True if device.type == 'cuda'
                  else torch.are_deterministic_algorithms_enabled(),
              'cuda_version': torch.version.cuda, 'cudnn_version': torch.backends.cudnn.version()}
    if device.type == 'cuda':
        result.update(gpu_name=torch.cuda.get_device_name(),
                      gpu_capability=list(torch.cuda.get_device_capability()),
                      cublas_workspace_config=os.environ['CUBLAS_WORKSPACE_CONFIG'],
                      cudnn_benchmark=False, cudnn_deterministic=True, allow_tf32=False)
    return result


def _resume(checkpoint, config, hashes, representation, selected, runtime):
    """Verify a completed run before carrying its immutable record forward.

    Interrupted-run recovery needs an atomic checkpoint bundle and is not
    promised by this continuation API. Device/backend changes are separate runs.
    """
    path = Path(checkpoint).resolve()
    if path.name != 'last.pt':
        raise ValueError('Resume requires a completed run last.pt checkpoint')
    manifest = json.loads((path.parent / 'manifest.json').read_text())
    if manifest.get('status') != 'completed' or manifest.get('schema_version') != SCHEMA:
        raise ValueError('Resume requires a completed development run')
    names = ('last.pt', 'best.pt', 'history.json', 'best_validation.json')
    artifact_hashes = {name: _sha256(path.parent / name) for name in names}
    if any(manifest.get('artifact_hashes', {}).get(name) != value
           for name, value in artifact_hashes.items()):
        raise ValueError('Resume artifact hash mismatch')
    state = torch.load(path, weights_only=True, map_location='cpu')
    previous = _config(state['config'])
    if ({key: value for key, value in config.items() if key != 'steps'} !=
            {key: value for key, value in previous.items() if key != 'steps'}):
        raise ValueError('Resume config differs from the original trajectory')
    if config['steps'] <= state['step'] or state['step'] != previous['steps']:
        raise ValueError('Resume requires a larger total step budget')
    if state['step'] != 1 and state['step'] % config['validation_every']:
        raise ValueError('Resume requires a scheduled validation boundary')
    if (_config(manifest['config']) != previous or manifest['completed_steps'] != state['step']
            or manifest['data_hashes'] != hashes or state['data_hashes'] != hashes
            or state['representation'] != representation or state['selection'] != selected
            or manifest['representation'] != representation or manifest['selection'] != selected
            or state['schema_version'] != SCHEMA):
        raise ValueError('Resume data, selection, representation, or manifest mismatch')
    old_runtime = manifest['runtime']
    for key in ('device', 'threads', 'python', 'numpy', 'torch'):
        if old_runtime.get(key) != runtime[key]:
            raise ValueError(f'Resume runtime differs: {key}')
    if runtime['device'] == 'cuda':
        for key in ('gpu_name', 'gpu_capability', 'cuda_version', 'cudnn_version',
                    'cublas_workspace_config', 'deterministic_algorithms', 'allow_tf32'):
            if old_runtime.get(key) != runtime[key]:
                raise ValueError(f'Resume CUDA runtime differs: {key}')
    best = torch.load(path.parent / 'best.pt', weights_only=True, map_location='cpu')
    if (best['step'] != state['best_step'] or best['best_score'] != state['best_score']
            or best['data_hashes'] != hashes or best['representation'] != representation
            or best['selection'] != selected
            or state['history'] != json.loads((path.parent / 'history.json').read_text())):
        raise ValueError('Resume selection/history mismatch')
    lineage = {'checkpoint': str(path), 'checkpoint_sha256': artifact_hashes['last.pt'],
               'manifest_sha256': _sha256(path.parent / 'manifest.json'),
               'prior_git_head': manifest['git_head'],
               'prior_source_sha256': manifest['source_sha256'],
               'prior_artifact_hashes': artifact_hashes,
               'prior_elapsed_seconds': manifest['elapsed_seconds']}
    return state, lineage


def train_probe(data_dir, run_dir, config, resume=None):
    c = _config(config)
    device = _device(c['device'])
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
    runtime = _runtime(device, c['num_threads'])
    resumed, lineage = (None, None) if resume is None else _resume(
        resume, c, hashes, representation, selected, runtime)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(c['num_threads'])
    source = {p.name: _sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    git = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).parents[2],
                         capture_output=True, text=True, check=True).stdout.strip()
    manifest = {'schema_version': SCHEMA, 'phase': 1, 'role': 'post-test development probe',
        'test_performance_access': False, 'config': c, 'data_hashes': hashes,
        'representation': representation, 'selection': selected,
        'metric_normalization': {'fitted_on': 'train', 'minimum': scaler['minimum'], 'maximum': scaler['maximum']},
        'optimizer': {'generator_lr': 1e-4, 'discriminator_lr': 2e-4, 'betas': [c['beta1'], .999], 'g_d_updates': [1, 1]},
        'discriminator_dropout': c['discriminator_dropout'],
        'generator_objective': {'adversarial': 'non-saturating BCE logits',
            'energy_weight': c['energy_weight'],
            'energy_distance': 'Euclidean conditional U-statistic in the TRAIN-fitted representation; equal eligible-condition weights',
            'energy_minimum_condition_count': 2, 'energy_negative_estimates': 'retained',
            'energy_uses_same_generator_batch': True},
        'sampling': 'shuffle all inner-training rows per pass; include final partial batch',
        'available_training_rows': len(training['x']), 'source_sha256': source, 'git_head': git,
        'runtime': runtime, 'resume': lineage,
        'started_at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'running',
        'prior_exposure': 'Development within original TRAIN; neither inner split is a newly untouched confirmatory holdout.'}
    _json(output / 'manifest.json', manifest)
    history, best, best_step, presentations, epoch, cursor = [], float('inf'), None, 0, 0, len(training['x'])
    start_step, elapsed_offset = 1, 0.
    if resumed is not None:
        start_step, history = resumed['step'] + 1, resumed['history']
        best, best_step = resumed['best_score'], resumed['best_step']
        presentations, epoch, cursor = (resumed['row_presentations'], resumed['passes_started'],
                                        resumed['sampler_cursor'])
        elapsed_offset = lineage['prior_elapsed_seconds']
        manifest['resumed_from_step'] = resumed['step']
        for name in ('best.pt', 'best_validation.json'):
            shutil.copyfile(Path(resume).parent / name, output / name)
        _json(output / 'history.json', history)
        _json(output / 'manifest.json', manifest)
    started = time.perf_counter()
    try:
        with _execution(device), torch.random.fork_rng(
                devices=[] if device.type == 'cpu' else [torch.cuda.current_device()]):
            torch.manual_seed(c['seed'])
            generator, discriminator = Generator().to(device), Discriminator().to(device)
            discriminator.model[-2].p = c['discriminator_dropout']
            optimizer_g = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(c['beta1'], .999))
            optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(c['beta1'], .999))
            sampler = torch.Generator().manual_seed(c['seed'] + 1)
            noise = torch.Generator().manual_seed(c['seed'] + 2)
            x = torch.from_numpy(transform(training['x'], representation).astype(np.float32)).to(device)
            y = torch.from_numpy(training['y']).to(device)
            if resumed is not None:
                generator.load_state_dict(resumed['generator'])
                discriminator.load_state_dict(resumed['discriminator'])
                optimizer_g.load_state_dict(resumed['optimizer_g'])
                optimizer_d.load_state_dict(resumed['optimizer_d'])
                sampler.set_state(resumed['sampler_rng_state'])
                noise.set_state(resumed['noise_rng_state'])
                order = resumed['sampler_order']
                torch.set_rng_state(resumed['model_rng_state'])
                if device.type == 'cuda':
                    torch.cuda.set_rng_state(resumed['cuda_rng_state'])
            loss_fn = nn.BCEWithLogitsLoss()
            sums = np.zeros(4)
            window = 0
            for step in range(start_step, c['steps'] + 1):
                if cursor >= len(x):
                    order = torch.randperm(len(x), generator=sampler)
                    cursor = 0
                    epoch += 1
                batch = order[cursor:cursor + c['batch_size']]
                cursor += len(batch)
                presentations += len(batch)
                batch_device = batch.to(device)
                real_batch, conditions = x[batch_device], y[batch_device]
                ones = torch.ones((len(batch), 1), device=device)
                optimizer_d.zero_grad(set_to_none=True)
                z = torch.randn(len(batch), 24, generator=noise).to(device)
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
                    z = torch.randn(len(batch), 24, generator=noise).to(device)
                    generated = generator(z, conditions)
                    adversarial_loss = loss_fn(discriminator(generated, conditions), ones)
                    if c['energy_weight']:
                        energy_loss = _conditional_energy(real_batch, generated, conditions)
                        g_loss = adversarial_loss + c['energy_weight'] * energy_loss
                    else:
                        energy_loss = adversarial_loss.new_zeros(())
                        g_loss = adversarial_loss
                    if not torch.isfinite(g_loss):
                        raise FloatingPointError('Nonfinite generator loss')
                    g_loss.backward()
                    optimizer_g.step()
                finally:
                    discriminator.requires_grad_(True)
                sums += [g_loss.item(), d_loss.item(), adversarial_loss.item(), energy_loss.item()]
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
                    'loss_adversarial': float(sums[2] / window),
                    'loss_energy': float(sums[3] / window),
                    'last_batch_real_logit_mean': float(real_logits.detach().mean()),
                    'last_batch_fake_logit_mean': float(fake_logits.detach().mean()),
                    'validation': metrics['macro'], 'diagnostics': metrics['diagnostics'],
                    'elapsed_seconds': elapsed_offset + time.perf_counter() - started}
                history.append(row)
                state = {'schema_version': SCHEMA, 'step': step, 'config': c,
                    'generator': generator.state_dict(), 'discriminator': discriminator.state_dict(),
                    'optimizer_g': optimizer_g.state_dict(), 'optimizer_d': optimizer_d.state_dict(),
                    'representation': representation, 'data_hashes': hashes, 'selection': selected,
                    'best_step': best_step, 'best_score': best, 'history': history,
                    'model_rng_state': torch.get_rng_state(), 'noise_rng_state': noise.get_state(),
                    'sampler_rng_state': sampler.get_state(), 'sampler_order': order,
                    'sampler_cursor': cursor, 'passes_started': epoch, 'row_presentations': presentations}
                if device.type == 'cuda':
                    state['cuda_rng_state'] = torch.cuda.get_rng_state()
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
            elapsed_seconds=elapsed_offset + time.perf_counter() - started,
            current_execution_seconds=time.perf_counter() - started,
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
    parser.add_argument('--resume', type=Path, help='Completed run last.pt; continue into a new output directory')
    args = parser.parse_args()
    train_probe(args.data, args.run, json.loads(args.config.read_text()), resume=args.resume)


if __name__ == '__main__':
    main()
