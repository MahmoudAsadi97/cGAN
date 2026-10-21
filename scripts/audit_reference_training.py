"""Reproduce read-only upstream parity and historical TRAIN mean-drift audit.

Requires an explicit TRAIN npz, conditional-run directory, and pinned upstream
cGAN.py source. Does not discover or open TEST/VAL files, train on real profiles,
or alter saved runs. Synthetic optimizer replay uses five batches. The original
captured result predates this CLI wrapper; see REFERENCE_TRAINING_AUDIT.md.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json

import numpy as np
import torch
from fluvius_cgan.models import Generator, Discriminator


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_SHA256 = '23c72dc32495c694417382fdb7674b5fbefa919da50813fe144442933690de53'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parity(reference):
    spec = importlib.util.spec_from_file_location('reference_audit_upstream', reference)
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    torch.manual_seed(73)
    g1 = upstream.Generator(24, 64, 64, 24)
    d1 = upstream.Discriminator(24, 64, 64)
    g2, d2 = Generator(), Discriminator()
    g2.load_state_dict(g1.state_dict())
    d2.load_state_dict(d1.state_dict())
    real = .32 + .03 * torch.randn(200, 24)
    labels = torch.randint(0, 64, (200,))
    conditions = torch.nn.functional.one_hot(labels, 64)
    ones, zeros = torch.ones(200, 1), torch.zeros(200, 1)

    def replay(g, d, local):
        torch.manual_seed(192)
        og = torch.optim.Adam(g.parameters(), lr=1e-4)
        od = torch.optim.Adam(d.parameters(), lr=2e-4)
        loss = torch.nn.BCEWithLogitsLoss() if local else torch.nn.BCELoss()
        losses = []
        for _ in range(5):
            od.zero_grad(set_to_none=True)
            latent = torch.randn(200, 24)
            c = labels if local else conditions
            fake = g(latent, c)
            if local:
                fake = fake.detach()
            fake_output, real_output = d(fake, c), d(real, c)
            ld = (loss(real_output, ones) + loss(fake_output, zeros)) / 2
            ld.backward()
            od.step()
            og.zero_grad(set_to_none=True)
            latent = torch.randn(200, 24)
            if local:
                d.requires_grad_(False)
            lg = loss(d(g(latent, c), c), ones)
            lg.backward(retain_graph=not local)
            og.step()
            if local:
                d.requires_grad_(True)
            losses.append([float(lg.detach()), float(ld.detach())])
        return np.array(losses), torch.get_rng_state()

    a, ra = replay(g1, d1, False)
    b, rb = replay(g2, d2, True)
    return {
        'upstream_sha256': sha(reference),
        'batches': 5, 'synthetic_batch_size': 200,
        'max_loss_difference': float(np.max(abs(a-b))),
        'max_generator_parameter_difference': max(float((p-q).abs().max().detach()) for p, q in zip(g1.parameters(), g2.parameters())),
        'max_discriminator_parameter_difference': max(float((p-q).abs().max().detach()) for p, q in zip(d1.parameters(), d2.parameters())),
        'equal_dropout_rng_state': torch.equal(ra, rb),
    }


def drift(train, runs):
    with np.load(train, allow_pickle=False) as data:
        x, y = data['x'], data['y']
        real = np.stack([x[y == c].mean(0) for c in range(64)])
    labels = torch.arange(64).repeat_interleave(256)
    latent = torch.randn(len(labels), 24, generator=torch.Generator().manual_seed(1781))
    records = []
    checkpoints = sorted(runs.glob('*/best.pt'))
    if not checkpoints:
        raise ValueError('No seed-directory best.pt checkpoints found in --runs')
    for best in checkpoints:
        points = []
        for name in ('best.pt', 'last.pt'):
            path = best.parent / name
            saved = torch.load(path, map_location='cpu', weights_only=True)
            model = Generator()
            model.load_state_dict(saved['generator'])
            model.eval()
            with torch.no_grad():
                fake = model(latent, labels).numpy().reshape(64, 256, 24)
            scale = saved['scaler']['maximum'] - saved['scaler']['minimum']
            fake_mean = fake.mean(1) * scale + saved['scaler']['minimum']
            error = fake_mean - real
            common = error.mean(0)
            points.append({
                'file': display_path(path), 'sha256': sha(path),
                'epoch': saved['epoch'],
                'condition_mean_error_rmse_kwh': float(np.sqrt(np.mean(error**2))),
                'common_error_energy_fraction': float(64*np.sum(common**2)/np.sum(error**2)),
                'common_error_by_hour_kwh': common.tolist(),
            })
        a, b = [np.asarray(p['common_error_by_hour_kwh']) for p in points]
        records.append({
            'seed': best.parent.name, 'checkpoints': points,
            'best_last_common_error_cosine': float(a@b/np.linalg.norm(a)/np.linalg.norm(b)),
            'best_last_common_error_change_rmse_kwh': float(np.sqrt(np.mean((a-b)**2))),
        })
    return {'train_file': display_path(train), 'train_sha256': sha(train),
            'noise_seed': 1781, 'draws_per_condition': 256, 'records': records}


def display_path(path):
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True,
                        help='Historical hourly TRAIN file, named train.npz')
    parser.add_argument('--runs', type=Path, required=True,
                        help='Conditional run root containing seed_*/best.pt and last.pt')
    parser.add_argument('--reference', type=Path, required=True,
                        help='Upstream cGAN.py at commit 885f78052a91fbc755e6b17a354c68849b5556ca')
    args = parser.parse_args()
    train, runs, reference = (p.resolve() for p in (args.data, args.runs, args.reference))
    if train.name != 'train.npz':
        parser.error('--data must explicitly name a TRAIN file named train.npz')
    if not train.is_file() or not runs.is_dir() or not reference.is_file():
        parser.error('TRAIN file, run directory and reference source must exist')
    if sha(reference) != REFERENCE_SHA256:
        parser.error('--reference does not match the pinned upstream cGAN.py digest')
    torch.set_num_threads(1)
    result = {'runtime': {'torch': str(torch.__version__), 'numpy': np.__version__},
              'synthetic_parity': parity(reference), 'train_mean_drift': drift(train, runs)}
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
