"""Execute a declared set of development probes, preserving every attempt."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import sys

from fluvius_cgan.development import _config
from fluvius_cgan.training import _json, _sha256
from evaluate_phase1 import _run_logged, _time


def run(plan_path, output, workers=3):
    if not 1 <= workers <= 4:
        raise ValueError('workers must be between 1 and 4')
    plan = json.loads(Path(plan_path).read_text())
    data = Path(plan['data']).resolve()
    if set(plan['data_hashes']) != {'train.npz', 'val.npz', 'scaler.json', 'metadata.json'}:
        raise ValueError('Plan must bind the four development data files')
    for name, digest in plan['data_hashes'].items():
        if _sha256(data / name) != digest:
            raise ValueError(f'Development data hash mismatch: {name}')
    jobs = plan['jobs']
    if not isinstance(jobs, dict) or not jobs:
        raise ValueError('Plan must declare named jobs')
    for name, config in jobs.items():
        if not re.fullmatch(r'[a-z0-9_]+', name):
            raise ValueError('Invalid job name')
        _config(config)
    resumes = plan.get('resume_from', {})
    if not isinstance(resumes, dict) or set(resumes) - set(jobs):
        raise ValueError('Resume entries must name declared jobs')
    for checkpoint in resumes.values():
        if not Path(checkpoint).is_file():
            raise FileNotFoundError(checkpoint)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'logs').mkdir()
    (output / 'configs').mkdir()
    _json(output / 'plan.json', plan)
    execution = {'status': 'running', 'started_at_utc': _time(), 'workers': workers,
                 'plan_sha256': _sha256(plan_path), 'test_performance_access': False, 'jobs': []}
    _json(output / 'execution.json', execution)
    def one(name, config):
        config_path = output / 'configs' / f'{name}.json'
        _json(config_path, config)
        command = [sys.executable, '-m', 'fluvius_cgan.development', '--data', str(data),
                   '--run', str(output / name), '--config', str(config_path)]
        if name in resumes:
            command += ['--resume', str(Path(resumes[name]).resolve())]
        record = _run_logged(name, command, output)
        _json(output / f'{name}.execution.json', record)
        print(f"{name}: exit={record['returncode']}", flush=True)
        return record
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, name, config) for name, config in jobs.items()]
        execution['jobs'] = [future.result() for future in futures]
    failed = any(job['returncode'] != 0 for job in execution['jobs'])
    execution.update(status='failed' if failed else 'completed', finished_at_utc=_time())
    _json(output / 'execution.json', execution)
    return int(failed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    raise SystemExit(run(args.plan, args.output, args.workers))
