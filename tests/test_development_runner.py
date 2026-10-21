"""Continuation plans must bind their inputs before any process is started."""
import importlib.util
import json
from pathlib import Path

import pytest

from fluvius_cgan.training import _sha256


def test_resume_plan_hashes_command_and_failed_attempt_are_preserved(tmp_path, monkeypatch):
    scripts = Path(__file__).parents[1] / 'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('development_runner', scripts / 'run_development.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    data = tmp_path / 'data'
    data.mkdir()
    names = ('train.npz', 'val.npz', 'scaler.json', 'metadata.json')
    for name in names:
        (data / name).write_bytes(name.encode())
    checkpoint = tmp_path / 'last.pt'
    checkpoint.write_bytes(b'checkpoint is checked but never loaded by the launcher')
    plan = {'data': str(data), 'data_hashes': {name: _sha256(data / name) for name in names},
            'jobs': {'continued': {'steps': 2}}, 'resume_from': {'continued': str(checkpoint)},
            'resume_checkpoint_sha256': {'continued': _sha256(checkpoint)}}
    path = tmp_path / 'plan.json'
    path.write_text(json.dumps(plan))
    calls = []
    def failed(name, command, output):
        calls.append(command)
        return {'name': name, 'command': command, 'returncode': 7}
    monkeypatch.setattr(runner, '_run_logged', failed)
    out = tmp_path / 'out'
    assert runner.run(path, out, workers=1) == 1
    assert calls[0][-2:] == ['--resume', str(checkpoint.resolve())]
    assert json.loads((out / 'execution.json').read_text())['status'] == 'failed'
    assert json.loads((out / 'continued.execution.json').read_text())['returncode'] == 7
    with pytest.raises(FileExistsError):
        runner.run(path, out, workers=1)
    checkpoint.write_bytes(b'changed checkpoint')
    with pytest.raises(ValueError, match='checkpoint hash mismatch'):
        runner.run(path, tmp_path / 'tampered', workers=1)
    assert len(calls) == 1
    assert not (tmp_path / 'tampered').exists()
    plan.update(method='flow', jobs={'flow': {'steps': 2, 'solver_steps': 2}},
                resume_from={}, resume_checkpoint_sha256={})
    path.write_text(json.dumps(plan))
    assert runner.run(path, tmp_path / 'flow', workers=1) == 1
    assert calls[-1][1:3] == ['-m', 'fluvius_cgan.flow']
    plan['jobs']['flow']['normalization'] = 'minmax'
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match='Unknown flow config'):
        runner.run(path, tmp_path / 'invalid_flow', workers=1)
    assert len(calls) == 2


def test_reflection_plan_dispatch_and_unsupported_resume(tmp_path, monkeypatch):
    scripts = Path(__file__).parents[1] / 'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('reflection_runner', scripts / 'run_development.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    data = tmp_path / 'data'
    data.mkdir()
    names = ('train.npz', 'val.npz', 'scaler.json', 'metadata.json')
    for name in names:
        (data / name).write_bytes(name.encode())
    plan = {'method': 'reflection', 'data': str(data),
            'data_hashes': {name: _sha256(data / name) for name in names},
            'jobs': {'reflected': {'steps': 2, 'solver_steps': 2}}}
    path = tmp_path / 'plan.json'
    path.write_text(json.dumps(plan))
    calls = []
    def process(name, command, output):
        calls.append(command)
        return {'name': name, 'command': command, 'returncode': 0}
    monkeypatch.setattr(runner, '_run_logged', process)
    assert runner.run(path, tmp_path / 'result', workers=1) == 0
    assert calls[0][1:3] == ['-m', 'fluvius_cgan.reflection']
    assert json.loads((tmp_path / 'result' / 'execution.json').read_text())['method'] == 'reflection'
    plan['resume_from'] = {'reflected': str(tmp_path / 'not-a-checkpoint')}
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match='continuation is not supported'):
        runner.run(path, tmp_path / 'rejected', workers=1)
    assert len(calls) == 1 and not (tmp_path / 'rejected').exists()
