import importlib.util
from pathlib import Path

import numpy as np
import pytest

from fluvius_cgan.calibration import fit
from fluvius_cgan.release import sha256_file

SPEC = importlib.util.spec_from_file_location(
    "generate_calibrated", Path(__file__).resolve().parents[1] / "scripts/generate_calibrated.py")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_bound_map_generation_and_rejection(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"synthetic checkpoint for mocked generation")
    labels = np.zeros(8, dtype=np.int64)
    raw = np.arange(8 * 24, dtype=float).reshape(8, 24)
    model = fit(2 * raw + 1, labels, raw, labels, quantile_knots=None,
                provenance={"checkpoint_sha256": sha256_file(checkpoint)})
    path = tmp_path / "map.npz"
    model.save(path)
    monkeypatch.setattr(module, "generate", lambda *a, **kw: raw.copy())
    result, metadata = module.generate_profiles(checkpoint, path, labels)
    np.testing.assert_allclose(result, 2 * raw + 1)
    assert metadata["calibrator_sha256"] == sha256_file(path)
    checkpoint.write_bytes(b"different checkpoint")
    with pytest.raises(ValueError, match="does not belong"):
        module.generate_profiles(checkpoint, path, labels)


def test_changed_artifact_and_output_guard(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"model")
    labels = np.zeros(8, dtype=np.int64)
    raw = np.arange(8 * 24, dtype=float).reshape(8, 24)
    path = tmp_path / "map.npz"
    fit(raw, labels, raw, labels, provenance={"checkpoint_sha256": sha256_file(checkpoint)}).save(path)
    def changed(*a, **kw):
        checkpoint.write_bytes(b"modified during generation")
        return raw
    monkeypatch.setattr(module, "generate", changed)
    with pytest.raises(ValueError, match="changed during"):
        module.generate_profiles(checkpoint, path, labels)
    output = tmp_path / "existing.npz"
    output.write_bytes(b"preserve")
    with pytest.raises(SystemExit):
        module.main(["--checkpoint", str(checkpoint), "--calibrator", str(path),
                     "--labels", "0", "--output", str(output)])
    assert output.read_bytes() == b"preserve"
