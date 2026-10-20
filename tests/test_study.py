import hashlib
import json

import numpy as np
import pytest

from fluvius_cgan.study import evaluate_run, freeze_protocol


def fixture_data(tmp_path):
    root = tmp_path / "prepared"
    root.mkdir()
    metadata = {"resolution": 24, "interval_hours": 1.0,
                "source_sha256": "a" * 64, "splits": {}}
    for split, prefix in [("train", "t"), ("val", "v"), ("test", "h")]:
        x = np.arange(8 * 24, dtype=np.float32).reshape(8, 24) / 50 - 1
        labels = np.repeat([0, 8], 4).astype(np.int64)
        np.savez_compressed(root / f"{split}.npz", x=x, y=labels,
                            meter_id=np.array([f"{prefix}{i}" for i in range(8)]),
                            date=np.array(["2024-01-01"] * 8))
        metadata["splits"][split] = {
            "sha256": hashlib.sha256((root / f"{split}.npz").read_bytes()).hexdigest()}
    (root / "metadata.json").write_text(json.dumps(metadata))
    (root / "scaler.json").write_text(json.dumps(
        {"minimum": float(x.min()), "maximum": float(x.max()), "fitted_on": "train"}))
    return root


def test_baseline_evaluation_records_equal_counts_and_seed_replicates(tmp_path):
    data = fixture_data(tmp_path)
    out = tmp_path / "evaluation.json"
    result = evaluate_run(data, "mean", "val", [31, 32, 33], out)
    assert result["split"] == "val"
    assert result["generation_seeds"] == [31, 32, 33]
    assert len(result["replicates"]) == 3
    assert result["replicates"][0]["metrics"]["macro"]["diversity_ratio"] == 0
    assert result["test_performance_access"] is False
    for rep in result["replicates"]:
        for value in rep["metrics"]["per_condition"].values():
            assert value["n_real"] == value["n_fake"] == 4
    assert json.loads(out.read_text()) == result
    with pytest.raises(FileExistsError):
        evaluate_run(data, "mean", "val", [31], out)


def test_test_evaluation_requires_frozen_protocol_before_reading_test(tmp_path):
    data = fixture_data(tmp_path)
    (data / "test.npz").unlink()
    with pytest.raises(ValueError, match="frozen protocol"):
        evaluate_run(data, "resample", "test", [31], tmp_path / "out.json")


def test_frozen_test_evaluation_binds_data_seeds_and_records_access(tmp_path):
    data = fixture_data(tmp_path)
    protocol = tmp_path / "protocol.json"
    freeze_protocol(data, protocol, training_config={"epochs": 5},
                    generation_seeds=[31, 32, 33])
    result = evaluate_run(data, "resample", "test", [31, 32, 33],
                          tmp_path / "result.json", protocol=protocol)
    assert result["test_performance_access"] is True
    assert result["protocol_sha256"] == hashlib.sha256(protocol.read_bytes()).hexdigest()
    events = [json.loads(line) for line in (tmp_path / "test_access.jsonl").read_text().splitlines()]
    assert events[0]["event"] == "test_evaluation_started"
    assert events[-1]["event"] == "test_evaluation_completed"
    with pytest.raises(ValueError, match="generation seeds"):
        evaluate_run(data, "mean", "test", [1], tmp_path / "other.json", protocol=protocol)


def test_modified_dataset_is_rejected(tmp_path):
    data = fixture_data(tmp_path)
    with (data / "val.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="hash"):
        evaluate_run(data, "mean", "val", [31], tmp_path / "result.json")


def test_protocol_cannot_be_silently_replaced(tmp_path):
    data = fixture_data(tmp_path)
    protocol = tmp_path / "protocol.json"
    freeze_protocol(data, protocol, training_config={"epochs": 5})
    with pytest.raises(FileExistsError):
        freeze_protocol(data, protocol, training_config={"epochs": 6})
