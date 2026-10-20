import json
from pathlib import Path
import numpy as np
import pytest


def test_download_verifies_existing_file(tmp_path):
    from energy_cgan.cli import fetch_data
    path = tmp_path / "raw.zip"
    path.write_bytes(b"not the released dataset")
    with pytest.raises(ValueError, match="checksum"):
        fetch_data(path)
    assert path.read_bytes() == b"not the released dataset"


def test_cli_has_safe_default_evaluation_split():
    from energy_cgan.cli import make_parser
    args = make_parser().parse_args(["evaluate", "--data", "d", "--checkpoint", "m.pt", "--output", "r.json"])
    assert args.split == "val"


def test_stratified_evaluation_sample_retains_each_condition():
    from energy_cgan.cli import evaluation_indices
    y = np.array([0] * 100 + [1] * 3)
    indices = evaluation_indices(y, max_per_condition=5, seed=42)
    assert (y[indices] == 0).sum() == 5
    assert (y[indices] == 1).sum() == 3
    np.testing.assert_array_equal(indices, evaluation_indices(y, 5, 42))
    with pytest.raises(ValueError):
        evaluation_indices(y, 0, 42)


def test_generated_archive_preserves_condition_and_checkpoint_provenance(tmp_path):
    from energy_cgan.cli import main, sha256
    from energy_cgan.training import train
    data = tmp_path / "data"
    data.mkdir()
    np.savez_compressed(data / "dataset.npz", x=np.ones((8, 24), dtype=np.float32),
                        y=np.zeros(8, dtype=np.int64),
                        split=np.array(["train"]*4 + ["val"]*2 + ["test"]*2),
                        meter_id=np.array([str(i) for i in range(8)]),
                        date=np.array(["2024-01-01"]*8))
    (data / "metadata.json").write_text(json.dumps({"classes":["weekday"], "unit":"kWh",
                                                    "interval_hours":1.0,"target":"import"}))
    run = train(data, tmp_path / "run", {"epochs":1,"hidden_dim":8,"device":"cpu"})
    output = tmp_path / "generated.npz"
    main(["generate", "--checkpoint", run["best_checkpoint"], "--condition", "0",
          "--num-samples", "2", "--seed", "17", "--output", str(output)])
    with np.load(output, allow_pickle=False) as generated:
        assert generated["x"].shape == (2, 24)
        provenance = json.loads(str(generated["provenance"]))
    assert provenance["seed"] == 17
    assert provenance["classes"] == ["weekday"]
    assert provenance["checkpoint_sha256"] == sha256(Path(run["best_checkpoint"]))
    assert provenance["unit"] == "kWh"
    assert provenance["interval_hours"] == 1.0
