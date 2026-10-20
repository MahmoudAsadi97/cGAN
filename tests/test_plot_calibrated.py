"""Synthetic-only checks for provenance-bound calibrated validation figures."""
import builtins
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
import pytest
import torch

from fluvius_cgan.calibration import ConditionalMarginalCalibrator, fit
from fluvius_cgan.evaluation import evaluate_conditional
from fluvius_cgan.training import CHECKPOINT_FORMAT, _config
import fluvius_cgan.training as training


pytest.importorskip("matplotlib")
SPEC = importlib.util.spec_from_file_location(
    "plot_checkpoint", Path(__file__).resolve().parents[1] / "scripts" / "plot_checkpoint.py")
plotting = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plotting)


def digest(values, dtype):
    return hashlib.sha256(np.asarray(values, dtype=dtype).tobytes(order="C")).hexdigest()


def generated(labels, seed):
    return (np.random.default_rng(seed).normal(size=(len(labels), 24))
            + np.asarray(labels)[:, None] / 7 + np.linspace(-.4, .2, 24))


@pytest.fixture(scope="module")
def experiment(tmp_path_factory):
    root = tmp_path_factory.mktemp("calibrated-plots")
    prepared = root / "prepared"
    prepared.mkdir()
    rng = np.random.default_rng(61)
    arrays, manifest = {}, {}
    days = np.arange("2024-01-01", "2025-01-01", dtype="datetime64[D]")
    months = days.astype("datetime64[M]").astype(np.int64) % 12 + 1
    weekends = ((days.astype(np.int64) + 3) % 7 >= 5).astype(np.int64)
    calendar_codes = weekends * 4 + (months % 12) // 3
    for split, per_label in (("train", 4), ("val", 3)):
        labels = np.repeat(np.arange(64), per_label).astype(np.int64)
        arrays[split] = {
            "x": (rng.normal(size=(len(labels), 24)) + labels[:, None] / 20).astype(np.float32),
            "y": labels,
            "meter_id": np.asarray([f"{split}-{label // 8}" for label in labels]),
            "date": np.concatenate([days[calendar_codes == label % 8][:per_label].astype("U10")
                                    for label in range(64)]),
        }
        path = prepared / f"{split}.npz"
        np.savez(path, **arrays[split])
        manifest[split] = {
            "profiles": len(labels), "meters": 8,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "meter_ids": sorted(set(arrays[split]["meter_id"].tolist())),
        }
    (prepared / "test.npz").write_bytes(b"sealed synthetic test sentinel")
    scaler = {"minimum": float(arrays["train"]["x"].min()),
              "maximum": float(arrays["train"]["x"].max())}
    metadata = {
        "resolution": 24, "target": "net", "units": "kWh/interval",
        "source_sha256": "a" * 64, "split_seed": 42, "splits": manifest,
        "label_formula": "((HP*4 + EV*2 + PV)*2 + weekend)*4 + season",
    }
    (prepared / "scaler.json").write_text(json.dumps(scaler))
    (prepared / "metadata.json").write_text(json.dumps(metadata))
    indices = np.arange(len(arrays["train"]["x"]), dtype=np.int64)
    hashes = {name: hashlib.sha256((prepared / name).read_bytes()).hexdigest()
              for name in ("train.npz", "val.npz", "metadata.json", "scaler.json")}
    checkpoint = root / "reference.pt"
    torch.save({
        "format_version": CHECKPOINT_FORMAT, "epoch": 1,
        "config": _config({"seed": 7}), "metadata": metadata,
        "scaler": scaler, "output_dim": 24, "data_hashes": hashes,
        "selection": {"train_profiles_available": len(indices), "train_profiles_used": len(indices),
                      "train_indices_sha256": digest(indices, "<i8")},
        "supported_labels": list(range(64)),
    }, checkpoint)
    calibration_labels = np.repeat(np.arange(64), 5).astype(np.int64)
    reference = generated(calibration_labels, 7301)
    model = fit(arrays["train"]["x"], arrays["train"]["y"], reference, calibration_labels,
                quantile_knots=5, provenance={
                    "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    "train_indices_sha256": digest(indices, "<i8"),
                    "prepared_data_hashes": hashes, "calibration_seed": 7301,
                    "calibration_samples_per_condition": 5, "calibration_profiles_total": 320,
                    "calibration_generated_profiles_sha256": digest(reference, "<f8"),
                    "calibration_labels_sha256": digest(calibration_labels, "<i8"),
                    "generator_inference_batch_size": 1024, "quantile_knots": 5,
                })
    calibrator = root / "calibration.npz"
    model.save(calibrator)
    return prepared, checkpoint, calibrator, arrays


@pytest.fixture
def guards(monkeypatch):
    path_open, builtin_open, numpy_load = Path.open, builtins.open, np.load

    def check(path):
        if isinstance(path, (str, bytes, Path)):
            assert Path(path.decode() if isinstance(path, bytes) else path).name != "test.npz", \
                "test split must not be opened, loaded, or hashed"

    def guarded_path_open(path, *args, **kwargs):
        check(path)
        return path_open(path, *args, **kwargs)

    def guarded_builtin_open(path, *args, **kwargs):
        check(path)
        return builtin_open(path, *args, **kwargs)

    def guarded_numpy_load(path, *args, **kwargs):
        check(path)
        return numpy_load(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_path_open)
    monkeypatch.setattr(builtins, "open", guarded_builtin_open)
    monkeypatch.setattr(np, "load", guarded_numpy_load)
    calls = []

    def generate(checkpoint, labels, seed=42, **kwargs):
        calls.append((Path(checkpoint), np.array(labels, copy=True), seed))
        return generated(labels, seed)

    monkeypatch.setattr(plotting, "generate", generate)
    # The unchanged raw plotting path calls the production evaluator as well.
    monkeypatch.setattr(training, "generate", generate)
    return calls


@pytest.fixture
def capture_figures(monkeypatch):
    captured = {"examples": [], "saved": []}

    def median(plt, Line2D, Patch, real, fake, labels, hours, method):
        captured["median"] = (real.copy(), fake.copy(), labels.copy(), method)
        return plt.figure()

    def examples(plt, Line2D, real, fake, labels, hours, group, method, **kwargs):
        captured["examples"].append((real.copy(), fake.copy(), labels.copy(), group, method, kwargs))
        return plt.figure()

    def heatmap(plt, TwoSlopeNorm, metrics, method):
        captured["raw_heatmap"] = (metrics, method)
        return plt.figure()

    def calibrated_heatmap(plt, TwoSlopeNorm, raw_metrics, calibrated_metrics, method, raw_method):
        captured["calibrated_heatmap"] = (raw_metrics, calibrated_metrics, method, raw_method)
        return plt.figure()

    def save(figure, destination, stem, plt):
        names = []
        for extension in ("svg", "png"):
            path = destination / f"{stem}.{extension}"
            path.write_bytes(b"synthetic orchestration artifact; rendering tested separately")
            names.append(path.name)
        captured["saved"].append(stem)
        plt.close(figure)
        return names

    monkeypatch.setattr(plotting, "_median_figure", median)
    monkeypatch.setattr(plotting, "_examples_figure", examples)
    monkeypatch.setattr(plotting, "_heatmap_figure", heatmap)
    monkeypatch.setattr(plotting, "_calibration_heatmap_figure", calibrated_heatmap)
    monkeypatch.setattr(plotting, "_save", save)
    return captured


def selected_validation(arrays, seed, cap):
    rng = np.random.default_rng(seed)
    return np.concatenate([
        rng.choice(np.flatnonzero(arrays["val"]["y"] == label), min(cap, 3), replace=False)
        for label in range(64)
    ])


def metric_cells(metrics):
    return {
        metric: [[metrics[str(group * 8 + offset)][metric] for offset in range(8)] for group in range(8)]
        for metric in ("pointwise_wasserstein_kwh", "diversity_ratio")
    }


def test_bound_map_drives_bands_examples_metrics_and_provenance(
        experiment, guards, capture_figures, tmp_path):
    prepared, checkpoint, calibrator, arrays = experiment
    before = calibrator.read_bytes()
    output = tmp_path / "plots"
    seed, cap = 23, 2
    report = plotting.plot_checkpoint(prepared, checkpoint, output, calibrator=calibrator,
                                      seed=seed, samples_per_condition=cap)
    selected = selected_validation(arrays, seed, cap)
    real, labels = (arrays["val"][name][selected] for name in ("x", "y"))
    raw = generated(labels, seed)
    expected = ConditionalMarginalCalibrator.load(calibrator).transform(raw, labels)
    assert not np.array_equal(raw, expected)
    plotted_real, plotted_fake, plotted_labels, method = capture_figures["median"]
    np.testing.assert_array_equal(plotted_real, real)
    np.testing.assert_array_equal(plotted_fake, expected)
    np.testing.assert_array_equal(plotted_labels, labels)
    assert method == "Calibrated cGAN"
    assert len(capture_figures["examples"]) == 8
    for real_examples, fake_examples, example_labels, group, example_method, _ in capture_figures["examples"]:
        np.testing.assert_array_equal(real_examples, real)
        np.testing.assert_array_equal(fake_examples, expected)
        np.testing.assert_array_equal(example_labels, labels)
        assert example_method == method
    raw_metrics = evaluate_conditional(real, labels, raw, labels, interval_hours=1)["per_condition"]
    calibrated_metrics = evaluate_conditional(real, labels, expected, labels, interval_hours=1)["per_condition"]
    assert capture_figures["calibrated_heatmap"][:2] == (raw_metrics, calibrated_metrics)
    assert report["heatmap_values"] == metric_cells(calibrated_metrics)
    assert report["raw_heatmap_values"] == metric_cells(raw_metrics)
    assert report["calibrator"]["sha256"] == hashlib.sha256(before).hexdigest()
    assert report["calibrator"]["provenance"]["checkpoint_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert report["calibrator"]["training_condition_counts"] == {str(label): 4 for label in range(64)}
    assert report["frozen_samples"]["raw_profiles_sha256"] == digest(raw, "<f8")
    assert report["frozen_samples"]["calibrated_profiles_sha256"] == digest(expected, "<f8")
    assert report["method"] == method and report["test_used"] is False
    assert report["selected_validation_indices_sha256"] == digest(selected, "<i8")
    assert len(report["artifacts"]) == 20 and len(guards) == 1
    assert calibrator.read_bytes() == before
    assert json.loads((output / "plot_manifest.json").read_text()) == report


@pytest.mark.parametrize("field", ["checkpoint_sha256", "train_indices_sha256"])
def test_wrong_map_is_rejected_before_generation(experiment, guards, tmp_path, field):
    prepared, checkpoint, calibrator, _ = experiment
    model = ConditionalMarginalCalibrator.load(calibrator)
    model.provenance[field] = "0" * 64
    invalid = tmp_path / "invalid.npz"
    model.save(invalid)
    with pytest.raises(ValueError, match=field):
        plotting.plot_checkpoint(prepared, checkpoint, tmp_path / "plots", calibrator=invalid)
    assert guards == []
    assert not (tmp_path / "plots").exists()


def test_calibration_seed_cannot_be_reused_for_plot_generation(experiment, guards, tmp_path):
    prepared, checkpoint, calibrator, _ = experiment
    with pytest.raises(ValueError, match="seeds must be distinct"):
        plotting.plot_checkpoint(prepared, checkpoint, tmp_path / "plots", calibrator=calibrator, seed=7301)
    assert guards == []


def test_calibrator_changed_during_plotting_prevents_manifest(
        experiment, guards, capture_figures, monkeypatch, tmp_path):
    prepared, checkpoint, calibrator, _ = experiment
    mutable = tmp_path / "mutable-map.npz"
    shutil.copyfile(calibrator, mutable)
    original_save = plotting._save
    changed = False

    def mutate_after_save(*args, **kwargs):
        nonlocal changed
        names = original_save(*args, **kwargs)
        if not changed:
            mutable.write_bytes(mutable.read_bytes() + b"changed during plotting")
            changed = True
        return names

    monkeypatch.setattr(plotting, "_save", mutate_after_save)
    output = tmp_path / "plots"
    with pytest.raises(ValueError, match="calibrat.*changed"):
        plotting.plot_checkpoint(prepared, checkpoint, output, calibrator=mutable, samples_per_condition=2)
    assert changed and not (output / "plot_manifest.json").exists()


def test_raw_default_never_loads_calibrator(experiment, guards, capture_figures, monkeypatch, tmp_path):
    prepared, checkpoint, _, arrays = experiment

    def forbidden(*args, **kwargs):
        raise AssertionError("raw plotting must not load a calibrator")

    monkeypatch.setattr(plotting, "load_verified_calibrator", forbidden)
    report = plotting.plot_checkpoint(prepared, checkpoint, tmp_path / "raw", seed=23, samples_per_condition=2)
    selected = selected_validation(arrays, 23, 2)
    np.testing.assert_array_equal(capture_figures["median"][1], generated(arrays["val"]["y"][selected], 23))
    assert report["method"] == "Conditional generator"
    assert "calibrator" not in report and "raw_heatmap_values" not in report
    assert "calibrated_heatmap" not in capture_figures
    assert len(report["artifacts"]) == 20 and report["test_used"] is False


def test_raw_calibrated_heatmap_exports_svg_and_png_with_shared_scales(tmp_path):
    _, plt, TwoSlopeNorm, _, _ = plotting._matplotlib()
    raw = {str(label): {"pointwise_wasserstein_kwh": .1 + label / 100,
                       "diversity_ratio": .25 + label / 32} for label in range(64)}
    calibrated = {str(label): {"pointwise_wasserstein_kwh": .01 + label / 1000,
                              "diversity_ratio": 1. + label / 128} for label in range(64)}
    figure = plotting._calibration_heatmap_figure(
        plt, TwoSlopeNorm, raw, calibrated, "Calibrated cGAN", "Raw cGAN")
    image_axes = [axis for axis in figure.axes if axis.images]
    assert len(image_axes) == 4
    distances = [axis.images[0] for axis in image_axes if "Wasserstein" in axis.get_title()]
    diversities = [axis.images[0] for axis in image_axes if "diversity" in axis.get_title().lower()]
    assert len(distances) == len(diversities) == 2
    assert distances[0].get_clim() == distances[1].get_clim()
    assert diversities[0].get_clim() == diversities[1].get_clim()
    names = plotting._save(figure, tmp_path, "synthetic_calibration_heatmaps", plt)
    assert set(names) == {"synthetic_calibration_heatmaps.svg", "synthetic_calibration_heatmaps.png"}
    assert "<svg" in (tmp_path / names[0]).read_text()
    assert (tmp_path / names[1]).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
