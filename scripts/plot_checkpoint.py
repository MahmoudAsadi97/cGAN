"""Create reproducible validation figures for a frozen generator checkpoint.

Example:
    python scripts/plot_checkpoint.py --prepared data/prepared96 \
        --checkpoint runs/candidate/best.pt --output-dir figures/candidate

All 64 conditions are required. Curves and example rows follow predeclared
selection rules; real and generated rows are independent, never paired. This
utility reads train/validation data only and has no test-evaluation option.
Matplotlib is an optional plotting dependency.

Optional --calibrator applies a previously fitted, provenance-verified marginal
map. It adds no fitting step: bands/examples show the calibrated sample, while
the heatmap compares the same raw and calibrated draws on common color scales.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_checkpoints import _checkpoint, _validate_manifest
from fluvius_cgan.calibration import load_verified_calibrator
from fluvius_cgan.cli import evaluate_checkpoint
from fluvius_cgan.evaluation import evaluate_conditional
from fluvius_cgan.release import sha256_file
from fluvius_cgan.training import _load_prepared, _read_split, generate


CONDITIONS = np.arange(64, dtype=np.int64)
WINTER_WEEKDAY_LABELS = np.arange(0, 64, 8, dtype=np.int64)
SEASONS = ("Winter", "Spring", "Summer", "Autumn")
REAL_COLOR = "#2369A5"
GENERATED_COLOR = "#D45B28"


def _array_hash(values):
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def _profile_hash(values):
    return hashlib.sha256(np.asarray(values, dtype="<f8").tobytes(order="C")).hexdigest()


def _technology(group):
    return f"HP={group // 4}  EV={(group // 2) % 2}  PV={group % 2}"


def _matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        from matplotlib.colors import TwoSlopeNorm
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
    except ImportError as exc:
        raise ImportError(
            "Plotting requires the optional matplotlib dependency. "
            "Install it with: python -m pip install matplotlib"
        ) from exc
    return matplotlib, plt, TwoSlopeNorm, Line2D, Patch


def _save(figure, destination, stem, plt):
    names = []
    for extension in ("svg", "png"):
        path = destination / f"{stem}.{extension}"
        options = {"metadata": {"Date": None}} if extension == "svg" else {"dpi": 180}
        figure.savefig(path, bbox_inches="tight", **options)
        names.append(path.name)
    plt.close(figure)
    return names


def _style_profile_axis(axis, title):
    axis.set_title(title, fontsize=10)
    axis.axhline(0, color="#777777", linewidth=.6, alpha=.65, zorder=0)
    axis.set_xlim(0, 24)
    axis.set_xticks([0, 6, 12, 18, 24])
    axis.grid(axis="y", alpha=.17, linewidth=.6)
    axis.spines[["top", "right"]].set_visible(False)


def _median_figure(plt, Line2D, Patch, real, fake, labels, hours, method):
    figure, axes = plt.subplots(2, 4, figsize=(14, 7), sharex=True, sharey=True)
    for axis, label in zip(axes.flat, WINTER_WEEKDAY_LABELS):
        mask = labels == label
        for values, color in ((real[mask], REAL_COLOR), (fake[mask], GENERATED_COLOR)):
            lower, median, upper = np.quantile(values, [.05, .5, .95], axis=0)
            axis.fill_between(hours, lower, upper, color=color, alpha=.17, linewidth=0)
            axis.plot(hours, median, color=color, linewidth=1.7)
        _style_profile_axis(axis, f"Label {label}: {_technology(int(label) // 8)}")
    figure.suptitle(f"{method}: winter weekdays across eight technology groups", fontsize=14, y=.99)
    figure.supxlabel("Local hour (Europe/Brussels)", y=.06)
    figure.supylabel("Net power (kW)", x=.01)
    figure.legend(handles=[
        Line2D([], [], color=REAL_COLOR, linewidth=1.7, label="Validation median"),
        Line2D([], [], color=GENERATED_COLOR, linewidth=1.7, label="Generated median"),
        Patch(facecolor="#777777", alpha=.17, label="Pointwise 5–95% range (not a confidence interval)"),
    ], loc="lower center", bbox_to_anchor=(.5, .005), ncol=3, frameon=False, fontsize=9)
    figure.subplots_adjust(left=.07, right=.99, top=.9, bottom=.16, wspace=.13, hspace=.27)
    return figure


def _heatmap_figure(plt, TwoSlopeNorm, metrics, method):
    distance = np.array([metrics[str(label)]["pointwise_wasserstein_kwh"]
                         for label in CONDITIONS]).reshape(8, 8)
    diversity = np.array([np.nan if metrics[str(label)]["diversity_ratio"] is None
                          else metrics[str(label)]["diversity_ratio"]
                          for label in CONDITIONS]).reshape(8, 8)
    finite = diversity[np.isfinite(diversity)]
    maximum = max(2.0, float(finite.max())) if len(finite) else 2.0
    figure, axes = plt.subplots(1, 2, figsize=(15, 7.3))
    column_names = [f"{day}\n{season}" for day in ("Weekday", "Weekend") for season in SEASONS]
    specifications = (
        (distance, "Interval Wasserstein distance", "kWh / interval", "viridis", None),
        (diversity, "Within-condition diversity ratio", "Generated / validation variance", "RdBu_r",
         TwoSlopeNorm(vmin=0, vcenter=1, vmax=maximum)),
    )
    for axis, (values, title, units, cmap_name, norm) in zip(axes, specifications):
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad("#CCCCCC")
        image = axis.imshow(np.ma.masked_invalid(values), cmap=cmap, norm=norm, aspect="auto")
        axis.set_title(title, fontsize=12, pad=12)
        axis.set_xticks(np.arange(8), column_names, fontsize=8)
        axis.set_yticks(np.arange(8), [_technology(group) for group in range(8)], fontsize=9)
        axis.tick_params(axis="both", length=0)
        axis.set_xticks(np.arange(-.5, 8, 1), minor=True)
        axis.set_yticks(np.arange(-.5, 8, 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=.7)
        axis.tick_params(which="minor", bottom=False, left=False)
        for row in range(8):
            for column in range(8):
                value = values[row, column]
                text = "NA" if not np.isfinite(value) else f"{value:.3f}"
                rgba = image.cmap(image.norm(value)) if np.isfinite(value) else (.8, .8, .8, 1)
                lightness = .2126 * rgba[0] + .7152 * rgba[1] + .0722 * rgba[2]
                axis.text(column, row, text, ha="center", va="center", fontsize=8,
                          color="black" if lightness > .55 else "white")
        figure.colorbar(image, ax=axis, shrink=.78, pad=.025, label=units)
    figure.suptitle(f"{method}: all 64 validation conditions", fontsize=14, y=.98)
    figure.text(.5, .025, "Each cell is one condition. Lower Wasserstein is better; diversity ratio 1 matches variance only. "
                "NA denotes zero real variance.", ha="center", fontsize=9)
    figure.subplots_adjust(left=.1, right=.96, bottom=.15, top=.88, wspace=.38)
    return figure


def _calibration_heatmap_figure(plt, TwoSlopeNorm, raw_metrics, calibrated_metrics, method, raw_method):
    """Compare all 64 conditions with one shared color scale per metric."""
    specifications = (
        ("pointwise_wasserstein_kwh", "Interval Wasserstein distance", "kWh / interval", "viridis"),
        ("diversity_ratio", "Within-condition diversity ratio", "Generated / validation variance", "RdBu_r"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(16, 12))
    column_names = [f"{day}\n{season}" for day in ("Weekday", "Weekend") for season in SEASONS]
    for row, (metric, title, units, cmap_name) in enumerate(specifications):
        grids = [np.array([np.nan if metrics[str(label)][metric] is None else metrics[str(label)][metric]
                           for label in CONDITIONS]).reshape(8, 8)
                 for metrics in (raw_metrics, calibrated_metrics)]
        finite = np.concatenate([grid[np.isfinite(grid)] for grid in grids])
        maximum = max(2.0, float(finite.max())) if metric == "diversity_ratio" and len(finite) else 2.0
        norm = TwoSlopeNorm(vmin=0, vcenter=1, vmax=maximum) if metric == "diversity_ratio" else None
        vmax = max(float(finite.max()), np.finfo(float).eps) if len(finite) else 1.0
        for column, (values, name) in enumerate(zip(grids, (raw_method, method))):
            axis = axes[row, column]
            cmap = plt.get_cmap(cmap_name).copy()
            cmap.set_bad("#CCCCCC")
            scale = {"norm": norm} if norm is not None else {"vmin": 0, "vmax": vmax}
            image = axis.imshow(np.ma.masked_invalid(values), cmap=cmap, aspect="auto", **scale)
            axis.set_title(f"{name}: {title}", fontsize=11, pad=12)
            axis.set_xticks(np.arange(8), column_names, fontsize=8)
            axis.set_yticks(np.arange(8), [_technology(group) for group in range(8)], fontsize=9)
            axis.tick_params(axis="both", length=0)
            axis.set_xticks(np.arange(-.5, 8, 1), minor=True)
            axis.set_yticks(np.arange(-.5, 8, 1), minor=True)
            axis.grid(which="minor", color="white", linewidth=.7)
            axis.tick_params(which="minor", bottom=False, left=False)
            for group in range(8):
                for condition in range(8):
                    value = values[group, condition]
                    text = "NA" if not np.isfinite(value) else f"{value:.3f}"
                    rgba = image.cmap(image.norm(value)) if np.isfinite(value) else (.8, .8, .8, 1)
                    lightness = .2126 * rgba[0] + .7152 * rgba[1] + .0722 * rgba[2]
                    axis.text(condition, group, text, ha="center", va="center", fontsize=8,
                              color="black" if lightness > .55 else "white")
        color_axis = figure.add_axes([.925, .56 if row == 0 else .16, .014, .30])
        figure.colorbar(image, cax=color_axis, label=units)
    figure.suptitle(f"{method}: raw and calibrated metrics across all 64 validation conditions", fontsize=14, y=.97)
    figure.text(.5, .045, "Same validation rows and raw generator draws; shared color scales within each metric. "
                "Lower Wasserstein is better; diversity ratio 1 matches variance only.", ha="center", fontsize=9)
    figure.text(.5, .025, "The fixed marginal map moves samples toward training marginals. "
                "Marginal improvement alone does not establish learned dependence. NA denotes zero real variance.",
                ha="center", fontsize=9)
    figure.subplots_adjust(left=.095, right=.895, bottom=.13, top=.91, wspace=.32, hspace=.40)
    return figure


def _examples_figure(plt, Line2D, real, fake, labels, hours, group, method, *, calibrated=False):
    figure, axes = plt.subplots(2, 4, figsize=(14, 7), sharex=True, sharey=True)
    for offset, axis in enumerate(axes.flat):
        label = group * 8 + offset
        positions = np.flatnonzero(labels == label)[:3]
        for values, color in ((real, REAL_COLOR), (fake, GENERATED_COLOR)):
            for position in positions:
                axis.plot(hours, values[position], color=color, linewidth=1, alpha=.75)
        day = "Weekend" if offset >= 4 else "Weekday"
        _style_profile_axis(axis, f"Label {label}: {day}, {SEASONS[offset % 4].lower()}")
    figure.suptitle(f"{method}: {_technology(group)} — first up to three profiles per condition", fontsize=14, y=.99)
    figure.supxlabel("Local hour (Europe/Brussels)", y=.065)
    figure.supylabel("Net power (kW)", x=.01)
    figure.legend(handles=[
        Line2D([], [], color=REAL_COLOR, label="Validation profiles"),
        Line2D([], [], color=GENERATED_COLOR, label="Generated profiles"),
    ], loc="lower center", bbox_to_anchor=(.5, .02), ncol=2, frameon=False)
    note = ("Independent real/generated samples; rows are not paired. First sampled rows; "
            "fixed marginal map applied. No ranking or smoothing." if calibrated else
            "Independent samples; rows are not paired. First sampled rows are shown without ranking, "
            "replacement, smoothing, or clipping.")
    figure.text(.5, .005, note, ha="center", fontsize=9)
    figure.subplots_adjust(left=.07, right=.99, top=.9, bottom=.17, wspace=.13, hspace=.27)
    return figure


def plot_checkpoint(prepared, checkpoint, output_dir, *, seed=2026, samples_per_condition=100, calibrator=None):
    """Write ten figures in SVG/PNG plus a manifest, using one plotting sample."""
    if samples_per_condition < 1 or not 0 <= seed < 2 ** 32:
        raise ValueError("samples-per-condition must be positive; seed must be in [0, 2**32)")
    prepared, checkpoint, output_dir = Path(prepared), Path(checkpoint), Path(output_dir)
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ValueError("output-dir must be new or empty; existing results are never overwritten")
    matplotlib, plt, TwoSlopeNorm, Line2D, Patch = _matplotlib()
    initial_checkpoint_hash = sha256_file(checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if sorted(state["supported_labels"]) != CONDITIONS.tolist():
        raise ValueError("these predeclared figures require checkpoint support for all 64 conditions")
    calibration, calibration_details = None, None
    method_key = "cgan" if calibrator is None else "calibrated_cgan"
    if calibrator is None:
        # Preserve the raw plotting path and its production-evaluator metrics.
        report = evaluate_checkpoint(prepared, checkpoint, split="val",
                                     samples_per_condition=samples_per_condition, seed=seed)
        if report["checkpoint_sha256"] != initial_checkpoint_hash:
            raise ValueError("checkpoint changed during evaluation")
        val, val_hash = _read_split(prepared / "val.npz")
        if val_hash != report["evaluation_sha256"]:
            raise ValueError("validation data changed after evaluation")
    else:
        # Reuse the comparison workflow's verification and exact training-row
        # reconstruction. The shared binder owns every calibration-map check.
        train, val, metadata, scaler, hashes = _load_prepared(prepared)
        _validate_manifest(metadata, hashes, train, val)
        details, indices = _checkpoint(checkpoint, train, metadata, scaler, hashes)
        if details["sha256"] != initial_checkpoint_hash:
            raise ValueError("checkpoint changed during calibration verification")
        calibration, calibration_details = load_verified_calibrator(
            calibrator, checkpoint_sha256=details["sha256"],
            train_indices_sha256=details["training_selection"]["train_indices_sha256"],
            prepared_data_hashes=hashes, generation_seed=seed,
            n_features=metadata["resolution"], train_labels=train["y"][indices],
        )
        val_hash = hashes["val.npz"]
        report = {
            "train_sha256": hashes["train.npz"], "evaluation_sha256": val_hash,
            "training_profiles": {"prepared": len(train["x"]), "model": len(indices), "calibration": len(indices)},
            "sampling": "Uniform without replacement within each observed condition; macro averages weight conditions equally.",
        }
        del train
    if np.unique(val["y"]).tolist() != CONDITIONS.tolist():
        raise ValueError("these predeclared figures require all 64 validation conditions")
    rng = np.random.default_rng(seed)
    selected = np.concatenate([
        rng.choice(np.flatnonzero(val["y"] == label),
                   min(samples_per_condition, int(np.sum(val["y"] == label))), replace=False)
        for label in np.unique(val["y"])
    ])
    labels = val["y"][selected]
    real = val["x"][selected].astype(np.float64)
    raw_generated = generate(checkpoint, labels, seed=seed).astype(np.float64)
    generated = raw_generated if calibration is None else calibration.transform(raw_generated, labels)
    if sha256_file(checkpoint) != initial_checkpoint_hash:
        raise ValueError("checkpoint changed during plotting-sample generation")
    counts = {str(label): int(np.sum(labels == label)) for label in CONDITIONS}
    resolution = real.shape[1]
    interval_hours = 24 / resolution
    if calibration is not None:
        report["methods"] = {
            "raw_cgan": evaluate_conditional(real, labels, raw_generated, labels, interval_hours=interval_hours),
            "calibrated_cgan": evaluate_conditional(real, labels, generated, labels, interval_hours=interval_hours),
        }
        if sha256_file(calibrator) != calibration_details["sha256"]:
            raise ValueError("calibrator changed during plotting-sample generation")
    if counts != report["methods"][method_key]["counts"]["real"]:
        raise ValueError("plot sample condition counts differ from verified evaluation")
    hours = np.arange(resolution) * interval_hours
    real_power, generated_power = real / interval_hours, generated / interval_hours
    method = "Unconditional generator" if state["config"]["unconditional"] else "Conditional generator"
    raw_method = "Raw unconditional generator" if state["config"]["unconditional"] else "Raw cGAN"
    if calibration is not None:
        method = "Calibrated unconditional generator" if state["config"]["unconditional"] else "Calibrated cGAN"
    if state["metadata"]["target"] != "net":
        raise ValueError("these net-power figures require the signed net target")
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise ValueError("output directory became nonempty; refusing to overwrite results")
    artifacts = []
    style = {
        "font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none",
        "svg.hashsalt": f"{initial_checkpoint_hash}:{seed}", "axes.titlesize": 10,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    }
    if calibration_details is not None:
        style["svg.hashsalt"] += f":{calibration_details['sha256']}"
    with matplotlib.rc_context(style):
        figure = _median_figure(plt, Line2D, Patch, real_power, generated_power, labels, hours, method)
        artifacts.extend(_save(figure, output_dir, "winter_weekday_quantiles", plt))
        figure = (_heatmap_figure(plt, TwoSlopeNorm, report["methods"][method_key]["per_condition"], method)
                  if calibration is None else
                  _calibration_heatmap_figure(plt, TwoSlopeNorm, report["methods"]["raw_cgan"]["per_condition"],
                                             report["methods"][method_key]["per_condition"], method, raw_method))
        artifacts.extend(_save(figure, output_dir, "conditional_metric_heatmaps", plt))
        for group in range(8):
            figure = _examples_figure(plt, Line2D, real_power, generated_power, labels, hours, group, method,
                                     calibrated=calibration is not None)
            artifacts.extend(_save(figure, output_dir, f"profile_examples_technology_{group}", plt))
    manifest = {
        "schema_version": 1, "split": "val", "test_used": False,
        "source_sha256": state["metadata"]["source_sha256"],
        "checkpoint_sha256": initial_checkpoint_hash, "checkpoint_epoch": state["epoch"],
        "checkpoint_config": state["config"], "method": method,
        "data_hashes": {
            "train.npz": report["train_sha256"], "val.npz": val_hash,
            "metadata.json": state["data_hashes"]["metadata.json"],
            "scaler.json": state["data_hashes"]["scaler.json"],
        },
        "training_profiles": report["training_profiles"],
        "train_indices_sha256": state["selection"]["train_indices_sha256"],
        "seed": seed, "samples_per_condition_cap": samples_per_condition,
        "label_counts": counts, "profile_count": len(selected),
        "selected_validation_indices_sha256": _array_hash(selected),
        "generated_label_order_sha256": _array_hash(labels),
        "generation_batch_size": 1024,
        "resolution": resolution, "interval_hours": interval_hours,
        "stored_units": "kWh/interval", "plotted_profile_units": "kW",
        "power_conversion": "interval energy divided by interval_hours; 0.25 hours for 96-slot data",
        "time_axis": "Local Europe/Brussels wall-clock interval start hours",
        "sampling": report["sampling"],
        "quantile_labels": WINTER_WEEKDAY_LABELS.tolist(), "quantiles": [.05, .5, .95],
        "example_selection": "First up to three rows in seeded sample order for every label; "
                             "generated rows selected independently in the same label order. "
                             "Real/generated rows are not paired. No ranking, smoothing, or clipping.",
        "example_sample_positions": {
            str(label): np.flatnonzero(labels == label)[:3].tolist() for label in CONDITIONS
        },
        "example_validation_indices": {
            str(label): selected[np.flatnonzero(labels == label)[:3]].tolist() for label in CONDITIONS
        },
        "condition_formula": state["metadata"]["label_formula"],
        "heatmap_rows": [_technology(group) for group in range(8)],
        "heatmap_columns": [f"{day}: {season.lower()}" for day in ("weekday", "weekend") for season in SEASONS],
        "heatmap_values": {
            metric: [[report["methods"][method_key]["per_condition"][str(group * 8 + offset)][metric]
                      for offset in range(8)] for group in range(8)]
            for metric in ("pointwise_wasserstein_kwh", "diversity_ratio")
        },
        "heatmap_metrics_source": "Verified evaluate_checkpoint report, methods.cgan.per_condition; "
                                  "the historical cgan key identifies the supplied checkpoint even when unconditional.",
        "interpretation": "Descriptive validation figures for a frozen checkpoint. Pointwise bands are "
                          "empirical ranges, not confidence intervals; example rows are unpaired. "
                          "Variance matching alone does not establish joint-profile fidelity.",
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "torch": str(torch.__version__), "matplotlib": matplotlib.__version__},
        "artifacts": {name: {"sha256": sha256_file(output_dir / name)} for name in artifacts},
    }
    if calibration_details is not None:
        if sha256_file(checkpoint) != initial_checkpoint_hash:
            raise ValueError("checkpoint changed during calibrated plotting")
        if sha256_file(calibrator) != calibration_details["sha256"]:
            raise ValueError("calibrator changed during plotting")
        manifest.update({
            "calibrator": calibration_details, "calibrator_sha256": calibration_details["sha256"],
            "raw_method": raw_method,
            "frozen_samples": {
                "hash_encoding": "Little-endian float64 C-order profile bytes",
                "real_profiles_sha256": _profile_hash(real),
                "raw_profiles_sha256": _profile_hash(raw_generated),
                "calibrated_profiles_sha256": _profile_hash(generated),
            },
            "raw_heatmap_values": {
                metric: [[report["methods"]["raw_cgan"]["per_condition"][str(group * 8 + offset)][metric]
                          for offset in range(8)] for group in range(8)]
                for metric in ("pointwise_wasserstein_kwh", "diversity_ratio")
            },
            "heatmap_metrics_source": "evaluate_conditional on one verified validation sample and identical "
                                      "generator draws before/after the saved map; heatmap_values are calibrated, "
                                      "raw_heatmap_values are raw. Color scales are shared within each metric.",
            "example_selection": "First up to three rows in seeded sample order for every label; independent "
                                 "real/generated profiles. Fixed saved marginal map applied to generated rows; "
                                 "no current-batch ranks, refitting, sample ranking, or smoothing.",
            "postprocessing": "Previously fitted conditional marginal map, bound to checkpoint and exact "
                              "training subset; no fitting on validation rows. Map interpolation may create "
                              "ties or endpoint saturation. Marginals are approximately calibrated toward training marginals.",
            "interpretation": "Exploratory validation-only figures for a frozen calibrated pipeline. "
                              "Empirical pointwise bands are not confidence intervals; examples are unpaired. "
                              "Marginal improvement is imposed by the map and does not independently establish "
                              "learned dependence or joint-profile fidelity. No final test data are used.",
            "plotting_script_sha256": sha256_file(Path(__file__)),
        })
    with (output_dir / "plot_manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--samples-per-condition", type=int, default=100)
    parser.add_argument("--calibrator", help="Previously fitted calibration NPZ; verified and applied without refitting")
    args = parser.parse_args()
    if args.samples_per_condition < 1 or not 0 <= args.seed < 2 ** 32:
        parser.error("samples-per-condition must be positive; seed must be in [0, 2**32)")
    manifest = plot_checkpoint(args.prepared, args.checkpoint, args.output_dir,
                               seed=args.seed, samples_per_condition=args.samples_per_condition,
                               calibrator=args.calibrator)
    print(f"Wrote {len(manifest['artifacts'])} validation figure files and plot_manifest.json to {args.output_dir}")


if __name__ == "__main__":
    main()
