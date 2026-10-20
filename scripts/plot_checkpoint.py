"""Create reproducible validation figures for a frozen generator checkpoint.

Example:
    python scripts/plot_checkpoint.py --prepared data/prepared96 \
        --checkpoint runs/candidate/best.pt --output-dir figures/candidate

All 64 conditions are required. Curves and example rows follow predeclared
selection rules; real and generated rows are independent, never paired. This
utility reads train/validation data only and has no test-evaluation option.
Matplotlib is an optional plotting dependency.
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
from fluvius_cgan.cli import evaluate_checkpoint
from fluvius_cgan.release import sha256_file
from fluvius_cgan.training import _read_split, generate


CONDITIONS = np.arange(64, dtype=np.int64)
WINTER_WEEKDAY_LABELS = np.arange(0, 64, 8, dtype=np.int64)
SEASONS = ("Winter", "Spring", "Summer", "Autumn")
REAL_COLOR = "#2369A5"
GENERATED_COLOR = "#D45B28"


def _array_hash(values):
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


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


def _examples_figure(plt, Line2D, real, fake, labels, hours, group, method):
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
    figure.text(.5, .005, "Independent samples; rows are not paired. First sampled rows are shown without ranking, "
                "replacement, smoothing, or clipping.", ha="center", fontsize=9)
    figure.subplots_adjust(left=.07, right=.99, top=.9, bottom=.17, wspace=.13, hspace=.27)
    return figure


def plot_checkpoint(prepared, checkpoint, output_dir, *, seed=2026, samples_per_condition=100):
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
    # Use the production evaluator to verify the checkpoint/prepared fingerprints,
    # model-selected training subset, and meter-disjoint validation data. Its
    # condition metrics supply the heatmaps; plotting does not re-evaluate them.
    report = evaluate_checkpoint(prepared, checkpoint, split="val",
                                 samples_per_condition=samples_per_condition, seed=seed)
    if report["checkpoint_sha256"] != initial_checkpoint_hash:
        raise ValueError("checkpoint changed during evaluation")
    # Reload validation once for all figures, verifying the bytes against that
    # evaluation. There are no data reloads inside any plotting function.
    val, val_hash = _read_split(prepared / "val.npz")
    if val_hash != report["evaluation_sha256"]:
        raise ValueError("validation data changed after evaluation")
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
    generated = generate(checkpoint, labels, seed=seed).astype(np.float64)
    if sha256_file(checkpoint) != initial_checkpoint_hash:
        raise ValueError("checkpoint changed during plotting-sample generation")
    counts = {str(label): int(np.sum(labels == label)) for label in CONDITIONS}
    if counts != report["methods"]["cgan"]["counts"]["real"]:
        raise ValueError("plot sample condition counts differ from verified evaluation")
    resolution = real.shape[1]
    interval_hours = 24 / resolution
    hours = np.arange(resolution) * interval_hours
    real_power, generated_power = real / interval_hours, generated / interval_hours
    method = "Unconditional generator" if state["config"]["unconditional"] else "Conditional generator"
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
    with matplotlib.rc_context(style):
        figure = _median_figure(plt, Line2D, Patch, real_power, generated_power, labels, hours, method)
        artifacts.extend(_save(figure, output_dir, "winter_weekday_quantiles", plt))
        figure = _heatmap_figure(plt, TwoSlopeNorm, report["methods"]["cgan"]["per_condition"], method)
        artifacts.extend(_save(figure, output_dir, "conditional_metric_heatmaps", plt))
        for group in range(8):
            figure = _examples_figure(plt, Line2D, real_power, generated_power, labels, hours, group, method)
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
            metric: [[report["methods"]["cgan"]["per_condition"][str(group * 8 + offset)][metric]
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
    args = parser.parse_args()
    if args.samples_per_condition < 1 or not 0 <= args.seed < 2 ** 32:
        parser.error("samples-per-condition must be positive; seed must be in [0, 2**32)")
    manifest = plot_checkpoint(args.prepared, args.checkpoint, args.output_dir,
                               seed=args.seed, samples_per_condition=args.samples_per_condition)
    print(f"Wrote {len(manifest['artifacts'])} validation figure files and plot_manifest.json to {args.output_dir}")


if __name__ == "__main__":
    main()
