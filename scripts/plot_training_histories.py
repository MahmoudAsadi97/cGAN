"""Plot complete, SHA-bound validation-selector histories from the reference collector.

Only JSON evidence is read. Matplotlib is imported only after all input checks.
No checkpoints, profile arrays, calibration maps or endpoint reports are opened.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import re

ARMS = ("candidate96", "bce96", "unconditional96", "bce24")
SEEDS = (11, 22, 33, 44, 55)
EPOCHS = 26
METRIC = "mean_condition_interval_wasserstein_normalized"
SCOPE = (
    "All 26 raw validation-selector epochs for all five declared seeds in each arm. "
    "Separate y-axis scales: scores with different standard/minmax normalization "
    "or 96/24-interval resolution are not directly comparable. Retained epochs are earliest minima, with no "
    "smoothing or seed selection. These histories do not establish convergence or TEST performance."
)


def check(value, message):
    if not value:
        raise ValueError(message)


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _decode(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            check(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"nonfinite JSON literal: {value}")

    return json.loads(raw, object_pairs_hook=unique, parse_constant=invalid)


def _json_path(value, base, basename=None):
    check(isinstance(value, (str, Path)) and str(value).strip(), "JSON path required")
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    check(path.suffix == ".json" and (basename is None or path.name == basename),
          "only the declared JSON input name is accepted")
    resolved = path.resolve()
    check(resolved.suffix == ".json", "JSON input resolves to a non-JSON file")
    return resolved


def extract(report, read_bound):
    """Validate and extract curves; read_bound(record, basename) supplies bound JSON.

    Keeping extraction independent of matplotlib permits small synthetic JSON
    contract checks without importing plotting or model packages.
    """
    check(isinstance(report, dict) and type(report.get("schema_version")) is int and
          report["schema_version"] == 1 and report.get("kind") == "completed_reference_validation_study"
          and report.get("status") == "complete" and report.get("split") == "val"
          and report.get("test_used") is False, "complete validation-only reference collector required")
    check(report.get("candidate_kind") in ("swd10", "capacity"), "unknown declared candidate")
    check(report.get("training_seeds") == list(SEEDS) and
          all(type(seed) is int for seed in report["training_seeds"]), "all five fixed training seeds required")
    runs, artifacts = report.get("runs"), report.get("artifacts")
    check(isinstance(runs, dict) and set(runs) == set(ARMS) and isinstance(artifacts, dict),
          "all four arms and collector artifact bindings required")
    # Reject coverage omissions before reading even the first history.
    for arm in ARMS:
        check(isinstance(runs[arm], dict) and set(runs[arm]) == set(map(str, SEEDS)),
              f"{arm}: all five distinct seeds required")
    panels, seen_paths = {}, set()
    for arm in ARMS:
        width = 24 if arm == "bce24" else 96
        normalization = "standard" if arm == "candidate96" else "minmax"
        traces = []
        for seed in SEEDS:
            run = runs[arm][str(seed)]
            check(run.get("arm") == arm and type(run.get("training_seed")) is int and
                  run["training_seed"] == seed and type(run.get("width")) is int and run["width"] == width,
                  f"{arm}/{seed}: arm, seed or resolution binding differs")
            config = run.get("config", {})
            check(type(config.get("seed")) is int and config["seed"] == seed and
                  config.get("normalization") == normalization and config.get("validation_selection") == "conditional_w1"
                  and config.get("validation_samples") == 4096 and config.get("epochs") == EPOCHS,
                  f"{arm}/{seed}: declared selector configuration differs")
            for field, expected in (("epochs_completed", EPOCHS), ("generator_updates", 30784),
                                    ("discriminator_updates", 30784)):
                check(type(run.get(field)) is int and run[field] == expected, f"{arm}/{seed}: incomplete run budget")
            loaded = {}
            for name in ("history.json", "summary.json"):
                record = run.get("artifacts", {}).get(name)
                check(isinstance(record, dict) and set(record) == {"path", "sha256", "bytes"}
                      and isinstance(record["path"], str) and
                      isinstance(record["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", record["sha256"])
                      and type(record["bytes"]) is int and record["bytes"] > 0,
                      f"{arm}/{seed}: invalid {name} binding")
                check(artifacts.get(record["path"]) == record, "run/global collector artifact records disagree")
                check(record["path"] not in seen_paths, "one history/summary path is reused across runs")
                seen_paths.add(record["path"])
                loaded[name] = read_bound(record, name)
            history, summary = loaded["history.json"], loaded["summary.json"]
            check(isinstance(history, list) and len(history) == EPOCHS and
                  all(isinstance(row, dict) and type(row.get("epoch")) is int for row in history)
                  and [row["epoch"] for row in history] == list(range(1, EPOCHS + 1)),
                  f"{arm}/{seed}: require all 26 ordered epochs")
            check(all(_finite(row.get("validation_score")) and row["validation_score"] >= 0 for row in history),
                  f"{arm}/{seed}: selector scores must be finite and nonnegative")
            check(isinstance(summary, dict) and summary.get("test_used") is False
                  and summary.get("stopped_early") is False and summary.get("validation_metric") == METRIC
                  and summary.get("resolution") == width and summary.get("target") == "net"
                  and summary.get("units") == "kWh/interval", f"{arm}/{seed}: summary scope/metric differs")
            for field, expected in (("epochs_completed", EPOCHS), ("generator_updates", 30784),
                                    ("discriminator_updates", 30784)):
                check(type(summary.get(field)) is int and summary[field] == expected,
                      f"{arm}/{seed}: summary is incomplete")
            check(summary.get("scaler", {}).get("kind", "minmax") == normalization and
                  summary.get("selection") == run.get("training_selection") and
                  isinstance(summary.get("selection"), dict) and
                  summary["selection"].get("validation_selection") == "conditional_w1" and
                  summary["selection"].get("validation_samples_requested") == 4096 and
                  summary.get("provenance") == run.get("provenance") and
                  isinstance(summary.get("provenance"), dict), f"{arm}/{seed}: summary/collector lineage differs")
            minimum = min(history, key=lambda row: row["validation_score"])
            check(type(run.get("best_epoch")) is int and type(summary.get("best_epoch")) is int and
                  run["best_epoch"] == summary["best_epoch"] == minimum["epoch"] and
                  _finite(summary.get("best_validation_score")) and
                  summary["best_validation_score"] == minimum["validation_score"] and
                  summary.get("last_epoch_metrics") == history[-1],
                  f"{arm}/{seed}: retained epoch must be the earliest minimum and match the summary")
            check(isinstance(run.get("checkpoint_sha256"), str) and
                  re.fullmatch(r"[0-9a-f]{64}", run["checkpoint_sha256"]), "retained checkpoint fingerprint missing")
            checkpoint_record = run.get("artifacts", {}).get("best.pt", {})
            check(checkpoint_record.get("sha256") == run["checkpoint_sha256"] and
                  artifacts.get(checkpoint_record.get("path")) == checkpoint_record,
                  "retained checkpoint fingerprint differs from collector artifact record")
            traces.append({"seed": seed, "epochs": list(range(1, EPOCHS + 1)),
                           "validation_scores": [row["validation_score"] for row in history],
                           "best_epoch": minimum["epoch"], "best_score": minimum["validation_score"],
                           "retained_checkpoint_sha256": run["checkpoint_sha256"]})
        panels[arm] = {"width": width, "normalization": normalization, "traces": traces}
    return panels


def load_histories(report_path):
    """Read collector + exactly 40 bound JSON artifacts; never follow endpoint paths."""
    report_path = _json_path(report_path, Path.cwd())
    raw = report_path.read_bytes()
    report = _decode(raw)
    records = {str(report_path): {"path": str(report_path), "sha256": _digest(raw), "bytes": len(raw)}}
    seen = {report_path}

    def read_bound(record, name):
        path = _json_path(record["path"], report_path.parent, name)
        check(path not in seen, "resolved input file is reused across runs")
        # Detect aliases among permitted JSON inputs without reading other files.
        check(not any(os.path.samefile(path, previous) for previous in seen), "hardlinked JSON evidence is reused")
        seen.add(path)
        content = path.read_bytes()
        check(len(content) == record["bytes"] and _digest(content) == record["sha256"],
              f"bound JSON content changed: {path}")
        records[str(path)] = {"path": str(path), "sha256": record["sha256"], "bytes": len(content)}
        return _decode(content)

    panels = extract(report, read_bound)
    return panels, records, report["candidate_kind"]


def _unchanged(records):
    for record in records.values():
        path = _json_path(record["path"], Path.cwd())
        raw = path.read_bytes()
        check(len(raw) == record["bytes"] and _digest(raw) == record["sha256"],
              f"JSON input changed during plotting: {path}")


def plot(report_path, output_dir, raster_dir=None):
    output_dir = Path(output_dir)
    raster_dir = Path(raster_dir) if raster_dir is not None else output_dir
    paths = {"svg": output_dir / "reference_training_histories.svg",
             "png": raster_dir / "reference_training_histories.png",
             "provenance": output_dir / "reference_training_histories.provenance.json"}
    for path in paths.values():
        if os.path.lexists(path):
            raise FileExistsError(path)
    script_sha = _digest(Path(__file__).read_bytes())
    panels, records, candidate_kind = load_histories(report_path)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#6556A3")
    titles = {"candidate96": "Capacity candidate" if candidate_kind == "capacity" else "SWD10 candidate",
              "bce96": "Conditional BCE96", "unconditional96": "Unconditional BCE96", "bce24": "Conditional BCE24"}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none",
                         "svg.hashsalt": "reference-training-histories-v1", "figure.facecolor": "white",
                         "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#A0A6AD"})
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.8), sharex=True, sharey=False)
    fig.subplots_adjust(left=.09, right=.975, bottom=.215, top=.80, wspace=.25, hspace=.45)
    for index, (ax, arm) in enumerate(zip(axes.flat, ARMS)):
        panel = panels[arm]
        for color, trace in zip(colors, panel["traces"]):
            ax.plot(trace["epochs"], trace["validation_scores"], color=color, linewidth=1.4, alpha=.9)
            ax.plot(trace["best_epoch"], trace["best_score"], marker="D", color=color,
                    markersize=5.5, markeredgecolor="white", markeredgewidth=.6, zorder=5)
        ax.set_title(f"{chr(65 + index)}  {titles[arm]} · {panel['width']} intervals · {panel['normalization']}",
                     loc="left", fontsize=10.5, pad=11)
        ax.set_xlim(1, EPOCHS)
        ax.set_xticks([1, 5, 10, 15, 20, 26])
        ax.set_xlabel("Training epoch")
        ax.set_ylabel("Condition-macro interval W₁\n(training-normalized units)", fontsize=9)
        ax.tick_params(axis="x", labelbottom=True)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3), useOffset=False)
        ax.grid(color="#E7EAED", linewidth=.55)
        ax.set_axisbelow(True)
        ax.margins(y=.12)
    fig.suptitle("Complete validation-selector histories", x=.09, y=.96, ha="left",
                 fontsize=18, fontweight="semibold", color="#20262D")
    fig.text(.09, .916, "All 26 epochs · five declared training seeds per arm · raw generator selector",
             fontsize=11, color="#515963")
    handles = [Line2D([0], [0], color=color, linewidth=2, label=f"Seed {seed}") for seed, color in zip(SEEDS, colors)]
    handles.append(Line2D([0], [0], color="#50565D", marker="D", linestyle="None", markersize=5,
                          label="Retained earliest minimum"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.085, .89), ncol=6, frameon=False,
               fontsize=9.3, handlelength=1.8, columnspacing=1.5)
    fig.text(.09, .133, "Panels use independent y-axis scales. Scores with different normalization or interval resolution are not directly comparable.",
             fontsize=9.6, color="#343A40")
    fig.text(.09, .093, "Lines show every recorded selector score without smoothing. Diamonds mark the retained best epoch for each seed.",
             fontsize=9.6, color="#343A40")
    fig.text(.09, .053, "Validation selection is adaptive. These fixed-budget histories do not establish convergence or TEST performance.",
             fontsize=9.6, color="#515963")
    buffers = {}
    try:
        for kind in ("svg", "png"):
            buffer = io.BytesIO()
            metadata = {"Date": None, "Title": "Complete validation-selector histories", "Description": SCOPE} if kind == "svg" else None
            fig.savefig(buffer, format=kind, dpi=200, metadata=metadata)
            buffers[kind] = buffer.getvalue()
    finally:
        plt.close(fig)
    _unchanged(records)
    check(_digest(Path(__file__).read_bytes()) == script_sha, "plotting source changed during execution")
    provenance = {"schema_version": 1, "figure_kind": "complete_reference_training_histories",
                  "created_utc": datetime.now(timezone.utc).isoformat(), "split": "val", "test_used": False,
                  "candidate_kind": candidate_kind, "metric": METRIC, "scope": SCOPE,
                  "collector": str(Path(report_path).resolve()), "input_artifacts": records,
                  "plotted_values": panels, "script_sha256": script_sha,
                  "python": platform.python_version(), "matplotlib": matplotlib.__version__,
                  "outputs": {kind: {"path": str(paths[kind]), "sha256": _digest(raw)} for kind, raw in buffers.items()}}
    buffers["provenance"] = (json.dumps(provenance, indent=2, allow_nan=False) + "\n").encode()
    for kind, raw in buffers.items():
        paths[kind].parent.mkdir(parents=True, exist_ok=True)
        with paths[kind].open("xb") as handle:
            handle.write(raw)
    return {kind: str(path) for kind, path in paths.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, help="completed reference collector JSON")
    parser.add_argument("--output-dir", required=True, help="new SVG and provenance sidecar directory")
    parser.add_argument("--raster-dir", help="PNG directory; defaults to --output-dir")
    args = parser.parse_args(argv)
    print(json.dumps(plot(args.report, args.output_dir, args.raster_dir)))


if __name__ == "__main__":
    main()
