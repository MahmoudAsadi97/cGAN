"""Plot four primary physical metrics from the compact robustness JSON only.

Five neural training-seed estimates and their mean/sample SD are shown alongside
one point per identical fixed control. No dataset, checkpoint or full run report
is read. SVG text remains editable, and the figure has no stochastic layout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("raw_cgan", "calibrated_cgan", "gaussian_copula", "conditional_resampling")
LABELS = ("Raw cGAN\n5 seeds", "Calibrated cGAN\n5 seeds", "Gaussian copula\nfixed control", "Resampling\nfixed control")
PANELS = (
    ("daily_energy_wasserstein_kwh", "Daily energy", "kWh"),
    ("peak_power_wasserstein_kw", "Peak power", "kW"),
    ("ramp_wasserstein_kw", "Ramps", "kW"),
    ("lag_one_acf_wasserstein", "Lag-one autocorrelation", "unitless"),
)
SEEDS = (11, 22, 33, 44, 55)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _check(condition, message):
    if not condition:
        raise ValueError(message)


def _finite(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def figure_data(report):
    """Extract plotted values and check their relation to the recorded summary."""
    _check(report.get("schema_version") == 2, "expected compact robustness schema version 2")
    _check(report.get("split") == "val" and report.get("test_used") is False, "validation-only report required")
    _check(report["frozen_protocol"]["training_seeds"] == list(SEEDS), "frozen five-seed protocol differs")
    runs = sorted(report["runs"], key=lambda row: row["training_seed"])
    _check([row["training_seed"] for row in runs] == list(SEEDS), "all five distinct neural seeds are required")
    coverage = report["coverage"]["caps"]["100"]
    _check(coverage["profiles_per_condition"] == 100 and report["coverage"]["conditions"] == 64 and
           coverage["selected_profiles"] == 6400, "primary condition/sample coverage differs")
    _check(coverage["identical_frozen_statistical_controls_verified"] is True, "identical fixed controls must be verified")
    panels = []
    for metric, title, unit in PANELS:
        column = report["metric_order"].index(f"physical/{metric}")
        target = report["descriptive_gate_thresholds"][metric]
        _check(_finite(target) and target > 0, "invalid practical target")
        methods = {}
        for method in METHODS:
            aggregate = report["aggregate_macro_by_cap"]["100"][method]
            mean, sd = aggregate["mean"][column], aggregate["sample_sd"][column]
            _check(_finite(mean) and _finite(sd) and mean >= 0 and sd >= 0, "invalid plotted W1 summary")
            if method in METHODS[:2]:
                points = [run["macro_by_cap"]["100"][method][column] for run in runs]
                _check(all(_finite(value) and value >= 0 for value in points), "invalid neural seed estimate")
                _check(math.isclose(statistics.mean(points), mean, rel_tol=1e-12, abs_tol=1e-12) and
                       math.isclose(statistics.stdev(points), sd, rel_tol=1e-12, abs_tol=1e-12),
                       "seed estimates disagree with recorded mean/sample SD")
            else:
                _check(sd == 0, "fixed control has nonzero across-run SD")
                points = [mean]
            methods[method] = {"displayed_points": points, "mean": mean,
                               "sample_sd": sd if method in METHODS[:2] else None}
        panels.append({"metric": metric, "title": title, "unit": unit, "target": target, "methods": methods})
    return panels, coverage


def plot(report_path, svg_path, png_path, provenance_path, *, overwrite=False):
    paths = [Path(path) for path in (svg_path, png_path, provenance_path)]
    _check(len({path.resolve() for path in paths}) == 3, "output paths must differ")
    _check(Path(report_path).resolve() not in {path.resolve() for path in paths}, "outputs must not overwrite the input report")
    for path in paths:
        if path.exists() and not overwrite:
            raise FileExistsError(f"output already exists: {path}; use --overwrite for an intentional regeneration")
    raw = Path(report_path).read_bytes()
    report = json.loads(raw)
    panels, coverage = figure_data(report)
    input_hash = _sha(raw)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none",
        "svg.hashsalt": "curated-primary-physical-v1", "axes.spines.top": False,
        "axes.spines.right": False, "axes.edgecolor": "#A3A8AF", "axes.linewidth": .7,
        "axes.labelcolor": "#343A40", "xtick.color": "#343A40", "ytick.color": "#343A40",
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    })
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.4))
    fig.subplots_adjust(left=.085, right=.98, bottom=.155, top=.79, hspace=.46, wspace=.24)
    colors = ("#386FA4", "#C76C2F", "#55616D", "#55616D")
    offsets = (-.17, -.10, -.03, .04, .11)
    first_errorbar = None
    for letter, ax, panel in zip("ABCD", axes.flat, panels):
        maximum = panel["target"]
        for index, method in enumerate(METHODS):
            values = panel["methods"][method]
            if method in METHODS[:2]:
                ax.scatter([index + offset for offset in offsets], values["displayed_points"],
                           s=34, c=colors[index], edgecolors="white", linewidths=.55, zorder=4)
                bars = ax.errorbar(index + .27, values["mean"], yerr=values["sample_sd"], fmt="s",
                                   markersize=4, color="#20262D", ecolor="#20262D", capsize=4,
                                   elinewidth=1.25, markeredgewidth=1.1, zorder=5)
                if first_errorbar is None:
                    first_errorbar = bars
                maximum = max(maximum, max(values["displayed_points"]), values["mean"] + values["sample_sd"])
            else:
                ax.scatter(index, values["mean"], marker="D", s=40, color=colors[index],
                           edgecolors="white", linewidths=.5, zorder=4)
                maximum = max(maximum, values["mean"])
        ax.axhline(panel["target"], color="#8D9298", linestyle=(0, (5, 4)), linewidth=1., zorder=1)
        ax.text(.985, panel["target"] + maximum*.025, f"target ≤ {panel['target']:g}",
                transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=9, color="#666D75")
        ax.set_title(f"{letter}   {panel['title']}", loc="left", fontsize=12.5, fontweight="semibold", pad=10)
        ax.set_ylabel(f"Condition-macro W₁ ({panel['unit']})", fontsize=10)
        ax.set_xticks(range(4), LABELS, fontsize=9.4)
        ax.tick_params(axis="x", length=0, pad=8)
        ax.tick_params(axis="y", labelsize=9, length=3, width=.7)
        ax.set_xlim(-.43, 3.42)
        ax.set_ylim(0, maximum * 1.18)
        ax.grid(axis="y", color="#E7EAED", linewidth=.6, zorder=0)
        ax.set_axisbelow(True)
    fig.suptitle("Primary validation: physical fidelity", x=.085, y=.965, ha="left", fontsize=17, fontweight="semibold", color="#20262D")
    fig.text(.085, .922, f"100 profiles per condition  ·  64 conditions  ·  6,400 profiles  ·  {coverage['selected_meters']} validation meters", fontsize=10.5, color="#515963")
    handles = [Line2D([], [], marker="o", color="none", markerfacecolor="#6E8294", markersize=5), first_errorbar,
               Line2D([], [], marker="D", color="none", markerfacecolor="#55616D", markersize=5),
               Line2D([], [], color="#8D9298", linestyle=(0, (5, 4)), linewidth=1)]
    fig.legend(handles, ["Training-seed estimate", "Mean ± sample SD (n = 5)", "Fixed control: one point", "Practical target"],
               loc="upper left", bbox_to_anchor=(.08, .888), frameon=False, ncol=4, fontsize=9.3,
               handletextpad=.7, columnspacing=1.8)
    fig.text(.085, .067, "Seeds 11, 22, 33, 44, 55. Error bars show sample SD, not confidence intervals.", fontsize=9.5, color="#343A40")
    fig.text(.085, .035, "Equal condition weighting; lower W₁ is better. Targets are descriptive project criteria. Validation remains exploratory.", fontsize=9.2, color="#515963")
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    svg_path, png_path, provenance_path = paths
    description = f"Primary validation only; five neural training seeds, single fixed controls, mean +/- sample SD (not CI). Input SHA256 {input_hash}."
    fig.savefig(svg_path, format="svg", metadata={"Date": None, "Title": "Curated primary physical validation", "Description": description})
    fig.savefig(png_path, dpi=150, metadata={"Description": description})
    plt.close(fig)
    provenance = {
        "schema_version": 1, "input_report": str(report_path), "input_sha256": input_hash,
        "script_sha256": _sha(Path(__file__).read_bytes()), "python": platform.python_version(),
        "matplotlib": matplotlib.__version__, "data_access": "Compact input JSON only; no dataset, checkpoint or full run report loaded",
        "validation_cap": 100, "training_seeds": list(SEEDS), "coverage": coverage,
        "method_order": list(METHODS), "panels": panels,
        "uncertainty_display": "Neural mean +/- sample SD across five training seeds; fixed controls appear once without error bars",
        "svg_fonttype": "none", "svg_sha256": _sha(svg_path.read_bytes()), "png_sha256": _sha(png_path.read_bytes()),
    }
    provenance_path.write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n")
    return provenance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "reports/curated_calibration_robustness.json")
    parser.add_argument("--svg", type=Path, default=ROOT / "reports/figures/curated_primary_physical.svg")
    parser.add_argument("--png", type=Path, default=ROOT / "runs/figures/curated_primary_physical.png")
    parser.add_argument("--provenance", type=Path, default=ROOT / "reports/figures/curated_primary_physical.provenance.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    result = plot(args.report, args.svg, args.png, args.provenance, overwrite=args.overwrite)
    print(json.dumps({"svg": str(args.svg), "png": str(args.png), "input_sha256": result["input_sha256"]}))


if __name__ == "__main__":
    main()
