"""Verify and summarize completed baseline artifacts; never load data or models."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPORT = Path(__file__).resolve().parent
ROOT = REPORT.parents[2]
SOURCE = ROOT / "artifacts/phase1-development/strong_baselines"
COPIED = (
    "manifest.json", "plan.json", "summary.json", "metrics.csv",
    "condition_diagnostics.csv", "real_proximity.json", "strong_beta_fit.json",
    "strong_t_df4_fit.json", "strong_t_df8_fit.json", "strong_t_df16_fit.json",
)
DERIVED = ("method_comparison.csv", "conditional_comparisons.csv", "comparison.svg", "comparison.png")
NEW = ("strong_beta", "strong_t_df4", "strong_t_df8", "strong_t_df16")
NEURAL = tuple(f"candidate_{i:03d}_{mode}" for i in range(4)
               for mode in ("raw", "calibrated"))
OLD = ("baseline_mean", "baseline_resample", "baseline_copula")
SEEDS = [20261023, 20261024]
ERROR_METRICS = (
    "profile_sw1_kwh", "daily_energy_w1_kwh", "peak_w1_kw", "ramp_w1_kw",
    "acf_lag1_w1", "acf_lag6_w1",
)


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text())


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def comparison_figure(reports, real_mean):
    methods = [
        ("baseline_resample", "Whole-day resampling", "#65717e"),
        ("baseline_copula", "Gaussian copula", "#65717e"),
        ("strong_beta", "Empirical beta copula", "#b26926"),
        ("strong_t_df4", "Student-t copula, df 4", "#417ca4"),
        ("strong_t_df8", "Student-t copula, df 8", "#417ca4"),
        ("strong_t_df16", "Student-t copula, df 16", "#417ca4"),
        ("candidate_002_calibrated", "Calibrated flow, seed 101", "#32806d"),
        ("candidate_003_calibrated", "Calibrated flow, seed 202", "#32806d"),
    ]
    panels = [
        ("metrics", "normalized_profile_sw1", "Normalized profile SW1\n(unitless; lower is closer)"),
        ("metrics", "ramp_w1_kw", "Signed-ramp W1\n(kW; lower is closer)"),
        ("proximity", "mean_rms_kwh", "Nearest same-condition TRAIN RMS\n(kWh; descriptive proximity)"),
    ]
    with matplotlib.rc_context({"svg.hashsalt": "strong-baselines-fixed-development",
                               "font.family": "DejaVu Sans", "font.size": 10}):
        fig, axes = plt.subplots(1, 3, figsize=(13, 5.7), sharey=True)
        for axis, (section, metric, title) in zip(axes, panels):
            for y, (method, _, color) in enumerate(methods):
                values = [draw[section]["macro"][metric] for draw in reports[method]["draws"]]
                axis.barh(y, np.mean(values), color=color, alpha=.25, height=.62)
                axis.plot(values[0], y - .11, "o", color=color, markersize=5,
                          label="Generation draw 20261023" if y == 0 else None)
                axis.plot(values[1], y + .11, "x", color=color, markersize=6,
                          label="Generation draw 20261024" if y == 0 else None)
                axis.plot(np.mean(values), y, "|", color="#18232b", markersize=12,
                          label="Two-draw mean" if y == 0 else None)
            axis.set_title(title, fontsize=10, pad=10)
            axis.set_xlim(left=0)
            axis.grid(axis="x", alpha=.22)
            axis.set_axisbelow(True)
            axis.spines[["top", "right"]].set_visible(False)
        axes[0].set_yticks(range(len(methods)), [item[1] for item in methods])
        axes[0].invert_yaxis()
        axes[2].axvline(real_mean, color="#423857", linestyle="--", linewidth=1.3,
                        label="Real inner-VAL to TRAIN mean")
        handles, labels = axes[2].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.57, .92),
                   ncol=2, fontsize=8, frameon=False)
        fig.suptitle("Stronger controls on post-test inner-validation development data", fontsize=14)
        fig.text(.02, .025,
                 "64 equally weighted conditions; 256 profiles per condition per draw. "
                 "Points are paired generation draws, not independent fits or confidence intervals.\n"
                 "Near-zero TRAIN distance can reflect copying or near-resampling; "
                 "it does not demonstrate privacy or useful novelty.", fontsize=9)
        fig.tight_layout(rect=(0, .11, 1, .84))
        fig.savefig(REPORT / "comparison.svg", metadata={"Date": None})
        fig.savefig(REPORT / "comparison.png", dpi=160)
        plt.close(fig)


def main():
    manifest = read(SOURCE / "manifest.json")
    assert manifest["status"] == "completed" and manifest["completed_draws"] == 30
    assert manifest["generation_seeds"] == SEEDS
    assert manifest["test_performance_access"] is False
    assert manifest["reselection"] is False and manifest["calibration_refitted"] is False
    assert manifest["clipping"] is False
    for name, expected in manifest["artifact_hashes"].items():
        assert Path(name).name == name and digest(SOURCE / name) == expected, name
    assert digest(SOURCE / "plan.json") == manifest["plan_sha256"]
    summary = read(SOURCE / "summary.json")
    assert set(summary["methods"]) == set(NEURAL + OLD + NEW)
    assert summary["selection_profiles"] == 16384 and summary["conditions"] == list(range(64))
    assert summary["generation_seeds"] == SEEDS
    reports = {key: read(SOURCE / f"{key}.json") for key in summary["methods"]}
    reference_reports = {}
    for reference_key in ("reference", "calibration_reference"):
        reference_directory = ROOT / "artifacts/phase1-development" / Path(
            manifest[reference_key]["directory"]).name
        reference_manifest_path = reference_directory / "manifest.json"
        assert digest(reference_manifest_path) == manifest[reference_key]["manifest_sha256"]
        reference_manifest = read(reference_manifest_path)
        names = OLD if reference_key == "reference" else NEURAL
        for method_id in names:
            path = reference_directory / f"{method_id}.json"
            assert digest(path) == reference_manifest["artifact_hashes"][path.name]
            reference_reports[method_id] = read(path)
    matched_reference_draws = 0
    for key, report in reports.items():
        assert [draw["generation_seed"] for draw in report["draws"]] == SEEDS
        for draw_index, draw in enumerate(report["draws"]):
            assert set(draw["metrics"]["per_condition"]) == {str(i) for i in range(64)}
            assert all(row["n_real"] == row["n_fake"] == 256
                       for row in draw["metrics"]["per_condition"].values())
            assert draw["reference_draw_match"] is (key not in NEW)
            matched_reference_draws += int(draw["reference_draw_match"])
            if key not in NEW:
                reference_draw = reference_reports[key]["draws"][draw_index]
                assert draw["generation_seed"] == reference_draw["generation_seed"]
                assert draw["generated_profile_bytes_sha256"] == reference_draw["generated_profile_bytes_sha256"]
                assert draw["metrics"] == reference_draw["metrics"]
        for metric, value in summary["methods"][key]["metrics"].items():
            values = [draw["metrics"]["macro"][metric] for draw in report["draws"]]
            assert np.mean(values) == value["mean"], (key, metric)
        for metric, value in summary["methods"][key]["proximity"].items():
            values = [draw["proximity"]["macro"][metric] for draw in report["draws"]]
            assert np.mean(values) == value["mean"], (key, metric)
    assert matched_reference_draws == 22
    for name in COPIED:
        shutil.copyfile(SOURCE / name, REPORT / name)
        assert digest(SOURCE / name) == digest(REPORT / name)

    rows = []
    for method_id, method in summary["methods"].items():
        report = reports[method_id]
        row = {"method_id": method_id, "method": method["method"],
               "training_seed": method["training_seed"], "draws": 2}
        for section in ("metrics", "tail_errors", "proximity"):
            row.update({f"{section}_{key}_mean": value["mean"]
                        for key, value in method[section].items()})
        for quantile in ("q01", "q05", "q50", "q95", "q99"):
            row[f"proximity_{quantile}_rms_kwh_mean"] = float(np.mean([
                draw["proximity"]["macro"][f"{quantile}_rms_kwh"] for draw in report["draws"]]))
        row["quantized_exact_training_match_fraction_mean"] = method[
            "quantized_exact_training_match_fraction"]["mean"]
        row["source_pv0_negative_daily_energy_fraction_mean"] = method[
            "source_pv0"]["fake"]["negative_daily_energy_fraction"]["mean"]
        row["source_pv0_export_day_fraction_mean"] = method[
            "source_pv0"]["fake"]["export_day_fraction"]["mean"]
        row["fit_seconds"] = report.get("fit_seconds", "")
        for timing in ("generation_seconds", "physical_metric_seconds", "nearest_neighbor_seconds"):
            row[f"{timing}_mean"] = float(np.mean([draw[timing] for draw in report["draws"]]))
        row["source_report_sha256"] = manifest["artifact_hashes"][f"{method_id}.json"]
        rows.append(row)
    write_csv(REPORT / "method_comparison.csv", rows)

    pairs = [(new, old) for new in NEW for old in OLD[1:]]
    pairs += [(f"candidate_{i:03d}_calibrated", new) for i in range(4) for new in NEW]
    comparisons = []
    for a, b in pairs:
        for index, generation_seed in enumerate(SEEDS):
            left = reports[a]["draws"][index]["metrics"]["per_condition"]
            right = reports[b]["draws"][index]["metrics"]["per_condition"]
            for metric in ERROR_METRICS:
                differences = np.array([left[str(c)]["metrics"][metric]
                                        - right[str(c)]["metrics"][metric] for c in range(64)])
                comparisons.append({"method_a": a, "method_b": b,
                    "generation_seed": generation_seed, "metric": metric,
                    "conditions_a_lower": int(np.sum(differences < 0)),
                    "conditions_tied": int(np.sum(differences == 0)),
                    "conditions_a_higher": int(np.sum(differences > 0)),
                    "macro_difference_a_minus_b": float(np.mean(differences))})
    write_csv(REPORT / "conditional_comparisons.csv", comparisons)
    real = read(SOURCE / "real_proximity.json")
    assert real["real_profile_bytes_sha256"] == manifest["real_profile_bytes_sha256"]
    assert real["label_bytes_sha256"] == manifest["label_bytes_sha256"]
    assert real["data_hashes"] == manifest["data_hashes"]
    comparison_figure(reports, real["proximity"]["macro"]["mean_rms_kwh"])
    receipt = {
        "schema_version": 1,
        "role": "Extraction from completed strong-baseline artifacts; no dataset/model access or reevaluation",
        "source_manifest": str((SOURCE / "manifest.json").relative_to(ROOT)),
        "source_manifest_sha256": digest(SOURCE / "manifest.json"),
        "source_evaluation_git_head": manifest["git_head"],
        "verified_draws": 30, "verified_methods": 15,
        "reference_generated_bytes_and_physical_metrics_matches": matched_reference_draws,
        "reference_reports_independently_compared": "22 existing-method draw hashes and complete physical-metric records matched the manifest-bound frozen calibration or fresh-diagnostic JSON reports.",
        "copied_byte_identical": {name: digest(REPORT / name) for name in COPIED},
        "input_artifact_sha256": manifest["artifact_hashes"],
        "derivation_script_sha256": digest(Path(__file__)),
        "derived_artifact_sha256": {name: digest(REPORT / name) for name in DERIVED},
        "aggregation": "Each draw equally averages 64 conditions; reported means average two paired generation draws within each training seed, not independent fits.",
        "proximity_definition": "Exact nearest same-condition inner-TRAIN Euclidean distance divided by sqrt(24), in kWh. Quantile columns are means of within-condition quantiles, not pooled quantiles.",
        "threshold_definition": "Generated distance strictly below the corresponding real inner-VAL condition's 5th percentile or median, using NumPy linear quantiles. Descriptive only; not a novelty or privacy guarantee.",
        "conditional_comparison_definition": "Strict lower/tied/higher comparisons in each of 64 conditions, separately for the same generation draw. Counts are descriptive, not independent tests; profile SW1 uses kWh, equivalent in ordering to fixed-range normalized SW1.",
        "figure_definition": "Bars and vertical ticks show two-draw means; circle/cross show the actual paired generation draws. No inferential error bars. Dashed TRAIN proximity line is the real inner-VAL mean.",
        "figure_runtime": {"matplotlib": matplotlib.__version__, "numpy": np.__version__},
    }
    (REPORT / "DERIVATION.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"Verified all {len(manifest['artifact_hashes'])} source artifacts and 30 draws; "
          f"copied {len(COPIED)} byte-identical artifacts, derived {len(rows)} method rows, "
          f"{len(comparisons)} conditional comparisons and one figure in two formats.")


if __name__ == "__main__":
    main()
