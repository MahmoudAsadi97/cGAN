"""Verify completed mixture artifacts and extract tables without loading arrays."""
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
SOURCE = ROOT / "artifacts/phase1-development/mixture_baselines"
NEW = tuple(f"gmm_k{k}" for k in (1, 2, 4, 8))
CONTROLS = ("baseline_resample", "baseline_copula", "strong_beta", "strong_t_df4",
            "candidate_002_calibrated", "candidate_003_calibrated")
COPIED = ("manifest.json", "plan.json", "summary.json", "metrics.csv",
          "condition_diagnostics.csv", "real_proximity.json",
          *(f"{name}_fit.json" for name in NEW))
DERIVED = ("method_comparison.csv", "fit_summary.csv", "initializations.csv",
           "component_diagnostics.csv", "marginal_support.csv",
           "conditional_comparisons.csv", "conditional_tail_quantiles.csv",
           "comparison.svg", "comparison.png")
SEEDS = [20261023, 20261024]
ERROR_METRICS = ("profile_sw1_kwh", "daily_energy_w1_kwh", "peak_w1_kw", "ramp_w1_kw",
                 "acf_lag1_w1", "acf_lag6_w1")


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text())


def write_csv(name, rows):
    with (REPORT / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def figure(reports, real_mean):
    methods = [
        ("baseline_resample", "Whole-day resampling", "#65717e"),
        ("baseline_copula", "Original Gaussian copula", "#65717e"),
        ("strong_beta", "Empirical beta copula", "#b26926"),
        ("strong_t_df4", "Student-t copula, df 4", "#417ca4"),
        *((f"gmm_k{k}", f"Normal-score mixture, K={k}", "#795b9a") for k in (1, 2, 4, 8)),
        ("candidate_002_calibrated", "Calibrated flow, seed 101", "#32806d"),
        ("candidate_003_calibrated", "Calibrated flow, seed 202", "#32806d"),
    ]
    panels = [("metrics", "normalized_profile_sw1", "Normalized profile SW1\n(unitless; lower is closer)"),
              ("metrics", "daily_energy_w1_kwh", "Daily-energy W1\n(kWh; lower is closer)"),
              ("metrics", "ramp_w1_kw", "Signed-ramp W1\n(kW; lower is closer)"),
              ("proximity", "mean_rms_kwh", "Nearest TRAIN RMS\n(kWh; descriptive proximity)")]
    with matplotlib.rc_context({"svg.hashsalt": "mixture-fixed-development",
                               "font.family": "DejaVu Sans", "font.size": 9}):
        fig, axes = plt.subplots(1, 4, figsize=(15, 6.5), sharey=True)
        for axis, (section, metric, title) in zip(axes, panels):
            for y, (method, _, color) in enumerate(methods):
                values = [draw[section]["macro"][metric] for draw in reports[method]["draws"]]
                axis.barh(y, np.mean(values), color=color, alpha=.25, height=.62)
                axis.plot(values[0], y - .11, "o", color=color, markersize=4.5,
                          label="Generation draw 20261023" if y == 0 else None)
                axis.plot(values[1], y + .11, "x", color=color, markersize=5.5,
                          label="Generation draw 20261024" if y == 0 else None)
                axis.plot(np.mean(values), y, "|", color="#18232b", markersize=11,
                          label="Two-draw mean" if y == 0 else None)
            axis.set_title(title, fontsize=9, pad=10)
            axis.set_xlim(left=0)
            axis.grid(axis="x", alpha=.22)
            axis.set_axisbelow(True)
            axis.spines[["top", "right"]].set_visible(False)
        axes[0].set_yticks(range(len(methods)), [item[1] for item in methods])
        axes[0].invert_yaxis()
        axes[-1].axvline(real_mean, color="#423857", linestyle="--", linewidth=1.3,
                         label="Real inner-VAL to TRAIN mean")
        handles, labels = axes[-1].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.57, .93),
                   ncol=2, fontsize=8, frameon=False)
        fig.suptitle("Conditional mixture controls: post-test inner-validation development", fontsize=14)
        fig.text(.02, .025,
                 "64 equally weighted conditions; 256 profiles per condition per draw. "
                 "Points are paired generation draws, not independent fits or confidence intervals.\n"
                 "Selected fits not converged: K=4, 1/64; K=8, 3/64. "
                 "Nearest-TRAIN distance alone establishes neither useful novelty nor privacy.", fontsize=9)
        fig.tight_layout(rect=(0, .11, 1, .86))
        fig.savefig(REPORT / "comparison.svg", metadata={"Date": None})
        fig.savefig(REPORT / "comparison.png", dpi=160)
        plt.close(fig)


def main():
    manifest = read(SOURCE / "manifest.json")
    assert manifest["status"] == "completed" and manifest["completed_draws"] == 8
    assert manifest["reused_draws"] == 30 and manifest["generation_seeds"] == SEEDS
    assert manifest["test_performance_access"] is False
    assert manifest["historical_models_regenerated"] is False and manifest["reselection"] is False
    assert manifest["physical_clipping"] is False
    assert not manifest.get("artifact_hash_errors")
    for name, expected in manifest["artifact_hashes"].items():
        assert Path(name).name == name and digest(SOURCE / name) == expected, name
    assert digest(SOURCE / "plan.json") == manifest["plan_sha256"]
    assert digest(SOURCE / "reference_manifest.json") == manifest["reference"]["manifest_sha256"]
    reference = read(SOURCE / "reference_manifest.json")
    old = read(SOURCE / "reference_summary.json")
    assert reference["status"] == "completed" and reference["completed_draws"] == 30
    assert digest(SOURCE / "reference_plan.json") == reference["artifact_hashes"]["plan.json"]
    assert digest(SOURCE / "reference_summary.json") == reference["artifact_hashes"]["summary.json"]
    summary = read(SOURCE / "summary.json")
    assert set(summary["methods"]) == set(old["methods"]) | set(NEW)
    assert len(summary["methods"]) == 19 and summary["selection_profiles"] == 16384
    assert summary["conditions"] == list(range(64)) and summary["generation_seeds"] == SEEDS
    reports = {key: read(SOURCE / f"{key}.json") for key in summary["methods"]}
    for key in old["methods"]:
        assert summary["methods"][key] == old["methods"][key]
        assert digest(SOURCE / f"{key}.json") == reference["artifact_hashes"][f"{key}.json"]
    for key, report in reports.items():
        assert [draw["generation_seed"] for draw in report["draws"]] == SEEDS
        for draw in report["draws"]:
            assert set(draw["metrics"]["per_condition"]) == {str(i) for i in range(64)}
            assert all(row["n_real"] == row["n_fake"] == 256
                       for row in draw["metrics"]["per_condition"].values())
        for section in ("metrics", "proximity"):
            for metric, value in summary["methods"][key][section].items():
                assert np.mean([draw[section]["macro"][metric] for draw in report["draws"]]) == value["mean"]
    for name in COPIED:
        shutil.copyfile(SOURCE / name, REPORT / name)
        assert digest(SOURCE / name) == digest(REPORT / name)

    method_rows = []
    for name, method in summary["methods"].items():
        report = reports[name]
        row = {"method_id": name, "method": method["method"], "training_seed": method["training_seed"], "draws": 2}
        for section in ("metrics", "tail_errors", "proximity"):
            row.update({f"{section}_{key}_mean": value["mean"] for key, value in method[section].items()})
        for q in ("q01", "q05", "q50", "q95", "q99"):
            row[f"proximity_{q}_rms_kwh_mean"] = float(np.mean([
                draw["proximity"]["macro"][f"{q}_rms_kwh"] for draw in report["draws"]]))
        row["quantized_exact_training_match_fraction_mean"] = method["quantized_exact_training_match_fraction"]["mean"]
        for metric in ("export_day_fraction", "negative_daily_energy_fraction"):
            row[f"source_pv0_{metric}_mean"] = method["source_pv0"]["fake"][metric]["mean"]
        row["fit_seconds"] = report.get("fit_seconds", "")
        for timing in ("generation_seconds", "physical_metric_seconds", "nearest_neighbor_seconds"):
            row[f"{timing}_mean"] = float(np.mean([draw[timing] for draw in report["draws"]]))
        row["source_report_sha256"] = manifest["artifact_hashes"][f"{name}.json"]
        method_rows.append(row)
    write_csv("method_comparison.csv", method_rows)

    fits, starts, components, atoms, tails, model_hashes = [], [], [], [], [], {}
    rank_hashes, input_hashes = None, None
    for name in NEW:
        report = reports[name]
        fit = read(SOURCE / f"{name}_fit.json")
        meta = report["fit_metadata"]
        assert report["status"] == "completed" and meta == fit["fit_metadata"]
        assert report["data_hashes"] == manifest["data_hashes"]
        assert meta["conditions"] == list(range(64)) and meta["profile_dim"] == 24
        assert meta["config"] == {key: value for key, value in report["specification"].items() if key != "method_id"}
        assert meta["training_profiles"] == 485212 and meta["copula_maximum_likelihood_fit"] is False
        assert report["model_sha256"] == fit["model_sha256"] == manifest["artifact_hashes"][report["model_artifact"]]
        assert read(SOURCE / f"{name}_worker.json")["returncode"] == 0
        model_hashes[report["model_artifact"]] = report["model_sha256"]
        current_ranks = {c: (v["ranks_sha256"], v["normal_scores_sha256"])
                         for c, v in meta["per_condition"].items()}
        current_inputs = {k: meta[k] for k in ("source_profile_bytes_sha256", "source_label_bytes_sha256")}
        assert rank_hashes is None or rank_hashes == current_ranks
        assert input_hashes is None or input_hashes == current_inputs
        rank_hashes, input_hashes = current_ranks, current_inputs
        nonconverged = []
        all_attempts, arm_components = [], []
        for condition, record in meta["per_condition"].items():
            assert record["status"] == "fitted" and len(record["initializations"]) == 2
            selected = record["selected_init"]
            assert selected == int(np.argmax([a["final_score"] for a in record["initializations"]]))
            if not record["converged"]:
                nonconverged.append(int(condition))
            for attempt in record["initializations"]:
                all_attempts.append(attempt)
                starts.append({"method_id": name, "condition": int(condition), "training_profiles": record["training_profiles"],
                    "selected": attempt["index"] == selected,
                    **{key: attempt[key] for key in ("index", "seed", "final_score", "lower_bound", "converged", "n_iter")},
                    "warning_count": len(attempt["warnings"]),
                    "warning_categories": ";".join(w["category"] for w in attempt["warnings"])})
            for component, weight in enumerate(record["weights"]):
                row = {"method_id": name, "condition": int(condition), "component": component,
                    "training_profiles": record["training_profiles"], "selected_init": selected,
                    "selected_converged": record["converged"], "weight": weight,
                    **{key: record[key][component] for key in ("effective_occupancies", "hard_occupancies",
                        "covariance_min_eigenvalues", "covariance_max_eigenvalues")}}
                components.append(row)
                arm_components.append(row)
        fits.append({"method_id": name, "n_components": meta["n_components"], "fit_seed": meta["config"]["fit_seed"],
            "conditions": 64, "attempts": len(all_attempts),
            "nonconverged_attempts": sum(not a["converged"] for a in all_attempts),
            "nonconverged_selected": len(nonconverged),
            "nonconverged_selected_conditions": ";".join(map(str, sorted(nonconverged))),
            "minimum_component_weight": min(c["weight"] for c in arm_components),
            "minimum_effective_occupancy": min(c["effective_occupancies"] for c in arm_components),
            "minimum_hard_occupancy": min(c["hard_occupancies"] for c in arm_components),
            "minimum_covariance_eigenvalue": min(c["covariance_min_eigenvalues"] for c in arm_components),
            "maximum_covariance_eigenvalue": max(c["covariance_max_eigenvalues"] for c in arm_components),
            "fit_seconds": report["fit_seconds"], "elapsed_seconds": report["elapsed_seconds"],
            "model_artifact": report["model_artifact"], "model_sha256": report["model_sha256"],
            "fit_metadata_artifact_sha256": manifest["artifact_hashes"][f"{name}_fit.json"]})
        for draw in report["draws"]:
            probabilities = draw["diagnostics"]["quantile_probabilities"]
            for condition, group in draw["diagnostics"]["per_condition"].items():
                row = {"method_id": name, "generation_seed": draw["generation_seed"], "condition": int(condition)}
                for quantity, unit in (("daily_energy", "kwh"), ("peak", "kw"), ("absolute_ramp", "kw")):
                    for quantile, probability in (("q01", .01), ("q99", .99)):
                        index = probabilities.index(probability)
                        for side in ("real", "fake"):
                            row[f"{side}_{quantity}_{quantile}_{unit}"] = group[side][f"{quantity}_quantiles_{unit}"][index]
                tails.append(row)
            for condition, group in draw["marginal_support"]["per_condition"].items():
                row = {"method_id": name, "generation_seed": draw["generation_seed"], "condition": int(condition),
                    "profiles": group["profiles"], "training_profiles": group["training_profiles"],
                    "outside_training_support_cells": sum(group["outside_training_support_counts"])}
                for label, generated, target in (("zero", "zero_counts", "training_zero_counts"),
                        ("negative", "negative_counts", "training_negative_counts"),
                        ("min", "at_training_min_counts", "training_min_counts"),
                        ("max", "at_training_max_counts", "training_max_counts")):
                    row[f"generated_{label}_cells"] = sum(group[generated])
                    row[f"training_{label}_cells"] = sum(group[target])
                    row[f"{label}_hourly_frequency_mae_vs_train"] = float(np.mean(np.abs(
                        np.asarray(group[generated]) / group["profiles"]
                        - np.asarray(group[target]) / group["training_profiles"])))
                atoms.append(row)
    write_csv("fit_summary.csv", fits)
    write_csv("initializations.csv", starts)
    write_csv("component_diagnostics.csv", components)
    write_csv("marginal_support.csv", atoms)
    write_csv("conditional_tail_quantiles.csv", tails)

    pairs = [(new, control) for new in NEW for control in CONTROLS]
    pairs += [(new, "gmm_k1") for new in NEW[1:]]
    comparisons = []
    for a, b in pairs:
        for index, seed in enumerate(SEEDS):
            left = reports[a]["draws"][index]["metrics"]["per_condition"]
            right = reports[b]["draws"][index]["metrics"]["per_condition"]
            for metric in ERROR_METRICS:
                delta = np.array([left[str(c)]["metrics"][metric] - right[str(c)]["metrics"][metric] for c in range(64)])
                comparisons.append({"method_a": a, "method_b": b, "generation_seed": seed, "metric": metric,
                    "conditions_a_lower": int(np.sum(delta < 0)), "conditions_tied": int(np.sum(delta == 0)),
                    "conditions_a_higher": int(np.sum(delta > 0)), "macro_difference_a_minus_b": float(np.mean(delta))})
    write_csv("conditional_comparisons.csv", comparisons)
    real = read(SOURCE / "real_proximity.json")
    assert real["real_profile_bytes_sha256"] == manifest["real_profile_bytes_sha256"]
    assert real["label_bytes_sha256"] == manifest["label_bytes_sha256"]
    figure(reports, real["proximity"]["macro"]["mean_rms_kwh"])
    receipt = {"schema_version": 1,
        "role": "Extraction of completed mixture JSON/CSV artifacts; no dataset or fitted-array deserialization, sampling or reevaluation",
        "source_manifest": str((SOURCE / "manifest.json").relative_to(ROOT)),
        "source_manifest_sha256": digest(SOURCE / "manifest.json"), "source_evaluation_git_head": manifest["git_head"],
        "verified_new_draws": 8, "verified_reused_draws": 30, "verified_methods": 19,
        "copied_byte_identical": {name: digest(REPORT / name) for name in COPIED},
        "input_artifact_sha256": manifest["artifact_hashes"], "fitted_models_verified_by_streaming_sha256_only": model_hashes,
        "derivation_script_sha256": digest(Path(__file__)),
        "derived_artifact_sha256": {name: digest(REPORT / name) for name in DERIVED},
        "rows": {"methods": len(method_rows), "initializations": len(starts), "components": len(components),
                 "marginal_condition_draws": len(atoms), "conditional_comparisons": len(comparisons),
                 "conditional_tail_quantiles": len(tails)},
        "aggregation": "Equal-weight mean of 64 conditions in each draw, then two paired generation draws for fixed fits; not independent training or meter replications.",
        "initialization_rule_verified": "Largest final TRAIN mean log density, earliest exact tie. Identical ranks and normal scores across K. All convergence outcomes retained.",
        "fit_snapshot_note": "Byte-identical *_fit.json files were written before verified reload and retain status=running. Completed method reports and manifest bind their identical fit_metadata and model hashes.",
        "marginal_definition": "Each condition row sums observed cells across 24 hours. Frequency MAE is the mean of per-hour absolute generated-versus-TRAIN atom frequency differences; no finite-draw noise subtraction. Min/max counts overlap for constant coordinates.",
        "proximity_definition": "Exact nearest same-condition TRAIN Euclidean distance divided by sqrt(24), kWh. Macro quantiles average condition quantiles rather than pooling; below-reference fractions use strict < against condition-specific real p05/median. No novelty or privacy guarantee.",
        "conditional_comparison_definition": "Strict lower/tied/higher condition counts, separately for each paired draw. Descriptive counts, not independent hypothesis tests; physical profile SW1 has the same ordering as fixed-range normalized SW1.",
        "tail_quantile_definition": "Real/generated condition-specific q01 and q99 of daily energy, peak, and absolute adjacent-hour ramps. Daily/peak quantiles use 256 profiles; ramp quantiles pool their 23 differences. NumPy linear interpolation; dependent observations and sparse tails preclude treating these as precise tail estimates.",
        "figure_definition": "Bars/ticks are means; circles/crosses are actual generation draws. No inferential error bars; dashed proximity line is the real inner-VAL mean.",
        "figure_runtime": {"matplotlib": matplotlib.__version__, "numpy": np.__version__}}
    (REPORT / "DERIVATION.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"Verified {len(manifest['artifact_hashes'])} source artifacts, 8 new and 30 reused draws; "
          f"copied {len(COPIED)} files, derived 19 method rows, {len(starts)} initialization records, "
          f"{len(components)} component records and one figure in two formats.")


if __name__ == "__main__":
    main()
