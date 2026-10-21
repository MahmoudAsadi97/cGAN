"""Derive a deterministic TRAIN-only report from completed, hash-bound fits.

No prepared dataset, generated samples, or validation scores are loaded. The
only NPZ members parsed are saved model metadata, weights, means and covariances
for the original-seed numerical comparison. Full model files are byte-hashed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np


REPORT = Path(__file__).resolve().parent
ROOT = REPORT.parents[2]
PARAMETERS = ("weights", "means", "covariances")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path):
    def invalid(value):
        raise ValueError(f"Nonfinite JSON in {path}: {value}")
    return json.loads(path.read_text(), parse_constant=invalid)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    require(bool(rows), f"No rows for {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def named(path):
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def inventory(directory, manifest, inputs):
    require(not manifest.get("artifact_hash_errors"), f"Inventory errors in {directory}")
    expected = manifest["artifact_hashes"]
    actual = {str(p.relative_to(directory)) for p in directory.rglob("*")
              if p.is_file() and p != directory / "manifest.json"}
    require(actual == set(expected), f"Inventory paths differ in {directory}")
    for name, value in sorted(expected.items()):
        path = directory / name
        require(path.resolve().is_relative_to(directory.resolve()), "Unsafe artifact path")
        observed = digest(path)
        require(observed == value, f"Artifact hash mismatch: {path}")
        inputs[named(path)] = observed
    inputs[named(directory / "manifest.json")] = digest(directory / "manifest.json")


def extract(job, role, receipt, metadata):
    initializations, components, conditions = [], [], []
    require(metadata["conditions"] == list(range(64)), f"Missing conditions: {job}")
    require(metadata["profile_dim"] == 24, f"Wrong profile dimension: {job}")
    for label in metadata["conditions"]:
        group = metadata["per_condition"][str(label)]
        attempts = group["initializations"]
        require(len(attempts) == 2 and len(group["weights"]) == 8, f"Wrong fit counts: {job}/{label}")
        require(group["selected_init"] == max(range(2), key=lambda i: attempts[i]["final_score"]),
                f"Incorrect TRAIN selection: {job}/{label}")
        selected = attempts[group["selected_init"]]
        require(group["converged"] == selected["converged"], "Convergence ledger mismatch")
        require(sum(group["hard_occupancies"]) == group["training_profiles"], "Hard occupancy sum differs")
        require(np.isclose(sum(group["effective_occupancies"]), group["training_profiles"], rtol=1e-10),
                "Effective occupancy sum differs")
        common = {"job": job, "role": role, "fit_seed": metadata["config"]["fit_seed"], "condition": label}
        for attempt in attempts:
            initializations.append({**common, "index": attempt["index"], "seed": attempt["seed"],
                "selected": attempt["index"] == group["selected_init"], "converged": attempt["converged"],
                "n_iter": attempt["n_iter"], "final_score": attempt["final_score"],
                "lower_bound": attempt["lower_bound"], "warning_count": len(attempt["warnings"]),
                "warnings_json": json.dumps(attempt["warnings"], sort_keys=True, separators=(",", ":"))})
        for index in range(8):
            minimum, maximum = group["covariance_min_eigenvalues"][index], group["covariance_max_eigenvalues"][index]
            components.append({**common, "component": index, "weight": group["weights"][index],
                "effective_occupancy": group["effective_occupancies"][index],
                "hard_occupancy": group["hard_occupancies"][index],
                "minimum_covariance_eigenvalue": minimum, "maximum_covariance_eigenvalue": maximum,
                "covariance_condition_number": maximum / minimum,
                "minimum_eigenvalue_at_regularizer": bool(np.isclose(minimum, metadata["config"]["reg_covar"], rtol=1e-8, atol=1e-12))})
        conditions.append({**common, "training_profiles": group["training_profiles"],
            "selected_init": group["selected_init"], "selected_converged": group["converged"],
            "selected_iterations": selected["n_iter"], "selected_final_train_score": selected["final_score"],
            "minimum_effective_occupancy": min(group["effective_occupancies"]),
            "minimum_hard_occupancy": min(group["hard_occupancies"]),
            "minimum_covariance_eigenvalue": min(group["covariance_min_eigenvalues"]),
            "maximum_covariance_eigenvalue": max(group["covariance_max_eigenvalues"]),
            **{key: group[key] for key in ("target_sha256", "ranks_sha256", "normal_scores_sha256",
                                         "weights_sha256", "means_sha256", "covariances_sha256")}})
    iterations = [r["n_iter"] for r in initializations]
    smallest = min(components, key=lambda r: r["effective_occupancy"])
    largest_eigenvalue = max(components, key=lambda r: r["maximum_covariance_eigenvalue"])
    summary = {"job": job, "role": role, "fit_seed": metadata["config"]["fit_seed"],
        "training_profiles": metadata["training_profiles"], "conditions": len(conditions),
        "initializations": len(initializations), "converged_initializations": sum(r["converged"] for r in initializations),
        "converged_selected": sum(r["selected_converged"] for r in conditions),
        "warnings": sum(r["warning_count"] for r in initializations),
        "iterations_minimum": min(iterations), "iterations_median": statistics.median(iterations),
        "iterations_maximum": max(iterations),
        "minimum_component_weight": min(r["weight"] for r in components),
        "minimum_effective_occupancy": min(r["effective_occupancy"] for r in components),
        "minimum_occupancy_condition": smallest["condition"], "minimum_occupancy_component": smallest["component"],
        "minimum_hard_occupancy": min(r["hard_occupancy"] for r in components),
        "components_below_25_effective_rows": sum(r["effective_occupancy"] < 25 for r in components),
        "components_below_100_effective_rows": sum(r["effective_occupancy"] < 100 for r in components),
        "components_at_regularizer": sum(r["minimum_eigenvalue_at_regularizer"] for r in components),
        "minimum_covariance_eigenvalue": min(r["minimum_covariance_eigenvalue"] for r in components),
        "maximum_covariance_eigenvalue": max(r["maximum_covariance_eigenvalue"] for r in components),
        "maximum_eigenvalue_condition": largest_eigenvalue["condition"], "maximum_eigenvalue_component": largest_eigenvalue["component"],
        "maximum_covariance_condition_number": max(r["covariance_condition_number"] for r in components),
        "fit_seconds": receipt["fit_seconds"], "elapsed_seconds": receipt["elapsed_seconds"],
        "model_sha256": receipt["model_sha256"]}
    return summary, initializations, components, conditions


def numerical_comparison(original_path, followup_path, old, new):
    changes, initializations = [], []
    with np.load(original_path, allow_pickle=False) as first, np.load(followup_path, allow_pickle=False) as second:
        require(json.loads(first["metadata"].item()) == old, "Original embedded metadata differs")
        require(json.loads(second["metadata"].item()) == new, "Follow-up embedded metadata differs")
        for label in old["conditions"]:
            before, after = old["per_condition"][str(label)], new["per_condition"][str(label)]
            for key in ("target_sha256", "ranks_sha256", "normal_scores_sha256"):
                require(before[key] == after[key], f"Numerical follow-up changed {key}: {label}")
            row = {"condition": label, "old_selected_init": before["selected_init"],
                "new_selected_init": after["selected_init"], "selected_init_changed": before["selected_init"] != after["selected_init"],
                "old_selected_converged": before["converged"], "new_selected_converged": after["converged"]}
            for name in PARAMETERS:
                a, b = first[f"{name}_{label}"], second[f"{name}_{label}"]
                for values, metadata in [(a, before), (b, after)]:
                    require(values.dtype == np.float64 and np.isfinite(values).all(), "Invalid saved numerical array")
                    require(hashlib.sha256(np.asarray(values, dtype="<f8").tobytes(order="C")).hexdigest() == metadata[name + "_sha256"],
                            f"Parameter hash mismatch: {label}/{name}")
                row.update({name + "_identical": before[name + "_sha256"] == after[name + "_sha256"],
                            name + "_maximum_absolute_difference": float(np.max(np.abs(a-b))),
                            "old_" + name + "_sha256": before[name + "_sha256"],
                            "new_" + name + "_sha256": after[name + "_sha256"]})
            row["selected_parameters_changed"] = any(not row[name + "_identical"] for name in PARAMETERS)
            changes.append(row)
            for a, b in zip(before["initializations"], after["initializations"], strict=True):
                require(a["index"] == b["index"] and a["seed"] == b["seed"], "Initialization seeds changed")
                initializations.append({"condition": label, "index": a["index"], "seed": a["seed"],
                    "old_selected": a["index"] == before["selected_init"], "new_selected": b["index"] == after["selected_init"],
                    "old_converged": a["converged"], "new_converged": b["converged"],
                    "old_n_iter": a["n_iter"], "new_n_iter": b["n_iter"],
                    "old_final_score": a["final_score"], "new_final_score": b["final_score"],
                    "final_score_difference": b["final_score"] - a["final_score"],
                    "old_lower_bound": a["lower_bound"], "new_lower_bound": b["lower_bound"],
                    "old_warnings_json": json.dumps(a["warnings"], sort_keys=True, separators=(",", ":")),
                    "new_warnings_json": json.dumps(b["warnings"], sort_keys=True, separators=(",", ":"))})
    return changes, initializations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=ROOT / "runs/phase1-development/mixture_replication")
    parser.add_argument("--original", type=Path, default=ROOT / "artifacts/phase1-development/mixture_baselines")
    parser.add_argument("--output", type=Path, default=REPORT)
    args = parser.parse_args()
    runs, original, output = args.runs.resolve(), args.original.resolve(), args.output.resolve()
    inputs = {}
    manifest, plan = read(runs / "manifest.json"), read(runs / "plan.json")
    require(manifest["status"] == "completed" and manifest["completed_jobs"] == 6 and not manifest["failed_jobs"], "All six jobs must finish successfully")
    require(manifest["validation_performance_access"] is False and manifest["test_performance_access"] is False,
            "Report requires fit-only provenance")
    inventory(runs, manifest, inputs)
    require(digest(runs / "plan.json") == manifest["plan_sha256"], "Root plan hash differs")
    require(plan["jobs"] == manifest["jobs"], "Root jobs differ from frozen plan")
    require(plan["numerical_followup_seed"] == 20261027 and plan["replication_seeds"] == list(range(20261101, 20261106)), "Unexpected declared seeds")
    old_manifest, old_report = read(original / "manifest.json"), read(original / "gmm_k8_fit.json")
    require(old_manifest["status"] == "completed", "Original screen was not completed")
    inputs[named(original / "manifest.json")] = digest(original / "manifest.json")
    for name in ("gmm_k8_fit.json", old_report["model_artifact"]):
        observed = digest(original / name)
        require(observed == old_manifest["artifact_hashes"][name], f"Original artifact mismatch: {name}")
        inputs[named(original / name)] = observed
    require(digest(original / old_report["model_artifact"]) == old_report["model_sha256"], "Original model receipt differs")
    old = old_report["fit_metadata"]
    estimator_sources = ["src/fluvius_cgan/mixture_baselines.py", "src/fluvius_cgan/strong_baselines.py", "src/fluvius_cgan/marginals.py"]
    require(all(old_manifest["source_sha256"][name] == manifest["source_sha256"][name] for name in estimator_sources),
            "Original/follow-up estimator source hashes differ")
    require(manifest["data_hashes"] == plan["data_hashes"] == old_report["data_hashes"], "Training source bindings differ")
    require(manifest["training_rows"] == old["training_profiles"] == 485212, "Full TRAIN row count differs")
    jobs, fits, all_initializations, all_components, all_conditions = {}, [], [], [], []
    for job, config in plan["jobs"].items():
        receipt, metadata = read(runs / job / "manifest.json"), read(runs / job / "fit_metadata.json")
        require(receipt["status"] == "completed", f"Incomplete fit: {job}")
        inventory(runs / job, receipt, inputs)
        require(receipt["config"] == metadata["config"] == config, f"Config differs: {job}")
        require(config["n_components"] == 8 and config["n_init"] == 2 and config["max_iter"] == 300,
                f"Unexpected fit recipe: {job}")
        for key in ("data_hashes", "plan_sha256", "training_profile_bytes_sha256", "training_label_bytes_sha256", "source_sha256", "runtime"):
            require(receipt[key] == manifest[key], f"Root/child provenance differs: {job}/{key}")
        require(metadata["source_profile_bytes_sha256"] == manifest["training_profile_bytes_sha256"] == old["source_profile_bytes_sha256"], "Profile hashes differ")
        require(metadata["source_label_bytes_sha256"] == manifest["training_label_bytes_sha256"] == old["source_label_bytes_sha256"], "Label hashes differ")
        require(receipt["model_sha256"] == receipt["artifact_hashes"][receipt["model_artifact"]], "Model receipt hash differs")
        require(metadata["library_versions"] == old["library_versions"], "Original/follow-up libraries differ")
        role = "numerical_followup" if config["fit_seed"] == plan["numerical_followup_seed"] else "fresh_replication"
        summary, starts, components, conditions = extract(job, role, receipt, metadata)
        fits.append(summary); all_initializations.extend(starts); all_components.extend(components); all_conditions.extend(conditions)
        jobs[job] = (receipt, metadata, starts)
    followup_job = next(job for job, config in plan["jobs"].items() if config["fit_seed"] == plan["numerical_followup_seed"])
    followup_receipt, new, _ = jobs[followup_job]
    require({k:v for k,v in old["config"].items() if k != "max_iter"} == {k:v for k,v in new["config"].items() if k != "max_iter"}, "Numerical follow-up changed settings beyond cap")
    require(old["config"]["max_iter"] == 100, "Unexpected original cap")
    changes, start_changes = numerical_comparison(original / old_report["model_artifact"], runs / followup_job / followup_receipt["model_artifact"], old, new)
    fresh = [fit for fit in fits if fit["role"] == "fresh_replication"]
    comparison = {"old_max_iter": 100, "new_max_iter": 300,
        "old_model_sha256": old_report["model_sha256"], "new_model_sha256": followup_receipt["model_sha256"],
        "old_fit_seconds": old_report["fit_seconds"], "new_fit_seconds": followup_receipt["fit_seconds"],
        "same_training_arrays_ranks_and_targets": True,
        "same_estimator_source_hashes": {name: manifest["source_sha256"][name] for name in estimator_sources},
        "changed_selected_parameter_conditions": [r["condition"] for r in changes if r["selected_parameters_changed"]],
        "changed_selected_initialization_conditions": [r["condition"] for r in changes if r["selected_init_changed"]],
        "previously_unconverged_initializations": [[r["condition"], r["index"]] for r in start_changes if not r["old_converged"]],
        "changed_initialization_records": [[r["condition"], r["index"]] for r in start_changes if any(r["old_" + key] != r["new_" + key] for key in ("n_iter", "final_score", "lower_bound", "converged", "warnings_json"))],
        "all_originally_converged_starts_identical": all(r["old_n_iter"] == r["new_n_iter"] and r["old_final_score"] == r["new_final_score"] and r["old_lower_bound"] == r["new_lower_bound"] for r in start_changes if r["old_converged"]),
        "parameter_difference_interpretation": "Direct component-index differences; no component matching. A selected-initialization switch can permute component meanings."}
    summary = {"schema_version": 1, "role": "TRAIN-only numerical follow-up and five fresh fit seeds",
        "training_profiles": 485212, "conditions": 64, "validation_scoring": False, "test_scoring": False,
        "fits": fits, "numerical_followup": comparison,
        "fresh_fit_aggregate": {"fits": len(fresh), "initializations": sum(r["initializations"] for r in fresh),
            "converged_initializations": sum(r["converged_initializations"] for r in fresh),
            "selected_conditions": sum(r["conditions"] for r in fresh), "converged_selected": sum(r["converged_selected"] for r in fresh),
            "minimum_effective_occupancy": min(r["minimum_effective_occupancy"] for r in fresh),
            "components_below_25_effective_rows": sum(r["components_below_25_effective_rows"] for r in fresh),
            "components_at_regularizer": sum(r["components_at_regularizer"] for r in fresh),
            "warnings": sum(r["warnings"] for r in fresh),
            "fit_seconds_mean": statistics.mean(r["fit_seconds"] for r in fresh),
            "fit_seconds_minimum": min(r["fit_seconds"] for r in fresh), "fit_seconds_maximum": max(r["fit_seconds"] for r in fresh)}}
    output.mkdir(parents=True, exist_ok=True)
    tables = {"fit_summary.csv": fits, "initializations.csv": all_initializations,
        "components.csv": all_components, "conditions.csv": all_conditions,
        "numerical_followup_conditions.csv": changes, "numerical_followup_initializations.csv": start_changes}
    for job, (_, _, starts) in jobs.items():
        tables[f"initializations/{job}.csv"] = starts
    for name, rows in tables.items():
        write_csv(output / name, rows)
    write_json(output / "summary.json", summary)
    lines = ["# TRAIN-only mixture replication", "",
        "Six fixed K8 fits use all 485,212 inner-TRAIN profiles, 64 conditions and two initializations per condition. The cap is 300 iterations for every fit, with tolerance .001 and covariance diagonal regularization .001. No validation/test profiles, generated draws or held-out scores were loaded for this report.", "",
        "| Fit seed | Role | Converged starts | Converged selected | Iterations min/median/max | Minimum effective rows (condition/component) | Components <25 rows | Fit seconds |",
        "|---|---|---:|---:|---|---:|---:|---:|"]
    for r in fits:
        lines.append(f"| {r['fit_seed']} | {r['role'].replace('_', ' ')} | {r['converged_initializations']}/128 | {r['converged_selected']}/64 | {r['iterations_minimum']}/{r['iterations_median']:g}/{r['iterations_maximum']} | {r['minimum_effective_occupancy']:.3f} ({r['minimum_occupancy_condition']}/{r['minimum_occupancy_component']}) | {r['components_below_25_effective_rows']} | {r['fit_seconds']:.2f} |")
    unchanged = 64 - len(comparison["changed_selected_parameter_conditions"])
    changed_labels = ", ".join(map(str, comparison["changed_selected_parameter_conditions"])) or "none"
    changed_starts = ", ".join(map(str, comparison["changed_selected_initialization_conditions"]))
    selected_text = f"Selected initialization changed in conditions {changed_starts}." if changed_starts else "No selected initialization changed."
    originally_converged = sum(r["old_converged"] for r in start_changes)
    retained_text = (f"All {originally_converged} previously converged starts retained identical iterations, final scores and lower bounds."
                     if comparison["all_originally_converged_starts_identical"] else
                     "Some previously converged starts changed; see the initialization comparison table.")
    resolved = [r for r in start_changes if not r["old_converged"]]
    resolved_text = "; ".join(f"condition {r['condition']}/initialization {r['index']}: {r['old_n_iter']} to {r['new_n_iter']}" for r in resolved)
    aggregate = summary["fresh_fit_aggregate"]
    widest = max(fresh, key=lambda r: r["maximum_covariance_eigenvalue"])
    lines += ["", f"The original-seed numerical follow-up changes selected parameter arrays in conditions **{changed_labels}**; {unchanged}/64 conditions retain byte-identical weights, means and covariances. {selected_text} The five previously capped starts now stop as follows: {resolved_text}. {retained_text} TRAIN arrays, empirical targets, rank permutations, normal scores and estimator source hashes are unchanged.", "",
        f"Across the five fresh fits, {aggregate['converged_initializations']}/{aggregate['initializations']} starts converge with {aggregate['warnings']} recorded warnings. There are {aggregate['components_below_25_effective_rows']} components with fewer than 25 effective rows and {aggregate['components_at_regularizer']} components whose minimum covariance eigenvalue is at the .001 regularizer (within the documented numerical tolerance). The largest fresh-fit covariance eigenvalue is {widest['maximum_covariance_eigenvalue']:.6f}, at seed {widest['fit_seed']}, condition {widest['maximum_eigenvalue_condition']}, component {widest['maximum_eigenvalue_component']}. Covariances are in latent normal-score units. Floor detection uses `isclose(eigenvalue, .001, rtol=1e-8, atol=1e-12)`.", "",
        "Seed 20261027 is a numerical follow-up and is excluded from the five-fresh-fit aggregate. Fresh seeds change tie permutations and EM initialization; these fits reuse the same TRAIN observations and are not independent household samples. Convergence satisfies the declared numerical stopping rule; it does not establish a global optimum or better held-out performance. Small components and covariance regularization remain visible in the tables.", "",
        "The estimator fits an ordinary Gaussian mixture to normal scores and selects by final TRAIN log density. It does not optimize copula likelihood. The reused validation screen informed the fixed K8 choice; these numerical results remain postselection development and establish no held-out quality improvement. Fit times are descriptive recorded wall times, not a controlled speed benchmark.", "",
        "[Fit summaries](fit_summary.csv), [all 768 initialization records](initializations.csv), [component diagnostics](components.csv), and [condition diagnostics](conditions.csv) preserve every fit. The `initializations/` directory also provides one 128-row table per fit. [Numerical initialization comparison](numerical_followup_initializations.csv) and [selected-parameter comparison](numerical_followup_conditions.csv) retain the original 100-iteration evidence. Direct array differences use stored component indices without component alignment.", "",
        "All six models remain in `runs/phase1-development/mixture_replication`; model hashes are recorded in the summary and derivation receipt. Root and child inventories were verified. For the original screen, only its manifest, K8 fit snapshot and model were inspected; its earlier fit snapshot has a running-stage status, while its enclosing manifest is completed. No original validation metric report was parsed.", "",
        "Rebuild from the repository root with:", "", "```bash",
        ".venv/bin/python reports/phase1-development/mixture_replication/derive_summary.py", "```", "",
        "[DERIVATION.json](DERIVATION.json) binds input/output hashes and the derivation script. [summary.json](summary.json) contains full-precision summaries. The script performs no fitting, sampling or dataset loading."]
    (output / "README.md").write_text("\n".join(lines) + "\n")
    generated = [*tables, "summary.json", "README.md"]
    receipt = {"schema_version": 1, "role": summary["role"], "script": named(Path(__file__).resolve()),
        "script_sha256": digest(Path(__file__).resolve()), "input_sha256": inputs,
        "output_sha256": {name: digest(output / name) for name in sorted(generated)},
        "verified_root_artifacts": len(manifest["artifact_hashes"]), "verified_child_manifests": len(jobs),
        "original_artifacts_verified": ["manifest.json", "gmm_k8_fit.json", old_report["model_artifact"]],
        "parsed_npz_members": ["metadata", "weights_<condition>", "means_<condition>", "covariances_<condition>"],
        "prepared_data_loaded": False, "validation_scores_loaded": False, "fitting_or_sampling_performed": False,
        "table_rows": {name: len(rows) for name, rows in tables.items()}}
    write_json(output / "DERIVATION.json", receipt)
    print(json.dumps({"report": str(output), "fits": len(fits), "initializations": len(all_initializations),
                      "changed_selected_parameter_conditions": comparison["changed_selected_parameter_conditions"]}, sort_keys=True))


if __name__ == "__main__":
    main()
