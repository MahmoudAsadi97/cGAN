"""Collect the prespecified five-run calibration robustness checks from JSON only.

No datasets, checkpoints, calibration artifacts, inference or fitting are opened.
Validation caps are summarized separately; sensitivity uses training seed 11 and
changes exactly one calibration setting. No confidence intervals are averaged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


PRIMARY = "calibration_validation.json"
PRECISION = "calibration_validation_1000.json"
SENSITIVITIES = {
    "calibration_seed7302": ("calibration_sensitivity_seed7302.json", 7302, 4096),
    "calibration_n8192": ("calibration_sensitivity_n8192.json", 7301, 8192),
}
BASELINES = ("gaussian_copula", "independent_marginals", "exact_marginal_copula", "conditional_resampling")
METHODS = ("raw_cgan", "calibrated_cgan", *BASELINES)
TRAINING_SEEDS = (11, 22, 33, 44, 55)
EVALUATION_SEED = 2026
QUANTILE_KNOTS = 1025
PHYSICAL = ("pointwise_wasserstein_kwh", "daily_energy_wasserstein_kwh", "peak_power_wasserstein_kw",
            "ramp_wasserstein_kw", "lag_one_acf_wasserstein", "diversity_ratio")
MEAN_BEHAVIOR = ("mean_profile_rmse_kwh", "pointwise_variance_rmse_kwh2", "real_mean_daily_energy_kwh",
                 "generated_mean_daily_energy_kwh", "real_mean_peak_power_kw", "generated_mean_peak_power_kw",
                 "real_mean_ramp_kw", "generated_mean_ramp_kw", "real_mean_pointwise_variance_kwh2",
                 "generated_mean_pointwise_variance_kwh2")
PREPARED = ("train.npz", "val.npz", "metadata.json", "scaler.json")
CORE = tuple(f"src/fluvius_cgan/{name}.py" for name in ("evaluation", "calibration", "copula", "joint", "battery"))


def _check(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _hash(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _same(left, right):
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_same(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right))
    if _finite(left) and _finite(right):
        return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)
    return left == right


def _numeric(values, keys):
    result = {key: values[key] for key in keys}
    _check(all(value is None or _finite(value) for value in result.values()), "nonfinite or nonnumeric metric")
    return result


def _physical(row):
    return _numeric({**row, "lag_one_acf_wasserstein": row["autocorrelation"]["1"]["wasserstein"]}, PHYSICAL)


def _metrics(report):
    output = {}
    for method in METHODS:
        physical = report["methods"][method]
        joint = report["joint_diagnostics"][method]
        battery = report["battery_diagnostics"][method]
        output[method] = {
            "physical": {"macro": _physical(physical["macro"]),
                         "per_condition": {label: _physical(row) for label, row in physical["per_condition"].items()}},
            "mean_behavior": {"macro": _numeric(physical["macro"], MEAN_BEHAVIOR),
                              "per_condition": {label: _numeric(row, MEAN_BEHAVIOR) for label, row in physical["per_condition"].items()}},
            "joint": {"macro": joint["macro"],
                      "per_condition": {label: row["mean_projected_wasserstein_kwh"] for label, row in joint["per_condition"].items()}},
            "battery": {"macro": _numeric(battery["macro"], battery["macro"]),
                        "per_condition": {label: _numeric(row, battery["macro"]) for label, row in battery["per_condition"].items()}},
        }
    return output


def _settings(report):
    return {
        "prepared_data_hashes": report["prepared_data_hashes"], "source_sha256": report["source_sha256"],
        "resolution": report["resolution"], "interval_hours": report["interval_hours"],
        "target": report["target"], "units": report["units"], "evaluation_seed": report["seed"],
        "quantile_knots": report["calibration"]["quantile_knots"], "baselines": report["baselines"],
        "conditions": report["validation_selection"]["conditions"],
        "training_selection": {key:report["checkpoint"]["training_selection"][key]
                               for key in ("train_profiles_available", "train_profiles_used", "train_indices_sha256")},
        "evaluation_script_sha256": report["provenance"]["evaluation_script_sha256"],
        "evaluation_core_files_sha256": {name: report["provenance"]["source_files_sha256"][name] for name in CORE},
        "joint_directions": report["joint_diagnostics"]["raw_cgan"]["directions"],
        "battery_scenario": report["battery_diagnostics"]["raw_cgan"]["scenario"],
    }


def _load(path, cap, calibration_seed, calibration_count):
    raw = path.read_bytes()
    def invalid(value):
        raise ValueError(f"nonfinite JSON constant: {value}")
    report = json.loads(raw, parse_constant=invalid)
    _check(report["split"] == "val" and report["test_used"] is False, "report must be validation-only with test_used=false")
    _check(report["seed"] == EVALUATION_SEED, f"frozen protocol requires evaluation seed {EVALUATION_SEED}")
    _check(report["target"] == "net" and report["units"] == "kWh/interval", "signed net physical units required")
    _check(report["resolution"] in (24, 96) and report["interval_hours"] == 24 / report["resolution"], "invalid resolution")
    _check(set(report["methods"]) == set(METHODS), "required method coverage differs")
    _check(set(report["joint_diagnostics"]) == set(METHODS) == set(report["battery_diagnostics"]), "joint/battery method coverage differs")
    hashes, checkpoint = report["prepared_data_hashes"], report["checkpoint"]
    _check(set(hashes) == set(PREPARED) and all(_hash(value) for value in hashes.values()), "invalid prepared hashes")
    _check(_hash(report["source_sha256"]) and _hash(checkpoint["sha256"]), "invalid source/checkpoint hash")
    _check(all(checkpoint["data_hashes"][key] == value for key, value in hashes.items()), "checkpoint/prepared hashes disagree")
    selection, frozen = report["validation_selection"], report["frozen_samples"]
    _check(selection["samples_per_condition_cap"] == cap, "unexpected validation cap")
    conditions = {str(label) for label in selection["conditions"]}
    _check(bool(conditions) and len(conditions) == len(selection["conditions"]) and
           all(isinstance(label, int) and not isinstance(label, bool) and 0 <= label < 64 for label in selection["conditions"]), "invalid condition support")
    counts = selection["condition_counts"]
    _check(set(counts) == conditions and all(isinstance(n, int) and not isinstance(n, bool) and 0 < n <= cap for n in counts.values()), "invalid condition counts")
    _check(frozen["profile_shape"] == [sum(counts.values()), report["resolution"]] and
           selection["selected_profiles"] == sum(counts.values()), "selected profile counts disagree")
    _check(all(_hash(value) for value in (selection["selected_indices_sha256"], frozen["labels_sha256"], frozen["real_profiles_sha256"])), "invalid frozen reference hashes")
    _check(set(frozen["profiles_sha256"]) == set(METHODS) and all(_hash(value) for value in frozen["profiles_sha256"].values()), "invalid generated hashes")
    calibration, provenance = report["calibration"], report["calibration"]["provenance"]
    _check(report["calibration_seed"] == calibration_seed == provenance["calibration_seed"] != report["seed"], "unexpected calibration seed")
    _check(calibration["samples_per_condition"] == provenance["calibration_samples_per_condition"] == calibration_count, "unexpected calibration sample count")
    support = checkpoint["supported_labels"]
    _check(set(support) >= set(selection["conditions"]), "checkpoint support omits validation conditions")
    _check(calibration["calibration_profiles"] == provenance["calibration_profiles_total"] == calibration_count * len(support), "calibration reference total differs")
    _check(provenance["checkpoint_sha256"] == checkpoint["sha256"] and provenance["prepared_data_hashes"] == hashes and
           provenance["train_indices_sha256"] == checkpoint["training_selection"]["train_indices_sha256"], "calibration provenance binding differs")
    _check(calibration["quantile_knots"] == provenance["quantile_knots"] == QUANTILE_KNOTS,
           f"frozen protocol requires {QUANTILE_KNOTS} calibration quantile knots")
    _check(all(report["baselines"][method]["quantile_knots"] == QUANTILE_KNOTS
               for method in ("gaussian_copula", "independent_marginals")) and
           report["baselines"]["exact_marginal_copula"]["quantile_knots"] is None,
           "frozen protocol baseline marginal settings differ")
    _check(report["baselines"]["conditional_resampling"]["seed"] == EVALUATION_SEED,
           "frozen protocol resampling seed differs")
    _check(all(_hash(provenance[key]) for key in ("train_indices_sha256", "calibration_generated_profiles_sha256", "calibration_labels_sha256")), "invalid calibration hashes")
    _check(report["seeds"]["calibration_reference"] == calibration_seed and
           all(report["seeds"][key] == report["seed"] for key in ("validation_selection", "evaluation_generation", "baseline_generation", "nearest_neighbor_subsampling")), "random-stream metadata disagree")
    settings = _settings(report)
    _check(_hash(settings["evaluation_script_sha256"]) and all(_hash(value) for value in settings["evaluation_core_files_sha256"].values()), "missing evaluation source fingerprints")
    directions, scenario = settings["joint_directions"], settings["battery_scenario"]
    _check(_hash(directions["sha256"]), "invalid joint directions hash")
    for method in METHODS:
        physical, joint, battery = (report[key][method] for key in ("methods", "joint_diagnostics", "battery_diagnostics"))
        _check(set(physical["per_condition"]) == set(joint["per_condition"]) == set(battery["per_condition"]) == conditions, "condition coverage differs")
        _check(joint["directions"] == directions and battery["scenario"] == scenario and battery["scenario_sha256"] == _sha(_canonical(scenario)), "joint/battery scenario differs")
        _check(_finite(joint["macro"]) and all(_finite(row["mean_projected_wasserstein_kwh"]) for row in joint["per_condition"].values()), "invalid joint values")
        for label in conditions:
            _check(physical["counts"]["real"][label] == physical["counts"]["generated"][label] == counts[label], "physical method counts differ")
            _check(battery["per_condition"][label]["n_real"] == battery["per_condition"][label]["n_generated"] == counts[label], "battery method counts differ")
    summaries = _metrics(report)
    return {"report": report, "path": str(path), "sha256": _sha(raw), "settings": settings, "metrics": summaries}


def _evidence(item):
    r = item["report"]
    return {"path": item["path"], "sha256": item["sha256"], "checkpoint": r["checkpoint"],
            "source_provenance": r["provenance"], "calibration_provenance": r["calibration"]["provenance"],
            "validation_selection": r["validation_selection"], "frozen_samples": r["frozen_samples"],
            "metrics": item["metrics"]}


def _checkpoint_pair(a, b):
    _check(a["settings"] == b["settings"], "evaluation settings, data, subset or diagnostic sources differ")
    for key in ("sha256", "data_hashes", "config", "training_selection", "epoch", "supported_labels"):
        _check(a["report"]["checkpoint"][key] == b["report"]["checkpoint"][key], f"checkpoint differs: {key}")
    varying = {"samples_per_condition", "calibration_profiles", "provenance", "fitting_diagnostics",
               "application_diagnostics", "saved_model"}
    contracts = [{key:value for key,value in item["report"]["calibration"].items() if key not in varying}
                 for item in (a, b)]
    _check(contracts[0] == contracts[1], "fixed calibration method or policy differs")


def _same_controls(a, b, methods):
    for method in methods:
        _check(a["report"]["frozen_samples"]["profiles_sha256"][method] == b["report"]["frozen_samples"]["profiles_sha256"][method], f"control profile hash differs: {method}")
        _check(all(_same(a["report"][key][method], b["report"][key][method]) for key in ("methods", "joint_diagnostics", "battery_diagnostics")), f"control diagnostics differ: {method}")


def _stats(values):
    if any(value is None for value in values):
        return {"n": len(values), "n_defined": sum(value is not None for value in values), "mean": None, "sample_sd": None}
    return {"n": len(values), "n_defined": len(values), "mean": statistics.mean(values), "sample_sd": statistics.stdev(values)}


def _aggregate(values):
    if isinstance(values[0], dict):
        _check(all(value.keys() == values[0].keys() for value in values), "metric coverage differs across reports")
        return {key: _aggregate([value[key] for value in values]) for key in values[0]}
    return _stats(values)


def _delta(value, reference):
    if isinstance(value, dict):
        _check(value.keys() == reference.keys(), "sensitivity metric coverage differs")
        return {key: _delta(value[key], reference[key]) for key in value}
    return None if value is None or reference is None else value - reference


def _collect(pairs, sensitivity_run):
    pairs = list(pairs)
    _check(len(pairs) == 5, "exactly five prespecified runs are required")
    _check(len({name for name, _ in pairs}) == 5 and all(name.strip() for name, _ in pairs), "run names must be nonempty and unique")
    directories = [Path(path).resolve() for _, path in pairs]
    _check(len(set(directories)) == 5, "run directories must be unique")
    sensitivity_path = Path(sensitivity_run).resolve()
    _check(sensitivity_path in directories, "sensitivity run must be one of the five named runs")
    loaded, all_items = [], []
    for (name, _), directory in zip(pairs, directories):
        primary = _load(directory / PRIMARY, 100, 7301, 4096)
        precision = _load(directory / PRECISION, 1000, 7301, 4096)
        _checkpoint_pair(primary, precision)
        _check(primary["report"]["calibration"]["provenance"] == precision["report"]["calibration"]["provenance"], "precision calibration provenance differs")
        _check(primary["report"]["calibration"]["fitting_diagnostics"] == precision["report"]["calibration"]["fitting_diagnostics"], "precision calibration fitting differs")
        # A primary report may name its saved map. Follow-up reports save no map;
        # path differences do not imply a different fit, and no NPZ is opened.
        _check(precision["report"]["calibration"].get("saved_model") is None, "precision report unexpectedly saves a map")
        seed = primary["report"]["checkpoint"]["config"]["seed"]
        _check(isinstance(seed, int) and not isinstance(seed, bool), "invalid training seed")
        loaded.append({"name": name, "directory": str(directory), "training_seed": seed, "caps": {"100": primary, "1000": precision}})
        all_items.extend((primary, precision))
    _check(len({run["training_seed"] for run in loaded}) == 5, "training seeds must be unique; do not duplicate runs")
    _check({run["training_seed"] for run in loaded} == set(TRAINING_SEEDS),
           f"frozen protocol requires training seeds {list(TRAINING_SEEDS)}")
    reference = loaded[0]["caps"]["100"]
    base_config = {key:value for key,value in reference["report"]["checkpoint"]["config"].items() if key != "seed"}
    for run in loaded:
        primary = run["caps"]["100"]
        _check(primary["settings"] == reference["settings"], "named runs do not share declared settings/data/subset")
        _check({key:value for key,value in primary["report"]["checkpoint"]["config"].items() if key != "seed"} == base_config, "named run training configurations differ beyond seed")
    precision_summary = {}
    for cap in ("100", "1000"):
        members = [run["caps"][cap] for run in loaded]
        first = members[0]["report"]
        for member in members[1:]:
            r = member["report"]
            _check(r["validation_selection"] == first["validation_selection"] and
                   all(r["frozen_samples"][key] == first["frozen_samples"][key] for key in ("labels_sha256", "real_profiles_sha256", "profile_shape")), "within-cap frozen real samples differ")
            _same_controls(members[0], member, BASELINES)
        precision_summary[cap] = {"validation_samples_per_condition_cap": int(cap), "n_training_seeds": 5,
                                  "methods": _aggregate([item["metrics"] for item in members]),
                                  "baseline_consistency_verified": True,
                                  "baseline_interpretation": "Identical deterministic controls repeated across training seeds are not independent replications"}
    selected = next(run for run in loaded if Path(run["directory"]) == sensitivity_path)
    _check(selected["training_seed"] == 11, "sensitivity run must have training seed 11")
    primary, variants = selected["caps"]["100"], {}
    for name, (filename, seed, count) in SENSITIVITIES.items():
        variant = _load(sensitivity_path / filename, 100, seed, count)
        _checkpoint_pair(primary, variant)
        a, b = primary["report"], variant["report"]
        _check(a["validation_selection"] == b["validation_selection"] and
               all(a["frozen_samples"][key] == b["frozen_samples"][key] for key in ("labels_sha256", "real_profiles_sha256", "profile_shape")), "sensitivity evaluation draws differ")
        _check(b["calibration"].get("saved_model") is None, "sensitivity report unexpectedly saves a map")
        allowed = {"calibration_generated_profiles_sha256"}
        if seed != 7301:
            allowed.add("calibration_seed")
        else:
            allowed.update(("calibration_samples_per_condition", "calibration_profiles_total", "calibration_labels_sha256"))
        ap, bp = a["calibration"]["provenance"], b["calibration"]["provenance"]
        _check(ap.keys() == bp.keys() and all(ap[key] == bp[key] for key in ap.keys() - allowed), "unexpected calibration provenance change")
        _same_controls(primary, variant, ("raw_cgan", *BASELINES))
        variants[name] = {"calibration_seed": seed, "calibration_samples_per_condition": count,
                          "evidence": _evidence(variant), "delta_from_primary": _delta(variant["metrics"], primary["metrics"]),
                          "same_evaluation_draws_and_controls_verified": True}
        all_items.append(variant)
    for item in all_items:
        _check(_sha(Path(item["path"]).read_bytes()) == item["sha256"], "input report changed during collection")
    return {
        "schema_version": 1, "split": "val", "test_used": False,
        "scope": "Prespecified exploratory validation robustness; five named training seeds, no seed selection",
        "aggregation": "Validation caps are separate. Mean and sample SD across all five training seeds; undefined inputs propagate. No confidence intervals are averaged or inferred.",
        "access_scope": "Input JSON reports only; no checkpoint, dataset, map, inference or fitting access. Report bindings are checked, not independently re-evaluated.",
        "frozen_protocol": {"training_seeds": list(TRAINING_SEEDS), "evaluation_seed": EVALUATION_SEED,
                            "quantile_knots": QUANTILE_KNOTS, "validation_caps": [100, 1000],
                            "primary_calibration_seed": 7301, "primary_calibration_samples_per_condition": 4096,
                            "sensitivity_training_seed": 11},
        "settings": reference["settings"],
        "runs": [{"name": run["name"], "directory": run["directory"], "training_seed": run["training_seed"],
                  "reports": {cap:_evidence(item) for cap,item in run["caps"].items()}} for run in loaded],
        "precision": precision_summary,
        "calibration_sensitivity": {"run_name": selected["name"], "training_seed": 11,
                                    "primary_report_sha256": primary["sha256"], "variants": variants,
                                    "interpretation": "Single prespecified training seed; descriptive changes from its primary fit, not additional training-seed replications"},
        "provenance": {"collector_sha256": _sha(Path(__file__).read_bytes())},
    }


def collect(pairs, sensitivity_run):
    try:
        return _collect(pairs, sensitivity_run)
    except (KeyError, TypeError, AttributeError, IndexError) as error:
        raise ValueError(f"missing or malformed robustness report: {error}") from error


def markdown(report):
    lines = ["# Calibration robustness", "", report["scope"], "", report["aggregation"], "",
             "All five declared runs are included. Values are mean ± sample SD, not confidence intervals."]
    for cap, group in report["precision"].items():
        lines += ["", f"## Validation cap {cap} per condition", "", "| Method | " + " | ".join(PHYSICAL) + " |",
                  "|---|" + "---:|" * len(PHYSICAL)]
        for method in METHODS:
            values = group["methods"][method]["physical"]["macro"]
            cells = ["undefined" if values[key]["mean"] is None else f"{values[key]['mean']:.6g} ± {values[key]['sample_sd']:.6g}" for key in PHYSICAL]
            lines.append("| " + method + " | " + " | ".join(cells) + " |")
    lines += ["", "## Seed 11 calibration sensitivity", "", "Values are calibrated-cGAN differences from the cap-100 primary report.", "",
              "| Change | " + " | ".join(PHYSICAL) + " |", "|---|" + "---:|" * len(PHYSICAL)]
    for name, variant in report["calibration_sensitivity"]["variants"].items():
        values = variant["delta_from_primary"]["calibrated_cgan"]["physical"]["macro"]
        lines.append("| " + name + " | " + " | ".join("undefined" if values[key] is None else f"{values[key]:+.6g}" for key in PHYSICAL) + " |")
    lines += ["", "Per-condition physical, joint, battery and mean-behavior summaries and input/source fingerprints are retained in JSON.", "",
              "Deterministic baselines match within each cap. Sensitivity runs retain identical real draws, raw cGAN draws and controls. No held-out test evaluation is included."]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="Prespecified NAME=DIRECTORY, repeated exactly five times")
    parser.add_argument("--sensitivity-run", required=True, help="Directory of the named training-seed-11 run")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args(argv)
    outputs = [path for path in (args.output, args.markdown) if path is not None]
    _check(len({path.resolve() for path in outputs}) == len(outputs), "output paths must differ")
    _check(args.output.suffix == ".json", "output must have .json extension")
    for path in outputs:
        if path.exists():
            raise FileExistsError(f"output already exists: {path}")
    pairs = []
    for value in args.run:
        _check("=" in value, "--run must be NAME=DIRECTORY")
        name, directory = value.split("=", 1)
        _check(bool(directory), "run directory must be nonempty")
        pairs.append((name, directory))
    result = collect(pairs, args.sensitivity_run)
    contents = [(args.output, json.dumps(result, indent=2, allow_nan=False) + "\n")]
    if args.markdown is not None:
        contents.append((args.markdown, markdown(result)))
    for path, value in contents:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            stream.write(value)
    print(f"Wrote report-only calibration robustness summary: {args.output}")


if __name__ == "__main__":
    main()
