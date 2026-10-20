"""Report-only protocol binding and aggregation checks using synthetic JSON."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics

import pytest


SPEC = importlib.util.spec_from_file_location("collect_calibration_robustness", Path(__file__).resolve().parents[1] / "scripts/collect_calibration_robustness.py")
collector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(collector)


def digest(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, allow_nan=False))


def report(seed, cap=100, calibration_seed=7301, calibration_count=4096):
    labels = [0, 4]
    prepared = {name:digest(name) for name in collector.PREPARED}
    subset = {"train_profiles_available":10000, "train_profiles_used":10000, "train_indices_sha256":digest("train-indices"),
              "training_seed":seed, "max_train_samples":None, "reconstruction":"Exact selected-index hash verified"}
    config = {"seed":seed, "hidden_dim":128, "normalization":"standard"}
    checkpoint = {"sha256":digest(f"checkpoint-{seed}"), "config":config, "epoch":7,
                  "supported_labels":labels, "training_selection":subset, "data_hashes":prepared}
    provenance = {"checkpoint_sha256":checkpoint["sha256"], "prepared_data_hashes":prepared,
                  "train_indices_sha256":subset["train_indices_sha256"], "calibration_seed":calibration_seed,
                  "calibration_samples_per_condition":calibration_count, "calibration_profiles_total":2*calibration_count,
                  "calibration_generated_profiles_sha256":digest(f"source-{seed}-{calibration_seed}-{calibration_count}"),
                  "calibration_labels_sha256":digest(f"labels-{calibration_count}"),
                  "generator_inference_batch_size":1024, "quantile_knots":1025}
    scenario = {"capacity_kwh":10., "interval_hours":1., "initial_soc_kwh":0.}
    scenario_hash = hashlib.sha256(json.dumps(scenario,sort_keys=True,separators=(",", ":")).encode()).hexdigest()
    directions = {"sha256":digest("directions"), "count":128, "seed":9182}
    methods, joint, battery, hashes = {}, {}, {}, {}
    for method in collector.METHODS:
        value = (seed / 10 if method in ("raw_cgan", "calibrated_cgan") else 10.) + (1. if cap == 1000 else 0.)
        if method == "calibrated_cgan":
            value += .2 if calibration_seed != 7301 else .4 if calibration_count != 4096 else 0.
        rows = {}
        for offset, label in enumerate(labels):
            v = value + offset
            rows[str(label)] = {**{key:v for key in collector.PHYSICAL if key != "lag_one_acf_wasserstein"},
                                **{key:v + .1 for key in collector.MEAN_BEHAVIOR},
                                "autocorrelation":{"1":{"wasserstein":v/10}}}
        macro = {**{key:value+.5 for key in collector.PHYSICAL if key != "lag_one_acf_wasserstein"},
                 **{key:value+.6 for key in collector.MEAN_BEHAVIOR},
                 "autocorrelation":{"1":{"wasserstein":(value+.5)/10}}}
        counts = {str(label):cap for label in labels}
        methods[method] = {"macro":macro, "per_condition":rows, "counts":{"real":counts,"generated":counts}}
        joint[method] = {"directions":directions, "macro":(value+.5)/10,
                         "per_condition":{str(label):{"mean_projected_wasserstein_kwh":(value+offset)/10} for offset,label in enumerate(labels)}}
        battery[method] = {"scenario":scenario, "scenario_sha256":scenario_hash,
                           "macro":{"post_grid_import_wasserstein_kwh":value+.5, "real_mean_end_soc_kwh":2.},
                           "per_condition":{str(label):{"post_grid_import_wasserstein_kwh":value+offset, "real_mean_end_soc_kwh":2., "n_real":cap, "n_generated":cap} for offset,label in enumerate(labels)}}
        method_seed = seed if method in ("raw_cgan","calibrated_cgan") else "fixed"
        map_id = f"-{calibration_seed}-{calibration_count}" if method == "calibrated_cgan" else ""
        hashes[method] = digest(f"generated-{method}-{method_seed}-{cap}{map_id}")
    return {
        "split":"val", "test_used":False, "target":"net", "units":"kWh/interval", "resolution":24, "interval_hours":1.,
        "prepared_data_hashes":prepared, "source_sha256":digest("release"), "checkpoint":checkpoint,
        "seed":2026, "calibration_seed":calibration_seed,
        "seeds":{"calibration_reference":calibration_seed, **{key:2026 for key in ("validation_selection","evaluation_generation","baseline_generation","nearest_neighbor_subsampling")}},
        "validation_selection":{"conditions":labels,"samples_per_condition_cap":cap,"condition_counts":{str(label):cap for label in labels}, "selected_profiles":2*cap, "selected_indices_sha256":digest(f"indices-{cap}")},
        "frozen_samples":{"labels_sha256":digest(f"evaluation-labels-{cap}"), "real_profiles_sha256":digest(f"real-{cap}"),"profile_shape":[2*cap,24],"profiles_sha256":hashes},
        "calibration":{"provenance":provenance,"samples_per_condition":calibration_count,"calibration_profiles":2*calibration_count,"quantile_knots":1025,"method":"fixed marginal map","tail_policy":"train endpoints",
                       "fitting_diagnostics":{"reference":digest(f"fit-{seed}-{calibration_seed}-{calibration_count}")},"saved_model":None},
        "baselines":{"gaussian_copula":{"quantile_knots":1025,"shrinkage":.01},"independent_marginals":{"quantile_knots":1025,"shrinkage":1.},"exact_marginal_copula":{"quantile_knots":None,"shrinkage":.01},"conditional_resampling":{"seed":2026}},
        "methods":methods,"joint_diagnostics":joint,"battery_diagnostics":battery,
        "provenance":{"evaluation_script_sha256":digest("eval-script"), "source_files_sha256":{name:digest(name) for name in collector.CORE},"git_commit":"a"*40},
    }


@pytest.fixture
def study(tmp_path):
    pairs = []
    for seed in (11, 22, 33, 44, 55):
        directory = tmp_path / f"seed{seed}"
        directory.mkdir()
        pairs.append((f"seed{seed}", directory))
        primary = report(seed)
        primary["calibration"]["saved_model"] = {"path":"never-open.npz", "sha256":digest("map")}
        write(directory / collector.PRIMARY, primary)
        write(directory / collector.PRECISION, report(seed, 1000))
        if seed == 11:
            for filename, calibration_seed, count in collector.SENSITIVITIES.values():
                write(directory / filename, report(seed, calibration_seed=calibration_seed, calibration_count=count))
    return pairs, pairs[0][1]


def mutate(path, change):
    value = json.loads(path.read_text())
    change(value)
    write(path, value)


def test_all_five_runs_caps_and_sensitivities_are_separate_and_report_only(study, monkeypatch):
    pairs, sensitivity = study
    original_open = Path.open
    def guarded_open(path, mode="r", *args, **kwargs):
        if "r" in mode:
            assert path.suffix in (".json", ".py"), "only JSON reports and own source may be read"
        return original_open(path, mode, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded_open)
    result = collector.collect(pairs, sensitivity)
    assert [row["training_seed"] for row in result["runs"]] == [11,22,33,44,55]
    assert result["frozen_protocol"]["training_seeds"] == [11,22,33,44,55]
    assert result["frozen_protocol"]["evaluation_seed"] == 2026
    assert result["frozen_protocol"]["quantile_knots"] == 1025
    assert set(result["precision"]) == {"100","1000"}
    for cap in ("100", "1000"):
        metrics = result["precision"][cap]["methods"]["calibrated_cgan"]
        expected = [seed/10+.5+(cap == "1000") for seed in (11,22,33,44,55)]
        point = metrics["physical"]["macro"]["pointwise_wasserstein_kwh"]
        assert point == {"n":5,"n_defined":5,"mean":pytest.approx(statistics.mean(expected)),"sample_sd":pytest.approx(statistics.stdev(expected))}
        assert metrics["physical"]["macro"]["lag_one_acf_wasserstein"]["mean"] == pytest.approx(statistics.mean(expected)/10)
        assert metrics["physical"]["per_condition"]["4"]["pointwise_wasserstein_kwh"]["mean"] == pytest.approx(statistics.mean(expected)+.5)
        assert metrics["mean_behavior"]["macro"]["generated_mean_daily_energy_kwh"]["mean"] == pytest.approx(statistics.mean(expected)+.1)
        assert metrics["joint"]["macro"]["mean"] == pytest.approx(statistics.mean(expected)/10)
        assert metrics["battery"]["macro"]["post_grid_import_wasserstein_kwh"]["mean"] == pytest.approx(statistics.mean(expected))
        assert result["precision"][cap]["baseline_consistency_verified"]
    variants = result["calibration_sensitivity"]["variants"]
    for name, expected in (("calibration_seed7302",.2),("calibration_n8192",.4)):
        delta = variants[name]["delta_from_primary"]
        assert delta["calibrated_cgan"]["physical"]["macro"]["pointwise_wasserstein_kwh"] == pytest.approx(expected)
        assert delta["raw_cgan"]["physical"]["macro"]["pointwise_wasserstein_kwh"] == 0
        assert variants[name]["same_evaluation_draws_and_controls_verified"]
    assert result["runs"][0]["reports"]["100"]["sha256"] == hashlib.sha256((sensitivity/collector.PRIMARY).read_bytes()).hexdigest()
    assert "source_files_sha256" in result["runs"][0]["reports"]["100"]["source_provenance"]
    assert result["test_used"] is False
    assert "confidence intervals" in result["aggregation"]
    assert "Validation cap 1000" in collector.markdown(result)
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("case", ["missing_report","wrong_cap","different_checkpoint","different_reference", "changed_knots", "test_used", "missing_method", "changed_source"])
def test_incomplete_or_incompatible_precision_is_rejected(study, case):
    pairs, sensitivity = study
    path = pairs[1][1] / collector.PRECISION
    if case == "missing_report":
        path.unlink()
    else:
        changes = {
            "wrong_cap":lambda r:r["validation_selection"].update(samples_per_condition_cap=999),
            "different_checkpoint":lambda r:r["checkpoint"].update(sha256=digest("other")),
            "different_reference":lambda r:r["calibration"]["provenance"].update(calibration_generated_profiles_sha256=digest("different")),
            "changed_knots":lambda r:r["calibration"].update(quantile_knots=2049),
            "test_used":lambda r:r.update(test_used=True),
            "missing_method":lambda r:r["methods"].pop("independent_marginals"),
            "changed_source":lambda r:r["provenance"]["source_files_sha256"].update({collector.CORE[0]:digest("different-source")}),
        }
        mutate(path, changes[case])
    with pytest.raises((ValueError, FileNotFoundError)):
        collector.collect(pairs, sensitivity)


@pytest.mark.parametrize("which", ["profile_hash","physical","joint","battery"])
def test_deterministic_baselines_must_match_within_each_cap(study, which):
    pairs, sensitivity = study
    path = pairs[2][1] / collector.PRECISION
    def change(r):
        method = "gaussian_copula"
        if which == "profile_hash":
            r["frozen_samples"]["profiles_sha256"][method] = digest("wrong-control")
        elif which == "physical":
            r["methods"][method]["macro"]["pointwise_wasserstein_kwh"] += 1
        elif which == "joint":
            r["joint_diagnostics"][method]["macro"] += 1
        else:
            r["battery_diagnostics"][method]["macro"]["post_grid_import_wasserstein_kwh"] += 1
    mutate(path, change)
    with pytest.raises(ValueError, match="control"):
        collector.collect(pairs, sensitivity)


@pytest.mark.parametrize("case", ["raw_draws","real_draws","seed_changes_count","reference_label_change","evaluation_seed","baseline_metric","tail_policy"])
def test_sensitivity_allows_only_planned_change_and_preserves_draws(study, case):
    pairs, sensitivity = study
    path = sensitivity / collector.SENSITIVITIES["calibration_seed7302"][0]
    changes = {
        "raw_draws":lambda r:r["frozen_samples"]["profiles_sha256"].update(raw_cgan=digest("new-draws")),
        "real_draws":lambda r:r["frozen_samples"].update(real_profiles_sha256=digest("new-real")),
        "seed_changes_count":lambda r:r["calibration"].update(samples_per_condition=8192),
        "reference_label_change":lambda r:r["calibration"]["provenance"].update(calibration_labels_sha256=digest("new-labels")),
        "evaluation_seed":lambda r:r.update(seed=2027),
        "baseline_metric":lambda r:r["methods"]["conditional_resampling"]["macro"].update(diversity_ratio=.1),
        "tail_policy":lambda r:r["calibration"].update(tail_policy="unplanned extrapolation"),
    }
    mutate(path, changes[case])
    with pytest.raises(ValueError):
        collector.collect(pairs, sensitivity)


def test_undefined_diversity_is_not_silently_dropped(study):
    pairs, sensitivity = study
    for filename in (collector.PRIMARY, collector.PRECISION):
        mutate(pairs[2][1]/filename, lambda r:r["methods"]["calibrated_cgan"]["macro"].update(diversity_ratio=None))
    result = collector.collect(pairs, sensitivity)
    assert result["precision"]["100"]["methods"]["calibrated_cgan"]["physical"]["macro"]["diversity_ratio"] == {"n":5,"n_defined":4,"mean":None,"sample_sd":None}


def test_exactly_five_unique_runs_and_prespecified_seed11_are_required(study):
    pairs, sensitivity = study
    for invalid in (pairs[:4], pairs+[pairs[0]], pairs[:4]+[pairs[0]]):
        with pytest.raises(ValueError):
            collector.collect(invalid, sensitivity)
    with pytest.raises(ValueError, match="seed 11"):
        collector.collect(pairs, pairs[1][1])


@pytest.mark.parametrize("setting", ["training_seeds", "evaluation_seed", "quantile_knots"])
def test_coherently_changed_protocol_across_all_reports_is_rejected(study, setting):
    pairs, sensitivity = study
    if setting == "training_seeds":
        # Preserve five distinct seeds and seed 11, but replace frozen seed 55.
        for filename, cap in ((collector.PRIMARY, 100), (collector.PRECISION, 1000)):
            write(pairs[-1][1] / filename, report(66, cap))
    else:
        def change(r):
            if setting == "evaluation_seed":
                r["seed"] = 2027
                for key in r["seeds"]:
                    if key != "calibration_reference":
                        r["seeds"][key] = 2027
                r["baselines"]["conditional_resampling"]["seed"] = 2027
            else:
                r["calibration"]["quantile_knots"] = 2049
                r["calibration"]["provenance"]["quantile_knots"] = 2049
                for method in ("gaussian_copula", "independent_marginals"):
                    r["baselines"][method]["quantile_knots"] = 2049
        for _, directory in pairs:
            for path in directory.glob("*.json"):
                mutate(path, change)
    with pytest.raises(ValueError, match="frozen protocol"):
        collector.collect(pairs, sensitivity)


def test_cli_writes_json_markdown_and_refuses_overwrite(study, tmp_path):
    pairs, sensitivity = study
    output, md = tmp_path/"robustness.json", tmp_path/"robustness.md"
    args = ["--sensitivity-run",str(sensitivity),"--output",str(output),"--markdown",str(md)]
    for name,path in pairs:
        args += ["--run",f"{name}={path}"]
    collector.main(args)
    assert json.loads(output.read_text())["split"] == "val"
    assert "Seed 11 calibration sensitivity" in md.read_text()
    with pytest.raises(FileExistsError):
        collector.main(args)
