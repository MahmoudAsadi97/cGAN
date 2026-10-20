# Completed curated validation study

All five fixed training seeds (11, 22, 33, 44, 55) now have primary physical evaluation, downstream utility, and both 500-replicate paired physical comparisons. They form **one compatible settings group**. The calibrated model meets the relative ramp comparison against the Gaussian copula and the descriptive utility RMSE tolerance, but **fails the primary mean absolute ramp gate and remains worse than conditional resampling on daily energy, peaks, ramps and lag-one ACF**. This is completed validation evidence, not a successful overall acceptance claim or an unseen-test result.

The primary setting remains 100 real and generated profiles per condition, 64 equally weighted conditions, evaluation seed 2026, and a TRAIN-only calibration reference of 4,096 draws per condition (seed 7301, 1,025 quantile knots). All five checkpoints and maps were verified by the collector; no prepared arrays were loaded during collection or summarization.

## Primary physical metrics

Neural entries are mean ± sample SD across all five training seeds. Controls are identical frozen draws and appear once; they are not five independent replications. W1 is lower-is-better. Diversity near 1 is only descriptive.

| Method | Pointwise W1 (kWh/interval) | Daily W1 (kWh) | Peak W1 (kW) | Ramp W1 (kW) | Lag-1 ACF W1 | Diversity ratio |
|---|---:|---:|---:|---:|---:|---:|
| Raw cGAN | 0.09038 ± 0.00136 | 2.99043 ± 0.04363 | 0.73261 ± 0.02157 | 0.20423 ± 0.01257 | 0.09406 ± 0.00720 | 0.99297 ± 0.03832 |
| Calibrated cGAN | 0.05461 ± 0.00096 | 3.27535 ± 0.07953 | 0.78329 ± 0.01325 | 0.10066 ± 0.00601 | 0.08330 ± 0.00640 | 1.12876 ± 0.06687 |
| Gaussian copula | 0.05363 | 2.93850 | 0.99430 | 0.14104 | 0.08536 | 1.08815 |
| Independent marginals | 0.05397 | 8.04477 | 3.42082 | 0.70573 | 0.59999 | 1.12633 |
| Exact-marginal copula | 0.05343 | 2.92722 | 0.97232 | 0.14013 | 0.08503 | 1.07452 |
| Conditional resampling | 0.05465 | 2.82569 | 0.50108 | 0.03417 | 0.04647 | 1.15849 |

Mean calibrated ramp W1 is 0.100655 kW: 28.63% below the Gaussian control, with the same direction in 5/5 seeds. This exceeds the stated 20% relative target. However, it is above the fixed 0.10 kW absolute target; only seed 22 meets all four daily ≤5 kWh, peak ≤1 kW, ramp ≤0.10 kW and ACF ≤0.10 macro gates. All-condition details remain in the full report. No favorable seed is substituted for the five-seed result.

| Seed | Daily W1 (kWh) | Peak W1 (kW) | Ramp W1 (kW) | Lag-1 ACF W1 | Ramp/Gaussian | Ramp/resampling | All four gates |
|---|---:|---:|---:|---:|---:|---:|---|
| 11 | 3.165317 | 0.776583 | 0.102262 | 0.081519 | 0.7251 | 2.9928 | Fail |
| 22 | 3.358924 | 0.794238 | 0.090725 | 0.092426 | 0.6433 | 2.6552 | Pass |
| 33 | 3.221380 | 0.788520 | 0.101320 | 0.086611 | 0.7184 | 2.9653 | Fail |
| 44 | 3.321075 | 0.763306 | 0.107070 | 0.075773 | 0.7592 | 3.1336 | Fail |
| 55 | 3.310058 | 0.793808 | 0.101900 | 0.080169 | 0.7225 | 2.9823 | Fail |

Daily-energy distance is worse than the Gaussian control in all five point estimates; peak and ramp distances improve. ACF has no stable advantage across the separate precision check. The [robustness report](CURATED_CALIBRATION_ROBUSTNESS.md) preserves the cap-1,000 and calibration-reference sensitivity results without replacing this primary cap-100 analysis. The [three-seed SWD attribution study](CURATED_SWD_ATTRIBUTION.md) addresses weight 0 versus 10 separately; this five-seed summary does not turn that comparison into a five-seed attribution study.

## Secondary joint and battery diagnostics

Both diagnostics have complete five-seed coverage in the full source report. With the same 128 unit-normalized projection directions (seed 9182), condition-macro projected-profile W1 is **0.06610 ± 0.00093 kWh** for calibrated cGAN (mean ± sample SD), versus 0.06345 for Gaussian copula and 0.06247 for resampling. Thus the primary ramp improvement does not establish a whole-profile advantage. The frozen controls remain single repeated references.

The hypothetical additional battery is 10 kWh / 4 kW with 90% round-trip efficiency, charged only from observed net export. Each 24-hour episode starts empty; leftover SOC is reported without future-day credit. For calibrated cGAN / Gaussian copula / resampling, five-seed mean condition-macro W1 values are **2.841 / 2.501 / 2.374 kWh** for post-battery grid import, **0.280 / 0.231 / 0.185 kWh** for grid-import reduction, and **0.764 / 0.979 / 0.489 kW** for post-battery peak grid import. These mixed distribution-fidelity results are not realized savings, PV attribution, annual operation or hardware validation; gross generation and existing batteries are unknown. Full five-seed post-export and end-SOC diagnostics, all means/SDs and per-condition values are retained under `groups[0].battery_diagnostics` and `groups[0].joint_diagnostics` in the hash-bound full report. These secondary details are omitted from the compact JSON and introduce no additional selection rule or acceptance gate.

## Downstream utility

The fixed task predicts the next 15-minute signed net-energy interval from 12 observed within-day intervals (3 hours), clock terms and condition indicators using the same linear ridge estimator (alpha 1). There are 84 prediction windows per daily profile. Each fitted method uses all 606,060 actual TRAIN-profile equivalents with matched labels (50,909,040 windows); evaluation uses the same 6,400 real validation profiles (537,600 windows). Macro RMSE/MAE is the equal-weight mean of the 64 condition-specific scores. This is observed-history prediction, not an annual or day-ahead forecasting evaluation.

| Training method / predictor | Macro RMSE (kWh/interval) | Macro MAE (kWh/interval) |
|---|---:|---:|
| Real training | 0.144882 | 0.072645 |
| Raw cGAN training | 0.145712 ± 0.000414 | 0.074312 ± 0.000633 |
| Calibrated cGAN training | 0.145489 ± 0.000289 | 0.073472 ± 0.000547 |
| Resampling training | 0.144881 | 0.072652 |
| Gaussian-copula training | 0.147528 | 0.076745 |
| Persistence | 0.150242 | 0.068753 |

| Seed | Calibrated RMSE | RMSE / real training | RMSE / resampling | MAE / real training | MAE / resampling | RMSE ≤1.10 both references |
|---|---:|---:|---:|---:|---:|---|
| 11 | 0.145298 | 1.002868 | 1.002878 | 1.013995 | 1.013904 | Pass |
| 22 | 0.145161 | 1.001926 | 1.001936 | 0.999563 | 0.999473 | Pass |
| 33 | 0.145457 | 1.003966 | 1.003976 | 1.010030 | 1.009940 | Pass |
| 44 | 0.145900 | 1.007021 | 1.007031 | 1.020011 | 1.019919 | Pass |
| 55 | 0.145629 | 1.005152 | 1.005162 | 1.013299 | 1.013208 | Pass |

All five calibrated RMSEs are within the stated 10% descriptive tolerance, but are slightly higher than both real-trained and resampling-trained references. MAE ratios are additional descriptions, not a second prespecified tolerance gate. The physical bootstrap below supplies **no utility confidence intervals**. Formal utility noninferiority, broader tasks and future-period transfer remain unestablished.

## Paired physical intervals, retained separately by seed

Each entry below is the candidate-minus-comparator W1 difference followed by its own [lower, upper] 95% percentile interval. Negative favors calibrated cGAN. The bootstrap samples whole real meters within the eight technology strata and recomputes condition-macro metrics; generated and comparator arrays remain frozen. All 10 comparisons have 500/500 valid draws, zero missing-condition draws, 353 observed meters and all 64 conditions. No intervals or endpoints are averaged.

These are descriptive marginal percentile intervals. Nominal coverage is not guaranteed, particularly near coincident CDFs where W1 is nonsmooth; their signs are not a multiple-testing-adjusted significance claim. The scope is real-meter sampling conditional on frozen models and reference arrays. It excludes training-seed, generated-sample, comparator-sample, preprocessing-fit, model-selection and future-period uncertainty.

### Compared with gaussian copula

| Seed | Pointwise Δ (kWh/interval) | Daily Δ (kWh) | Peak Δ (kW) | Ramp Δ (kW) | Lag-1 ACF Δ |
|---|---:|---:|---:|---:|---:|
| 11 | +0.00038 [-0.00116, +0.00130] | +0.22681 [-0.03955, +0.39935] | -0.21771 [-0.33553, -0.08523] | -0.03878 [-0.04305, -0.03078] | -0.00385 [-0.01441, +0.00659] |
| 22 | -0.00008 [-0.00188, +0.00104] | +0.42042 [+0.08043, +0.62621] | -0.20006 [-0.34410, -0.06452] | -0.05031 [-0.05726, -0.04074] | +0.00706 [-0.00602, +0.01840] |
| 33 | +0.00069 [-0.00080, +0.00169] | +0.28288 [-0.00877, +0.46691] | -0.20578 [-0.33448, -0.07519] | -0.03972 [-0.04443, -0.03135] | +0.00125 [-0.01062, +0.01252] |
| 44 | +0.00224 [+0.00072, +0.00313] | +0.38257 [+0.09030, +0.54377] | -0.23099 [-0.34599, -0.09332] | -0.03397 [-0.03738, -0.02668] | -0.00959 [-0.01995, +0.00143] |
| 55 | +0.00170 [-0.00058, +0.00280] | +0.37155 [+0.02521, +0.56636] | -0.20049 [-0.33141, -0.07224] | -0.03914 [-0.04393, -0.03149] | -0.00520 [-0.01551, +0.00513] |

### Compared with conditional resampling

| Seed | Pointwise Δ (kWh/interval) | Daily Δ (kWh) | Peak Δ (kW) | Ramp Δ (kW) | Lag-1 ACF Δ |
|---|---:|---:|---:|---:|---:|
| 11 | -0.00064 [-0.00239, +0.00035] | +0.33962 [+0.03085, +0.52116] | +0.27551 [+0.16824, +0.31364] | +0.06809 [+0.05568, +0.07013] | +0.03505 [+0.02158, +0.03594] |
| 22 | -0.00111 [-0.00324, +0.00023] | +0.53323 [+0.15210, +0.73741] | +0.29316 [+0.16913, +0.33142] | +0.05656 [+0.04470, +0.05730] | +0.04596 [+0.03095, +0.04727] |
| 33 | -0.00033 [-0.00207, +0.00080] | +0.39569 [+0.08002, +0.61119] | +0.28744 [+0.17389, +0.32301] | +0.06715 [+0.05487, +0.06955] | +0.04014 [+0.02622, +0.04149] |
| 44 | +0.00122 [-0.00043, +0.00213] | +0.49538 [+0.16923, +0.65821] | +0.26223 [+0.16015, +0.30949] | +0.07290 [+0.05978, +0.07525] | +0.02930 [+0.01631, +0.03038] |
| 55 | +0.00067 [-0.00187, +0.00179] | +0.48436 [+0.11587, +0.67844] | +0.29273 [+0.17234, +0.32545] | +0.06773 [+0.05512, +0.06971] | +0.03370 [+0.02009, +0.03490] |

Against the Gaussian control, peak and ramp intervals lie below zero in every seed; daily-energy intervals lie above zero in three seeds and span zero in two. All five ACF intervals span zero. Against resampling, **all five daily-energy, peak, ramp and ACF intervals lie above zero**; all pointwise intervals span zero. This negative resampling result is part of the completed evidence, not an omitted secondary comparison.

## Coverage, provenance and limits

The fixed seed-11 figure run is complete: 20 SVG/PNG files cover quantile bands,
raw/calibrated heatmaps and unpaired examples for all eight technology groups.
Its [verification record](../reports/original_figures_verification.json) retains
every file hash and the full manifest's hash. The complete files are under
`runs/curated_swd10/seed_11/figures_verified`; the earlier interrupted plotting
directory remains separate. These figures illustrate validation behavior and
do not replace the five-seed quantitative comparisons.

The [compact JSON](../reports/curated_completed_validation.json) retains every run’s 10 input artifact hashes, all source-code hash variants and the compact source/input SHA-256 list (excluding per-meter identity fingerprints), common protocol settings, per-seed physical and utility macros/ratios, aggregate metrics, every paired interval endpoint and complete coverage counts. Exact per-condition details, source histories and metadata associations remain in the full collector report.

Full report: `runs/curated_swd10/completed_validation_study.json` (15,744,128 bytes). SHA-256: `c5e773c3e32f985b96a6b2fed7ab1da1b351572710a432a41f8851b7fafd5fd0`. Its companion Markdown is `runs/curated_swd10/completed_validation_study.md`. Collector SHA-256: `6ad46ce7f8bb2dd35018555fc76c338f10bf8d6cab3fad57cd2c6cfceef965c8`.

Reproduce collection without prepared-profile reads:

```bash
python scripts/collect_calibrated_study.py \
  --run seed11=runs/curated_swd10/seed_11 \
  --run seed22=runs/curated_swd10/seed_22 \
  --run seed33=runs/curated_swd10/seed_33 \
  --run seed44=runs/curated_swd10/seed_44 \
  --run seed55=runs/curated_swd10/seed_55 \
  --output runs/curated_swd10/completed_validation_study.json \
  --markdown runs/curated_swd10/completed_validation_study.md
```

Validation supported development and checkpoint selection. No test performance is reported. Exact-trace curation removes the declared full-trace overlaps but does not prove household independence or exclude near-duplicates. Calibration directly targets training marginal distributions, so marginal improvement is not independent evidence that joint behavior is correct. Further utility uncertainty and ablations remain separate work.
