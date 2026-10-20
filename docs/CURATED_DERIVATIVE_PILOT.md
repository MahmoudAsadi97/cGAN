# Derivative-distribution pilot: failed promotion screen

**The completed three-seed pilot failed and is not eligible for promotion.** Calibrated ramp W1 improves in all three paired seeds, but its mean reduction is only **8.79%**, below the required 10%. Mean lag-one ACF W1 worsens by **15.56%**, beyond the allowed 10%. Passing the remaining gates and crossing the older 0.10 kW ramp target does not override either failure.

This is a validation-only exploratory follow-up, declared after inspecting the original SWD10 validation results. The original arm and the failed pilot remain in the record; no TEST performance is used. The [fixed protocol](TEMPORAL_REFINEMENT.md) determines the outcome.

## Exact promotion gates

Candidate means below use all matched seeds 11, 22 and 33 and equal weight for the 64 conditions. Limits are fixed ratios of the same three baseline seeds, not the five-seed baseline mean.

| Gate | Baseline | Candidate | Fixed limit | Result |
|---|---:|---:|---:|---|
| Ramp lower in every paired seed | — | 3/3 lower | Strictly lower in all 3 | Pass |
| Mean ramp W1 (kW) | 0.098102228 | 0.089475790 | ≤0.088292005 (0.90×) | **Fail** |
| Mean daily-energy W1 (kWh) | 3.248540262 | 3.291910086 | ≤3.573394288 (1.10×) | Pass |
| Mean peak W1 (kW) | 0.786446823 | 0.784350475 | ≤0.865091505 (1.10×) | Pass |
| Mean lag-1 ACF W1 | 0.086851975 | 0.100362376 | ≤0.095537173 (1.10×) | **Fail** |
| Mean projected-profile W1 (kWh) | 0.065446587 | 0.065533910 | ≤0.071991246 (1.10×) | Pass |
| Utility macro RMSE in each seed | Real and resampling fits | 3/3 within both | ≤1.10× each reference | Pass |

The JSON retains the unrounded gate values and every paired ratio. The ramp threshold is 0.08829200540531076 kW; the observed mean is 0.08947579044546178 kW. The ACF threshold is 0.09553717294842969; the observed mean is 0.10036237647146432.

| Seed | Ramp baseline → candidate (kW) | Relative ramp change | ACF baseline → candidate | Selected epoch baseline → candidate |
|---|---:|---:|---:|---:|
| 11 | 0.102262 → 0.092309 | -9.73% | 0.081519 → 0.100628 | 25 → 24 |
| 22 | 0.090725 → 0.086733 | -4.40% | 0.092426 → 0.102041 | 25 → 22 |
| 33 | 0.101320 → 0.089385 | -11.78% | 0.086611 → 0.098419 | 24 → 24 |

## Physical, joint and battery tradeoffs

The following are calibrated condition-macro values, mean ± sample SD across three seeds. Sample SD is not a confidence interval. Distances are lower-is-better; diversity near 1 is descriptive.

| Metric | Baseline | Candidate |
|---|---:|---:|
| Pointwise W1 (kWh/interval) | 0.053956 ± 0.000388 | 0.054028 ± 0.000686 |
| Daily-energy W1 (kWh) | 3.248540 ± 0.099620 | 3.291910 ± 0.079445 |
| Peak W1 (kW) | 0.786447 ± 0.009008 | 0.784350 ± 0.017244 |
| Ramp W1 (kW) | 0.098102 ± 0.006406 | 0.089476 ± 0.002789 |
| Lag-1 ACF W1 | 0.086852 ± 0.005458 | 0.100362 ± 0.001826 |
| Diversity ratio | 1.092629 ± 0.029930 | 1.119547 ± 0.016517 |
| Projected-profile W1 (kWh) | 0.065447 ± 0.000349 | 0.065534 ± 0.000328 |
| Post-battery grid-import W1 (kWh) | 2.805069 ± 0.117331 | 2.851171 ± 0.110512 |
| Post-battery grid-export W1 (kWh) | 0.284809 ± 0.037186 | 0.292475 ± 0.033336 |
| Grid-import-reduction W1 (kWh) | 0.287906 ± 0.008192 | 0.282557 ± 0.016636 |
| Post-battery peak-import W1 (kW) | 0.767681 ± 0.007172 | 0.765180 ± 0.016561 |
| End-SOC W1 (kWh) | 0.240333 ± 0.023153 | 0.225963 ± 0.042544 |

Ramps improve while daily-energy and ACF errors increase. The projected-profile mean changes little; finite projections do not certify the complete joint distribution. Frozen Gaussian-copula, independent-marginal, exact-marginal-copula and resampling references match across all six reports and are retained once in the compact JSON, not counted as independent seed repetitions. Raw-generator diagnostics are also retained there.

Battery outcomes are secondary and are not promotion gates. The hypothetical additional device is 10 kWh / 4 kW with 90% round-trip efficiency, charged only from observed net export. Each day starts empty, and leftover SOC receives no future-day credit. These are distribution distances, not realized savings or annual operation; gross generation and any pre-existing batteries are unknown.

## Utility

All methods use the full 606,060-row actual TRAIN pool as their reference and matched real/generated profile and label counts. Each fitted ridge predictor uses 50,909,040 windows; the same 6,400 validation profiles supply 537,600 windows. The task predicts the next 15-minute interval from 12 observed intervals, clock terms and condition indicators, with alpha 1 and 81 features. There is no cap on the downstream training pool.

| Seed | Calibrated RMSE baseline → candidate | Candidate / real-trained RMSE | Candidate / resampling-trained RMSE | Calibrated MAE baseline → candidate |
|---|---:|---:|---:|---:|
| 11 | 0.145298 → 0.145065 | 1.001258 | 1.001268 | 0.073662 → 0.072909 |
| 22 | 0.145161 → 0.145038 | 1.001072 | 1.001082 | 0.072613 → 0.072625 |
| 33 | 0.145457 → 0.145254 | 1.002564 | 1.002574 | 0.073374 → 0.072969 |

Candidate mean calibrated RMSE is 0.145118617 kWh/interval, versus real-trained 0.144882250 and resampling-trained 0.144880840. It improves against the paired original calibrated model in all three seeds, but remains slightly above both fixed controls. Candidate MAE is slightly worse than baseline for seed 22. The narrow descriptive RMSE tolerance is met; no paired pilot utility or physical uncertainty intervals are available, and no significance or noninferiority claim follows.

## Implementation, budgets and provenance

The candidate adds `derivative_swd_weight=1.0` to profile SW2-squared weight 10. It matches distributions of the 95 adjacent within-day differences of globally TRAIN-standardized profiles, divided by one TRAIN-only scalar. That scalar is 0.4571263982135081 in float64 and 0.4571264088153839 in float32 training. It is fitted on all 606,060 actual training rows with equal condition weighting, without centering or midnight wraparound. The generator objective is adversarial +10×profile SW2-squared +1×difference SW2-squared; the new term has its own seed+5 projection RNG.

Architecture, balanced batches, optimizers, TRAIN rows, scaler, calibration and checkpoint-selection rules remain matched. All completed runs have 26 epochs, 30,784 generator and discriminator updates each, and 15,761,408 sampled profiles. Per-epoch class exposures match within paired seeds. Selected checkpoint epochs differ, so selected-prefix budgets are recorded separately. Training source hashes differ to implement the new term; physical/joint/battery evaluation and utility core source hashes match. No claim of byte-identical training implementation is made.

Calibration remains checkpoint-specific, using 4,096 separate generated reference draws per condition (seed 7301) against the actual TRAIN marginals and 1,025 knots. Support/extrapolation event counts, fractions and fitting tie summaries for every run remain in the compact JSON; full per-slot/per-condition tie and support arrays remain in the original report. These diagnostics do not create replacement success criteria.

The [compact report](../reports/curated_derivative_pilot.json) records exact gates, configs, all input artifact hashes, source fingerprints, budgets, raw/calibrated macro values, frozen controls and all 64 calibrated condition means across physical/joint/battery/utility metrics. It also retains every seed’s 64 ramp and ACF differences. Full source: `runs/temporal_pilot.json` (22,574,906 bytes), SHA-256 `1b7748e5f3c7f78ad2a7b3b27e91f3b8539229994fb9d48ca2ec7ac7d27b1a9c`; companion `runs/temporal_pilot.md`. The original files remain unchanged.

Independent report-only checks recomputed all promotion decisions from the recorded seed macros, checked existing JSON input hashes and checkpoint/map cross-report bindings, verified full TRAIN and validation hashes, matched completed budgets and class exposures, and confirmed repeated control metrics/array fingerprints. No data arrays, model/map archives, fitting, generation or TEST access were needed. Recorded hashes establish consistency of the evidence chain, not proof that execution was correct.

The failed arm is retained as exploratory evidence. The result does not justify expansion to seeds 44/55 under this protocol. Any later capacity or architecture study requires its own declared comparison and cannot erase this failed screen.
