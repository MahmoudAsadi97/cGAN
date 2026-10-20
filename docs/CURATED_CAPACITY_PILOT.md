# Capacity pilot: failed promotion screen

**The complete three-seed capacity package failed both fixed ramp criteria and is not eligible for promotion.** Mean calibrated ramp W1 is **32.34% higher** than the matching original SWD10 mean, and ramp error is higher in every paired seed. Seeds **44 and 55 are not added under this protocol**. Passing the other physical and utility gates does not override either failure.

The comparison uses all fixed seeds 11, 22 and 33 against the original curated SWD10 arm. It is a validation-only exploratory refinement after the failed derivative pilot, with no TEST performance access. The [frozen capacity protocol](CAPACITY_REFINEMENT.md) determines the decision.

| Promotion gate | Original mean | Capacity mean | Fixed limit | Outcome |
|---|---:|---:|---:|---|
| Ramp W1 (kW) | 0.098102228 | 0.129828606 | ≤0.088292005 | **Fail** |
| Daily-energy W1 (kWh) | 3.248540262 | 3.034799207 | ≤3.573394288 | Pass |
| Peak W1 (kW) | 0.786446823 | 0.770612288 | ≤0.865091505 | Pass |
| Lag-one ACF W1 | 0.086851975 | 0.069717497 | ≤0.095537173 | Pass |
| Projected-profile W1 (kWh) | 0.065446587 | 0.064743182 | ≤0.071991246 | Pass |
| Ramp lower in every paired seed | — | 0/3 lower | Strictly lower in all 3 | **Fail** |
| Utility RMSE in every seed | Real/resampling references | 3/3 within both | ≤1.10× each | Pass |

| Seed | Original ramp → capacity (kW) | Ramp change | Selected epoch original → capacity |
|---|---:|---:|---:|
| 11 | 0.102262 → 0.135061 | +32.07% | 25 → 26 |
| 22 | 0.090725 → 0.125690 | +38.54% | 25 → 26 |
| 33 | 0.101320 → 0.128735 | +27.06% | 24 → 25 |

The exact mean ramp limit is 0.08829200540531076 kW; capacity produces 0.12982860632983112 kW. The complete failed package remains in the record. This does not establish that capacity cannot help under another prospectively declared package or budget.

Calibrated physical, projected and battery macros below are mean ± sample SD across three seeds. Distances are lower-is-better; diversity near 1 is descriptive. SD is not a confidence interval. Raw and calibrated values, all battery means and every seed are retained in the [compact JSON](../reports/curated_capacity_pilot.json).

| Metric | Original SWD10 | Capacity |
|---|---:|---:|
| Pointwise W1 (kWh/interval) | 0.053956 ± 0.000388 | 0.054402 ± 0.000948 |
| Ramp W1 (kW) | 0.098102 ± 0.006406 | 0.129829 ± 0.004780 |
| Daily-energy W1 (kWh) | 3.248540 ± 0.099620 | 3.034799 ± 0.132142 |
| Peak W1 (kW) | 0.786447 ± 0.009008 | 0.770612 ± 0.028885 |
| Lag-one ACF W1 | 0.086852 ± 0.005458 | 0.069717 ± 0.002517 |
| Projected-profile W1 (kWh) | 0.065447 ± 0.000349 | 0.064743 ± 0.001221 |
| Diversity ratio | 1.092629 ± 0.029930 | 1.153237 ± 0.012429 |
| Post-battery import W1 (kWh) | 2.805069 ± 0.117331 | 2.623943 ± 0.118864 |
| Post-battery export W1 (kWh) | 0.284809 ± 0.037186 | 0.276432 ± 0.005659 |
| Import-reduction W1 (kWh) | 0.287906 ± 0.008192 | 0.267310 ± 0.014184 |
| Post-battery peak W1 (kW) | 0.767681 ± 0.007172 | 0.757422 ± 0.029327 |
| End-SOC W1 (kWh) | 0.240333 ± 0.023153 | 0.228467 ± 0.029702 |

Daily-energy, peak, ACF and projected-profile means improve while calibrated ramp error worsens. Finite projected distances do not certify the full joint distribution. Marginal/diversity behavior is mechanically affected by the fixed map and is not independent evidence that the generator learned those properties. The shared Gaussian, independent-marginal, exact-marginal and resampling controls remain in the JSON once, without counting identical controls as independent seeds.

The battery diagnostic is secondary: an additional 10 kWh / 4 kW device with 90% round-trip efficiency, export-only charging, import-only discharge and zero starting SOC each day. It does not determine promotion or establish annual savings; gross generation and pre-existing batteries are unknown.

| Seed | Calibrated RMSE original → capacity | Capacity / real RMSE | Capacity / resampling RMSE | Capacity MAE |
|---|---:|---:|---:|---:|
| 11 | 0.145298 → 0.145588 | 1.004872 | 1.004882 | 0.074054 |
| 22 | 0.145161 → 0.145346 | 1.003200 | 1.003210 | 0.073761 |
| 33 | 0.145457 → 0.145551 | 1.004613 | 1.004622 | 0.073909 |

Capacity mean calibrated RMSE is 0.145494861 kWh/interval, compared with real-trained 0.144882250 and resampling-trained 0.144880840. Every seed meets both descriptive 10% tolerances, but every capacity seed has higher RMSE than its original calibrated counterpart. Each ridge fit uses all 606,060 TRAIN/generated profiles and 50,909,040 within-day windows; the same 6,400 validation profiles supply 537,600 windows. The fixed task predicts the next 15-minute value from 12 observed lags, clock terms and condition indicators, with alpha 1. No pilot paired confidence intervals or formal noninferiority claim are made.

Capacity changes generator hidden width 64→128 and latent dimension 24→96 together; discriminator width 128, profile SW2-squared weight 10 and derivative weight 0 remain fixed. Generator parameters increase from 28,512 to 98,912. The package cannot attribute an outcome separately to width or latent dimension, and matching seed labels do not imply identical initialization or latent noise.

All six runs completed 26 epochs, 30,784 updates per adversary and 15,761,408 sampled profiles, using the same 606,060-row actual TRAIN pool. Checkpoint selection remains the earliest minimum of raw condition-macro interval W1 on the fixed 4,096 validation rows. Selected epochs differ as shown above; prefix budgets and elapsed times are retained. Equal updates do not mean equal compute. Calibration uses checkpoint-specific 4,096-per-condition independent reference draws, seed 7301, 1,025 knots and production seed 2026. All 64 condition results, ties and endpoint diagnostics remain referenced by exact source-report hashes.

The capacity numerical execution uses clean commit `1ca62d0fbe1d77f916f8904e812db7d720f8cf53`; evaluation/utility core bytes and frozen statistical controls match the original comparison. Training source differences are retained. The matched old interpolation implementation precedes the later float64 correction; the later frozen-TEST source does not retrospectively change these validation results.

Full assessment: `runs/capacity_pilot.json`, SHA256 `477d62a38e6b536a8d844388abf81e768cb8da0f4013d66389e4970072742c29`. The compact JSON binds every input report, checkpoint/map recorded digest, full budget, source record and location of all 64 condition results. Arrays, checkpoint/map archives and TEST were not opened for this summary.

The [original independent review](../reports/capacity_pilot_independent_review.json) recomputed all 64 condition macros and the gates. It passed 1,468 checks and flagged one source-location assumption: it compared the assessor collector against later main. The [append-only correction](../reports/capacity_pilot_independent_review_correction.json) verifies actual collector, assessor and protocol files against clean declared assessor commit `64a2e9aee98b0e3552ad3775df99f9cee35b2f8c` and its Git blobs. The first receipt is preserved; there is no unresolved discrepancy or changed scientific decision.

These are repeated-validation, identity-curated internal source/year results. Exact trace deduplication does not establish independent households or external validity. No significance, novel-method, universal capacity-failure or TEST-confirmation claim follows. The failed arm and both ramp failures remain visible; the fixed capacity protocol does not support adding 44/55.
