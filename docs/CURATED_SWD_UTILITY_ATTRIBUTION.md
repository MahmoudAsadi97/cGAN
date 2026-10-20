# Matched SWD utility attribution

The completed utility reports show lower macro RMSE and MAE with SWD weight 10 than weight 0 in **all three matched seeds (11, 22, 33)**, both before and after calibration. This is descriptive three-seed evidence under the fixed balanced hinge-SN training and checkpoint-selection procedure. **No paired utility uncertainty intervals are available for this weight comparison.**

This separate report adds utility evidence to the [historical SWD attribution snapshot](CURATED_SWD_ATTRIBUTION.md), which remains unchanged. It does not expand the weight comparison to the five seeds in the [completed SWD10 validation study](CURATED_COMPLETED_VALIDATION.md).

## Matched results

Values are condition-macro errors in kWh/interval, averaged equally over 64 conditions. Entries are mean ± sample SD across three seeds; SD is not a confidence interval. Δ is SWD10 minus SWD0.

| Generator | Metric | SWD0 | SWD10 | Paired Δ | Lower in seeds |
|---|---|---:|---:|---:|---:|
| Raw cGAN | RMSE | 0.147855 ± 0.001027 | 0.145459 ± 0.000041 | -0.002396 ± 0.001051 | 3/3 |
| Raw cGAN | MAE | 0.076211 ± 0.001397 | 0.073990 ± 0.000462 | -0.002221 ± 0.001752 | 3/3 |
| Calibrated cGAN | RMSE | 0.146769 ± 0.000624 | 0.145305 ± 0.000148 | -0.001463 ± 0.000771 | 3/3 |
| Calibrated cGAN | MAE | 0.074876 ± 0.000596 | 0.073216 ± 0.000542 | -0.001660 ± 0.001082 | 3/3 |

| Seed | Raw RMSE 0 → 10 | Raw MAE 0 → 10 | Calibrated RMSE 0 → 10 | Calibrated MAE 0 → 10 |
|---|---:|---:|---:|---:|
| 11 | 0.147691 → 0.145424 | 0.076260 → 0.074359 | 0.146719 → 0.145298 | 0.074763 → 0.073662 |
| 22 | 0.148954 → 0.145448 | 0.077583 → 0.073472 | 0.147416 → 0.145161 | 0.075520 → 0.072613 |
| 33 | 0.146920 → 0.145505 | 0.074790 → 0.074139 | 0.146171 → 0.145457 | 0.074345 → 0.073374 |

The identical real-trained control is RMSE 0.144882 / MAE 0.072645; the identical resampling-trained control is 0.144881 / 0.072652. Their training-profile, label and fitted-coefficient hashes, as well as all per-condition and overall scores, match across all six reports. Both controls are single deterministic references, not six replications. The Gaussian-trained control and persistence scores also match exactly.

| Seed | Calibrated RMSE / real, SWD0 → SWD10 | Calibrated RMSE / resampling, SWD0 → SWD10 |
|---|---:|---:|
| 11 | 1.012675 → 1.002868 | 1.012685 → 1.002878 |
| 22 | 1.017491 → 1.001926 | 1.017500 → 1.001936 |
| 33 | 1.008895 → 1.003966 | 1.008905 → 1.003976 |

Both weights satisfy the previously stated descriptive RMSE ratio ≤1.10 against both controls, while remaining slightly worse than the controls. The tolerance therefore does not establish that SWD is necessary. MAE is reported as an additional error metric, not a separate prespecified gate. Per-condition changes and counts are retained in the compact JSON; conditions are not independent experimental replicates.

## Counts, budget and source bindings

Every utility evaluation uses the full actual TRAIN pool of 606,060 profiles from 1,665 retained meters as its reference. The real, raw, calibrated, resampling and Gaussian fits each receive 606,060 real or generated training profiles with identical label counts and 50,909,040 prediction windows. Validation is the same 6,400 profiles from 353 meters, 100 profiles per condition and 537,600 windows. There is no downstream training-pool cap. The task is one-step, 15-minute prediction from 12 observed past intervals (3 hours), clock terms and conditions, with the same 81-feature linear ridge model and alpha 1.

Only `swd_weight` differs between matched training configurations. All completed runs have 26 epochs, 30,784 generator and discriminator updates each, and 15,761,408 sampled profiles; per-epoch class exposures match within each seed. The selected epochs are SWD0 **25/26/25** versus SWD10 **25/25/24**, respectively. Thus full-run budgets match, while selected-checkpoint prefix budgets differ under the same condition-macro validation-selection rule. Elapsed time is recorded, not assumed equal.

The utility runner and all recorded utility source-file hashes match across all six reports, as do runtime versions, data/scaler hashes, actual TRAIN indices and frozen validation indices/profiles/labels/meter IDs. Each utility checkpoint and calibration-map hash matches the historical attribution ledger; complete checkpoint-training and calibration provenance fingerprints also match. The maps remain separate checkpoint-specific fits using 4,096 generated reference draws per condition against the same TRAIN marginals, seed 7301 and 1,025 quantile knots. No map was refitted for this summary.

## Provenance and limits

The [compact JSON](../reports/curated_swd_utility_attribution.json) records all six utility-report SHA-256 hashes, the earlier artifact bindings, common source hashes, budgets, per-seed values/ratios, paired summaries and all 64-condition error differences. Its historical source is `reports/curated_swd_attribution.json`, SHA-256 `30f346cf4b5475804c69b8756d3dbe7eaf19a862015ab40aaea4e0bda4827bb8`. Full per-condition source scores remain in each `runs/curated_swd0/seed_{11,22,33}/utility_validation.json` and corresponding `curated_swd10` report.

This report reads JSON only. Existing JSON artifact hashes were rechecked; checkpoint/map bindings rely on the previously verified historical ledger, without reopening model or map archives. No arrays were loaded, models fitted, profiles generated or metrics recomputed from data.

The result is limited to the fixed training package, checkpoint selection and this observed-history ridge task. Calibration is fitted separately for each checkpoint, so the calibrated contrast compares complete train-and-calibrate procedures. Validation was used for development; unseen-test behavior and broader application transfer remain unknown. No significance, formal noninferiority or overall acceptance claim follows from these three paired point estimates.
