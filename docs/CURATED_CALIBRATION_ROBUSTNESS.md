# Curated calibration robustness

The calibrated cGAN retains its ramp and peak advantages over the Gaussian copula at both validation sample sizes. Daily-energy and projected joint errors remain higher; the small ACF advantage in the primary sample reverses in the larger sample. Conditional resampling remains a strong reference. These results support a specific temporal tradeoff, not general superiority or paper readiness.

## Frozen settings and verification

All five declared training seeds (11, 22, 33, 44, 55) are included. Each checkpoint is unchanged. The primary reports use 100 profiles per condition; the separate precision reports use 1,000. Calibration uses 4,096 reference draws per condition, seed 7301, and 1,025 target/source knots; evaluation seed is 2026. Each sample includes all 64 conditions and all 353 validation meters: 6,400 primary profiles or 64,000 precision profiles. Conditions receive equal macro weight.

The report-only collector verified matching checkpoint, prepared-data, exact training-row, calibration-reference, source-code and diagnostic-setting bindings. Within each cap, selected real rows and statistical-control profiles and metrics match across the five training seeds. Precision runs reproduce each primary calibration provenance and fitting diagnostics. The seed-11 sensitivities preserve the real evaluation sample, raw cGAN draws, all four controls and their diagnostics. No mismatch or missing report was accepted. No dataset or checkpoint was loaded by this collection.

## Primary result: 100 profiles per condition

Values are condition-macro mean ± sample SD across the five training seeds. Zero control SD reflects repeated identical controls, not independent replication. W1 is lower-is-better; diversity is a variance ratio, with 1 as the descriptive reference.

| Method | Point W1 (kWh) | Daily W1 (kWh) | Peak W1 (kW) | Ramp W1 (kW) | Lag-1 ACF W1 | Diversity ratio |
|---|---:|---:|---:|---:|---:|---:|
| Raw cGAN | 0.09038 ± 0.00136 | 2.99043 ± 0.04363 | 0.73261 ± 0.02157 | 0.20423 ± 0.01257 | 0.09406 ± 0.00720 | 0.99297 ± 0.03832 |
| Calibrated cGAN | 0.05461 ± 0.00096 | 3.27535 ± 0.07953 | 0.78329 ± 0.01325 | 0.10066 ± 0.00601 | 0.08330 ± 0.00640 | 1.12876 ± 0.06687 |
| Gaussian copula | 0.05363 ± 0.00000 | 2.93850 ± 0.00000 | 0.99430 ± 0.00000 | 0.14104 ± 0.00000 | 0.08536 ± 0.00000 | 1.08815 ± 0.00000 |
| Independent margins | 0.05397 ± 0.00000 | 8.04477 ± 0.00000 | 3.42082 ± 0.00000 | 0.70573 ± 0.00000 | 0.59999 ± 0.00000 | 1.12633 ± 0.00000 |
| Exact-marginal copula | 0.05343 ± 0.00000 | 2.92722 ± 0.00000 | 0.97232 ± 0.00000 | 0.14013 ± 0.00000 | 0.08503 ± 0.00000 | 1.07452 ± 0.00000 |
| Conditional resampling | 0.05465 ± 0.00000 | 2.82569 ± 0.00000 | 0.50108 ± 0.00000 | 0.03417 ± 0.00000 | 0.04647 ± 0.00000 | 1.15849 ± 0.00000 |

## Separate precision check: 1,000 profiles per condition

These reports use larger empirical samples under the same frozen model and calibration settings. They do not replace the primary result, and lower W1 at the larger cap must not be described as a training improvement.

| Method | Point W1 (kWh) | Daily W1 (kWh) | Peak W1 (kW) | Ramp W1 (kW) | Lag-1 ACF W1 | Diversity ratio |
|---|---:|---:|---:|---:|---:|---:|
| Raw cGAN | 0.07666 ± 0.00095 | 1.97144 ± 0.04466 | 0.65517 ± 0.01823 | 0.20396 ± 0.01138 | 0.09311 ± 0.00798 | 1.00506 ± 0.01762 |
| Calibrated cGAN | 0.03218 ± 0.00014 | 2.35043 ± 0.05617 | 0.68160 ± 0.00964 | 0.09784 ± 0.00594 | 0.08170 ± 0.00798 | 1.14352 ± 0.00830 |
| Gaussian copula | 0.03167 ± 0.00000 | 2.04501 ± 0.00000 | 0.92506 ± 0.00000 | 0.14165 ± 0.00000 | 0.07791 ± 0.00000 | 1.11719 ± 0.00000 |
| Independent margins | 0.03175 ± 0.00000 | 7.63500 ± 0.00000 | 3.36177 ± 0.00000 | 0.70324 ± 0.00000 | 0.59756 ± 0.00000 | 1.12599 ± 0.00000 |
| Exact-marginal copula | 0.03151 ± 0.00000 | 2.03308 ± 0.00000 | 0.90091 ± 0.00000 | 0.14078 ± 0.00000 | 0.07759 ± 0.00000 | 1.10300 ± 0.00000 |
| Conditional resampling | 0.03140 ± 0.00000 | 1.85189 ± 0.00000 | 0.32610 ± 0.00000 | 0.02376 ± 0.00000 | 0.02695 ± 0.00000 | 1.12268 ± 0.00000 |

Ramp W1 is **28.63% lower** than the Gaussian copula in the primary reports and **30.92% lower** in the precision reports, with the same direction in all five training seeds. The across-seed mean is lower in 52/64 and 53/64 conditions, respectively. Against resampling, it is lower in only 1/64 primary conditions and 0/64 precision conditions.

The fixed project ramp gate is 0.10 kW. The primary mean (0.100655 kW) remains above it; only seed 22 meets all four descriptive daily/peak/ramp/ACF gates. The precision mean (0.097844 kW) falls below it, with seeds 11, 22 and 55 meeting all four. This sample-sensitive gate outcome is disclosed without selecting the favorable cap or seeds.

Daily-energy W1 remains worse than the Gaussian copula at both caps (3.275 vs 2.939 kWh; 2.350 vs 2.045 kWh), despite better peaks and ramps. ACF W1 changes from 0.08330 vs 0.08536 in the primary reports to 0.08170 vs 0.07791 in the precision reports. The data therefore do not support a stable ACF advantage.

## Secondary joint and battery diagnostics

The fixed joint check averages 128 unit-direction W1 values (seed 9182). Battery columns are W1 for daily outcomes under the fixed 10 kWh, 4 kW additional export-charged battery with charge/discharge efficiency sqrt(0.9), empty initial SOC each day, and no future-day credit. Battery peak is nonnegative grid-import power. These are descriptive diagnostics; neither test is a full joint-distribution certificate.

Cap **100**; across-seed means:

| Method | Projected W1 (kWh) | Post import (kWh) | Post export (kWh) | Import reduction (kWh) | Post peak (kW) | End SOC (kWh) |
|---|---:|---:|---:|---:|---:|---:|
| Raw cGAN | 0.06581 | 2.55360 | 0.33461 | 1.66571 | 0.72507 | 0.32232 |
| Calibrated cGAN | 0.06610 | 2.84116 | 0.29348 | 0.27970 | 0.76409 | 0.23407 |
| Gaussian copula | 0.06345 | 2.50068 | 0.24630 | 0.23146 | 0.97915 | 0.26004 |
| Independent margins | 0.09323 | 6.67839 | 0.78257 | 1.04821 | 3.14849 | 0.70973 |
| Exact-marginal copula | 0.06320 | 2.49097 | 0.24605 | 0.23136 | 0.95708 | 0.25990 |
| Conditional resampling | 0.06247 | 2.37382 | 0.30360 | 0.18527 | 0.48902 | 0.19832 |

Cap **1000**; across-seed means:

| Method | Projected W1 (kWh) | Post import (kWh) | Post export (kWh) | Import reduction (kWh) | Post peak (kW) | End SOC (kWh) |
|---|---:|---:|---:|---:|---:|---:|
| Raw cGAN | 0.04348 | 1.70611 | 0.24471 | 1.65946 | 0.65355 | 0.24642 |
| Calibrated cGAN | 0.04269 | 2.04640 | 0.19484 | 0.21390 | 0.65943 | 0.15298 |
| Gaussian copula | 0.03873 | 1.77969 | 0.17036 | 0.15130 | 0.91165 | 0.16329 |
| Independent margins | 0.07885 | 6.27318 | 0.79847 | 1.01997 | 3.09721 | 0.70153 |
| Exact-marginal copula | 0.03852 | 1.76891 | 0.16979 | 0.15107 | 0.88781 | 0.16299 |
| Conditional resampling | 0.03505 | 1.59982 | 0.19994 | 0.08227 | 0.32011 | 0.11704 |

The calibrated pipeline has higher projected W1 than the Gaussian copula at both caps. Its battery post-import, post-export and import-reduction errors are also higher; battery peak and end-SOC errors are lower. Relative to raw cGAN, calibration greatly reduces the battery import-reduction error (1.666 to 0.280 kWh primary; 1.659 to 0.214 kWh precision), while increasing post-import error. These mixed outcomes do not establish an overall application advantage.

Mean behavior is retained in JSON. At the larger cap, calibrated macro mean daily net energy is 11.6026 kWh versus 11.4917 real; mean signed-net peak power is 2.8939 kW versus 3.0667 real. Its diversity ratio is 1.1435. Calibration improves marginal alignment without exactly matching validation means or variances.

## Prespecified seed-11 calibration sensitivities

Both sensitivities retain the primary 100-profile-per-condition evaluation draws. The following are calibrated-cGAN changes from the original seed-11 fit; they are not additional GAN training replications or confidence intervals. No sensitivity fit replaces the primary map.

| Calibration change | Point W1 (kWh) | Daily W1 (kWh) | Peak W1 (kW) | Ramp W1 (kW) | ACF W1 | Diversity ratio |
|---|---:|---:|---:|---:|---:|---:|
| 4,096 reference draws, seed 7302 | -0.000231 | -0.048901 | -0.014886 | -0.000085 | +0.000083 | +0.002974 |
| 8,192 reference draws, seed 7301 | -0.000207 | +0.011537 | -0.012761 | -0.000632 | +0.000223 | -0.021112 |

These two perturbations change ramp W1 by less than 0.00064 kW and projected W1 by less than 0.00052 kWh. Daily-energy changes are −0.04890 and +0.01154 kWh. The larger reference lowers diversity ratio by 0.02111. This is limited descriptive stability evidence for one fixed checkpoint, not a general guarantee against calibration uncertainty.

## Provenance and limits

`reports/curated_calibration_robustness.json` retains both caps, all 12 input report hashes, the full-report hash, exact scenario/settings, implementation hashes, all five raw/calibrated run macros, all-method aggregate macros, sensitivity deltas, and ordered ramp-condition contrasts. Numeric vectors follow its explicit metric order without rounding. Complete source histories, calibration provenance and per-run/per-condition metrics remain in `runs/curated_swd10/calibration_robustness.json`; its SHA-256 is `1bd661562ba16cf25eb1a820136dd1b2a30b4ecbcb9f3dcbe078bda9325f57fe`. The accompanying Markdown at that path is the unabridged collector table output.

Validation was reused for method development and checkpoint selection. Training-seed SD is not a confidence interval; no intervals are pooled or averaged. The larger evaluation sample is a precision check under the same data source, not independent confirmation. The battery is hypothetical and additional to any unobserved existing storage; no PV attribution, annual-operation, cost or hardware claim is made. No held-out test performance has been evaluated by these reports or this collection.
