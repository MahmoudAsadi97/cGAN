# Curated profile-SWD attribution: weight 10 versus 0

On matched seeds **11, 22 and 33**, profile SWD10 improves several raw-generator distribution distances substantially. After the same fixed marginal-calibration procedure, **peak and ramp distances improve in all three seeds**, and projected-profile W1 improves modestly in all three. The result is a trade-off: calibrated lag-1 ACF is worse on average, and battery grid-import-reduction fidelity is worse in every seed. This supports a narrow attribution to the declared auxiliary-loss package, not uniform joint-quality dominance.

All results are exploratory validation diagnostics. This analysis read no test split, no confidence intervals were averaged, and no significance claim is made.

## Matched protocol and binding checks

Each run completed **26 epochs** on the same **606,060 actual training profiles**. Training uses global standardization, hinge loss with spectral normalization, balanced batches of 512 (four conditions × 128 samples), and the same conditional-W1 checkpoint-selection rule. Within each paired seed, the only configuration difference is `swd_weight: 0 → 10`; architecture, optimizer, learning rates, projections and sampling settings match.

All six checkpoint and calibration archive hashes match their reports. The five-key checkpoint provenance (including curation manifest) and four-key prepared-data provenance agree with their respective records; training subset/scaler, fixed validation rows, labels and real-profile fingerprints match. All four statistical baselines have identical generated-array hashes and metrics across the six runs. No datasets or weight tensors were loaded for this attribution.

Calibration uses 4,096 independent generated profiles per condition, seed 7301 and 1,025 quantile knots. Evaluation uses seed 2026 and 100 real/generated profiles per condition (6,400 total). Both calibrated cGAN arms and the primary Gaussian/independent controls use the same target-quantile setting.

| Seed | SWD0 selected epoch | SWD10 selected epoch |
|---|---:|---:|
| 11 | 25 | 25 |
| 22 | 26 | 25 |
| 33 | 25 | 24 |

Training-core source hashes match for each pair. Controls 22/33 were evaluated after the derivative-ablation module/configuration extension: their evaluation `training.py` hash differs, while the evaluation runner and physical/copula/calibration/joint/battery modules match. The recorded `generate`, `_conditions` and `_normalized` function ASTs are identical between commits `555b09d` and `1ca62d0`; trained configurations have no derivative term. This source difference is retained in the JSON rather than described as whole-package identity. Added joint/battery modules also explain unrelated training-provenance differences for controls 22/33.

## Physical and whole-profile diagnostics

Cells are mean ± sample SD across the three training seeds. Δ is the mean paired SWD10−SWD0 difference; “lower” counts seed-level distance reductions and is descriptive only. Lower distances/RMSE are preferable.

### Raw cGAN

| Metric | SWD0 | SWD10 | Paired Δ |
|---|---:|---:|---:|
| Pointwise W1 (kWh) | 0.10957 ± 0.00189 | 0.08962 ± 0.00119 | -0.01995 (3/3 lower) |
| Daily energy W1 (kWh) | 6.01753 ± 0.25166 | 2.98954 ± 0.02477 | -3.02799 (3/3 lower) |
| Peak power W1 (kW) | 0.93644 ± 0.03905 | 0.74765 ± 0.00639 | -0.18879 (3/3 lower) |
| Ramp W1 (kW) | 0.23750 ± 0.01597 | 0.19743 ± 0.01172 | -0.04008 (3/3 lower) |
| Mean-profile RMSE (kWh) | 0.07819 ± 0.00636 | 0.04771 ± 0.00035 | -0.03048 (3/3 lower) |
| Pointwise-variance RMSE (kWh²) | 0.09551 ± 0.00187 | 0.06668 ± 0.00071 | -0.02883 (3/3 lower) |
| Lag-1 ACF W1 | 0.09417 ± 0.00338 | 0.09823 ± 0.00569 | +0.00407 (0/3 lower) |
| Lag-4 ACF W1 | 0.09906 ± 0.00147 | 0.08483 ± 0.00391 | -0.01423 (3/3 lower) |
| Lag-24 ACF W1 | 0.04143 ± 0.00320 | 0.03577 ± 0.00329 | -0.00566 (3/3 lower) |
| Projected-profile W1 (kWh/projection) | 0.10215 ± 0.00162 | 0.06494 ± 0.00029 | -0.03721 (3/3 lower) |
| Diversity ratio (near 1 diagnostic) | 1.88477 ± 0.10700 | 0.96737 ± 0.02116 | -0.91740 |

### Calibrated cGAN

| Metric | SWD0 | SWD10 | Paired Δ |
|---|---:|---:|---:|
| Pointwise W1 (kWh) | 0.05378 ± 0.00098 | 0.05396 ± 0.00039 | +0.00018 (2/3 lower) |
| Daily energy W1 (kWh) | 3.29830 ± 0.08828 | 3.24854 ± 0.09962 | -0.04976 (2/3 lower) |
| Peak power W1 (kW) | 0.85988 ± 0.01438 | 0.78645 ± 0.00901 | -0.07343 (3/3 lower) |
| Ramp W1 (kW) | 0.12143 ± 0.00420 | 0.09810 ± 0.00641 | -0.02332 (3/3 lower) |
| Mean-profile RMSE (kWh) | 0.04636 ± 0.00134 | 0.04647 ± 0.00045 | +0.00011 (2/3 lower) |
| Pointwise-variance RMSE (kWh²) | 0.07504 ± 0.00187 | 0.07636 ± 0.00146 | +0.00132 (0/3 lower) |
| Lag-1 ACF W1 | 0.07988 ± 0.00490 | 0.08685 ± 0.00546 | +0.00697 (1/3 lower) |
| Lag-4 ACF W1 | 0.08234 ± 0.00841 | 0.08019 ± 0.00716 | -0.00215 (2/3 lower) |
| Lag-24 ACF W1 | 0.02859 ± 0.00345 | 0.02990 ± 0.00275 | +0.00131 (1/3 lower) |
| Projected-profile W1 (kWh/projection) | 0.06669 ± 0.00063 | 0.06545 ± 0.00035 | -0.00125 (3/3 lower) |
| Diversity ratio (near 1 diagnostic) | 1.11226 ± 0.00776 | 1.09263 ± 0.02993 | -0.01963 |

Calibration removes much of the raw marginal mismatch in both arms: calibrated pointwise W1 is approximately 0.054 for both. Its mean is slightly higher with SWD10, not lower. Mean daily-energy improvement after calibration is small and mixed across seeds; seed 22 worsens. Calibrated lag-1 and lag-24 ACF distances worsen in two of three seeds. Marginal/diversity diagnostics are directly targeted by calibration and do not independently validate dependence.

### Every calibrated seed

| Seed | Daily W1 0→10 | Peak W1 0→10 | Ramp W1 0→10 | Lag-1 ACF 0→10 | Projected W1 0→10 |
|---|---:|---:|---:|---:|---:|
| 11 | 3.34380 → 3.16532 | 0.87343 → 0.77658 | 0.11778 → 0.10226 | 0.08408 → 0.08152 | 0.06672 → 0.06504 |
| 22 | 3.19655 → 3.35892 | 0.86140 → 0.79424 | 0.12601 → 0.09072 | 0.08106 → 0.09243 | 0.06605 → 0.06565 |
| 33 | 3.35454 → 3.22138 | 0.84479 → 0.78852 | 0.12049 → 0.10132 | 0.07450 → 0.08661 | 0.06731 → 0.06565 |

The shared primary Gaussian copula remains better on daily energy W1 (2.93850 versus SWD10 calibrated mean 3.24854) and projected-profile W1 (0.06345 versus 0.06545). SWD10 calibrated is better on peak and ramp than that copula, but conditional resampling is still better on daily/peak/ramp/lag-1/projected distances. These baselines prevent a claim that profile SWD10 produces the best general joint model.

## Fixed battery-scenario fidelity

The hypothetical additional battery is 10 kWh / 4 kW, with 90% round-trip efficiency, empty initial SOC each day and charging only from observed net export. These are distances between simulated output distributions, not realized savings or annual utility.

### Raw cGAN battery outputs

| Metric | SWD0 | SWD10 | Paired Δ |
|---|---:|---:|---:|
| Post-battery grid import W1 (kWh) | 4.58024 ± 0.33388 | 2.56439 ± 0.10066 | -2.01585 (3/3 lower) |
| Post-battery grid export W1 (kWh) | 0.83213 ± 0.21019 | 0.31212 ± 0.04563 | -0.52001 (3/3 lower) |
| Grid-import reduction W1 (kWh) | 1.23140 ± 0.11950 | 1.65721 ± 0.09189 | +0.42581 (0/3 lower) |
| Post-battery peak import W1 (kW) | 0.93087 ± 0.02820 | 0.74300 ± 0.01348 | -0.18787 (3/3 lower) |
| End-of-day SOC W1 (kWh) | 0.88395 ± 0.13645 | 0.32893 ± 0.02202 | -0.55502 (3/3 lower) |

### Calibrated cGAN battery outputs

| Metric | SWD0 | SWD10 | Paired Δ |
|---|---:|---:|---:|
| Post-battery grid import W1 (kWh) | 2.78089 ± 0.04830 | 2.80507 ± 0.11733 | +0.02418 (2/3 lower) |
| Post-battery grid export W1 (kWh) | 0.37605 ± 0.03655 | 0.28481 ± 0.03719 | -0.09125 (3/3 lower) |
| Grid-import reduction W1 (kWh) | 0.25823 ± 0.01618 | 0.28791 ± 0.00819 | +0.02967 (0/3 lower) |
| Post-battery peak import W1 (kW) | 0.82354 ± 0.01193 | 0.76768 ± 0.00717 | -0.05586 (3/3 lower) |
| End-of-day SOC W1 (kWh) | 0.24784 ± 0.03511 | 0.24033 ± 0.02315 | -0.00751 (1/3 lower) |

After calibration, export and peak-import fidelity improve in every seed. Import-reduction fidelity worsens in every seed; post-battery grid-import W1 is slightly worse on average, and end-SOC results are mixed. The battery diagnostic therefore does not support a blanket application-fidelity improvement.

## Utility availability and limits

**No paired utility attribution is available.** All three SWD0 utility reports are queued/incomplete at this snapshot. Available SWD10 utility reports are unpaired and cannot establish a weight effect. The report makes no utility-improvement or tolerance-pass claim.

This comparison identifies the effect of weight 10 versus 0 within the stated training and checkpoint-selection package; it does not isolate architecture, prove GAN novelty, or establish unseen-test performance. Only three training seeds are paired, calibration/evaluation streams are fixed, and validation has been reused. Exact-trace curation does not establish household independence or exclude near-duplicates. The 128 fixed projections and single battery scenario are secondary diagnostics with limited scope.

All scalar physical means, three ACF lags, battery-output means/distances, every paired seed, source differences and input hashes are retained in compact metric vectors in [the JSON report](../reports/curated_swd_attribution.json). The vector key and summary-column orders are explicit. Full condition-level arrays and identity history remain in the fingerprint-bound source reports; conditions are not treated as independent significance evidence.
