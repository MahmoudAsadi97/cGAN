# Conditional energy-distance auxiliary-loss screen

Adding a conditional energy-distance term improves selected profile SW1 over
the matched cGAN control in both seeds at 6,000 updates. It remains substantially
behind training-day resampling and the empirical Gaussian copula, and does not
establish a reliable model for publication.

The four runs share global TRAIN standardization, the reference generator and
discriminator topology, Adam beta1=0, discriminator dropout 0.4, batch size 512,
training seeds 101 and 202, and a budget of 6,000 discriminator/generator update
pairs. Each presents 3,071,016 real profiles. The new generator objective adds
weight 1 times a conditional Euclidean energy-distance U-statistic to its usual
non-saturating adversarial loss. The control weight is zero. Eligible conditions
have at least two batch examples and receive equal weights; same-sample
diagonals are excluded and negative finite estimates are retained. The term
uses the existing real and generated generator-update batch.

This statistical energy distance is measured in the TRAIN-standardized
representation. It is not physical daily electricity energy, and it is not a
reproduction of an Energy Matching method. Minibatches contain potentially
dependent meter-day observations, so U-statistic terminology supplies no
inferential calibration for this study. The changed objective is an adaptation
of reference training despite retaining its topology.

## Selected inner-validation results

All comparisons use the same 256 real days per condition, generation seed
20261022 and common physical normalization. Checkpoints minimize profile SW1
on those validation curves; their scores are not independent performance
estimates. Errors are lower-is-better. A diversity ratio of 1 denotes equal
generated and observed aggregate variance.

| Generator objective | Training seed | Selected step | Normalized profile SW1 | Daily-energy W1 (kWh) | Peak W1 (kW) | Ramp W1 (kW) | Diversity ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Adversarial + energy distance, weight 1 | 101 | 4,000 | 0.009086 | 3.402 | 0.818 | 0.168 | 0.780 |
| Adversarial + energy distance, weight 1 | 202 | 4,000 | 0.008282 | 3.548 | 0.804 | 0.175 | 0.910 |
| Adversarial control, weight 0 | 101 | 5,000 | 0.012346 | 5.742 | 0.917 | 0.174 | 0.627 |
| Adversarial control, weight 0 | 202 | 6,000 | 0.012232 | 5.559 | 0.948 | 0.154 | 0.665 |
| Training-day resampling | — | — | 0.005269 | 2.208 | 0.344 | 0.046 | 1.271 |
| Empirical Gaussian copula | — | — | 0.005414 | 2.365 | 0.523 | 0.123 | 1.245 |

Both auxiliary-loss runs select step 4,000 rather than the final step. Profile,
daily-energy and peak-distribution errors improve relative to their same-seed
controls, and variance is closer to that of the validation sample. Ramp error
slightly improves for seed 101 but worsens for seed 202. Both remain worse than
the strong controls on all four error columns above. The learning curves do not
support an assumption that simply extending these settings will keep improving
the selected metric.

## Recorded objective components and runtime

The following are recorded means over the updates since the previous logged
checkpoint. They are optimization diagnostics, not validation metrics. At weight
1, generator loss equals adversarial loss plus the auxiliary component, up to
floating-point arithmetic.

| Training seed | Window end step | Mean adversarial loss | Mean energy-distance loss | Mean total generator loss |
| --- | ---: | ---: | ---: | ---: |
| 101 | 1 | 0.683584 | 2.703560 | 3.387145 |
| 101 | 4,000 | 1.392178 | 0.252487 | 1.644664 |
| 101 | 6,000 | 1.460904 | 0.213723 | 1.674627 |
| 202 | 1 | 0.669968 | 2.503195 | 3.173163 |
| 202 | 4,000 | 1.342125 | 0.237536 | 1.579660 |
| 202 | 6,000 | 1.456425 | 0.217865 | 1.674290 |

`objective_losses.csv` retains every logged window for all four runs. Component
columns are empty for the historical controls because they were not separately
recorded; their actual generator and discriminator losses are retained. The
decrease in the auxiliary training component does not guarantee improvement in
the selected validation profile score or in every temporal metric.

The new runs recorded **209.19 and 209.67 seconds**, versus **103.08 and 102.29
seconds** for the historical controls. These are complete elapsed run times,
including validation and artifact writing, under concurrent CPU workloads.
The auxiliary objective adds distance calculations, but these observations are
not an isolated overhead benchmark, a GPU speed estimate or an equal-wall-time
comparison. Both arms have equal update counts and real-row presentations, not
equal computational work per update.

## Saved record and limits

`checkpoints.csv`, `learning_curves.csv` and `manifest.json` are byte-identical
copies from `artifacts/phase1-development/energy_summary`. The receipt binds all
four run manifests, the matched baseline manifest, the renderer and generated
artifacts. Full condition-level metrics, baseline draws and plots remain in
that ignored directory. `objective_loss_receipt.json` binds the four original
history files and the extracted component-loss CSV; it records the arithmetic
consistency check applied to the saved components.

The two weight-zero controls are reused trajectories from the momentum screen,
not additional independent training seeds. These are two-seed development
comparisons selected on previously used inner validation data. There is no
independent final-validation result, calibrated significance claim or claim of
superiority. The summary read saved reports only and performed no training,
checkpoint deserialization, data-array access or model evaluation. Phase 2
remains deferred.

Regenerate the core tables and plots into a new directory:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
.venv/bin/python scripts/summarize_development.py \
  --runs \
  runs/phase1-development/energy_screen/standard_b0_energy1_s101 \
  runs/phase1-development/energy_screen/standard_b0_energy1_s202 \
  runs/phase1-development/momentum_screen/standard_b0_s101 \
  runs/phase1-development/momentum_screen/standard_b0_s202 \
  --baselines artifacts/phase1-development/baselines_screen \
  --output artifacts/phase1-development/energy_summary_rerender
```
