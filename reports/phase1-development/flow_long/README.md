# Common 30,000-update flow-source comparison

All four flow runs completed the common 30,000-update budget. Longer training
reduces profile and ramp errors relative to their short screens, but the
correlated source still produces larger ramp and lag-1 autocorrelation errors
than the isotropic source in both seeds. Its slightly smaller selected profile
SW1 does not justify adopting the correlated source as an overall improvement.

Both source settings use training seeds 101 and 202, batch size 512, learning
rate 0.001, the same velocity network and 32-step Heun sampling for selection.
Each run presents 15,354,916 real profiles. The source correlation fits inner
TRAIN only. Runs restart deterministically from initialization because the flow
trainer does not implement checkpoint continuation. They reuse the original
two seeds and trajectories; they are not four additional independent replicates.

## Selected and final inner-validation scores

Checkpoint selection minimizes profile SW1 on the same 256 real days per
condition and generation seed 20261022. All metrics use the same physical
normalization. Lower errors are better. The final checkpoint is always step
30,000; it is reported separately rather than replacing the selected checkpoint.

| Source | Seed | Selected step | Selected SW1 | Final SW1 | Selected daily-energy W1 (kWh) | Selected ramp W1 (kW) | Final ramp W1 (kW) | Selected lag-1 ACF W1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Isotropic | 101 | 21,000 | 0.005482 | 0.005491 | 2.381 | 0.135880 | 0.133609 | 0.0530 |
| Isotropic | 202 | 15,000 | 0.005298 | 0.005706 | 2.176 | 0.154073 | 0.134490 | 0.0575 |
| Correlated | 101 | 27,000 | 0.005481 | 0.005831 | 2.364 | 0.199342 | 0.205010 | 0.1295 |
| Correlated | 202 | 27,000 | 0.005274 | 0.005637 | 2.272 | 0.189160 | 0.191907 | 0.1171 |
| Training-day resampling | — | — | 0.005269 | — | 2.208 | 0.045946 | — | 0.0316 |
| Empirical Gaussian copula | — | — | 0.005414 | — | 2.365 | 0.123084 | — | 0.0592 |

Full selected and final macro metrics, including peaks, diversity, variance,
slot distributions and lag-6 autocorrelation, are in `selected_last_metrics.csv`.
The selected diversity ratios are 1.199/1.229 for isotropic and 1.218/1.126 for
correlated sources. At the final checkpoint they are 1.297/1.274 and 1.387/1.301.

All four selected profile SW1 values remain above the matched resampling draw.
Both seed-202 runs have selected SW1 below the copula draw, while seed-101 runs
do not. This mixed ordering on reused selection data is not evidence of
superiority. Correlated-source ramp errors remain materially higher than the
isotropic errors at both selected and final checkpoints, and all four exceed
the resampling and copula ramp errors. None of these runs establishes adequate
temporal fidelity merely through a competitive profile SW1.

## Exact verification of reused prefixes

All four long runs reproduce their corresponding short-run histories exactly
at steps 1, 1,000, 2,000, 3,000, 4,000, 5,000 and 6,000 after excluding only
these inspected timing fields:

- `elapsed_seconds`
- `validation_generation_seconds`
- `validation_metric_seconds`
- `training_update_seconds`

The last field is the cumulative `time.perf_counter()` difference around the
training-update blocks; it is a timing measurement, not a training-step count.
Every other saved field, including velocity loss, row presentations, pass
counts, all validation metrics and diagnostic vectors, matches with exact
equality. No scientific field or numeric tolerance was excluded to obtain the
match. `prefix_verification.json` records both complete history hashes, bound
manifest hashes, checked steps and equal canonical hashes of each non-timing
prefix. This checks saved histories; checkpoint weights were not deserialized
or compared for this report.

## Source-only and solver diagnostics

Both source-only baselines were saved for these long runs. They inverse-scale
the source noise without applying a trained velocity map. Values are identical
between seeds within a source type because the fitted source and generation
draw are identical. They are shared diagnostics, not independent replicates.

| Untrained source | Normalized profile SW1 | Daily-energy W1 (kWh) | Ramp W1 (kW) | Diversity ratio |
| --- | ---: | ---: | ---: | ---: |
| Isotropic Gaussian | 0.021670 | 10.107 | 1.157 | 3.235 |
| Training-correlated Gaussian | 0.020812 | 12.392 | 0.421 | 3.229 |

Learning improves substantially over either source alone. That finding does
not show improvement over the trained empirical comparators.

The saved numerical checks keep each selected checkpoint and initial noise
fixed while refining Heun integration from 32 to 64 steps, doubling network
evaluations from 64 to 128. These checks were not used for selection.

| Source | Seed | SW1, 32 steps | SW1, 64 steps | Ramp W1, 32 steps | Ramp W1, 64 steps | Paired profile RMSE (kWh) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Isotropic | 101 | 0.005482 | 0.005486 | 0.135880 | 0.135704 | 0.002518 |
| Isotropic | 202 | 0.005298 | 0.005302 | 0.154073 | 0.153920 | 0.002202 |
| Correlated | 101 | 0.005481 | 0.005484 | 0.199342 | 0.199378 | 0.001957 |
| Correlated | 202 | 0.005274 | 0.005278 | 0.189160 | 0.189196 | 0.001828 |

Refinement changes these metrics only slightly and does not repair the observed
ramp mismatch. This is a sensitivity check at the selected checkpoints, not a
formal integration-convergence guarantee. `diagnostics.json` retains every
source-only macro score and solver macro score, source-file hashes, selected
checkpoint hashes and paired-profile changes.

## Saved record and limits

`checkpoints.csv`, `learning_curves.csv` and `manifest.json` are byte-identical
copies from `artifacts/phase1-development/flow_long_summary`. The receipt binds
all four long-run manifests, the matched baseline manifest, the renderer and
generated artifacts. Full per-condition metrics, baseline draws and plots
remain in that ignored directory. `extraction_receipt.json` binds the additional
selected/final table, prefix checks and diagnostic extraction. Every input
history and diagnostic hash was checked against its run manifest; each saved
32-step solver report was checked against its selected validation report.

These are budget-sensitivity development results on previously used validation
data. Longer budgets and repeated prefixes are not additional seed replicates.
There is no independent final-validation result, calibrated significance claim,
adoption decision or publication-readiness claim. This summary read saved
reports only and performed no training or model evaluation. Phase 2 remains
deferred.

Regenerate the core tables and plots into a new directory:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
.venv/bin/python scripts/summarize_development.py \
  --runs \
  runs/phase1-development/flow_long/isotropic_flow_s101 \
  runs/phase1-development/flow_long/isotropic_flow_s202 \
  runs/phase1-development/flow_long/correlated_flow_s101 \
  runs/phase1-development/flow_long/correlated_flow_s202 \
  --baselines artifacts/phase1-development/baselines_screen \
  --output artifacts/phase1-development/flow_long_summary_rerender
```
