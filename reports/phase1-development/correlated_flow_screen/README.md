# Correlated-source flow development screen

The training-derived correlated Gaussian source worsened profile-distribution
and ramp errors relative to the isotropic source in both training seeds at the
6,000-update budget. It is not a successful improvement in this screen.

All four runs completed 6,000 velocity-network optimizer updates with batch
size 512 and 3,071,016 real-profile presentations. Both arms use training seeds
101 and 202, the same inner-training and validation data, and 32-step Heun
sampling for checkpoint selection. The correlated source uses the unconditional
hourly correlation estimated on inner TRAIN with 0.01 shrinkage toward identity;
the velocity network remains conditional. Correlation is a source-distribution
change, not a change to the original reference cGAN. The earlier isotropic flow
trajectories are reused controls, not newly trained replicates.

## Selected inner-validation results

Each checkpoint minimizes profile SW1 on the same 256 real days per condition
and generation seed 20261022. All scores use a common physical-unit
normalization. Wasserstein errors are lower-is-better. The diversity ratio is
generated to real variance, with 1 denoting equality of that aggregate statistic.

| Source / comparator | Training seed | Selected step | Normalized profile SW1 | Daily-energy W1 (kWh) | Ramp W1 (kW) | Lag-1 ACF W1 | Diversity ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Isotropic flow | 101 | 6,000 | 0.005768 | 2.434 | 0.191 | 0.0595 | 1.376 |
| Isotropic flow | 202 | 4,000 | 0.005716 | 2.427 | 0.196 | 0.0525 | 1.298 |
| Correlated-source flow | 101 | 6,000 | 0.006084 | 2.314 | 0.289 | 0.1840 | 1.335 |
| Correlated-source flow | 202 | 4,000 | 0.006445 | 2.309 | 0.320 | 0.1891 | 1.334 |
| Training-day resampling | — | — | 0.005269 | 2.208 | 0.046 | 0.0316 | 1.271 |
| Empirical Gaussian copula | — | — | 0.005414 | 2.365 | 0.123 | 0.0592 | 1.245 |

Correlation reduces daily-energy W1 in both seeds, but profile SW1, ramp W1 and
lag-1 autocorrelation error all worsen. Neither correlated run improves on
resampling's selected-draw scores for those endpoints. The small daily-energy
improvement does not compensate for the poorer temporal behavior. Selection at
the final budget for seed 101 does not establish convergence.

## Source-only baseline and solver sensitivity

The correlated source alone, inverse-scaled without the trained velocity map,
has normalized profile SW1 **0.020812**, daily-energy W1 **12.392 kWh**, ramp W1
**0.421 kW**, lag-1 ACF W1 **0.1349**, and diversity ratio **3.229**. Its saved
metrics are identical in the two run records because they use the same fitted
unconditional source and the same generation draw. They constitute one shared
source-only diagnostic, not two independent estimates. The trained flow improves
several errors over this starting distribution; that improvement does not imply
superiority over the stronger controls. Source-only reports were absent from
the original isotropic runs and were not recomputed for this summary.

All four saved solver checks keep the selected checkpoint and initial noise
fixed while doubling Heun integration from 32 to 64 steps (64 to 128 network
evaluations). These checks were not used to select a checkpoint or solver.

| Source | Training seed | SW1, 32 steps | SW1, 64 steps | Ramp W1, 32 steps | Ramp W1, 64 steps | Paired profile RMSE (kWh) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Isotropic | 101 | 0.005768 | 0.005773 | 0.190862 | 0.190740 | 0.001998 |
| Isotropic | 202 | 0.005716 | 0.005718 | 0.195638 | 0.195506 | 0.001504 |
| Correlated | 101 | 0.006084 | 0.006087 | 0.288894 | 0.288915 | 0.001487 |
| Correlated | 202 | 0.006445 | 0.006447 | 0.319655 | 0.319662 | 0.001129 |

The small changes under this solver refinement do not explain or repair the
large ramp mismatch. This comparison is a numerical sensitivity check at one
selected checkpoint per run, not a formal convergence guarantee for integration.

## Record and limits

`checkpoints.csv`, `learning_curves.csv` and `manifest.json` are byte-identical
copies from `artifacts/phase1-development/correlated_flow_summary`. The receipt
binds all four run manifests, the baseline manifest, renderer source and saved
plots/tables. Full per-condition metrics and baseline draws remain in that
ignored artifact directory. `diagnostics.json` extracts the source-only and
solver results above from the original saved JSON reports, with their source
hashes and selected-checkpoint hashes. All diagnostic file hashes were checked
against their run manifests; the 32-step solver metrics were checked against
the selected validation reports.

The renderer and diagnostic extraction read saved reports only. No training,
checkpoint deserialization, data-array access or model evaluation occurred.
These are two-seed development comparisons selected on previously used inner
validation data. They provide no independent final-validation result, calibrated
significance claim or evidence of publication readiness. Phase 2 remains deferred.

To regenerate the core tables and plots, use a new output directory:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
.venv/bin/python scripts/summarize_development.py \
  --runs \
  runs/phase1-development/flow_screen/flow_s101 \
  runs/phase1-development/flow_screen/flow_s202 \
  runs/phase1-development/correlated_flow/correlated_flow_s101 \
  runs/phase1-development/correlated_flow/correlated_flow_s202 \
  --baselines artifacts/phase1-development/baselines_screen \
  --output artifacts/phase1-development/correlated_flow_summary_rerender
```
