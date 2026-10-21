# Common 30,000-update discriminator-dropout study

All six declared cGAN trajectories completed 30,000 generator/discriminator
update pairs: dropout 0, 0.2 and 0.4 at training seeds 101 and 202. The longer
budget reduces profile-distribution error, but reducing dropout provides no
consistent improvement across seeds and does not resolve underdispersion.

The four dropout-0/0.2 runs continue their 6,000-update screen checkpoints. The
two dropout-0.4 controls reuse the completed `momentum_long/standard_b0_s101`
and `standard_b0_s202` trajectories. Continuation creates no additional
independent seeds. All six share the same inner TRAIN, global standardization,
Adam beta1=0, batch size 512, original network topology and adversarial objective.
Each completes 15,354,916 real-profile presentations, approximately 31.646 passes
through inner TRAIN. Elapsed times reflect concurrent CPU runs and are not
controlled speed measurements.

Each checkpoint below was selected by minimum profile SW1 on the same inner
validation sample: 256 days per condition, all 64 conditions, generation seed
20261022 and fixed projections. Last means the 30,000-update checkpoint.
Diversity is the macro average of condition-specific generated/real variance
ratios; 1 denotes equality of that statistic. Lower SW1 is better.

| Dropout | Seed | Selected step | Selected normalized SW1 | Last normalized SW1 | Selected diversity | Last diversity |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 101 | 28,000 | 0.006869 | 0.006985 | 0.640 | 0.640 |
| 0 | 202 | 28,000 | 0.007085 | 0.007176 | 0.636 | 0.626 |
| 0.2 | 101 | 29,000 | 0.006960 | 0.007001 | 0.628 | 0.629 |
| 0.2 | 202 | 30,000 | 0.006963 | 0.006963 | 0.644 | 0.644 |
| 0.4 | 101 | 25,000 | 0.006982 | 0.006988 | 0.633 | 0.617 |
| 0.4 | 202 | 30,000 | 0.006873 | 0.006873 | 0.675 | 0.675 |

At the saved 6,000-update points, normalized SW1 ranged from 0.010912 to
0.012460; all final 30,000-update scores are lower. Dropout 0 and 0.2 have
slightly lower selected SW1 than dropout 0.4 for seed 101, while dropout 0.4
has the lowest score for seed 202. These small, validation-selected differences
do not establish a winning dropout setting. Two selected checkpoints remain at
the budget boundary, and the curves do not establish convergence.

The dispersion deficit persists in every setting at both selected and last
checkpoints. Selected daily-energy W1 is 3.264–3.447 kWh and peak W1 is
0.563–0.598 kW. The matched training-day resampling control has normalized SW1
0.005269, daily-energy W1 2.208 kWh and peak W1 0.344 kW; empirical Gaussian
copula has 0.005414, 2.365 kWh and 0.523 kW. Both controls therefore retain
smaller errors on these endpoints. The cGANs' selected ramp W1 is
0.096–0.106 kW, below copula's 0.123 kW but above resampling's 0.046 kW.
The controls themselves have diversity ratios above 1 on this finite cohort
(resampling 1.271; copula 1.245); they are comparison methods, not ground truth.

These are two-seed development results selected on previously used validation
curves. They are not independent confirmation, significance tests, or evidence
of publication readiness. This summary reads only saved reports and verifies
artifact hashes; it does not open data arrays, deserialize checkpoints, train,
or recompute scores. No original VAL or TEST data are accessed. Fresh-generation
diagnostics are separate from this selection-draw summary.

`checkpoints.csv`, `learning_curves.csv` and `manifest.json` are byte-identical
copies from `artifacts/phase1-development/dropout_long_summary`. The manifest
binds all six source run manifests, the matched baseline manifest, renderer
source and artifact hashes. Complete per-condition scores, baseline sampling
draws, summary JSON and both learning-curve figures remain in that ignored
artifact directory; their hashes are retained in the copied receipt.

Reproduce into a new directory using the existing renderer:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
.venv/bin/python scripts/summarize_development.py \
  --runs \
  runs/phase1-development/dropout_long/standard_b0_d0_s101 \
  runs/phase1-development/dropout_long/standard_b0_d0_s202 \
  runs/phase1-development/dropout_long/standard_b0_d2_s101 \
  runs/phase1-development/dropout_long/standard_b0_d2_s202 \
  runs/phase1-development/momentum_long/standard_b0_s101 \
  runs/phase1-development/momentum_long/standard_b0_s202 \
  --baselines artifacts/phase1-development/baselines_screen \
  --output artifacts/phase1-development/dropout_long_summary_rerender
```
