# Conditional marginal calibration: fixed development experiment

Declared 21 October 2026, before fitting or evaluating this calibration.
This continues Phase 1 on the existing inner split of original TRAIN. It is
post-test development, not independent confirmation or a new reference cGAN.
The [machine-readable plan](../reports/phase1-development/MARGINAL_CALIBRATION_PLAN.json)
binds the datasets, selected checkpoints, historical comparison artifacts and
sampling settings. Original VAL/TEST performance is not accessed.

## Question and fixed arms

Can TRAIN-only hourly marginal calibration remove the recorded excess negative
readings while improving whole-day quality? Does dependence learned by the
cGAN or basic flow add value over the conditional Gaussian copula, when both
use the same empirical marginal inverse?

Use both training seeds, 101 and 202, for each of two completed 30,000-update
families: global-standardized cGAN with Adam beta1=0 and discriminator dropout
0.4; and isotropic-source conditional flow with its fixed 32-step Heun sampler.
Use each saved SW1-best checkpoint without reselection or retraining. The
dropout alternatives had no consistent advantage across seeds; the correlated
flow source worsened ramps; the auxiliary energy-loss experiment has only a
6,000-update budget. Their results remain preserved, but they are not additional
arms in this bounded diagnostic.

Compare each raw generator with its calibrated version and with TRAIN-fitted
conditional mean, complete-day resampling and Gaussian copula. These are
separate adapted generators; none replaces the original reported samples.

## Frozen transformation

For each of the 64 observed conditions and 24 hours, sort the real inner-TRAIN
values and independently generated calibration values. Generate **16,384
calibration days per condition**, in ascending condition order, using seed
**20261025**. The total is 1,048,576 generated profiles per checkpoint. This
count exceeds the 4,025–11,088 real TRAIN days in each condition, giving finer
source rank resolution than the smallest empirical target jump. It does not
eliminate finite-sample error or validate rare-event frequencies.

For a new generator value x, use the deterministic mid-ECDF
`u = (count(source < x) + count(source <= x)) / (2*m)`. Decode with the real
TRAIN empirical step inverse, index `ceil(n*u)-1` bounded to `[0,n-1]`, exactly
as in the existing Gaussian-copula baseline. Values below/above the fitted
source range map to observed target minima/maxima. There is no interpolation
through target jumps, extrapolation, random tie breaking, or physical clipping.

Source ties remain tied. A constant source coordinate with a varying TRAIN
target causes explicit failure; random noise must not manufacture spread.
A constant target maps to its constant value. Fitted arrays are copied, saved
without pickle, hashed, reloaded with hash verification and fixed for inference.
Neither validation data nor the current generated evaluation batch fits ranks.

The mapping is coordinatewise nondecreasing; output ties mean exact rank
correlations need not be preserved. Pearson correlations, ramps and daily
energy can change. Empirical quantiles preserve available zero/negative support
but truncate each hourly tail to observed TRAIN extrema. This enforces only
hourly marginal support, not realistic joint profiles or unseen tail behavior.
In particular, rare negative PV=0 TRAIN readings remain possible.

## Evaluation and interpretation

Reuse the exact saved 16,384 inner-validation rows, 256 per condition, and
generation seeds **20261023 and 20261024**. These sampling draws were used by
the previous diagnostic: they are paired comparisons, not new validation
meters, independent training replicates or fresh unseen diagnostics. Calibration
draws are separate from these and from checkpoint-selection seed 20261022.
Verify regenerated raw and baseline array hashes and metric records against
the preserved fresh-draw study before interpreting transformed results.

Report normalized profile SW1, daily-energy/peak/ramp Wasserstein distances,
mean-profile error, ACF, covariance, diversity, and conditional tail quantiles.
Report raw negative-hour/day frequencies and export-energy magnitudes for
PV=0 groups; zero atoms, target endpoint mass and source-range exceedances;
and per-condition/hour probability decile histograms on independent generator
draws. These are descriptive diagnostics, not calibrated significance tests.
Flat probability deciles are not expected for constant source coordinates or
source atoms under the deterministic mid-ECDF; ties are reported explicitly.
Retain precision-specific full-TRAIN exact-row copy checks without calling
zero matches a privacy or novelty guarantee.

Marginal/export improvements alone are expected from a TRAIN lookup. The useful
signal is consistent joint-quality improvement for both training seeds, assessed
against the same-inverse Gaussian copula and complete-day resampling. Report
tradeoffs and every declared arm; do not tune the mapping or reselect checkpoints
using these results. No threshold here creates a publication-readiness claim.
If dependence remains weak, the next separately declared comparator is a
conditional reflection copula, informed by the
[verified source review](SUPPORT_AWARE_GENERATORS.md). Phase 2 and federated
learning remain deferred.

## Implementation and verification

1. Implement the calibration transform and safe serialization with synthetic
   tests for known quantiles/ties, constant and signed support, monotonicity,
   input immutability, evaluation batch/order invariance and tampered artifacts.
2. Implement the plan-bound evaluator with tiny actual cGAN/flow fixtures,
   forbidden TEST files, provenance/seed checks and failure preservation.
3. Review the code, run the full software suite and commit this plan and code
   before the real experiment. Preserve the execution source commit and hashes.
4. Run the four fixed fits and paired comparisons, save all artifacts in a new
   ignored directory, and commit compact results and their derivation receipts.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python scripts/evaluate_marginal_calibration.py \
  --plan reports/phase1-development/MARGINAL_CALIBRATION_PLAN.json \
  --output artifacts/phase1-development/marginal_calibration
```
