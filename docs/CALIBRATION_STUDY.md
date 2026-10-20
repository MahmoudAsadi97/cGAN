# Curated-data calibration study

This protocol was written before retraining on the identity-curated preparation.
Earlier original-split experiments informed the method choice; they contain
cross-split exact trace aliases and remain exploratory. Validation has already
been used for development, so the corrected validation split is not a new
confirmatory holdout. Final test performance remains unseen.

## Method and comparison

Fit a conditional cGAN with train-only standard normalization, a spectral-normalized
hinge discriminator of width 128, generator width 64, and conditional squared
sliced-Wasserstein auxiliary weight 10. Use the already implemented balanced
sampler (four conditions, 128 profiles each) and conditional validation W1 for
checkpoint selection. Raw marginal W1 is an indirect selector of calibrated
temporal quality; retain this fixed rule rather than choosing retrospectively
among calibrated diagnostics. No new architecture or hidden clipping is introduced.

After freezing each checkpoint, estimate its conditional, per-slot generated CDF
from 4,096 independent draws per condition, seed 7301. Map new samples through that
fixed CDF and the corresponding training quantile function (1,025 knots). Target
marginals use only the model's actual training rows. Evaluation draws use seed
2026 and 100 profiles per condition. Never compute ranks from the current output
batch. Constant generated margins cannot be repaired into nonconstant targets.
The generated CDF uses a generalized inverse of its quantile grid: ties map
to the midpoint of the plateau probability interval; interpolation between distinct
values uses the upper edge of the lower plateau and the lower edge of the upper
plateau. Target quantiles are linearly interpolated, with endpoints held fixed
outside the source support. Report ties, endpoint saturation and out-of-CDF-support rates.

Calibration approximately aligns generated margins toward training margins. Finite
reference draws, quantile compression, ties and saturation prevent exact matching.
It supplies marginal information mechanically; it does not learn new rank
dependence. Improved marginal W1,
means, variance and diversity alone are not evidence of learned joint quality.
Primary evidence for the calibrated pipeline concerns daily totals, peaks, ramps, autocorrelations and
train-synthetic/test-real forecasting utility. Compare the same real validation
sample with raw cGAN, calibrated cGAN, Gaussian copula (shrinkage
0.01), independent margins (shrinkage 1), and conditional resampling. Both primary
copula controls use the same 1,025-knot training marginals as calibration. Also
report an all-order-statistic, linearly interpolated quantile copula as a secondary sensitivity reference.
This avoids conflating different marginal approximations with learned dependence. Baselines
must use the same actual training rows. Bounded nearest-neighbor diagnostics do
not establish privacy or absence of memorization.

## Prospective retraining budget

Use `configs/curated_swd10.json` on `data/curated96`. The full retained training
split contains 606,060 profiles. Batch 512 gives 1,184 updates per epoch; 26 epochs
give 30,784 updates per adversary. An epoch is a replacement-sampling budget,
not one traversal of unique profiles. No early stopping or resume is used.
Run all five seeds 11, 22, 33, 44 and 55 under the same budget, scheduled
within available CPU/memory. Initial three-process measurements showed about
3 GiB anonymous memory in total with an 8 GiB limit, permitting all five
one-thread jobs concurrently. Scheduling does not change per-run seeds, numerical
settings, sampling or update budgets. Retain every
attempt, and do not replace unsuccessful seeds. Additional methods or budgets
require an explicit new exploratory protocol entry.

The earlier practical gates remain: daily-energy W1 <= 5 kWh, peak W1 <= 1 kW,
ramp W1 <= 0.10 kW and lag-one ACF W1 <= 0.10, alongside all-condition inspection.
Marginal gates are reported but cannot certify the calibrated model. Compare
joint tradeoffs against the copula and resampling controls, and require utility
and uncertainty evidence before describing a publication direction as promising.
These project gates are not statistical significance or publication guarantees.

## Uncertainty and final evaluation

Report each seed, across-seed dispersion and paired technology-stratified meter
bootstrap comparisons. The bootstrap holds fitted models and generated samples
fixed; it excludes training, calibration and model-selection uncertainty. Assess
calibration-sample/seed sensitivity before freezing the final method. Do not tune
on final test results. The test subset is identity-audited and performance-unseen;
exact deduplication does not establish independent households or remove possible
near-duplicates. Empirical calibration is bounded by training support and cannot
validate unseen-tail extrapolation.

The fixed-CDF idea is related to implicit generative copulas (Janke, Ghanmi and
Steinke, NeurIPS 2021). This post-hoc adaptation does not reproduce their training
algorithm and is not claimed as a novel calibration method.

## Workspace recovery record

An execution-workspace reset removed ignored prepared arrays/checkpoints and the
then-uncommitted curation/calibration files. Tracked code and earlier reports were
recovered from V2 commit `066b6b54021995d4124adc08e29a7edbad6264f9`. The latest
implementation files were reconstructed from retained source patches and must
pass fresh tests. Raw data is re-downloaded and verified against its pinned hash;
preparation, curation and subsequent experiments are rerun. Summarized pre-reset
calibration observations are historical exploratory notes, not substituted for
lost machine-readable artifacts or fresh curated-data evidence.

## Commands

```bash
python scripts/run_experiments.py --prepared data/curated96 --config configs/curated_swd10.json --output runs/curated_swd10 --seeds 11 22 33 44 55 --samples-per-condition 100
python scripts/evaluate_calibration.py --prepared data/curated96 --checkpoint runs/curated_swd10/seed_11/best.pt --output runs/curated_swd10/seed_11/calibration_validation.json --calibration-output runs/curated_swd10/seed_11/calibration.npz
python scripts/generate_calibrated.py --checkpoint runs/curated_swd10/seed_11/best.pt --calibrator runs/curated_swd10/seed_11/calibration.npz --labels 0 8 56 --samples-per-label 100 --seed 9001 --output runs/example_calibrated.npz
```

Repeat calibration/evaluation for every declared seed. The inference command binds
the fixed map to the checkpoint hash and produces original-unit profiles without
reading validation or test data. Keep both checkpoint and map when distributing
the calibrated generator; a raw checkpoint alone represents a different method.

## Attribution control (declared before curated endpoint evaluation)

Run `configs/curated_swd0.json` with seeds 11, 22 and 33 on the same curated
training pool and 26-epoch budget. It differs only in auxiliary SWD weight (zero
instead of ten); balanced sampling, conditional checkpoint selection, generator,
discriminator and optimizer settings are shared. Compare both raw and calibrated
outputs with the same fixed references. This three-seed control supports
preliminary attribution; it does not replace the remaining reference-architecture,
unconditional and resolution ablations required for a completed manuscript.
Training exposure and elapsed time differ only as recorded (the weight-zero arm
skips SWD computation); update counts remain matched. All attempted seeds remain
in the ledger. This addition was chosen on methodological grounds before any
curated model's final temporal/utility metrics were evaluated.

## Primary evidence and secondary joint check

Before any curated final endpoint evaluation, designate condition-macro **ramp W1**
as the primary physical comparison with the matched-knot Gaussian copula. A useful
exploratory signal is at least 20% lower mean ramp W1 over the five fixed seeds,
with the same direction in at least four seeds. This is a project-specific
practical target, not a p-value, a preregistered external trial or a publication
acceptance rule. Require the existing absolute physical gates and inspect daily
energy, peak and ACF tradeoffs, each condition and all seeds. Report resampling
even if it remains better. Never describe nonidentical synthetic profiles as a
privacy advantage.

The fixed real-validation ridge task supplies a complementary utility check.
As a descriptive tolerance, calibrated synthetic training should have macro RMSE
no more than 10% above both real-trained and resampling-trained references. This
is not a formal noninferiority test; paired utility uncertainty and broader
application validation remain separate requirements. All methods use the full
actual training pool and matched downstream profile/label counts in primary runs.

Add a secondary whole-profile diagnostic using 128 fixed unit-normalized Gaussian
projection directions, seed 9182. Compute exact empirical one-dimensional W1 on
each projected profile distribution, average directions within each condition,
then average the 64 conditions equally. Freeze these directions across methods
and seeds and record their hash. Original-unit projections have kWh units but
are not daily-energy totals; compare within the same resolution only. Finite
projections cannot certify equality of the complete joint distribution. This
diagnostic is distinct from the training loss's fresh squared-W2 projections.

## Evaluation precision and calibration sensitivity

Keep the declared 100-profile-per-condition primary reports. Also evaluate every
completed candidate seed at a cap of 1,000 real and generated profiles per
condition, with the same checkpoints and fixed calibration settings. This
separate precision check addresses finite-sample metric noise; do not replace
the primary result with whichever sample size looks better or select a seed.
Use the same seed2026 and selection rule for all methods at each sample size,
and report actual counts if a condition has fewer available rows.

For the predetermined candidate seed11, assess calibration-reference sensitivity
with (4,096 draws/condition, seed7302) and (8,192 draws/condition, seed7301), keeping
checkpoint, target knots and evaluation draws fixed. These checks quantify
reference-estimation stability, not additional GAN-training repetitions. Any
resulting method change requires a new exploratory entry before final testing.

## Secondary battery-operation diagnostic

Before curated endpoint evaluation, add one fixed hypothetical additional-battery
scenario: 10 kWh capacity, 4 kW input/output power limits, charge/discharge
efficiencies each sqrt(0.9), and zero initial stored energy for each independent
day. Charge only from observed net export; discharge against net import. Report
post-battery import/export, import reduction, post-battery peak and terminal
stored energy distributions and their errors against the same real validation
profiles. Preserve energy conservation and interval-duration units.

This probes chronological nonlinear dependence with a transparent engineering
rule. It is a secondary diagnostic, not a tuned policy or an additional primary
success criterion. It does not estimate gross PV self-consumption, costs, annual
battery performance or optimal sizing. The original net records may already
include unobserved storage; the scenario is an additional hypothetical device.
Terminal stored energy is reported rather than credited to an unseen next day.
Power is represented as an interval average, so sub-interval effects are absent.
