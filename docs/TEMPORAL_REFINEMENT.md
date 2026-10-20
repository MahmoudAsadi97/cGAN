# Derivative-distribution refinement: exploratory protocol

This follow-up is declared after inspecting all five curated SWD10 primary
validation reports and before training the new arm. Validation is development
data; final test performance remains unseen. Retain the original arm, its failed
gates and every attempted refinement in the study ledger.

## Evidence motivating the change

The fixed calibrated SWD10 arm has condition-macro ramp W1
0.100655 +/- 0.006007 kW (mean +/- sample SD over seeds 11, 22, 33, 44, 55).
This is 28.63% below the matched Gaussian copula, but the mean and four individual
seeds exceed the existing 0.10 kW practical target. Peak error improves against
that copula, whereas daily-energy and projected-profile errors worsen. The
additional-battery diagnostic also overstates daily import reduction. Resampling
remains stronger on the main joint metrics. These are reasons to investigate
temporal dependence, not evidence of general superiority or a publication claim.

## One controlled change

Retain every setting in `configs/curated_swd10.json` and add
`derivative_swd_weight: 1.0`. Keep adversarial training, profile SW2-squared
weight 10, architecture, latent dimension 24, balanced batches, optimizer, data,
checkpoint selection, calibration and evaluation unchanged.

For globally TRAIN-standardized profiles z, form the 95 adjacent within-day
differences d(z)[t] = z[t+1] - z[t]. Do not join midnight to the next day.
Estimate a single positive scalar s_delta from the actual model training pool:
take the mean squared difference over rows and slots within each supported
condition, average those means equally over conditions, then take the square
root. Accumulate in float64 in bounded batches. Record the estimate, the value
used by float32 training and its fitting provenance. Do not center differences
or fit separate slot/condition scales. No validation values fit this scalar.

The new auxiliary term is the equal-condition mean empirical SW2-squared between
d(real)/s_delta and d(fake)/s_delta. Use 128 fresh unit projection directions and
the existing four-condition batches with 128 rows per condition. Average over
samples, directions and conditions; do not add a dimension sum or square root.
There is no additional interval-duration factor: conversion to power cancels
in the standardized difference scale. Use a separate RNG seeded training_seed+5;
zero weight must preserve the previous training path and random streams exactly.

The generator objective is adversarial +10*profile_SW2_squared
+1*difference_SW2_squared. This is an established distribution-matching idea
applied to a deterministic temporal transform, not a claimed new transport
method. Isotropic profile projections are not blind to ramps; this term changes
their relative importance. Finite minibatch transport can favor reduced
dispersion. The fixed marginal map changes physical differences, so a smaller
raw training loss need not improve the deployed calibrated pipeline.

## Fixed pilot and promotion rule

Run seeds 11, 22, 33, each on all 606,060 eligible TRAIN profiles with the same
26 epochs and 30,784 generator/discriminator updates. Compare against the matching
three original SWD10 seeds. No early stopping, resuming an interrupted run,
discarding seeds or tuning the auxiliary weight on these endpoints is allowed.
Infrastructure interruptions may restart the same seed from initialization with
the full budget, retaining the interrupted attempt separately.

Promote this single setting to seeds 44 and 55 only if the first three show:

- At least 10% lower mean calibrated ramp W1 than the matching original seeds,
  with lower error in all three paired seeds.
- No more than 10% increase in mean daily-energy, peak, lag-one ACF or fixed
  projected-profile errors relative to those same original seeds.
- The existing forecasting tolerance: calibrated synthetic-training macro RMSE
  at most 1.10 times both real-trained and resampling-trained references in each
  of the three pilot seeds. Also report the three-seed aggregate.

This is a project-specific exploratory screen, not statistical significance or
an external preregistration. It demands more than barely crossing 0.10 kW.
Inspect and retain all 64 conditions, battery outcomes, diversity and support/tie
diagnostics. Battery performance is secondary; do not redefine the primary
success criterion after seeing it. If promoted, retain the first three runs,
complete all five, and apply the unchanged final five-seed criteria plus the
declared uncertainty, 1,000-profile precision and calibration-reference sensitivity
checks. If the screen fails, report the failed arm and reconsider the model
structure before declaring another exploratory protocol. Never tune on TEST.
