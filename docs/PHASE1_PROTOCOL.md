# Frozen Phase 1 experiment protocol

Scope: the reference hourly architecture on Fluvius. No auxiliary loss,
calibration, architecture improvement or federated learning is permitted.

## Data and prior access

Use the verified release and the original V2 allocation, with complete-trace
identity curation at quarter-hour resolution followed by hourly aggregation.
The counts are 1,665/353/349 train/validation/test meters and
606,060/128,492/127,036 daily profiles. Every transform fits training only.
Preserve the data and access history described in PROVENANCE.md.

The primary target is signed hourly net grid-import energy. Conditions encode
HP/EV/PV, weekend and meteorological season. Reported equipment indicators are
not guaranteed proof that undetected equipment is absent.

## Fixed neural experiment

Conditional and unconditional arms each use seeds **11, 22, 33, 44, 55**.
All runs use 100 epochs, batch size 2,000, no training-row cap, one discriminator
and one generator update per batch, and the reference widths/activations/loss
and Adam parameters. The unconditional control omits both conditioning inputs.
This is 30,400 updates of each network per run, including the last partial
batch; it is 60,606,000 training-row presentations per run.

This budget is an adaptation, not a reproduction of upstream's 10,000 epochs
on an unavailable original dataset. A separate full-data seed-101, 20-epoch
pilot measured the cost and showed nonmonotonic but improving validation
behavior: normalized SW1 at epochs 1/5/10/15/20 was approximately
0.15717/0.08698/0.06438/0.04946/0.08025. Twenty epochs did not establish
convergence. The declared budget extends exposure fivefold and is fixed across
all ten runs. It does not guarantee convergence; learning curves and
boundary-selected checkpoints must be reported as potential limitations.

Do not terminate successful runs early, select a favorable seed, add seeds in
response to significance, or change the budget after inspecting test results.
Infrastructure failures are recorded; any restarted attempt uses the same
configuration and seed, with the earlier attempt retained.

Checkpoint selection minimizes equally weighted conditional normalized SW1
over 128 unit Gaussian projections (seed 2026). Select at epoch 1 and every
fifth epoch thereafter. Fixed validation indices use seed 2027 and fixed
selection noise uses seed 2028, shared across all arms and training seeds.
Each condition uses up to 1,024 validation profiles without replacement and an
equal generated count. Ties retain the earliest checkpoint. Final evaluation
accepts only the selected artifact of a completed full-budget run.

## Baselines and reporting

Comparators are conditional mean, complete-training-day resampling, and the
fixed conditional empirical Gaussian copula specified in V3_DESIGN.md.
Use identical real evaluation samples, output counts, projection directions,
physical units and metrics. Do not clip model outputs or delete difficult
conditions. Retain the mean control's low diversity and the resampling
control's copying behavior in the interpretation.

Generation seeds **31001, 31002, 31003** are independent reporting replicates.
Use all eligible real rows, with equal generated counts per condition.
The primary endpoint is condition-macro normalized profile SW1. Report
daily-energy W1, peak-power W1, signed-ramp W1, slot W1, ACF, mean/quantile
curves, variance/diversity and condition counts as diagnostic secondary
endpoints. Five training-seed outcomes remain visible; report their mean and
sample standard deviation separately from generation variability.

## Uncertainty and limitations

Use stratified whole-meter bootstrap for real-data uncertainty, preserving
each sampled meter's complete history. Use 499 bootstrap replicates, seed
20261020, for paired primary-profile and daily-energy distance comparisons
against resampling and the copula. The bootstrap is paired through identical
real-meter draws. Its intervals condition on each fitted model and fixed
generation draws; they do not include all training or generation uncertainty.
Report those components separately instead of treating three generation
replicates as three independently trained models.

The first declared training seed, 11, and generation seed, 31001, are the fixed
bootstrap and memorization exemplar; neither is chosen for performance.
Normalize physical profile SW1 differences and interval endpoints by the same
frozen training-only range used for the primary endpoint.
Report paired point differences
for every training seed, their mean and sample SD, and the exemplar's cluster
intervals. The exemplar interval is not an interval for the five-model mean.
This allocation keeps the expensive whole-profile bootstrap inspectable and
bounded. More seeds or external data would strengthen generality.

For memorization, compare conditional nearest-training-profile RMSE for
generated and held-out real days using at most 256 queries per condition and
seed 41001. Report exact/tolerance matches and distance quantiles. This is a
diagnostic, not a differential-privacy guarantee.

Effect sizes and uncertainty are primary. Do not interpret large daily sample
sizes as independent households or use unadjusted thousands of significance
tests to claim superiority. Equipment groups were sampled equally by design;
sample-weighted results do not estimate population equipment prevalence.
All splits share the 2024 calendar, so no future-weather or future-year
generalization claim is supported.

## Completion and phase boundary

Freeze a machine-readable protocol binding data hashes and these settings
before final test-performance access. Preserve every final report and an
append-only test-access log. No test-informed tuning follows.

A functioning implementation, a falling GAN loss, or attractive example curves
does not establish publication readiness. The final assessment must distinguish
software correctness, empirical fidelity, comparator performance, convergence,
uncertainty and remaining novelty/data-access limitations. Negative findings
are valid Phase 1 outcomes. Present the complete record to the user and wait
for approval before Phase 2.
