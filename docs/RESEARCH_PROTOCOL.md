# Research protocol and publication gates

This repository implements an adaptation, not an exact reproduction and not an
established publishable result. The upstream raw training data are absent. Its
paper concerns a different population. A change of dataset alone does not establish
scientific novelty. Freeze this protocol before looking at final test outcomes.

## Question and estimand

Can the Li et al. conditional MLP GAN reproduce distributions of daily **net grid
import** in the supplied Fluvius 2024 residential sample, conditional on reported
HP/EV/PV category, weekend and season, for meters never used during training?
The prediction target is kWh per interval. It is not gross appliance demand,
PV generation, EV charging energy or a calibrated individual-household forecast.
The equal sampling of 300 meters per technology group is not population prevalence.

## Frozen data protocol

Apply [DUPLICATE_REMEDIATION.md](DUPLICATE_REMEDIATION.md) before confirmatory runs.
Meter-ID disjointness alone is insufficient: exact full-trace aliases must be
removed across all splits while preserving the original held-out membership.
All earlier original-split scores are exploratory and qualified by this finding.

Use the pinned release digest and metadata.json. Exclude the two DST transition
calendar days, preserve negative net export, and do not clip outliers. Keep all days
of each meter in one of the 70/15/15 partitions, stratified by technology, seed42.
The full expected assignment is 1,680 train, 360 validation and 360 test meters.
Fit all transforms on train. Save IDs in the manifest so all methods use the same
partitions. Do not resplit until a favorable result appears. The model never receives
meter identifiers. Missing or invalid ordinary days must be counted separately.

Train/validation/test all cover the same 2024 seasons. This estimates held-out-meter
performance under shared calendar conditions; it does not prove future-year or
future-weather generalization. A future-time benchmark needs a separately defined
blocked temporal split and gap policy; no such result is claimed by this release.

## Mandatory experiments

| Experiment | Configuration | Purpose |
|---|---|---|
| Conditional resampling | Draw complete train days matching each condition | Strong distribution baseline; deliberately not a privacy method |
| Conditional mean | Mean train profile for each condition | Detect a model that merely recovers average shape and loses diversity |
| Unconditional MLP GAN | Same width/loss/budget, condition dimension0 | Isolate the benefit of conditioning |
| Conditional MLP, quarter-hour | 96 output values, reference hidden widths and BCE | Primary adaptation |
| Conditional MLP, hourly | Aggregate four quarters to hourly energy before fitting; output24 | Resolution sensitivity and closer architecture comparison |

Run neural methods with at least five declared seeds, e.g.11/22/33/44/55. The scripts
keep separate run directories and a fixed evaluation sampling seed. An unconditional
model still receives condition IDs at the evaluation interface so conditional failure
can be measured; those IDs are omitted from both networks. Refit the train-only scaler
for the hourly prepared dataset. Compare 24 and96 outputs on a common hourly scale
(sum quarters), as well as reporting their native-resolution metrics. Native interval
metrics with different units per slot must not be compared directly.

Use identical validation selection rules, data, epoch budget, and stopping patience
for comparable GANs; record actual best epochs and computation. The supplied full
configs are starting experimental budgets, not tuned hyperparameters. The smoke
config truncates training observations and runs only3epochs; it is never paper
quality evidence. A more advanced GAN or diffusion model can be added after a
literature-grounded research question motivates it; it must not replace these controls.

## Selection and final evaluation

The trainer minimizes mean pointwise Wasserstein distance in normalized train units
on a fixed validation subset with fixed noise. This is a marginal selection measure,
not proof of temporal fidelity. Inspect validation temporal and diversity measures
before freezing hyperparameters. Report how many candidate settings were tried.
Do not select a seed: report all planned seeds, including failed or collapsed runs.
Do not select checkpoints using the adversarial loss alone.

After freezing choices, evaluate each model's best validation checkpoint on test
once. The CLI defaults to validation; test requires `--split test`. This operational
choice cannot prevent a researcher from repeatedly viewing test results, so maintain
an experiment log and declare any reuse. Never tune to numbers in the final report.

Report overall and per-condition counts plus equally weighted condition means:

- Wasserstein distance for each interval's energy, then average over intervals.
- Wasserstein distance of daily net energy, daily peak interval-average kW, and
  signed within-day power changes in kW (not kW/hour ramp rates).
- Mean-profile RMSE, variance error and diversity ratio.
- Per-profile biased autocorrelation distributions at one slot, one hour and six
  hours where representable; constant profiles have ACF0 and separate counts.
- Per-condition quantile/mean profiles, daily-energy and peak distributions, and
  examples sampled with a declared seed. Do not cherry-pick visually favorable days.
- Negative-output rate for nonnegative import/injection experiments. Signed net
  import can legitimately be negative. Raw outputs are never silently clipped.

Upstream KL on sorted raw values is not a probability-distribution KL and is not
ported. Upstream scalar feature Fréchet distances are not image Inception FID;
we do not label a replacement metric as FID. If a reviewer requests legacy numbers,
define and report them separately with limitations rather than equating them.

## Uncertainty, usefulness and privacy

Days from the same meter are dependent. Use a meter-level cluster bootstrap for
sampling uncertainty, and report variation across training seeds separately. The
provided `meter_bootstrap_confidence_intervals` resamples real meter clusters while
holding the generated set fixed; it does not cover training or generation randomness.
For paired method differences use the SAME resampled meters and matching evaluation
conditions for both methods; a daily IID t-test is inappropriate. Avoid claiming
statistical significance from a single seed or treating all daily profiles as
independent households. Predeclare primary metrics and multiple-comparison handling.

An application claim needs an application experiment. For example, a separately
specified train-on-synthetic/test-on-real forecasting task can quantify utility,
but its forecast horizon, leakage controls, downstream model and baselines must be
fixed first. No downstream utility experiment is currently claimed.

Nearest-neighbor distances and exact-copy fractions are descriptive diagnostics.
The bounded reference subsample can miss memorized records. A privacy claim requires
additional threat-model-specific attacks (e.g. membership inference), appropriate
train/holdout matching, and an explicit formal mechanism for any differential-privacy
claim. This baseline is not differentially private. Do not publish generated profiles
as anonymous merely because a GAN generated them.

## Completion criteria for a manuscript

1. Full data audit and legend interpretation reviewed; data use/license and source
   provenance confirmed for redistribution/publication. Repository release access
   alone does not settle dataset licensing.
2. Full-data training finished for all planned methods/seeds with logs, checkpoints,
   source commit, environment versions, hardware, duration, configuration and hashes.
3. Validation decisions frozen before test; all final conditions/seeds reported,
   including negative results and failures.
4. Baseline comparisons, temporal fidelity/diversity checks, meter-level uncertainty,
   and a predeclared primary hypothesis support the stated conclusion.
5. Claims stay within the observed population, year, conditions and evaluated task;
   no weather, capacity, causal, population-prevalence or privacy claims without evidence.
6. Any methodological novelty is justified against current literature, not inferred
   from applying the published architecture to another dataset.
7. A clean environment reproduces preprocessing and reported results from the pinned
   archive. Full training checkpoints/results are archived separately from source git.

Code tests and a short CPU smoke run satisfy software checks only. Publication
readiness depends on the outstanding experiments and the scientific contribution.
