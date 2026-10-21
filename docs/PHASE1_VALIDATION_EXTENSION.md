# Validation of the Phase 1 development extension

This is a prospective methodological plan for the continuation, not a revision
of the frozen benchmark. The original TEST results informed development and
remain a disclosed negative result. The only available release is 2024. Work
can establish useful internally validated behavior with explicit limits; it
cannot create an untouched external confirmation by repartitioning these meters.

This document proposes the later validation stages. Machine-readable experiment
plans identify what has actually been fixed for execution. The initial
[cohort-reference plan](../reports/phase1-development/COHORT_REFERENCE_PLAN.json)
fixes 100 fitting-cohort comparisons at the current 256-profile selection cap.
Its quantiles describe cohort variation; they are not a publication gate or
externally justified accuracy requirement.

## Target and available information

The model generates a 24-hour signed aggregate net-import profile conditional
on reported HP/EV/PV flags, season and weekend. It does not identify heat-pump
type, building type, component-specific consumption, or equipment effects
causally. Missing component labels do not invalidate this aggregate conditional
target. They limit subgroup explanations and the scope of quality claims.
The target is the balanced release's condition-specific daily distribution,
not population equipment prevalence or a coherent synthetic household year.

## Finite stages and locked comparisons

1. Finish the declared global-standardization momentum comparison: beta1
   0/0.5/0.9, seeds 101/202, common 30,000-update budget, unchanged selection
   schedule, fresh-draw diagnostics and all failures retained. The previous
   6,000-update errors and approximately 0.53 diversity do not establish adequate
   quality, and boundary selections leave convergence unresolved.
2. Use inner development evidence to nominate at most one challenger against
   global-standard beta1=0.9. Fix the final budget and checkpoint rule before
   additional seed outcomes. Fit each retained arm at five equal-budget seeds;
   the two existing seeds may count if their complete trajectories/settings
   match exactly, so this needs at most six new 30,000-update runs. Do not pool
   the 6,000-update outcomes with longer-budget outcomes.
3. Before reading a locked audit, generate the TRAIN-only adequacy references
   below and freeze all endpoint/margin/aggregation settings. Select one model
   procedure using only inner development data. Fit five seeds on original
   TRAIN, with original VAL used for the audit only, never checkpoint selection.
   Checkpoints can use a fixed update count chosen from inner development, or
   use an inner split inside TRAIN under the declared selection procedure.
   Baselines must consume the same fitting data as the corresponding model.
4. If resources permit, perform the small internal robustness study below.
   Do not require new data or revisit original TEST to complete these stages.

Use measured update/row-presentation and elapsed-time budgets rather than epoch
comparisons across batch sizes. Estimate the cost from the ongoing runs before
each stage. For equal architecture, stage 2 adds at most 180,000 update pairs;
the five-seed original-VAL audit adds 150,000. One further extension is defensible
if a prespecified plateau/boundary rule triggers it, but it must apply at the
same budget to compared arms and be logged before new outcomes. Endless budget
extension until a good seed appears is not a validation strategy.

## Adequacy margins from real-meter variability

Practical adequacy is a distinct question from beating another algorithm.
Do not pick an error percentage because the eventual winner happens to pass it.
Instead, define a transparent operational reference: disagreement between two
independent meter cohorts of the intended evaluation size under this calendar.

Before final seed/audit results, fix 100 equipment-stratified disjoint pairs of
meter subsets drawn from the 1,333 inner-fit meters. Each side should have the
same per-equipment meter counts as the 332-meter inner validation cohort. For
the later 353-meter original-VAL audit, form a separate reference matched to
that cohort's declared equipment counts. Each paired cohort retains all days;
the two sides of a pair share no meter. Different repetitions may overlap, so
the 100 values are reference draws, not 100 independent experiments.

For every declared error endpoint calculate real-versus-real discrepancy,
using the identical physical units, condition weighting, directions, profile
counts and sampling rule used for model assessment. Predeclare the 90th
percentile of these discrepancies as an operational tolerance. The choice of
90% describes coverage of ordinary cohort variation, not 90% accuracy or a
calibrated confidence level. Report its median, 10/90% spread, full definition,
and the effect of using the median tolerance. This quantile is a transparent
design choice, not an externally validated application requirement.

Use all eligible daily profiles for the main audit or a fixed adequate cap if
compute requires it; calibrate the reference at that exact cap. The current
256-day-per-condition checkpoint sample has an elevated sampling floor and
must not be compared directly with all-row baseline numbers. Equal numbers of
synthetic days do not reproduce real within-meter dependence, so the reference
is an operational cohort-variation yardstick, not an exact null distribution
for a fitted generator. Shared calendar/weather also restrict its scope.

For SW1, daily-energy W1, peak W1, ramp W1, ACF errors and mean-profile RMSE,
report the model error divided by its matching real-real tolerance. Where a
tolerance is zero or numerically negligible, retain physical values and mark
the ratio undefined. Do not alter denominators after seeing model results.
Use TRAIN-only references for variance/diversity as well: compare absolute
log variance ratios, which treat doubling and halving symmetrically, rather
than rewarding a macro ratio near one despite offsetting condition failures.

An operational adequacy claim requires the primary macro profile error and
the declared macro guardrails to be within these tolerances for at least four
of five equal-budget seeds, while showing all five outcomes. Also require
each of the eight equipment groups to pass primary and energy tolerances in
at least four seeds; otherwise qualify the claim by the failed groups.
Condition-level failures, tail/quantile curves, and the worst-condition results
remain visible. Four-of-five is a declared robustness rule, not a confidence
statement. An unconditional global claim must not hide failed subgroups.

Keep resampling, copula and mean control results beside these margins. A model
may be adequate at the stated cohort scale while copula is equally good or
better. That supports adequacy, not superiority. If it loses materially on
these diagnostics, a lower neural SW1 than the original failed reference is
insufficient. A comparative advantage needs evidence on a separately declared
criterion such as controllability or a stated downstream task; neural training
and lack of exact copies alone do not constitute a scientific advantage.

## Internal replication without manufacturing a new holdout

Original VAL has prior development use; a single locked audit reduces further
adaptation but does not erase that history. If it informs another technique,
label it development evidence and do not present later VAL scores as untouched.

A workable stronger check is three equipment-stratified outer meter folds
inside original TRAIN. Fix the method, budget and any checkpoint-selection
algorithm before seeing any outer results. For each outer fold, fit preprocessing
and baselines on the other meters and select checkpoints only using an inner
meter subset. Evaluate the excluded fold once. Run the chosen model at three
fixed seeds per fold: nine additional runs, at most 270,000 update pairs at
30,000 each. Preserve the separate five-seed main audit rather than claiming
that overlapping folds yield nine independent household datasets.

This small design evaluates robustness of a locked learning procedure across
meter partitions. It is not full nested hyperparameter selection. A claim about
the performance of the entire search procedure would require rerunning that
selection inside each outer fold. All these meters and the 2024 setting have
historical exposure, so even this result remains post-test internal evidence.
Do not compute a naive t interval across three overlapping-training folds.

## Distances, bootstrap and distribution tests

There is no general unbiased correction for empirical W1. Its positive
finite-sample floor, absolute-CDF nonsmoothness, near-equality behavior and
cluster dependence make ordinary percentile-bootstrap coverage uncertain.
Recentering the bootstrap or switching to BCa after seeing a favorable result
does not establish coverage. Preserve the old intervals and their caveat.

Continue to report paired whole-meter bootstrap sensitivity summaries for
fixed fitted models, with training-seed and generation variability separate.
Use identical real-meter draws across methods. Call these conditional
sensitivity intervals unless coverage has been validated for the stated
estimand. They omit training uncertainty and future-calendar/weather variation.
Five seeds estimate algorithmic variation only coarsely; they are not five
independent Fluvius cohorts. No day-level resampling or inference treating
120,848 days as independent homes is acceptable.

Energy distance or a characteristic-kernel MMD can supplement SW1 because they
have unbiased U-statistic forms under independent sampling; this does not make
an ordinary implementation valid for clustered daily rows. For the random-meter,
random-condition-day target, real-real terms must average across distinct
meters, excluding all same-meter profile pairs. Real-fake and fake-fake terms
must use the corresponding independently generated samples and proper diagonal
exclusions. A finite unbiased estimate may be negative; do not clip it or
interpret a negative value as better-than-perfect fidelity.
The unbiasedness claim is conditional on independent sampled meters and the
fixed calendar, with finite required moments; shared feeder behavior or residual
near-duplicates can violate that assumption and must remain a limitation.

A practical diagnostic is conditional energy distance on the physical 24-vector,
macro-averaged across the 64 conditions. Use fixed Monte Carlo cross-meter
profile pairs (for example 50,000 per condition) and independently drawn
real-fake/fake-fake pairs, shared across methods where applicable. Report
Monte Carlo variation separately and choose pair counts by a prespecified
precision target. Alternatively, bounded Gaussian-kernel mean embeddings with
TRAIN-fixed bandwidths admit efficient cross-meter U-statistics; finite random
Fourier features approximate the kernel and introduce another declared error.
These should be secondary diagnostics, never a replacement selected because
their ranking is favorable.

An off-the-shelf daily permutation test is invalid here. Permuting whole real
meters against artificial collections of independently generated days is also
not automatically valid: equality of daily marginals does not make those
histories exchangeable, since the generator does not model household history.
Correct cluster multiplier/bootstrap inference for the appropriate potentially
degenerate U-statistic needs methodological implementation and clustered-null
simulation checks. Unless that work is done, report energy/MMD effect sizes
and cluster sensitivity without calibrated significance claims. Non-rejection
by a two-sample test does not establish equivalence or adequacy, and rejection
with many observations need not imply a practically important error.

## Claim boundaries

A defensible positive statement is narrow: at the stated sample/calendar scale,
the declared conditional daily-profile diagnostics meet prespecified empirical
cohort-variation tolerances across seeds and internally excluded meters, with
specific remaining failures reported. This requires the results to pass; it
cannot be guaranteed prospectively. It supports a technically valid aggregate
generator foundation even if a simpler copula is competitive, while superiority,
novelty and application value remain separate questions.

Do not claim appliance-level validity, heat-pump-type validity, causal effects,
future-year generalization, household-year coherence, or privacy. A rigorous
negative or competitive-baseline result can support research, but a promise of
a publishable positive result would not be evidence-based. The available 2024
data permit continued internal model assessment without making external data
an unmet prerequisite.
