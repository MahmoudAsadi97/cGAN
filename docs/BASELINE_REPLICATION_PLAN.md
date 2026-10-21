# Fixed-recipe Phase 1 replication

The completed mixture screen supports replication before another method search.
At eight components, the two-draw means improve Gaussian-copula profile SW1,
daily energy, peaks, ramps and lag-1 ACF. Three of 64 selected fits hit the
100-iteration cap. All four component counts and every initialization remain
preserved. The same validation cohort has informed this choice, so the next
runs remain post-test development, not independent confirmation.

## Bounded numerical follow-up and mixture replication

Fix eight components, full covariance, two k-means++ initializations,
normal-score tie handling, tolerance 0.001 and diagonal regularization 0.001.
Increase the maximum EM iterations to **300 for every condition and every
initialization**. Keep this budget regardless of validation ranking and retain
any residual warnings; do not keep extending it until a favorable result.
Already-converged initializations stop early under the same rule. Do not reduce
component count or change regularization solely because one component has two
effective training rows. Report small components and convergence separately.

Fit the original seed **20261027** under the increased cap to diagnose numerical
stability, then five fresh fit seeds **20261101–20261105**. All fits use all
inner-TRAIN rows. The original seed is a numerical follow-up, excluded from the
five-fit replication summary. Each fit preserves its two initialization records
and selects only by final TRAIN log density. This is the same surrogate
normal-score estimator, not copula-likelihood optimization.

The [fit-only plan](../reports/phase1-development/MIXTURE_REPLICATION_PLAN.json)
does not select or score validation data. Save, hash and reload every fitted
artifact before later generation. No bootstrap over random initializations
will be represented as independent household evidence.

## Reference-architecture cGAN replication

Freeze the existing developed training recipe and run five fresh, from-scratch
seeds **301, 401, 501, 601 and 701**, all for **30,000 update pairs**, batch 512.
Use global TRAIN standardization and Adam beta1=0, preserving the existing
reference generator/discriminator architecture, learning rates, BCE loss and
discriminator dropout. These normalization/optimizer choices differ from the
original benchmark; that frozen benchmark and its five original seeds remain
separate. This does not reproduce the author's full original training budget.

The [cGAN plan](../reports/phase1-development/CGAN_REPLICATION_PLAN.json) retains
all five outcomes. Select the earliest minimum normalized profile SW1 among the
saved checkpoints, with the existing fixed 16,384-row inner-VAL selection,
128 projections and generation seed 20261022. Save the last checkpoint and all
learning curves too. A new training seed does not undo selection on these meters.

Later evaluate both raw outputs and the already specified marginal-calibration
combination for each selected model. Fit each mapping using 16,384 independent
generated profiles per condition, calibration seed **20261121**, and the same
full-TRAIN empirical target inverse. Preserve the raw arm; no physical clipping,
calibration on evaluation-batch ranks, or checkpoint reselection is allowed.
Bindings to actual checkpoint/calibration hashes must precede the evaluation.

## Next evaluation, to be bound before execution

Use the [proper-score specification](PROPER_SCORE_EVIDENCE_PLAN.md), including
independent estimator tests and synthetic verification of the meter-bootstrap
algorithm before reporting real-data intervals. The intended ensemble size is
1,024 profiles per condition with generation seeds **20261111, 20261112 and
20261113**; the nested first 512 are a numerical-sensitivity comparison.
Evaluate all eligible inner-VAL days and verify each meter's calendar and all
eight season/day-type bins. Preserve the original screening distances and
dependence/tail diagnostics alongside these different scoring endpoints.

Retain Gaussian, beta, all three Student-t settings and full-day resampling.
Existing two-seed flow models may be contextual comparisons, not five-seed
replications. The primary baseline comparison is the average score of the
five fixed mixture fits versus Gaussian. Compare raw and calibrated cGAN using
the same five training seeds, and compare their fixed-fit averages with the
strong baselines. A later execution plan must enumerate every model binding,
comparison, metric, seed and output before scoring; no favorable-only reporting.

Report fit-seed, generation and meter variation separately. Averaging scores
across fits is not equivalent to concatenating draws into a mixture generator.
Any interval on reused validation meters is postselection descriptive,
conditional on the saved score functions and ignoring selection. Even successful
synthetic coverage checks cannot establish actual new-meter coverage here.

Only the current Fluvius release is available. The study concerns conditional
whole-house signed net-import profiles, not appliance disaggregation,
future-year generalization or privacy guarantees. Phase 2 and federated learning
remain deferred. Replication outcomes cannot guarantee journal acceptance.
