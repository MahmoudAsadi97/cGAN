# Retained experiment and decision ledger

This ledger distinguishes software checks, development screens and full curated
experiments. Validation has informed repeated method choices; it is development
evidence. The identity-audited TEST partition remains performance-unseen at this
stage. No failed method is removed because another method looks better.

## Original, uncurated preparation: historical development only

The original meter-ID split contained exact annual trace aliases across splits.
These observations motivated development, but they are not independent-trace
validation for the corrected study. Their checkpoints were lost in a workspace
reset; retained reports and configs are preserved rather than presented as fresh
curated runs.

| Stage | Configurations / evidence | Declared execution |
|---|---|---|
| Full-data initial pilot | `configs/pilot.json`; `reports/pilot_full_seed42.json`, `reports/pilot_diagnosis.json` | Seed 42; budget and completion in the retained report |
| Normalization | `screen_minmax`, `screen_standard` | Seed 42, 65,536 training profiles, 120 epochs, batch 512 |
| Discriminator capacity / stabilization | `screen_standard_wide`, `screen_standard_sn` | Same screen budget; width-only and hinge/spectral-normalization package separately declared |
| Conditional-mean residuals | `screen_standard_residual`, `screen_standard_sn_residual` | Same screen budget; two objective-specific comparisons |
| Conditional distribution matching | `screen_conditional_swd0`, `screen_conditional_swd1`, `screen_conditional_swd10` | Same screen budget; balanced batches and conditional checkpoint selection shared across weights |

The nine screen configurations are JSON files in `configs/`. Their retained
results are in `reports/screening_results.json`; the first two also appear in
`reports/normalization_screen.json` and are not additional training replications.
`reports/normalization_paired_uncertainty.json` and `reports/copula_screen.json`
are additional evaluations, not new GAN fits. Protocol decisions and outcomes
are described in `docs/IMPROVEMENT_STUDY.md`.

Early fixed-marginal calibration observations preceded the workspace reset and
the corrected full-data study. Their summarized observations are historical;
missing machine-readable artifacts are not reconstructed as measurements. The
recovery record in `docs/CALIBRATION_STUDY.md` identifies that limitation. This
is an inventory of retained evidence, not a claim that every pre-reset temporary
artifact survives. Smoke runs and synthetic unit-test models are software checks,
not additional scientific training seeds.

## Identity-curated preparation: current development study

All completed runs below use the same 606,060 eligible TRAIN profiles and
meter/trace-disjoint validation membership. Each neural run has 26 epochs and
30,784 updates per adversary; an epoch is a replacement-sampling budget. All
models are selected by the declared raw conditional validation W1. Fixed marginal
calibration is a separate TRAIN-target fitting stage with a generated reference.

| Arm / attempt | Seeds | Status and retained evidence |
|---|---|---|
| Profile SWD weight 10 | 11, 22, 33, 44, 55 | All training, primary calibration, precision and declared calibration-sensitivity reports complete; utility/uncertainty completion is recorded by the collector |
| Profile SWD weight 0 | 11, 22, 33 | All full-budget training and primary calibration reports complete; matched three-seed attribution is preliminary |
| Interrupted weight-zero attempts | 22, 33 | Interrupted after 6 and 5 epochs by resource failure; retained under `runs/interrupted_oom_20261020/`; not accepted as completed runs |
| Derivative-distribution weight 1, profile weight 10 | 11, 22, 33 | Complete; promotion failed. Ramp error improved 8.79% (required 10%); ACF error worsened 15.56% (allowed 10%). No seeds 44/55 are authorized for this arm. |
| Larger generator, profile weight 10 | 11, 22, 33 | Activated after the complete derivative failure; fixed width 128 and latent dimension 96 under `docs/CAPACITY_REFINEMENT.md`. Extend to 44/55 only if every pilot gate passes. |
| Reference conditional BCE96 | 11, 22, 33, 44, 55 | Declared under `docs/REFERENCE_COMPLETION.md`; complete package comparison with the candidate |
| Reference unconditional BCE96 | 11, 22, 33, 44, 55 | Declared matched network-conditioning comparison within the BCE reference method |
| Reference conditional BCE24 | 11, 22, 33, 44, 55 | Declared resolution sensitivity; verified derived hourly development preparation required before fitting |

Completed controls 22/33 restart from initialization with the same seeds and full
budget. The interruption is documented in `reports/resource_incident.json` and
was unrelated to a performance-based stopping or seed-replacement decision.
Completed artifacts are never overwritten to hide failed evaluations; for example,
the partial first plotting directory remains separate from any finished rerun.

The original five-seed SWD10 primary ramp mean is 0.100655 kW, above the fixed
0.10 kW target despite a 28.63% improvement over the matched Gaussian copula.
That failure and other tradeoffs remain in
`docs/CURATED_CALIBRATION_ROBUSTNESS.md`. The larger-sample sensitivity does not
replace the primary evaluation. The derivative pilot was declared after these
observations in `docs/TEMPORAL_REFINEMENT.md`, before its training, with fixed
seeds, budget, weight and promotion criteria. It is exploratory refinement, not
an independent confirmation of the original hypothesis.

The complete derivative assessment is retained in
`docs/CURATED_DERIVATIVE_PILOT.md`; its two failed gates were independently
recomputed from the source reports. The capacity package was prepared as a
contingency before that endpoint decision and activated afterward. It tests a
different structural hypothesis against the original SWD10 comparator. These
sequential validation decisions remain part of the reported selection history.

The hourly preparation and common-hourly physical comparison are implemented in
`docs/HOURLY_IMPLEMENTATION.md`. Their synthetic verification is separate from
actual data derivation, training and endpoint reports. The hourly child preserves
the quarter-hour canonical membership and carries an inherited-audit
qualification; no additional hourly trace-uniqueness claim is made.

## Evidence needed before a manuscript claim

Keep the full source-hashed reports, best/last checkpoints, saved calibration maps,
training logs, attempted-run records and environment alongside compact git reports.
Git contains source and selected reports, not the ignored model archives. The
source commit/tree and exact source-file hashes in each artifact resolve changes
made during development; `reports/provenance_commits.json` documents relevant
content-equivalent Git-data publication mappings.

Official CSV/legend content identity and the current custom Fluvius licence are
confirmed in [DATA_PROVENANCE.md](DATA_PROVENANCE.md). Publication must comply
with its attribution and other reuse conditions. This does not establish privacy,
household independence or methodological novelty.

The reference BCE architecture, unconditional control and common-membership
hourly sensitivity remain manuscript experiments. They are not satisfied by the
weight-zero attribution control or by software tests. A promising current model
would still require the frozen final-test assessment, justified contribution,
and claims confined to this population, year and observed-history utility task.
No TEST result may be used to reopen model tuning
while retaining the same partition as an independent final holdout.
