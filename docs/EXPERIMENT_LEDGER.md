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
| Larger generator, profile weight 10 | 11, 22, 33 | Complete; promotion failed. Mean ramp error increased 32.34% and every paired ramp error worsened. Other mean gates and utility tolerances pass. No seeds 44/55; see `docs/CURATED_CAPACITY_PILOT.md`. |
| Fixed conditional TRAIN quantile decoder, profile weight 10 | 11, 22, 33 | Activated after complete capacity failure; implementation and synthetic verification before any fit. Same fixed gates; no generated-CDF posthoc map. See `docs/QUANTILE_DECODER_REFINEMENT.md`. |
| Reference conditional BCE96 | 11, 22, 33, 44, 55 | All five full-budget fits complete; validation reports are being completed under `docs/REFERENCE_COMPLETION.md` |
| Reference unconditional BCE96 | 11, 22, 33, 44, 55 | All five full-budget fits complete; matched network-conditioning validation remains in progress |
| Reference conditional BCE24 | 11, 22, 33, 44, 55 | All five full-budget fits complete; hourly TRAIN/VAL derivation verified at clean source ec10d83. Evaluations are serialized |

The [reference training receipt](../reports/reference_training_completion.json)
retains all fifteen full budgets, selected epochs, source pins and artifact
hashes. It does not substitute for the pending complete reference validation.

Completed controls 22/33 restart from initialization with the same seeds and full
budget. The interruption is documented in `reports/resource_incident.json` and
was unrelated to a performance-based stopping or seed-replacement decision.
A later synthetic frozen-study verification was killed under memory pressure;
`reports/resource_incident_synthetic_20261020.json` records the incomplete
software check and resource-aware retry. Actual research training continued.
The native BCE96 seed11 cap-1,000 calibration evaluation later exited with
signal9 and no saved report while training remained active. Its failed receipt
is retained in `reports/resource_incident_reference_precision_20261020.json`.
The same evaluation is deferred until the current reference training phase completes.
Any later decoder training phase must hold the evaluation lock throughout,
excluding heavy evaluations. Completed primary
reports, model checkpoints, sample counts and random seeds are unchanged.
Completed artifacts are never overwritten to hide failed evaluations; for example,
the partial first plotting directory remains separate from any finished rerun.

The execution service disconnected around 16:47 UTC on 2026-10-20. The
[recovery receipt](../reports/reference_service_disconnection_recovery.json)
records 23 completed reference evaluations and the fixed 47-job remainder.
Native BCE96 seed55 calibration had started but saved neither report nor map.
The cause is undetermined; this event is distinct from the earlier signal9
precision failure. Recovery preserves completed outputs, source pins and exact
commands, with historical hashes distinguished from hashes verified only at
recovery. The receipt records a launch, not completion of the remaining study.

The recovered hourly seed33 map subsequently failed its binding check and ZIP
integrity. Its unchanged report retains the original expected map hash; the
damaged bytes and execution evidence are preserved in
`reports/reference_recovery_map_binding_failure.json`. No dependent utility job
ran with that map. Reconstruction used the original source and TRAIN inputs;
the [restoration receipt](../reports/reference_recovery_exact_map_restoration.json)
confirms exact original-file SHA-256 equality before canonical restoration.
Physical and utility metrics were not recomputed. The cause is undetermined
and is not attributed to the service disconnection.

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
different structural hypothesis against the original SWD10 comparator. Its
complete screen failed both ramp criteria. An independent audit initially compared
the collector hash to a later checkout; the retained correction verifies the
actual clean assessor pin and resolves that source-location mistake without
changing endpoints or the failed decision.

The quantile-decoder contingency was prepared before capacity endpoints were
read and activated only after that complete failure. It is a new exploratory
training/inference package, using physical TRAIN quantiles within the generator
and no later marginal correction. Original checkpoint/map identities are fixed
in `configs/quantile_decoder_baseline_bindings.json`. Both sides will be evaluated
with the same corrected numerical implementation; the old comparisons remain
unchanged. These sequential validation decisions remain part of the selection history.

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
