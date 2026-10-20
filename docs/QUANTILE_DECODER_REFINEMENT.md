# Quantile-decoder refinement: activated exploratory protocol

Protocol identifier: `conditional_physical_quantile_decoder_v1`.

Activated on 2026-10-20 after the complete three-seed capacity pilot failed its
fixed promotion rule. This activation follows the user's standing instruction
to continue improving and validating the model. It fixes one new exploratory
package before any real decoder fit, training or endpoint evaluation. Publication,
verified numerical source and the execution declaration below are prerequisites
to a launch. This document reports no decoder results and authorizes no TEST
access.

## Activation evidence and retained failures

The completed [capacity assessment](../runs/capacity_pilot.json), SHA256
`477d62a38e6b536a8d844388abf81e768cb8da0f4013d66389e4970072742c29`, reports mean
calibrated ramp W1 **0.129828606 kW**, against **0.098102228 kW** for the matched
original SWD10 seeds: a **32.34% increase**. Ramp error worsens in each paired
seed. Both ramp promotion requirements therefore fail. Daily-energy, peak,
lag-one ACF and projected-profile mean guards, and each seed's real/resampling
utility tolerances, pass; those outcomes do not override the failed conjunction.
The capacity arm is retained as a failed completed screen, without seeds 44 or
55. The independent assessment's source-path discrepancy was resolved against
its declared pinned assessor, with the correction retained.

Retain the preceding [derivative failure](CURATED_DERIVATIVE_PILOT.md), the
capacity attempts, and the [original completed validation](CURATED_COMPLETED_VALIDATION.md).
The original five-seed mean ramp W1 of 0.100655 kW still exceeds its 0.10 kW
absolute practical gate. None of these outcomes is relabeled as success.

The underlying contingency, `runs/proposed_dependence_refinement.md`, was
prepared before capacity endpoint inspection. Its hypothesis is that training
through the same fixed physical map used at inference may reduce a mismatch in
the present posthoc-calibrated pipeline. A coordinatewise nonlinear map can
change physical ramps, totals, autocorrelations and battery trajectories even
when an idealized strictly increasing map preserves ranks. This is motivation
for an experiment, not a diagnosis proving why either earlier arm failed.

## One frozen package

Use [curated_quantile_decoder.json](../configs/curated_quantile_decoder.json).
The substantive intervention is a fixed conditional TRAIN quantile decoder in
the generator forward path. Return to the original generator architecture; do
not combine this package with the failed larger-capacity arm.

| Setting | Fixed value |
|---|---|
| Generator hidden width / latent dimension | 64 / 24 |
| Output / condition dimensions | 96 signed quarter-hour energies / 64 conditions |
| Discriminator | Width 128, spectral-normalized hinge loss |
| Generator objective | Existing hinge objective + 10 times conditional profile SW2-squared |
| Profile projections | 128 fresh directions under the existing independent RNG stream |
| Derivative SWD / conditional mean residual | 0 / disabled |
| Adam learning rates / betas | G=0.0001, D=0.0002 / (0.5, 0.999) |
| Batch | 512: four distinct conditions, 128 sampled rows per condition |
| Decoder knots | Fixed 1,025 uniform probability knots; no tunable knot-count setting |
| Normalization | Existing scalar global TRAIN population mean/std |
| Device / numerical threads | CPU / one per job |
| First-stage seeds | 11, 22, 33, all from initialization |

The generator retains 28,512 trainable parameters and the discriminator 185,729.
The decoder adds fixed buffers, not trainable parameters or independent latent
variables. No new temperature, initialization rule, rank loss, marginal loss,
gradient clipping, derivative penalty, optimizer change or additional
discriminator is included.

Use all **606,060** profiles in the identity-curated TRAIN preparation, with the
same labels, row order, meter split, prepared-file hashes and curation manifest
as the original comparator. `max_train_samples` and `patience` remain null.
Fit the fixed decoder once before optimization in each run, using exactly the
actual selected TRAIN rows, condition by condition. The full-pool experiment
must retain all 64 conditions. The same inputs must produce the same decoder
table hash across seeds; never substitute a pooled map for an unsupported
condition. No validation or TEST values fit the decoder, normalization or
statistical controls.

## Forward arithmetic and deployed output

For each condition and interval, form empirical linear quantiles from sorted
signed physical TRAIN energies, using float64 fitting arithmetic and the fixed
1,025-point probability grid. Preserve the observed endpoints, tied values and
constant margins. Do not winsorize, smooth, jitter, truncate tails or alter the
grid after inspecting outputs.

The numerical order is fixed:

1. The existing float32 generator returns unconstrained logits.
2. Cast logits to float64, apply sigmoid, and interpolate the fixed physical
   float64 quantiles with versioned signed arithmetic.
3. Cast the decoded physical profile to float32 once.
4. For discriminator, SWD and checkpoint selection, cast that physical float32
   profile to float64, subtract the saved global TRAIN mean, divide by its
   saved effective scale, then cast to float32. The effective scale is std,
   with the existing scale-one convention when std is zero.
5. Inference returns the same direct physical float32 result from step 3.
   It does not invert a rounded standardized output through another affine
   roundtrip.

Thus the candidate's assessed and deployed output is its **direct decoded
profile**. There is no candidate generated-CDF fit, posthoc calibration map,
evaluation-batch ranking or fabricated identity-map artifact. The original
comparator remains its frozen calibrated pipeline. Do not select between raw,
decoded or additionally calibrated candidate variants after seeing endpoints.

The interpolator gathers only the lower and upper scalar knots needed for each
batch row and interval. It must never materialize an N-by-1,025-by-96 table by
indexing the entire condition table with batch labels. Fixed table storage is
64-by-1,025-by-96 float64 values, about 48.05 MiB, excluding copies and runtime
overhead. Fitting holds only one condition's float64 observations at a time;
inference and probability diagnostics use bounded batches.

Same-sign open segments use the fixed left anchor `a + w * (b - a)`;
crossing-sign segments use `(1 - w) * a + w * b`. At the exact right endpoint,
use `b - (1 - w) * (b - a)` to retain both the exact knot and its segment
derivative. There is no midpoint anchor switch or physical-output clipping.
This replaces a demonstrated one-ULP undershoot from right anchoring a negative
segment at a tiny positive weight. The left anchor trades some relative
accuracy near right-endpoint cancellation for stable support boundaries;
rounding and endpoint tests retain the observed counterexample explicitly.

Repeated values create genuinely flat segments with zero interior gradient.
At probability zero and one, choose the interior one-sided segment derivative;
strictly outside probability support the bounding operation has zero
derivative. These are explicit conventions at nonsmooth boundaries, not unique
two-sided derivatives. The synthetic prototype identified a zero derivative at
exactly one in the installed `torch.clamp`; production must preserve the declared
one-sided convention explicitly. Saturated sigmoid logits still have zero chain
gradient. No epsilon slopes, surrogate ranks or detached endpoint replacements
may hide plateaus or create a different objective.

Require finite-gradient checks away from knots, ties, constant and signed
margins, exact endpoints, large finite values, selected-TRAIN leakage sentinels,
strict state reconstruction, normalization/inference parity and deterministic
replay. Near-knot numerical tests are evidence for tested cases, not a universal
floating-point monotonicity proof. Old checkpoints must remain generatable.
Nonfinite values, unsupported conditions and incompatible state fail explicitly.

Sigmoid probabilities need not be uniform, so this construction does **not**
guarantee empirical marginals or identify a pure dependence model. Its bounded
TRAIN support cannot establish extrapolation to unseen tails. Quantile
plateaus, steep segments, sigmoid saturation and the retained 24-dimensional
latent input may still impair learning. This is a conditional quantile
parameterization tested as one package, not a new generative-model claim.

## Budget and unchanged checkpoint-selector function

Every seed completes **26 epochs**, with **1,184** batches per epoch,
**30,784** updates per adversary and **15,761,408** sampled profiles. These are
replacement-sampling budgets, not 26 unique TRAIN traversals. Retain the
existing uniform choice of four distinct supported conditions and independent
within-condition sampling with replacement. Preserve the sampler and projection
RNG meanings and seed offsets. The decoder fit itself uses no random draws.

Retain the conditional interval-W1 selector on the fixed **4,096** validation
profiles, 64 per condition, its existing per-seed real-row and noise rules, and
earliest-best strict-improvement tie handling. The score remains the equally
weighted mean of condition-wise interval marginal W1 in normalized coordinates.
For this arm it sees decoded profiles; the original selector saw raw profiles
before later calibration. Disclose this representation difference. The screen
therefore estimates a training/inference package under the same selector
function, not an isolated causal effect with identical representation semantics.
Do not replace the selector with ramp, battery, utility or a composite score.

There is no early stopping, performance-based restart, replacement seed or
warm-start. Preserve every attempted seed and its complete history. Any allowed
infrastructure recovery must retain the failed attempt and restart that same
seed from initialization with the full budget. Equal updates do not imply equal
compute; record fit/training wall time, peak memory and chosen epochs. Shared
seed labels alone do not establish identical model draws or numerical source.

## Frozen original artifacts and common corrected evaluation

The exact original SWD10 comparators are bound in
[quantile_decoder_baseline_bindings.json](../configs/quantile_decoder_baseline_bindings.json),
SHA256 `046ddf6d3ba66ac16507539c494422c9bc322505da00bb9a35e848dc4223f9e9`.
That file preserves each seed's checkpoint, map, configuration, selected epoch,
development-data hashes and historical training-source provenance. It is fixed
before any decoder fit or endpoint.

| Original seed | Checkpoint SHA256 | Frozen calibration SHA256 |
|---|---|---|
| 11 | `12159e0495b1a6a53e5814eebd1cff54ad50d53a259fdbe3e64cbc4d6e703a4b` | `74fd16d428d98e93f6bf1fa6469cc0638d59da979a77991fc5571d84da91aca5` |
| 22 | `71600382449bdd109a1f05ce2898c4eac02b17d155d7e4363d863b7fefb23765` | `7ce4fc08449755f4e653783bb2f3760ca0bfb74403e6ca74cd71d19ee6a336e3` |
| 33 | `fd866bc7e2b032997576609575a953750f6d5a5af13448d8518476e213bdc1aa` | `62794b732cd2dd3ea09db7b8d3738e02871e42cad8d310785c46441a8be73dde` |

Reevaluate these exact original raw and frozen-map outputs alongside the direct
decoder candidate under **one verified corrected evaluation source**. Retain
the original maps and their original fit provenance; do not refit them or change
their reference seed, sample size or knots. The original map settings remain
seed 7301, 4,096 generated reference profiles per condition and 1,025 knots.
Those settings describe the original comparator only. The decoder candidate
has no such generated-reference fitting step.

Historical original, derivative and capacity reports remain historical evidence.
Do not mix their older interpolation-source outputs into the new common-source
comparison, or assume a synthetic roundoff correction leaves actual-study
results unchanged. Preserve historical training-source differences explicitly.
The evaluation/control code, real validation selections, statistical-control
arrays and utility task must match across the newly compared arms. All baseline
artifact hashes must match the frozen binding file; a matching seed label is
insufficient.

## Fixed pilot screen and complete reporting

Evaluate all three declared seeds before a promotion decision. Compare the
direct decoded candidate to the matching **original calibrated SWD10** outputs
from the common corrected evaluator. Keep the existing conjunction:

1. Candidate mean ramp W1 is at most **90%** of the original three-seed mean,
   and candidate ramp W1 is **strictly lower in each paired seed**.
2. Candidate mean daily-energy, peak, lag-one ACF and fixed projected-profile
   W1 are each at most **110%** of their corresponding original means.
3. In **each** seed, candidate synthetic-trained macro RMSE is at most **110%**
   of both the same-draw real-trained and resampling-trained references.

No threshold, comparator, metric definition or conjunction is weakened. Missing
or incompatible evidence is incomplete assessment, not success. Retain all 64
conditions, seed values, means, sample SDs, ranges and paired differences. Also
report MAE, physical marginal errors, joint diagnostics, diversity, support,
bounded nearest-neighbor diagnostics and adverse resampling comparisons.

Use the unchanged primary **100-profile-per-condition cap**, generation/evaluation
seed **2026**, all 64 conditions with equal macro weights, and identical real
validation selections. Preserve conditional resampling, TRAIN conditional means,
Gaussian copula with shrinkage .01, independent marginals with shrinkage 1 and
the separately identified exact-order-statistic copula sensitivity. Primary
quantile-based controls use the same 1,025-knot convention. Preserve the 128
unit-direction projected-profile diagnostic at seed **9182**.

Probability diagnostics reuse the candidate's same cap-100, seed-2026 draws.
Report exact zero/one probabilities, fixed near-saturation fractions
`u <= 1e-6` and `u >= 1-1e-6`, physical marginal errors and sampled tied-output
frequencies. Label fixed-table flat-segment fractions as fractions of the
uniform probability grid, not learned atom mass. Probability uniformity and
bounded support are diagnostics, not additional promotion gates. No separate
4,096-draw candidate CDF fit or diagnostic reference is required.

Keep the hypothetical additional-battery diagnostic: 10 kWh capacity, 4 kW
charge/discharge limits, efficiencies sqrt(.9) each, empty initial SOC for each
24-hour episode, charging only from observed net export and discharging against
import. Report leftover SOC without future-day credit. This does not identify
gross PV production, existing batteries, annual performance, costs or real
hardware policy. Battery results remain mandatory secondary outcomes, never an
alternative favorable route through the primary gates.

Keep the observed-history one-step ridge task: 12 prior quarter-hour intervals,
clock and condition terms, intercept, alpha 1, all 606,060 real or generated
TRAIN profiles with matched labels, and the fixed 6,400 validation profiles.
There is no cap on the downstream training pool. Preserve RMSE and MAE and the
real/resampling references. The 10% tolerance is descriptive, not formal
noninferiority; physical confidence intervals do not establish utility
uncertainty.

Only a separately recorded promotion after all three rules pass may add seeds
**44 and 55**, retaining the first three runs and the identical package and full
budget. A failed screen retains this package as failed and authorizes no
post-result temperature, knot, architecture, loss, optimizer or selector sweep.

Any later full-five-seed assessment retains the original final criteria: mean
ramp W1 at least **20% below** the matched Gaussian control, lower ramp error in
at least **four of five** seeds, daily-energy W1 **<=5 kWh**, peak W1 **<=1 kW**,
ramp W1 **<=0.10 kW** and lag-one ACF W1 **<=0.10**, together with the retained
utility requirements and complete tradeoffs. Passing this relative three-seed
screen does not certify those absolute criteria.

The pilot decision uses the fixed point metrics above; there is no added or
removed CI gate. For any later full-five-seed assessment, retain separate
per-seed paired technology-stratified whole-meter bootstrap comparisons with
500 replicates under the existing uncertainty scope. Never
average CI endpoints or present seed SD as a confidence interval. Fixed model
selection, fitted maps/decoder, generated draws and downstream fits are outside
fixed-sample interval scope. The cap-1,000 precision check remains separate from
the primary cap-100 results for every promoted seed. Candidate calibration-seed
or generated-CDF-size sensitivities are inapplicable because there is no
candidate generated-CDF fit; do not invent a posthoc map to create them. Retain
the original comparator's recorded calibration sensitivities as qualified
historical evidence, not substitutes for this frozen common-source baseline.

## State, source and execution prerequisites

Checkpoints must retain fixed physical tables, support masks and training
counts; the probability-grid, quantile-value, selected-profile, label, index,
support and count hashes; the global scaler; prepared-file and curation-manifest
hashes; full source hashes and runtime provenance. The decoder's fit metadata,
checkpoint metadata and provenance must agree. Reconstruction validates the
saved state and declared actual-TRAIN binding without refitting. No TEST values
or freshly generated CDFs are used to reconstruct an inference model.

Publish this protocol and config, complete synthetic source verification, then
write a mandatory execution declaration **before the first real decoder fit**.
It binds the clean numerical `source_commit`, `source_tree` and
`source_files_sha256`; `config_sha256`, `protocol_sha256` and
`baseline_bindings_sha256`; exact prepared/checkpoint development-data hashes;
seeds 11/22/33; 606,060 TRAIN profiles, 26 epochs, 1,184 updates per epoch and
30,784 updates per adversary; evaluation seed 2026 and cap 100;
`candidate_posthoc_calibration: false`; and
`baseline_calibration: frozen_original_saved_map`. Any calibration seed/sample/
knot fields in that declaration describe the **original comparator** only.
Retain publication and numerical-source pins, config copies, source verification
and phase receipts. Reject a missing, dirty or mismatched execution source and
incompatible evidence rather than silently merging it.

The tracked `scripts/run_quantile_decoder_pilot.py` controls a guarded
training phase. The existing 15 reference fits must finish before the first
real decoder fit. The **entire decoder phase holds
`runs/heavy_evaluation.lock`**, excluding heavy evaluations, including native
resolution precision work. It writes the execution declaration and frozen
per-seed config copies before fitting. Seed **11 runs alone for all 26 epochs**.
Then seeds 22 and 33 run in parallel only when
`2 * measured_child_peak_RSS(seed11) + 1.5 GiB <= 8 GiB`; otherwise they run
sequentially. Convert and record RSS units explicitly, with GiB equal to
2^30 bytes. The scheduling decision uses memory alone, not endpoint performance;
all seeds, numerical settings and budgets remain fixed. Record actual scheduling,
memory and interruptions. Heavy evaluations resume after the phase releases the
shared lock. Creating this document or the inactive queue does not launch a job.

No real decoder fitting, training, evaluation or TEST access occurs as part of
drafting or synthetic verification. TEST performance remains outside this
refinement. A later TEST run requires the separately frozen method, all required
artifacts and sealed harness; it cannot refit ranks, tune a map, select a
checkpoint or alter a loss after seeing TEST outcomes.

This is another adaptive refinement on already reused validation data. Preserve
the complete search and failure history; eventual validation success would not
be fresh confirmation or external preregistration. Exact-trace curation does not
prove independent households or eliminate near-duplicates. Neither this
parameterization nor a favorable screen establishes novelty, privacy, external
validity or publication readiness. The broader reference, unconditional,
hourly-resolution and application requirements remain in force.
