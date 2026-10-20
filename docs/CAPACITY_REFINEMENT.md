# Capacity refinement: activated exploratory protocol

Activated on 2026-10-20 after the complete derivative pilot failed its preset
promotion rule. This is a new prospective validation refinement, declared before
capacity training. Activation follows the user's standing instruction to continue
improving and validating the model; it is an implementation decision, not a
separate user review of this experiment. This document reports no capacity
results. TEST performance remains unseen.

## Activation evidence and independent audit

The completed assessment is [temporal_pilot.json](../runs/temporal_pilot.json),
with its [readable report](../runs/temporal_pilot.md), under the prior
[TEMPORAL_REFINEMENT.md](TEMPORAL_REFINEMENT.md). The assessment JSON SHA256 is
`1b7748e5f3c7f78ad2a7b3b27e91f3b8539229994fb9d48ca2ec7ac7d27b1a9c`.
The retained [independent review](../reports/temporal_pilot_independent_review.json)
(SHA256 `714d2c3e84e988c38357aaba997e5b68b5f3ddae74ae28716d226c61ce08f48c`)
passed 150 JSON-only binding/arithmetic checks; a separate reviewer recomputed
all 64-condition macro means from the six source endpoint reports. Neither
review accessed arrays, checkpoints or TEST. All six runs completed 26 epochs
and 30,784 updates per adversary; this is a completed screen failure.

| Promotion metric | Derivative mean | Original matched-three mean | Required limit | Outcome |
|---|---:|---:|---:|---|
| Ramp W1 (kW) | 0.08947579045 | 0.09810222823 | <=0.08829200541 | Fail: 8.7933% improvement; at least 10% required |
| Daily-energy W1 (kWh) | 3.29191008560 | 3.24854026180 | <=3.57339428798 | Pass |
| Peak W1 (kW) | 0.78435047492 | 0.78644682304 | <=0.86509150534 | Pass |
| Lag-one ACF W1 | 0.10036237647 | 0.08685197541 | <=0.09553717295 | Fail: 15.5557% increase; at most 10% allowed |
| Projected-profile W1 | 0.06553391041 | 0.06544658709 | <=0.07199124580 | Pass |

All three paired ramp directions improve, and all three calibrated forecasting
RMSE values satisfy both reference tolerances. These do not override the two
failed aggregate gates. ACF error worsens in every seed. The derivative arm is
retained as a failed three-seed refinement; seeds 44 and 55 are not added under
that promotion protocol. The original five-seed model's mean ramp W1
0.100655 kW still exceeds the original 0.10 kW practical gate. Neither outcome
is relabeled as success.

The contingency was prepared before derivative endpoints were inspected in
`runs/proposed_capacity_refinement.md`. The present activation retains its
package, comparator and promotion rules. It tests a capacity hypothesis after
the derivative-only intervention failed, without attributing that failure to
capacity or claiming that capacity will solve it.

## Question and rationale

Ask whether one larger generator package improves the existing calibrated
pipeline under the same training-update budget and evaluation procedure.

The current generator ends in a 96-by-64 linear map. With the conditional-mean
residual option off, its raw outputs lie in a shared affine subspace of dimension
at most 64. For a fixed condition, the 24-dimensional latent input also bounds
the local input-output rank by 24. Increasing generator width to 128 and latent
dimension to 96 removes both enforced bounds: hidden widths become
128 -> 256 -> 128, followed by 96 outputs. This is a plausible capacity
hypothesis, not a demonstrated cause of the original model's temporal errors.
Learned weights need not achieve the maximum available rank.

The fixed coordinatewise calibration is nonlinear and can already remove the
raw affine-subspace restriction. It does not add independent latent variables.
Thus neither raw rank arguments nor improved raw marginal fit establish better
calibrated temporal dependence. The original calibrated model's ramp, daily
energy, projected-profile, battery and resampling tradeoffs remain relevant.
Increasing width/noise dimension is an established modeling choice; no novel
method is claimed.

## One frozen package

Use [curated_capacity.json](../configs/curated_capacity.json), derived from
[curated_swd10.json](../configs/curated_swd10.json) with only these two substantive
changes and one explicit zero:

| Setting | Original SWD10 | Capacity package |
|---|---:|---:|
| Generator hidden width | 64 | 128 |
| Latent dimension | 24 | 96 |
| Discriminator hidden width | 128 | 128, explicitly retained |
| Profile SW2-squared weight | 10 | 10 |
| Derivative SW2-squared weight | 0 | 0, explicitly retained |
| Generator trainable parameters | 28,512 | 98,912 |
| Discriminator trainable parameters | 185,729 | 185,729 |

Parameter counts include all linear weights and biases for 96 output intervals
and 64 condition indicators. Spectral-normalization buffers are not trainable
parameters. Generator parameter count rises about 3.47-fold; this is not a claim
that total training time rises by that factor.

Retain standard normalization fitted on actual TRAIN data, conditional labels,
unbounded linear generator output, residual means off, hinge loss with spectral
normalization, Adam learning rates G=1e-4 and D=2e-4, betas (0.5, 0.999), batch
512, four conditions with 128 rows each, and 128 profile projections. Use CPU,
one numerical thread per job, and the same sampler, loss definitions and data.
No warm-start, clipping, optimizer change, derivative term or other architecture
change is included.

Use all 606,060 actual TRAIN profiles from the identity-curated preparation
(`max_train_samples: null`), with its existing hashes and meter split. Fit all
statistical controls and calibration target marginals on that same verified
training pool. No validation or TEST values fit any transform.

## Budget, random streams and checkpoint selection

Every declared seed starts from initialization and completes 26 epochs:
1,184 updates per epoch, 30,784 updates per adversary, and 15,761,408 sampled
profiles. An epoch is the existing replacement-sampling budget, not a unique
traversal of the 606,060 available rows. There is no early stopping, resume,
performance-based restart, or replacement seed. Preserve infrastructure-failed
attempts separately; any permitted recovery restarts the same seed from scratch
with the same full budget and records the reason.

Equal updates and training exposure do not mean equal compute. Record wall time,
parameter counts and resource use. A wider generator changes optimization and
adversarial balance; retaining the original learning rates is part of this
package. A failed screen would concern this package, budget and selection policy,
not prove that additional capacity cannot help.

Keep the current raw conditional interval-W1 checkpoint selector on the fixed
4,096 validation profiles, with the existing seed rules and earliest-best-epoch
tie handling. Do not select among calibrated ramp, battery or utility outcomes.
Record the chosen epoch and full history. Completed budgets are equal, while
selected-checkpoint prefix budgets may differ. Raw marginal selection remains
imperfectly aligned with calibrated dependence quality; changing it would be a
separate factor and requires a separate prospective entry.

Matching seed numbers blocks the comparison by the existing seed labels, but
does not imply identical initial discriminator weights or latent realizations.
Generator initialization consumes more global RNG values before discriminator
initialization. Latent and fixed validation-noise shapes change. The independently
seeded batch-order and profile-projection streams remain unchanged under the
same update schedule. Preserve all RNG settings and provenance; do not claim
exact common-noise pairing across architectures.

The implementation already constructs dimensions from saved configuration.
Keep discriminator width explicitly 128, since its fallback follows generator
width. New checkpoints need their own bound calibration maps. Generation draws
all latent vectors at once, so the latent tensor is four times larger. Schedule
heavy calibration/utility evaluations serially within the existing resource
queue; changing the schedule must not change numerical settings. Use the clean
`cGAN-temporal` checkout pinned to commit
`1ca62d0fbe1d77f916f8904e812db7d720f8cf53` for training, calibration, physical
evaluation and utility evaluation. Load the capacity configuration externally
from the separately published protocol/config commit; record both commits, the
configuration hash and runtime versions before launching jobs. The zero-weight
derivative implementation has exact-replay evidence against its prior path.
This does not imply equal random draws across the changed architecture.

The pinned evaluator preserves the original SWD10 comparison's numerical
implementation, including `copula.py` SHA256
`5739c8d0cb438ebd3fa06bc67ba79d6037493f8406a4e7ac1ba50ebba29d6939`.
Cross-arm evaluation and utility core source hashes, real selections and
statistical-control array hashes must match. Checkpoint-validation labels and
selected rows remain matched; latent noise has the expected new width and
therefore a different hash, reconstructed under the original seed rules.
Training-source differences from the original SWD10 implementation remain
explicit in provenance. Do not silently update the pinned evaluator.

A later inverse-quantile interpolation correction addresses a synthetic
float64 monotonicity edge case. It is deliberately absent from this matched
validation comparison and remains part of the later frozen-TEST source with
the numerical change disclosed. Existing reports keep their original source
hashes; this protocol makes no universal interpolation-robustness claim.
A numerical failure must remain visible and cannot be bypassed by substituting
a different calibration map or evaluator for one arm.

## Fixed pilot and promotion rule

Run seeds **11, 22 and 33**. The comparator is the matching three
**original curated SWD10** seeds, not the derivative arm, an individually chosen
best seed, or whichever prior arm appears favorable. Evaluate all pilot seeds
before deciding promotion.

Apply the same conjunction of rules already fixed for the derivative pilot:

1. At least 10% lower mean calibrated ramp W1 than the matching original three
   seeds, and strictly lower ramp W1 in each of the three paired seeds.
2. No more than a 10% increase in mean daily-energy, peak, lag-one ACF or fixed
   projected-profile errors relative to those same original seeds.
3. In each pilot seed, calibrated synthetic-training macro RMSE at most 1.10
   times both the real-trained and resampling-trained references. Also report
   the three-seed aggregate.

Only if all rules pass, retain these runs and add seeds **44 and 55** with the
same package and full budget. Do not alter thresholds, compare against a new
baseline, retune width/noise/learning rates, or substitute a different checkpoint
after seeing endpoints. If the pilot fails, retain the complete failed package;
no follow-on sweep is authorized by this protocol.

This screen asks whether the complete package improves the pipeline. It cannot
attribute any gain to width, latent dimension or their interaction. A subsequent
2-by-2 attribution study would require the four cells (64,24), (128,24),
(64,96), (128,96), with discriminator width 128 throughout. The single package
saves those additional runs at the cost of that attribution. It is a
project-specific exploratory decision, not a significance test or external
preregistration. Repeated validation use and accumulated prior screens remain
part of the selection history.

## Unchanged calibration, evaluation and final criteria

Follow [CALIBRATION_STUDY.md](../docs/CALIBRATION_STUDY.md):

- Fit a separate generated CDF for each frozen checkpoint from 4,096 generated
  reference profiles per condition, seed 7301, using 1,025 source/target quantile
  knots and the same actual TRAIN target marginals. Production draws use seed
  2026. Keep maps fixed during evaluation; never rank the evaluation batch.
  Retain ties, source exceedance, saturation and bounded-support diagnostics.
- Preserve the primary cap of 100 real/generated profiles per condition, all
  64 equally weighted conditions, and the same real validation selection and
  statistical controls. Report raw and calibrated models, Gaussian copula
  (shrinkage .01), independent marginals (shrinkage 1), conditional resampling,
  and the separate exact-marginal copula sensitivity. Primary marginal settings
  remain matched at 1,025 knots.
- Keep the 128 unit-direction whole-profile diagnostic at seed 9182 and the
  existing hypothetical daily battery scenario: 10 kWh, 4 kW input/output,
  efficiencies sqrt(.9) each, zero initial SOC daily, export-only charging and
  discharge against import. Battery outcomes remain secondary and cannot
  redefine the primary criterion or establish annual/application validity.
- Keep the observed-history one-step ridge task: 12 prior quarter-hour intervals,
  clock and condition terms, alpha 1, all 606,060 real/generated downstream
  training profiles with matched labels, and the fixed 6,400 validation rows.
  Retain RMSE and MAE, real/resampling references, and uncertainty limitations.
  The 10% RMSE tolerance is descriptive, not formal noninferiority.
- After promotion, retain all five seeds and apply the original final criteria:
  at least 20% lower five-seed mean ramp W1 than the matched Gaussian control,
  with lower error in at least four seeds; daily-energy W1 <=5 kWh, peak W1
  <=1 kW, ramp W1 <=.10 kW, and lag-one ACF W1 <=.10, with per-seed and
  all-condition results visible. Keep utility requirements and all tradeoffs.
  Resampling must remain reported even if it is stronger.
- Retain separate per-seed paired technology-stratified meter-bootstrap
  comparisons with 500 replicates under the existing protocol. Do not average
  CI endpoints or present seed SD as a CI; physical intervals do not supply
  utility uncertainty. Fixed generated samples, calibration and model selection
  remain outside those physical sampling intervals.
- Perform the separate cap-1,000 precision check on every promoted candidate
  seed, using the same checkpoints/maps and seed 2026. Preserve the primary
  cap-100 findings. For predetermined seed 11, retain calibration sensitivities
  (4,096 references, seed 7302) and (8,192 references, seed 7301), with knots and
  evaluation draws fixed. These sensitivities cannot replace the primary map.

Keep every attempted arm, seed, interruption, checkpoint history, calibration
map and endpoint report, including the original model's failed gates and the
derivative pilot's outcome. Inspect all conditions, diversity, support, bounded
nearest-neighbor diagnostics and secondary tradeoffs without inventing new
post-result success gates. Calibration-induced marginal improvement is not
independent evidence of learned dependence, novelty, privacy or broad utility.

No TEST data or outcomes enter this protocol, activation decision, training,
selection or refinement. Final TEST evaluation can occur only after the chosen
method and all five artifacts are frozen under the separate sealed harness.
Exact-trace curation does not establish independent households or external
validity. This protocol does not resolve the outstanding reference-architecture,
unconditional, hourly-resolution or broader application requirements for a
complete manuscript.

## Execution record

Publish this document and `configs/curated_capacity.json` before any capacity
job starts. Record that publication commit separately from the pinned numerical
source commit above. The study owner controls the training/evaluation queue.
Creating this protocol launches no jobs and changes no source implementation.
Retain the original arm, derivative failure, capacity attempts and all later
decisions; this is an adaptive exploratory research sequence, not a fresh
confirmatory study or external preregistration.
