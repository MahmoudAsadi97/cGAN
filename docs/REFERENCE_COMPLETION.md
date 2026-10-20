# Curated reference and resolution experiments

Declared on 20 October 2026, before these runs. This protocol completes the
reference BCE, unconditional and hourly comparisons required by
[RESEARCH_PROTOCOL.md](RESEARCH_PROTOCOL.md). It adds no candidate promotion
gate and does not authorize method selection using TEST. At declaration, the
three arms below have not trained; hourly preparation and common-resolution
evaluation still require verified implementation.

## Fixed arms and comparison scope

All three arms use seeds **11, 22, 33, 44 and 55**, including failed or collapsed
runs. The total is fifteen new neural fits, without a hyperparameter sweep.

| Arm | Configuration | Preparation | Main comparison |
|---|---|---|---|
| Conditional BCE96 | [curated_reference_bce96.json](../configs/curated_reference_bce96.json) | Existing `data/curated96` | Quarter-hour adaptation of the upstream architecture |
| Unconditional BCE96 | [curated_reference_unconditional96.json](../configs/curated_reference_unconditional96.json) | Same curated96 | Network conditioning within the reference BCE architecture |
| Conditional BCE24 | [curated_reference_bce24.json](../configs/curated_reference_bce24.json) | Hourly sums of the same curated96 development rows | Resolution sensitivity of that reference architecture |

The unconditional config differs from conditional BCE96 only in
`unconditional=true`. The BCE24 and conditional BCE96 config contents are
identical: output dimension comes from verified prepared metadata, not the
filename. A BCE24 run requires a 24-slot preparation; passing curated96 would
train another 96-slot model and would not satisfy this protocol.

This unconditional comparison isolates network conditioning in the **reference
BCE method**. It does not isolate conditioning in the current hinge/SWD/capacity
candidate. Comparing the candidate with BCE96 evaluates a complete method
package; it cannot separately attribute effects to normalization, Adam betas,
width, loss, spectral normalization, capacity or calibration. The existing
matched SWD and capacity studies retain their own narrower interpretation.

## Fixed architecture, training and selection

Use the documented upstream generator widths 64/128/64, discriminator widths
64/128/256, LeakyReLU 0.2, discriminator dropout 0.4, linear output and 24-dimensional
Gaussian noise. Use non-saturating BCE with the existing numerically stable
logit formulation, generator/discriminator learning rates 1e-4/2e-4, and Adam
betas 0.9/0.999. Fit global min/max on the full eligible TRAIN partition. Keep
negative net exports; no arbitrary value filter, output clipping, residual
profile, spectral normalization, profile SWD or derivative SWD is added.

| Setting | Fixed value |
|---|---:|
| Eligible TRAIN profiles | 606,060 |
| Epochs | 26 |
| Batch size | 512 |
| Balanced batch | 4 conditions × 128 replacement draws |
| Updates per epoch, per adversary | 1,184 |
| Total updates, per adversary | 30,784 |
| Validation selection sample | 4,096 profiles |
| Selection criterion | Raw condition-macro marginal W1 in training-normalized units |
| Device / threads | CPU / 1 |
| Training cap / early stopping / resume | None / none / none |

The sampler draws 606,208 profiles per epoch with replacement. An epoch is an
update budget, not a traversal of each profile. The same seed and ordered labels
give the same sampled row indices across these three arms. For an unconditional
run, labels still determine balanced-mixture sampling and conditional validation
selection but enter neither network. Describe it as an unconditional network
trained on a condition-balanced mixture, not a fully label-agnostic procedure.

Record every epoch, best epoch, elapsed time, parameter count and full source
provenance. The architecture parameter counts are:

| Arm | Generator | Discriminator |
|---|---:|---:|
| Conditional BCE96 | 28,512 | 51,905 |
| Unconditional BCE96 | 24,416 | 47,809 |
| Conditional BCE24 | 23,832 | 47,297 |

Removing condition inputs reduces input-layer parameters; this is not an exactly
parameter-matched ablation. Changing output resolution also changes parameters.
Equal updates do not imply equal computation, convergence or the original
paper's training budget. Do not extend a budget in response to unfavorable
endpoint results. The older 200-epoch configs are not used in this study.

The data, labels, sampling, selection and resolution changes make this an
adaptation of upstream commit `885f78052a91fbc755e6b17a354c68849b5556ca`, not a
reproduction of its published numbers. See
[UPSTREAM_ADAPTATION.md](UPSTREAM_ADAPTATION.md) and
[LITERATURE_CONTEXT.md](LITERATURE_CONTEXT.md).

## Hourly derivation and TEST boundary

Derive development data directly from verified curated96 TRAIN/VAL rows:

```python
x24 = x96.astype(np.float64).reshape(-1, 24, 4).sum(axis=2).astype(np.float32)
```

Keep labels, meter IDs, dates and row order unchanged. Refit the hourly TRAIN
min/max. Store kWh per hourly interval: sum four quarter-hour energies, do not
average them. Native hourly mean power uses interval duration 1 hour.

The derived-preparation manifest must bind the parent development-file hashes,
curation manifest, canonical membership, exact operation/dtypes and new output
hashes. Verify this lineage through an explicit derived-preparation guard.
Preserve the actual release digest; do not bypass curation safeguards or copy
the original curation record as though a new hourly trace audit occurred.
Quarter-hour canonical representatives remain fixed. Aggregation can create
additional exact coincidences; quarter-hour uniqueness does not prove hourly
uniqueness. Any development-only diagnostic of such coincidences must not move
or drop rows after outcomes are seen.

Hourly development preparation must not open, hash, aggregate or copy
`test.npz`. Retain its expected parent digest and membership from existing
metadata only, marking hourly TEST as unmaterialized. Do not label the parent
96-slot digest as a 24-slot artifact digest. Synthetic checks must cover
aggregation order, conservation within declared float32 rounding, row identity,
TRAIN-only scaling, tamper rejection and forbidden TEST access.

Final reference evaluation will need a sealed implementation that verifies the
reserved TEST96 input and applies this exact aggregation without fitting. The
current calibrated final-test workflow does not supply a multi-resolution
reference comparison merely by accepting a new checkpoint path. That capability
must be implemented and verified before any reference TEST evaluation.

## Fixed validation comparisons

Use the existing primary cap of 100 profiles per condition, seed 2026, plus a
separate 1,000-profile precision check for all arms and seeds. Retain both;
neither replaces an unfavorable primary result. Preserve all 64 conditions and
full TRAIN rows for controls. Record and verify real row identities across
resolutions; equal random seeds alone are not proof of equal selection.

The primary reference contrasts are:

- Raw conditional BCE96 versus raw unconditional BCE96, with matched seeds.
- Raw BCE96 and the current 96-slot candidate versus conditional mean,
  conditional resampling and the Gaussian reference, with complete package
  differences disclosed.
- Raw BCE24 versus hourly sums of raw BCE96 outputs, on exactly the same real
  validation rows aggregated to 24 slots. Native 96-slot results remain separate.

For both conditional BCE arms, also fit the established fixed conditional map:
4,096 independent reference draws per condition, seed 7301, 1,025 quantile knots,
TRAIN targets only. Report raw and calibrated outputs. This supplies a calibrated
reference for the current calibrated candidate. For common-hourly comparisons,
sum the already calibrated 96-slot outputs; do not recalibrate the summed output
batch or use VAL to fit another map. Calibration and aggregation do not commute;
directly trained/calibrated BCE24 is a distinct pipeline.

The primary unconditional comparison has **no condition-specific calibrator**.
Such a map would inject conditional TRAIN marginals and make the deployed
pipeline conditional. If a later separately declared diagnostic uses it, label
it "unconditional generator + conditional map"; it cannot show that the network
learned conditioning.

Report daily energy, peaks, signed ramps, ACF, diversity, whole-profile
projections, per-condition failures and the observed-history ridge task. Match
physical time lags when comparing ACF. Common-hourly metrics use duration 1 hour
and common 24-dimensional projection directions; native 96-dimensional
directions and quarter-hour ramp/ACF endpoints are different metrics. If battery
outputs are compared, rerun the same scenario on every common-hourly input;
native dispatch results at different resolutions are not interchangeable.

For common-hourly utility, fit every representation's ridge model on 24-slot
profiles with three observed hourly lags, four clock harmonics, 64 condition
indicators, and the existing fixed ridge rule/full TRAIN counts. Native96 uses
12 lags and predicts one quarter-hour ahead; native24 predicts one hour ahead.
Their native RMSE values are not a direct resolution comparison. Conditioning
inside the downstream ridge model is shared task information, not evidence that
an unconditional generator used labels.

Report all five neural results and their mean/sample SD. The declared
within-resolution physical contrasts use paired whole-meter bootstrap draws
within technology strata, 500 replicates, with models/generated references held
fixed. Use the same meter multiplicities for each pair. Do not average per-seed
interval endpoints or treat days as independent meters. Keep real-trained and
resampling-trained utility references and the existing descriptive uncertainty
scope. Reference results do not alter the candidate's practical thresholds.

## Publication before execution and pending work

Publish this protocol and all three configs to V2 before starting these jobs.
For BCE96 and unconditional BCE96, use the already published numerical source
commit **`1ca62d0fbe1d77f916f8904e812db7d720f8cf53`** from a clean pinned worktree,
passing the newly published configs as external files. Keep that pin for their
native-resolution validation/calibration/utility computations, including its
original copula interpolation. This preserves numerical attribution alongside
the existing study; any later implementation correction must be recorded
separately rather than silently substituting reports.

Hourly training requires a new clean source pin after the derived preparation
and guard have been tested and published. Its SHA is deliberately pending.
Disclose that provenance/loader extension; do not claim byte-identical training
source across resolutions. Use one subsequently fixed evaluator version for all
common-hourly comparisons, binding every model, input and transformation.

Before each arm starts, retain an execution record with its numerical commit/tree,
source hashes, the separate published protocol/config commit and config hashes,
environment and prepared-data binding. The new protocol/config publication SHA
can only be recorded after publication. The trainer also records each run's
source and data provenance. Retain content-equivalent publication mappings if
local and published commit IDs differ. Both source and external configs must
remain immutable throughout a run.

Run directories must be new; retain unsuccessful and interrupted attempts.
Schedule training within resource limits and serialize heavy evaluation.
Quarter-hour commands, after publishing the protocol/configs and recording
their hashes, are below. Explicit `PYTHONPATH` selects the pinned package rather
than an editable install in the development checkout. The example creates a
new worktree; an already verified clean worktree at the same commit is equivalent.

```bash
study_root="$(pwd)"
git worktree add --detach ../cGAN-reference-numeric \
  1ca62d0fbe1d77f916f8904e812db7d720f8cf53
training_root="$(cd ../cGAN-reference-numeric && pwd)"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

PYTHONPATH="$training_root/src" python "$training_root/scripts/run_experiments.py" \
  --prepared "$study_root/data/curated96" \
  --config "$study_root/configs/curated_reference_bce96.json" \
  --output "$study_root/runs/curated_reference_bce96" --seeds 11 22 33 44 55 \
  --samples-per-condition 100
PYTHONPATH="$training_root/src" python "$training_root/scripts/run_experiments.py" \
  --prepared "$study_root/data/curated96" \
  --config "$study_root/configs/curated_reference_unconditional96.json" \
  --output "$study_root/runs/curated_reference_unconditional96" --seeds 11 22 33 44 55 \
  --samples-per-condition 100
```

Use `curated_reference_bce24.json` only after the derived hourly preparation and
its verification path are complete, with its exact output path recorded in the
execution ledger. Common-hourly evaluation, precision reports, calibration,
utility and paired comparisons are separate stages; the training runner's raw
validation report does not complete them. Their implementation and execution
remain pending at declaration. No smoke test or historical uncurated run
substitutes for the fifteen declared fits and their stated comparisons.
