# Frozen final-test evaluation

This workflow evaluates all five saved calibrated cGANs after the method is
frozen. It separates preparation of the experiment from access to TEST outcomes.
It does not retrain a GAN, choose a seed, tune on TEST, or turn the exploratory
validation study into a confirmatory result retrospectively. See
[CALIBRATION_STUDY.md](CALIBRATION_STUDY.md) for the method and its development
history.

## Seal before TEST access

`freeze_test(prepared, runs, output_dir, *, decision_rationale, protocol_files,
evidence_files=(), utility=True, bootstrap_replicates=500)` takes the five
checkpoint/calibrator pairs for training seeds 11, 22, 33, 44 and 55. The output
must be new. Preserve every declared seed; there is no best-seed option.

The helper requires matching configurations and relevant training source hashes
across all five runs, and agreement between summaries and checkpoint provenance.
It verifies completion of each checkpoint's configured epoch budget; it does not
globally hardcode 26 epochs. For this selected study, the sealed protocol and
decision rationale must record the prescribed 26-epoch budget.

Supply a nonempty decision rationale and at least one protocol document. The
bundle copies and hashes these documents, any explicitly supplied evidence, and
each run's `config.json`, `summary.json`, `history.json` and JSON files whose
names contain `validation`, `uncertainty`, `robustness`, `comparison` or
`sensitivity`, including the separate per-seed paired-comparison reports. Aggregate development
reports outside run directories must be supplied explicitly. Record the selection
reason and remaining weaknesses; preserving evidence does not certify its
scientific adequacy or convert validation results into TEST evidence.

The existing preparation loader reads TRAIN and VAL for provenance checks;
fitting and synthetic-reference construction use TRAIN only. Freezing must not
open, load, or hash `test.npz`. The expected TEST SHA256 and membership/calendar
specification come from the existing curated metadata and curation manifest.
The resolved TEST path is pinned. The original four preparation hashes
(`train.npz`, `val.npz`, `metadata.json`, `scaler.json`) remain separate from this
TEST binding; neither is silently rewritten to accommodate a different file.

The bundle contains the precomputed, original-unit generated reference arrays
for every declared method and seed, plus the artifacts needed to replay the
evaluation. Calibration maps remain fixed. Statistical baselines use only TRAIN
rows from the verified model-training pool. Utility is enabled by default: its
real-trained and synthetic-trained ridge states are fitted before sealing, using
TRAIN and the declared synthetic training sets. No TEST outcomes enter fitting.
Calibration fitting diagnostics and diagnostics on each fresh sealed raw
reference are saved as JSON artifacts before sealing and included in each seed's
evaluation report. These checks do not refit maps on TEST.

The frozen settings are:

| Setting | Value |
|---|---:|
| Maximum real and generated profiles per condition | 1,000 |
| Real-profile selection seed | 9201 |
| Physical-reference generation seed | 9202 |
| Meter-bootstrap seed | 9203 |
| Utility seed | 9204 |
| Calibration-reference profiles per condition | 4,096 |
| Calibration-reference seed | 7301 |
| Calibration quantile knots | 1,025 |
| Bootstrap replicates by default | 500 |
| Utility by default | On |

The calibration settings are enforced; calibration-sensitivity maps are not
accepted as substitutes.

Utility may be disabled, or bootstrap replicates set to zero, only as an explicit
choice recorded before TEST evaluation. Omitting utility supplies no utility
evidence; omitting bootstrap leaves point estimates without intervals. Metric
definitions, condition weighting, comparisons, projection directions and the
battery scenario are fixed in the bundle.

Source and runtime checks are strict: the Git commit, package/CLI source hashes,
Python/library versions, recorded platform and BLAS/thread settings must match
at evaluation. Finalize the code and commit before sealing, and preserve that
runtime. The seal records these requirements; it is not a portable environment
installer.

Retain the SHA256 of the exact bytes of `manifest.json` outside the bundle. This
is the seal supplied to the evaluator; it is not the hash of `test.npz`.

## Commands

Template only: replace the rationale with the actual recorded decision. Add
`--evidence PATH` for each aggregate report or other development artifact to
retain. `--protocol PATH` and `--evidence PATH` are repeatable; each `--run`
takes two separate paths, not a `CHECKPOINT=CALIBRATOR` string.

```bash
python scripts/freeze_test_manifest.py \
  --prepared data/curated96 \
  --run runs/curated_swd10/seed_11/best.pt runs/curated_swd10/seed_11/calibration.npz \
  --run runs/curated_swd10/seed_22/best.pt runs/curated_swd10/seed_22/calibration.npz \
  --run runs/curated_swd10/seed_33/best.pt runs/curated_swd10/seed_33/calibration.npz \
  --run runs/curated_swd10/seed_44/best.pt runs/curated_swd10/seed_44/calibration.npz \
  --run runs/curated_swd10/seed_55/best.pt runs/curated_swd10/seed_55/calibration.npz \
  --decision-rationale "REPLACE with the method-freeze decision and remaining limitations" \
  --protocol docs/CALIBRATION_STUDY.md --protocol docs/FROZEN_TEST.md \
  --output runs/frozen_test_bundle

python scripts/evaluate_frozen_test.py \
  --bundle runs/frozen_test_bundle \
  --expected-seal-sha256 "REPLACE with the independently retained manifest SHA256" \
  --output runs/frozen_test_attempt_01
```

The freeze command prints the manifest path and seal. `--no-utility` explicitly
omits utility; `--bootstrap-replicates 0` omits intervals. Otherwise the replicate
count must be at least two; the default is 500. These options are unavailable at
evaluation. The evaluation API is
`evaluate_frozen_test(bundle, expected_seal_sha256, output_dir)`.

## Evaluate the sealed bundle once

The evaluator requires the bundle, the expected manifest SHA256, and a new
attempt directory. It first verifies the seal and bound artifacts. It then opens
the pinned TEST file and verifies its hash, full retained-meter membership,
per-meter calendar, calendar-derived labels and condition/technology counts
before selecting rows. These checks apply to the entire TEST partition, not only
the capped sample. The manifest records retained original TEST representatives;
it does not move development meters into TEST.

Selection uses the frozen seed and a cap of 1,000 profiles per condition. Every
method and training seed is compared with exactly the same selected real rows.
Report actual condition and meter counts when fewer rows are available. Missing
support or incompatible provenance must not trigger silent condition exclusion,
substitution of another dataset, or a changed denominator.

The evaluator performs no model fitting, generator inference, calibration fitting
or baseline fitting. Utility predictions use the already frozen ridge states and
the fixed observed-history task. It must not adjust transformations, clipping,
hyperparameters or evaluation settings in response to TEST results.

Attempt outputs are immutable: use a new directory and retain successful and
failed attempts. Freeze failures are retained, and evaluation writes attempt,
TEST-access, completion or failure records with an append-only bundle ledger.
This records access through this harness; it cannot prove that no other process
previously inspected TEST. A software-correction rerun must identify the original
attempt, the defect and the correction. It does not justify choosing favorable results or
concealing earlier TEST access. Further method tuning after these outcomes are
seen uses this TEST set for development and requires a new independent holdout
for a clean final assessment.

## Read the results correctly

Report all five seed results and their mean, sample standard deviation and range.
When utility is enabled, five-seed RMSE/MAE summaries include these same statistics.
These summarize five frozen trained models. Repeated identical baseline arrays
are one frozen control, not five independent baseline repetitions. Do not pool
generated profiles across seeds: that would evaluate a different mixture model.

The current uncertainty scope is **per-seed paired meter bootstrap only**, for
physical errors against Gaussian copula/resampling and, when utility is enabled,
forecasting RMSE/MAE against real-trained and resampling-trained ridge models.
Joint-projection and battery results have no bootstrap intervals here. Sample
whole real meters with replacement within the fixed technology strata, retaining
the original number of meters per stratum and applying the same multiplicities
to both methods in a comparison. Generated and comparator references remain
fixed. Report differences as candidate minus comparator, so negative values
favor the candidate. For utility, compute RMSE inside each condition before
averaging conditions equally; resample the same observed-history windows for
both predictors. Invalid support or insufficient meters must be reported;
there is no silent fallback to independent-row resampling. The utility intervals
do not convert the descriptive 10% RMSE tolerance into a noninferiority test.

These are descriptive marginal percentile intervals, conditional on the frozen
models, generated references and selected TEST rows. They describe variation from
real-meter resampling under the independence assumptions of the strata. They
exclude training-seed, model-selection, calibration-fitting and generated-sample
uncertainty. They are not simultaneous intervals, and nominal coverage is not
guaranteed for the nonsmooth Wasserstein functional. Meter independence is an
assumption; exact-trace curation does not establish independent households or
eliminate near-duplicates. This is an internal meter holdout from the same source
and year, not external or future-period validation.

There is **no confidence interval for the five-seed mean** in this workflow. Do
not average per-seed interval endpoints, combine seed SD with meter intervals,
or treat both as independent sources of replication. A future mean-contrast
interval would require shared meter draws across every seed and comparator,
with the seed contrasts averaged inside each replicate.

Calibration directly targets TRAIN marginals. Marginal improvements alone are
not evidence of learned temporal dependence, privacy or utility. Keep physical,
battery and utility tradeoffs visible; the fixed hypothetical battery diagnostic
does not establish annual operation, cost savings or gross PV self-consumption.
