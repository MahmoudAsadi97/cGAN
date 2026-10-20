# Frozen candidate and reference TEST study

The schema-2 harness seals one selected candidate and the three declared
reference arms before any TEST performance access. It is separate from the
existing strict schema-1 candidate-only harness; accepting a different path in
the old command does not supply this study. See `REFERENCE_COMPLETION.md` for
the scientific contracts. This document describes implementation, not completed
TEST results or authorization to inspect them during development.

The four arm IDs are `candidate96`, `bce96`, `unconditional96` and `bce24`.
Each must contain seeds 11, 22, 33, 44 and 55, including unsuccessful results.
The selected candidate kind is either the published `swd10` or `capacity`
package. No ensemble or seed selection is performed. All twenty selected
checkpoints must have complete 26-epoch histories, matching final checkpoints
and the full 606,060-row TRAIN pool. The fixed reference configs, conditioning
dimensions, native widths, update budgets, raw validation selector and source
bindings are verified. All fifteen conditional pipelines require their saved
4,096-draw / seed 7301 / 1,025-knot maps. The unconditional pipeline is raw-only.

## Freeze inputs

Use `freeze_reference_study.py` with curated96, its verified hourly development
derivative, a JSON specification and a new output directory. Paths in the
specification are interpreted relative to the process working directory; use
absolute paths for an unambiguous execution record.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
flock runs/heavy_evaluation.lock python scripts/freeze_reference_study.py \
  --prepared96 data/curated96 --prepared24 data/curated24 \
  --specification runs/final_reference_specification.json \
  --output runs/final_reference_bundle
```

The specification contains:

```json
{
  "candidate_kind": "swd10",
  "decision_rationale": "The fixed decision and limitations, with development references.",
  "arms": {
    "candidate96": [{"checkpoint": ".../best.pt", "calibrator": ".../calibration.npz"}],
    "bce96": [{"checkpoint": ".../best.pt", "calibrator": ".../calibration.npz"}],
    "unconditional96": [{"checkpoint": ".../best.pt", "calibrator": null}],
    "bce24": [{"checkpoint": ".../best.pt", "calibrator": ".../calibration.npz"}]
  },
  "protocol_files": ["docs/REFERENCE_COMPLETION.md", "...selected candidate protocol..."],
  "attempt_history": ["...complete provided attempted-arm record..."],
  "development_evidence": {
    "native": {}, "common_hourly": {}, "paired": {}, "additional": []
  }
}
```

Each arm array in this abbreviated illustration must actually have five run
records. The mandatory evidence index has these exact roles:

| Index | Required entries |
|---|---|
| `native[arm][seed]` | `primary`, `precision`, `utility`, `utility_precision`, `raw_primary` |
| `common_hourly[seed]` | `primary`, `precision`, `utility`, `utility_precision` |
| `paired[contrast][seed]` | One paired comparison path |
| `additional` | List of other retained evidence paths, possibly empty |

Seed keys are strings. All four arms and all five seeds are mandatory. The
four paired contrast names are `conditioning`, `candidate_reference`,
`resolution_raw` and `resolution_calibrated`. `primary`, `raw_primary` and
`utility` use cap 100 / seed 2026; `precision` and `utility_precision` use
cap 1000 / seed 2026. All paired reports use cap 100 and 500 replicates. Native
conditional primary/precision reports contain raw and mapped outcomes;
unconditional reports contain no conditional calibration. `raw_primary` retains
the original raw CLI report with its conditional-mean and resampling controls.
Common-hourly reports must include both raw/mapped BCE representations and the
selected candidate's raw/mapped hourly sums.

The freezer verifies report type, checkpoint roles, prepared-file bindings,
caps/seeds, ordered paired contrasts and maps. Utility reports must use the
full TRAIN pool and the correct lag/task width. Native calibration primary
reports must identify the chosen saved-map digest. Precision reports that did
not save another map retain matching provenance. This is a coverage/binding
check, not a new statistical selection procedure. The validation collector's
complete report and any robustness or sensitivity reports should be included
in `additional`; sibling JSON reports are automatically retained as well.

The supplied attempted-arm history and protocols are mandatory and archived.
The harness cannot prove that an operator has disclosed every attempt outside
the supplied records. It records that limit rather than claiming to establish
the absence of undisclosed screening.

## What the seal contains

The freeze stage opens TRAIN/VAL only. It verifies the curated parent and exact
hourly derivation, including portable parent lineage and the actual full TRAIN
row order. Prepared-data maps retain the original four-file binding; checkpoint
curation/derivation manifests are bound separately. Every explicit input and
implicit sibling `config.json`, `summary.json`, `history.json`, `last.pt` and
discovered JSON evidence file is checked for reserved TEST aliases before its
contents are read. A child TEST24 archive is forbidden.

The bundle retains twenty selected and twenty final checkpoints, fifteen maps,
the native metadata/scalers/manifests, source/environment fingerprints, decision
and development evidence, all physical generated arrays, fixed ridge states,
map fitting/application diagnostics, one generated-label vector and an exact
artifact inventory. All generated/control arrays and downstream predictors are
created before the manifest is sealed. Sources/environment and input hashes
are checked again before completion. There is no reduced-utility or reduced-
bootstrap option in this schema.

Fresh physical generation uses seed 9202. Each condition has up to 1,000 generated
rows, with counts determined from reserved TEST metadata only. Fixed TRAIN-only
controls at 96 slots are Gaussian copula, independent marginals, exact-marginal
Gaussian sensitivity, conditional resampling and conditional mean. At 24 slots, fit
Gaussian, independent, resampling and mean controls directly on verified
TRAIN24. A directly fitted hourly copula differs from summed copula96 outputs.
Controls are shared across models/seeds, not treated as five independent draws.

Each conditional96 model supplies native raw/mapped arrays and their hourly
sums. Each BCE24 model supplies native raw/mapped arrays reused in common-hourly
comparisons. Native maps are applied before summing. Hourly sums accumulate in
float64 and cast the result to float32; mapped96 values are not prematurely cast
before summing. Native mapped24 retains float64. Uniform float64 artifact storage
does not erase the recorded computation dtypes or transformation hashes.

Utility fitting uses fresh seed 9204 and complete TRAIN label counts. The same
native generated draw supplies separate native96 and common24 ridge fits.
Native96 uses 12 observed lags and predicts one quarter-hour ahead; common24
uses three observed hourly lags and predicts one hour ahead. All use the same
four clock-feature columns, 64 condition indicators and fixed ridge rule. Real,
resampling and Gaussian ridge controls are fitted once per resolution. Hourly
BCE24 predictors are reused, not refitted for another report section. No ridge
coefficient transformation or aggregation of native96 predictions substitutes
for fitting the hourly task.

Freeze is a heavy, serialized development operation. Run it after training
resource queues permit; it never needs TEST contents. The full-count generated
utility arrays are temporary; their hashes, training metadata and ridge states
are retained. Failed freeze directories remain and cannot be overwritten.

## One reserved TEST session

The scoring command requires the externally retained exact manifest SHA:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
flock runs/heavy_evaluation.lock python scripts/evaluate_reference_study.py \
  --bundle runs/final_reference_bundle \
  --expected-seal-sha256 THE_RETAINED_MANIFEST_SHA256 \
  --output runs/final_reference_test
```

All bundle contracts, artifacts and source/environment fingerprints are checked
before access. One private full TEST96 load verifies the expected digest,
archive types, all 64 conditions, canonical membership, development exclusions,
calendar completeness and technology counts. Pre/post/final integrity hashes
remain; “once” means one shared logical loading/selection context, not removal
of mutation checks. One cap 1000 selection with seed 9201 supplies identical
rows, labels, meter IDs and dates to every model, metric and representation.

Only after this verification is real TEST24 created in memory from the selected
TEST96 rows using exact declared sum4. It is not a separately selected cohort,
new identity audit or new TEST archive. The report distinguishes the parent
file digest from selected96/24 profile digests and retains selection/identity
hashes. No generation, calibration, control fitting, predictor fitting or
checkpoint selection occurs during scoring.

Physical, joint and fixed daily-battery diagnostics use 0.25-hour native96 or
1-hour common24 intervals. Projection directions are shared within each
dimension. Hourly battery dispatch is recomputed from hourly inputs. Native
quarter-hour ACF/ramps, 96D projections and 15-minute utility errors are not
directly comparable to native hourly endpoints. The common-hourly section uses
the same 24-dimensional task for every representation.

For each training seed, the four reference contrasts receive physical and
utility paired technology-stratified whole-meter uncertainty. The existing
selected-candidate96 comparisons against Gaussian/resampling physical controls
and real/resampling utility controls remain. All use 500 draws with fixed seed 9203,
holding generated arrays, maps and predictors fixed. Missing-support draws
withhold bounds under the existing helper policy. Native24 outputs/predictions
are reused in common24 comparisons. Confidence intervals remain per-seed;
the five-seed summaries contain point-estimate means, sample SDs and ranges,
never averaged CI bounds or a synthetic ensemble.

An exclusive per-bundle TEST access claim prevents concurrent or subsequent
TEST attempts. Every invocation has a new output directory and append-only
attempt records; failures remain, no report is marked complete after partial
failure, and no automatic retry occurs. A pre-access failure can be addressed
by an explicit new invocation. A claim after attempted access is retained even
if scoring fails. Runtime methods, paths, seeds and settings cannot be changed
through evaluator options.

Practical thresholds apply only to the selected calibrated native96 candidate.
References add no promotion gates, and quarter-hour ramp/ACF thresholds are not
transferred to hourly results. The study remains an identity-audited internal
meter holdout from one source/year. Exact trace deduplication does not establish
independent households; calibration does not establish tail extrapolation or
independently learned margins; descriptive conditional intervals do not account
for all model-selection or generation uncertainty. Observed-history ridge and
one hypothetical daily battery scenario do not establish general forecasting,
annual battery or economic validity. Subsequent tuning on these results exhausts
the cohort as performance-unseen confirmation.
