# Collecting the declared reference evidence

`scripts/collect_reference_study.py` collects saved **validation** evidence for
the reference experiment declared in [REFERENCE_COMPLETION.md](REFERENCE_COMPLETION.md).
It adds no metric threshold, selects no method and performs no evaluation.
Completion of the inventory is not a claim of publication readiness or a
successful result. Failed runs and unfavorable results must remain in the study;
an incomplete run prevents a complete-study report instead of disappearing from
an average.

## Required inventory

Supply one JSON object with exactly these fields:

```json
{
  "schema_version": 1,
  "candidate_kind": "swd10",
  "prepared96": "../data/curated96",
  "prepared24": "../data/curated24",
  "runs": {},
  "development_evidence": {
    "native": {},
    "common_hourly": {},
    "paired": {},
    "additional": []
  }
}
```

This skeleton is deliberately incomplete. Populate every entry below before
collection. Paths are resolved relative to the inventory file, and the candidate
must explicitly be `swd10` or `capacity`. No seed or candidate is inferred from
a directory name or from performance.

| Inventory section | Required keys and values |
|---|---|
| `runs` | Four arms `candidate96`, `bce96`, `unconditional96`, `bce24`; each maps seed strings `11`, `22`, `33`, `44`, `55` to run directories |
| `development_evidence.native` | The same four arms and five seeds; each seed maps the five roles below to report paths |
| `development_evidence.common_hourly` | Each of the five seeds maps `primary`, `precision`, `utility`, `utility_precision` to report paths |
| `development_evidence.paired` | Four contrasts `conditioning`, `candidate_reference`, `resolution_raw`, `resolution_calibrated`; each maps all five seeds to report paths |
| `development_evidence.additional` | Explicit extra JSON report paths to retain and hash; these cannot replace a required role |

Every run directory must contain `config.json`, `summary.json`, `history.json`,
`best.pt` and `last.pt`. Each of the fifteen conditional runs also requires its
fixed native `calibration.npz` in that directory. The unconditional arm must
remain raw in its primary comparison.

| Native role | Conditional arms | Unconditional BCE96 |
|---|---|---|
| `raw_primary` | Original raw `validation_report.json`, cap 100 | Same original raw report, retained independently |
| `primary` | Native calibration report containing raw and calibrated methods, cap 100 | Supplemental raw unconditional report, cap 100 |
| `precision` | Same native map and protocol, cap 1,000 | Supplemental raw unconditional report, cap 1,000 |
| `utility` | Native utility with raw and mapped generators, cap 100 | Native raw-only utility, cap 100 |
| `utility_precision` | Same full TRAIN fits, cap 1,000 | Same full TRAIN fits, cap 1,000 |

There are 140 required report slots across the twenty runs, common hourly
comparisons and four physical contrasts. Common-hourly reports must include
`candidate96`, `bce96`, `bce24`, their raw and mapped outputs and the declared
controls. They regenerate all methods in common hourly coordinates; they do not
compare native 15-minute distances to hourly distances. The physical paired
contrasts use cap 100 and 500 bootstrap draws. Common-hourly utility includes
its declared paired uncertainty at both caps.

## Running the collector

Run after every required saved report is available:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
flock runs/heavy_evaluation.lock \
python scripts/collect_reference_study.py \
  --inventory runs/reference_inventory.json \
  --output runs/reference_validation_study.json \
  --evidence-index runs/reference_evidence_index.json
```

Both output files must be new JSON paths. An invalid or incomplete inventory
raises an error; the collector does not emit a partial successful report.
Existing evidence is never overwritten. All explicit paths are checked against
prepared TRAIN/VAL/TEST paths and their hardlinks before artifact access. The
control inventory must itself be a JSON manifest with no hardlinks.
All evidence entries, including `additional`, must resolve to JSON reports;
profile archives and arbitrary binary files are not accepted as extra evidence.

The collector reads saved JSON, checkpoint weights and the scalar metadata
member of calibration archives. It hashes complete checkpoint/map artifacts.
It does **not** open prepared profile arrays, load calibration quantile arrays,
generate profiles, fit maps/predictors, rerun metrics, or open TEST. Prepared
hashes are cross-checked among metadata, manifests, checkpoints and reports;
this is evidence consistency, not a new audit of profile bytes or hourly sums.
The freezer separately validates the saved map arrays before deployment.

## What is checked and retained

All twenty runs must match their explicit configurations, 26-epoch histories,
30,784 updates per adversary, full 606,060-row canonical TRAIN selection, and
balanced sampler exposure accounting. Best/last checkpoints must agree with the
history, selection criterion, finite model tensors, metadata, scaler and source
records. BCE96 and unconditional96 require the declared clean native source pin.
Hourly evidence must bind the derived metadata and the original curated parent.

Report checks bind checkpoint/map SHA-256 values, selected real rows, all 64
conditions, caps 100/1,000, fixed generation/calibration seeds, TRAIN-fitted
controls, physical units, per-condition/macro consistency and bootstrap draw
accounting. Missing-condition bootstrap draws retain their documented withheld
status and null bounds; the collector never redraws them or substitutes a more
favorable interval. Conditional maps are applied in native coordinates before
96-to-24 summation.

Changing the utility validation cap must leave TRAIN profile/label hashes,
condition counts and fitted coefficient hashes unchanged for every method.
Observed-TRAIN and conditional-resampling utility controls must match across
arms at each resolution, and native24 controls must match common-hourly
controls. All predictors retain the declared observed-history task, features,
ridge objective and full ordered TRAIN labels.

Numerical evaluator and runner/helper fingerprints must agree across seeds and
caps within each comparison family. Supplemental unconditional, common-hourly
and paired physical evaluation share overlapping current numerical modules.
Historical native96 evaluation remains separately attributed to its original
copula interpolation. Original raw reports lack their own evaluator source
fingerprints; their source attribution remains protocol-declared, not recovered
from the report. Their exact files are retained, including the sixth original
raw artifact bound by each unconditional supplemental report.

The unconditional native utility Gaussian baseline uses exact empirical
marginals; conditional native utility uses 1,025-knot marginals. Both settings
and source versions remain visible. They are **not** asserted to be a common
Gaussian reference. The unconditional supplemental physical controls use the
declared 1,025-knot current implementation and do not replace the original raw
report.

The aggregate contains every seed's endpoints, per-condition results,
uncertainty results and source/input hashes. Neural endpoints and paired point
differences receive five-seed means, sample standard deviations, ranges and all
five values. A repeated deterministic control is not treated as five independent
experiments. Bootstrap confidence bounds remain attached to their individual
seed reports; they are never averaged, pooled across seeds or turned into a
confidence interval for the five-seed mean.

## Evidence index and final freezing

The index exposes the same path-string `development_evidence` object expected
by `fluvius_cgan.frozen_reference_study`, plus SHA-256/size records and the
collector report itself in `additional`. Use that object in the freeze input;
the freeze specification also requires its own protocol files, explicit
candidate kind, decision rationale and retained attempt history. Collection
does not choose those values or authorize final TEST use.

The indexed report is an immutable, reviewable record of declared validation
coverage. It cannot establish that undisclosed attempts never occurred, that
exact-trace curation proves independent households, or that validation results
generalize to future periods. The BCE conditioning contrast is an ablation
within the reference architecture; candidate-versus-reference remains a method
package comparison. All reused validation evidence is exploratory.

## Implementation verification record

The first locked synthetic report-contract run on 20 October 2026 completed
with 72 passing checks and one failure. The failing preflight check found that
the collector hashed its own Python source files before rejecting an explicitly
supplied forbidden-file alias. No prepared profile content was read. Source
hashing was moved after complete input-path preflight, retaining the stronger
test. Three further checks reject binary/profile archives supplied as extra
evidence. The locked report-contract rerun passed all 76 checks in 12.59 seconds;
the separate synthetic checkpoint completion suite passed all 34 checks in
10.87 seconds. These 110 checks use artificial metadata/reports/checkpoints,
not actual data evaluation. They do not substitute for collecting the full
twenty-run study when its reports are available.

A subsequent [saved-artifact compatibility receipt](../reports/reference_collector_actual_binding_check.json)
records a successful check of completed BCE96 seed 11: training/history and
best/last checkpoints, map metadata, original raw primary, native primary
calibration and native primary utility, with unchanged source/input hashes.
No profile or TEST archive was opened. This is explicitly partial coverage;
other seeds/arms, precision, common-hourly and paired reports were not checked
by that inspection, and no model metric was rerun.
