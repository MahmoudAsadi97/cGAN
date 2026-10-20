# Frozen Phase 1 review record

Read the [results and scientific assessment](../../docs/PHASE1_RESULTS.md)
alongside the [methods](../../docs/PHASE1_METHODS.md) and
[frozen protocol](../PHASE1_FROZEN_PROTOCOL.json).

This directory contains compact extracts and byte-for-byte copies of completed
experiment outputs. No model is retrained or reselected to produce these files.

| Record | Contents |
| --- | --- |
| `report_summary.json` | All 13 report-level summaries, three generation-draw macro/weighted summaries per report, and five-seed neural summaries |
| `selected_runs.json` | All ten run configurations, selected epochs, validation minima and artifact hashes |
| `histories/` | All 100 epochs for each of the ten neural runs |
| `inference.json` | Fixed-exemplar clustered comparisons, paired training-seed differences and nearest-training-profile diagnostics |
| `figures/` | PNG/PDF figures, CSV exports and renderer input/output hashes |
| `provenance/` | Training/evaluation execution records and append-only test-access logs |
| `record_manifest.json` | SHA-256 inventory of this review record and the separate full archive |

The scalar summaries are extracted without rounding; narrative tables round
values for readability. Full per-model JSON reports also retain curves and
validation row indices. Those larger reports and the selected/final neural
checkpoints are in `artifacts/phase1-record-reviewed.zip`, outside Git. The ZIP contains
its own `bundle_manifest.json` with each member's size and SHA-256. Source
datasets are excluded. The archive is available from the working session;
its hash and size are recorded here for transfer verification.

Archive paths are portable: after extraction, a conditional checkpoint is
`runs/conditional/seed_11/best.pt`, for example. Provenance may also contain
absolute paths from the original execution machine; these identify source
artifacts and need not exist on a receiving machine. Install the repository
package to use a checkpoint with the documented `generate` command.

Training used commit `67911e40c827c91dd540559f3a4680ec1c2c2816`;
evaluation used code at `48475e0770d8078ae03fed29091fe585238a9e4b`.
Final figure review found clipped shared-axis ranges in the initial learning
and technology-group figures. Commit
`ccda7e2e44a5170939ee1e80da0a5c15231775de` corrects axis ordering and adds
regression checks. The reviewed figures and archive use that commit; the
original archive is retained separately. No training or metric calculation was
repeated, and every CSV export is byte-identical between the two renders.
The training launch record discloses
an untracked evaluator test file at launch; executable training source hashes
are recorded separately. Later results commits add the review record and
scientific interpretation without changing the frozen experiment.

The primary error is condition-macro normalized profile SW1. Neural error bars
summarize five **training seeds**; baseline bars summarize three **generation
draws**. Clustered intervals instead concern the predeclared
training-seed-11/generation-seed-31001 exemplar. Profile bands show pointwise
percentiles, not confidence intervals. Nearest-neighbor diagnostics do not
establish privacy.
