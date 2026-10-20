# Phase 1 implementation and validation plan

**Goal:** Validate the reference 24-hour conditional GAN on the identity-curated
Fluvius data, with reproducible software and an honest research assessment.

**Architecture:** A small Python package separates data preparation, reference
networks, training, baselines and evaluation. The 96-slot source representation
is used to identify aliases; all Phase 1 models use 24 hourly energy values.

**Technology:** Python 3.12, NumPy, pandas, SciPy, PyTorch, pytest and Matplotlib.
CPU execution is supported. Dependencies and experiment settings are recorded.

**Design:** [V3_DESIGN.md](V3_DESIGN.md).

## Constraints

- Work and push only on `V3`; preserve `main` and unrelated projects.
- Owner author/committer identity; retain upstream MIT notices and citation.
- No changed GAN architecture, auxiliary losses, calibration or federated
  learning in Phase 1. Phase 2 requires a separate user approval.
- Five training seeds: 11, 22, 33, 44, 55. All runs, including failures, count.
- Train-only transforms; preserve original held-out membership and remove
  exact complete-trace aliases before hourly aggregation.
- Do not claim publication readiness from software tests or a short pilot.
- Keep source data, model weights and large generated arrays outside Git.

## Review focus

1. Missing tracked package modules: install and test a built wheel outside the
   source checkout.
2. Duplicate traces with different IDs or signed zeros: compare full canonical
   values and retain original split priority.
3. Resume and evaluation randomness: resumed CPU training must match uninterrupted
   training, and reporting draws must differ from selection draws.
4. Degenerate conditions and tied copula values: fail clearly on missing support,
   preserve constant slots, never manufacture nonfinite metrics.
5. Unit and uncertainty errors: sum quarter-hour energy, convert peaks to kW,
   resample meter clusters rather than independent daily rows.

## Task 1: Audited ingestion and canonical datasets

Files: `src/fluvius_cgan/{release,data,curation,prepared}.py`,
`tests/test_{release,data,curation,prepared}.py`.

Interfaces:

- `prepare_archive(archive_path, output_dir, resolution=96, seed=42, ...)`
  creates original meter-disjoint NPZ splits and metadata.
- `curate_prepared(prepared_dir, output_dir, expected_dates=None)` removes
  full-trace aliases without moving development traces into test.
- `load_split(path)` returns arrays `x`, `y`, `meter_id`, `date`.
- `aggregate_hourly(source_dir, output_dir)` keeps curated membership and
  sums groups of four; fits a new hourly scaler on training values only.

- [ ] Port the existing data tests first; verify absent implementation fails.
- [ ] Reuse reviewed V2 ingestion/curation with explicit source provenance.
- [ ] Add hourly-aggregation tests checking energy conservation, negative
  exports, unchanged meter membership and train-only scaler fitting.
- [ ] Run data tests, prepare the actual release, verify expected retained
  counts and independently check the output identity manifests.
- [ ] Review and commit the bounded data implementation.

## Task 2: Reference networks and resumable trainer

Files: `src/fluvius_cgan/{models,training}.py`,
`tests/test_{models,training}.py`, `configs/phase1_*.json`.

Interfaces:

- `Generator(profile_dim=24, num_classes=64, latent_dim=24)` and
  `Discriminator(profile_dim=24, num_classes=64)` consume tensors plus integer
  condition labels. `num_classes=0` represents the unconditional comparator.
- `train(data_dir, run_dir, config, resume=False)` reads only train/validation,
  saves `best.pt`, `last.pt`, `history.json` and a run manifest.
- `generate(checkpoint, labels, seed, batch_size=2048)` returns unscaled
  physical hourly-kWh profiles as a NumPy array.

- [ ] Write topology, gradient, reproducibility and resume tests before code.
- [ ] Implement exact hidden widths, activations, dropout, BCE-with-logits,
  Adam rates and explicit default betas 0.9/0.999.
- [ ] Add fixed conditional validation selection using Task 3's SW1 metric.
- [ ] Check input validation, deterministic CPU replay, saved normalization,
  finite losses, overwrite protection and incompatible-resume rejection.
- [ ] Run focused tests; review and commit.

## Task 3: Distribution metrics and clustered uncertainty

Files: `src/fluvius_cgan/{evaluation,uncertainty}.py`,
`tests/test_{evaluation,uncertainty}.py`.

Interfaces:

- `projection_directions(dimension, count=128, seed=2026)` returns fixed
  unit vectors.
- `sliced_wasserstein(real, fake, directions)` computes empirical SW1.
- `profile_metrics(real, fake, interval_hours=1.0, directions=None)` returns
  finite, explicitly named distribution, energy, peak, ramp and ACF metrics.
- `conditional_metrics(real, fake, real_labels, fake_labels, ...)` returns
  per-condition counts, condition-macro and sample-weighted summaries.
- `cluster_bootstrap_indices(meter_ids, technology, rng)` resamples complete
  meter clusters within technology strata.

- [ ] Test known translated/scaled distributions, row-order invariance,
  constant profiles, missing labels, physical units and clustered sampling.
- [ ] Implement the metrics with bounded memory and no invalid raw-value KL.
- [ ] Record ACF conventions and constant-profile counts.
- [ ] Run focused tests; review and commit.

## Task 4: Strong non-neural baselines and study interface

Files: `src/fluvius_cgan/{baselines,study,cli,__main__}.py`,
`tests/test_{baselines,study,cli}.py`, `pyproject.toml`.

Interfaces:

- `fit_baseline(kind, x, labels)` returns a model with
  `sample(labels, seed)` for conditional mean, resampling and Gaussian copula.
- `evaluate_run(data_dir, checkpoint_or_baseline, split, seeds, output)`
  uses equal real/generated counts and shared estimator settings.
- CLI: `download`, `prepare`, `curate`, `hourly`, `train`, `generate`,
  `evaluate`, `study`, `summarize`.

- [ ] Test support validation, tied/constant copula data and deterministic
  sampling before baseline implementation.
- [ ] Implement the frozen train-only copula specification.
- [ ] Test and implement the complete offline fixture workflow.
- [ ] Separate selection and reporting seeds; require an explicit frozen
  protocol record before test-performance evaluation.
- [ ] Build/install a wheel and run tests outside the source directory.
- [ ] Review and commit.

## Task 5: Resource pilot and frozen experiment budget

Files: `configs/phase1_pilot.json`, `configs/phase1_confirmatory.json`,
`docs/PHASE1_PROTOCOL.md`, `reports/phase1_pilot.json`.

- [ ] Run the actual curated hourly pipeline, checking recorded hashes/counts.
- [ ] Benchmark one full-training-data pilot; record wall time, losses and
  validation trajectories. A pilot is not a publication result.
- [ ] Set a feasible fixed confirmatory update budget from measured throughput
  and validation convergence, documenting the departure from upstream's
  10,000 epochs on an unavailable original dataset.
- [ ] Freeze architecture, seeds, stopping/selection rule, sample counts,
  baseline settings and primary endpoints before final test evaluation.
- [ ] Commit the frozen protocol and configurations before confirmatory runs.

## Task 6: Multi-seed experiments and research assessment

Files: `scripts/run_phase1.py`, `reports/phase1_*.json`,
`docs/PHASE1_RESULTS.md`, generated figures under `reports/figures/`.

- [ ] Train conditional and unconditional reference arms with all five seeds
  under the same budget. Record unsuccessful runs without replacing seeds.
- [ ] Evaluate all declared baselines on the same validation data.
- [ ] Inspect convergence and failure diagnostics; document any development
  changes before freezing the final choices. Never tune on test.
- [ ] Run the frozen test evaluation with three generation replicates;
  compute paired baseline differences, meter-cluster intervals, seed
  variability, diversity and nearest-neighbor memorization diagnostics.
- [ ] Generate inspectable tables and figures from saved metrics.
- [ ] State which scientific claims are supported, unsupported or unresolved.
  If validation exposes a Phase 1 limitation, report it rather than changing
  the architecture or claiming superiority.

## Task 7: Verification, reproducibility and handoff

Files: `README.md`, `.github/workflows/tests.yml`, pinned requirements,
`docs/PROVENANCE.md`, `docs/UPSTREAM_LICENSE.txt`.

- [ ] Run full tests, installed-package smoke checks and source-diff review.
- [ ] Have a separate reviewer inspect methodology and implementation.
- [ ] Fix confirmed issues and repeat affected checks.
- [ ] Record dependency versions, source/split/checkpoint hashes, runtime,
  commands and experiment access history.
- [ ] Push verified commits only to `V3`, confirm `main` is unchanged.
- [ ] Present Phase 1 for user review, explicitly stopping before Phase 2.
