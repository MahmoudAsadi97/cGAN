# Executed validation — 20 October 2026

**Subsequent audit qualification:** exact full-year traces were later found under
different IDs across the original train/validation split. The historical model
results below are exploratory and do not establish independent-trace validation.
See [DUPLICATE_REMEDIATION.md](DUPLICATE_REMEDIATION.md) for the correction protocol.

All source changes are on V2. Main remains
`af20fdc02f95f64d8b098d4271bafe12b6e0a314`. This document separates execution evidence
from planned research. The implementation source used for the pilot corresponds to
published commit `3eca990e3a46a3705672a79b2bedf47ad9d533ae`.

## Software verification

- `python -m pytest -q`: **98 passed**.
- Independent review checked splitting, units, conditioning, architecture, metrics,
  safe output handling and provenance. Reproduced review findings were fixed with
  failing-then-passing regressions: test-byte checksums; output/sidecar protection;
  constant-range inverse scaling. Smoke baselines now use the actual model subset.
- GitHub Actions run 39354790824: both Python 3.10 and 3.12 jobs passed installation,
  tests and CLI help on the published implementation.
  https://github.com/MahmoudAsadi97/cGAN/actions/runs/39354790824
- Editable package installation succeeded; generation, checkpoint reload, validation
  evaluation and all three methods ran end to end on actual data.
- A synthetic test fixture exercises final-test evaluation and rejects changed test
  values. The **real test partition was not used for training, selection or evaluation**.

## Source audit and full preparation

The pinned 395,085,174-byte ZIP matched SHA256
`934cc871e9fe18dbe30530472816d711ece878ec7cd8725d9d8417bbdb389bd8`.
The source audit scanned all 84,326,400 rows; an independent calendar scan checked
every timestamp. Both reusable audit CLIs reproduced the results. There are no
unexpected missing, nonfinite, negative raw-channel, metadata or timestamp values.
See DATASET_AUDIT.md and reports/dataset_audit.json for the actual checks and caveats.

Full preparation processed all 2,400 meters, excluded 4,800 DST meter-days, and retained
873,600 profiles. Train/validation/test contain 611,520/131,040/131,040 profiles from
1,680/360/360 disjoint meters. Normalization uses only train.

Preparation completed before the review added NPZ digests to metadata. The resulting
array files were left unchanged and hashed using the same SHA256 routine to complete
the integrity manifest before training. The current prepare command writes those
hashes directly. Full split hashes are recorded in reports/pilot_full_seed42.json.

## Actual CPU pilot

Command:

```bash
fluvius-cgan train --prepared data/prepared96 --config configs/pilot.json --output runs/pilot_full_seed42
fluvius-cgan evaluate --prepared data/prepared96 --checkpoint runs/pilot_full_seed42/best.pt --output runs/pilot_full_seed42/validation.json --samples-per-condition 100
fluvius-cgan generate --checkpoint runs/pilot_full_seed42/best.pt --labels 0 8 56 --samples-per-label 10 --output runs/pilot_full_seed42/generated.npz
```

This used **all 611,520 training days**, 10 epochs, seed 42, batch 1024, hidden 64,
latent 24, two CPU threads and a fixed 2,048-profile validation selection sample.
Training took 153.33 seconds in the recorded environment. Best validation checkpoint
was epoch 9 (normalized mean-interval Wasserstein 0.0523061).

A separate seeded descriptive validation sample contained 100 days for each of 64
conditions: 6,400 profiles spanning all 360 validation meters. All methods used the
same labels and all 611,520 training profiles. Values below are equally weighted
condition means; lower distances are better.

| Metric | cGAN pilot | Conditional resampling | Conditional mean |
|---|---:|---:|---:|
| Interval Wasserstein, kWh | 0.537458 | 0.054542 | 0.176770 |
| Daily net energy Wasserstein, kWh | 14.189360 | 2.986623 | 10.189749 |
| Peak power Wasserstein, kW | 4.232336 | 0.495113 | 2.229067 |
| Signed power-change Wasserstein, kW | 2.523131 | 0.032723 | 0.239389 |
| Generated/real diversity ratio | 0.451274 | 1.159173 | 0.000000 |

**This pilot underperforms both empirical baselines on these distances.** A successful
training loop is not successful distribution learning. These results indicate that
more training, validation-driven investigation and the planned multi-seed comparisons
are necessary; improvement is not guaranteed. No significance test or final-test
performance claim follows from this single short run. The diversity ratio alone is
not a quality score. Negative values are valid for the signed net-import target.

The machine-readable report includes provenance, environment, hashes, epoch history,
per-condition primary metrics, overall and macro diagnostics. Thirty generated
profiles were verified finite with shape 30×96 and original-unit metadata. Checkpoints,
prepared data and generated arrays remain run artifacts excluded from source git;
commands above reproduce them. Observed package versions are in reports/environment.json.

## Still required before a paper claim

Full declared training budgets and at least five seeds; unconditional and hourly
ablations; common-hourly comparisons; meter-level uncertainty; frozen decisions
followed by final-test evaluation; and a
scientific contribution justified against current literature. See RESEARCH_PROTOCOL.md.

Official CSV/legend content identity and the current custom Fluvius licence are
now documented in [DATA_PROVENANCE.md](DATA_PROVENANCE.md). Attribution and other
reuse conditions still apply; this verification does not establish privacy or
household independence.
