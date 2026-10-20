# Complete validation-selector histories

After `collect_reference_study.py` produces a **complete** reference report,
render the supplement with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python scripts/plot_training_histories.py \
  --report runs/reference_study/completed_reference_validation.json \
  --output-dir reports/figures \
  --raster-dir runs/reference_study/figures
```

The example collector path is a placeholder for the actual completed report.
The outputs are `reference_training_histories.svg`, the matching PNG, and
`reference_training_histories.provenance.json`; existing outputs are refused.
Rendering remains pending until the full reference study is collected.

The four panels retain **all 26 epochs and all five seeds** for `candidate96`,
`bce96`, `unconditional96` and `bce24`. Diamonds mark each retained best epoch,
verified as the earliest minimum. The script checks the collector's complete,
validation-only status, run coverage, SHA256 and byte bindings for every
`history.json` and `summary.json`, ordered epochs, finite scores, completed
budgets, selector configuration and agreement between summary and collector.
It reads these JSON files only; it does not open checkpoints, profile arrays,
calibration maps or endpoint reports. Source and input hashes, exact plotted
values, runtime versions and output hashes are retained in the sidecar.

The y-axis is **condition-macro interval W1 in training-normalized units**,
not physical-unit performance. Each panel uses a separate y-axis scale:
scores with different standard/min/max normalization or 96/24-interval
resolution are not directly comparable. Conditional and unconditional BCE96
share the same normalization, resolution and selector metric. Lines show unsmoothed raw-generator selector values,
not calibrated endpoint metrics. These adaptively selected validation histories
do not establish convergence, comparable optimization difficulty or TEST
performance. Adversarial losses are not plotted.
