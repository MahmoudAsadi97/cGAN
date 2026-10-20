# Reference validation runners

These runners implement the fixed comparisons in
[REFERENCE_COMPLETION.md](REFERENCE_COMPLETION.md). They evaluate VAL only and
do not supply completed five-seed evidence merely by existing. Use a clean
published worktree and retain its source hashes with every actual report.

## Raw unconditional reference

```bash
python scripts/evaluate_unconditional_reference.py \
  --prepared data/curated96 \
  --checkpoint runs/curated_reference_unconditional96/seed_11/best.pt \
  --training-seed 11 --samples-per-condition 100 \
  --output runs/curated_reference_unconditional96/seed_11/unconditional_reference_supplemental_validation.json
```

The original raw training-run report remains required and hash-bound. This
separately named supplement adds conditional mean, resampling, Gaussian and
independent-marginal controls, whole-profile projections and battery diagnostics
around the raw unconditional output. It never fits a conditional calibrator.
The checkpoint, complete history, final checkpoint and summary must establish
26 epochs, 30,784 updates and the declared full TRAIN cohort. The chosen best
checkpoint can precede epoch 26; that does not shorten the completed fit.

The supplement uses the current verified numerical implementation, including
the documented monotone quantile-interpolation correction. It does not replace
the older pinned native96 raw report or claim identical evaluator provenance.
The native96 utility runner without a calibrator uses exact empirical Gaussian
margins; calibrated native utility uses 1,025 knots. Keep that difference visible
and do not pool those Gaussian utility controls. Real-trained and resampling
utility controls remain directly matched.

Repeat at cap 1,000 with the separately named
`unconditional_reference_supplemental_precision.json`; preserve the primary
cap-100 result. Execute both for all five declared seeds.

## Matched physical contrasts

```bash
python scripts/compare_reference_uncertainty.py \
  --contrast conditioning --prepared96 data/curated96 \
  --candidate runs/curated_reference_bce96/seed_11/best.pt \
  --comparator runs/curated_reference_unconditional96/seed_11/best.pt \
  --output runs/reference_conditioning_seed11.json
```

The four explicit contrast modes are:

| Mode | Candidate minus comparator |
|---|---|
| `conditioning` | Raw conditional BCE96 minus raw unconditional BCE96 |
| `candidate_reference` | Calibrated selected candidate96 minus calibrated BCE96 |
| `resolution_raw` | Raw BCE24 minus hourly sum of raw BCE96 |
| `resolution_calibrated` | Calibrated BCE24 minus hourly sum of calibrated BCE96 |

Provide both native saved-map paths for calibrated modes; maps are rejected for
the raw conditioning contrast. Resolution modes also require `--prepared24`.
The candidate package must be declared as `--candidate-kind swd10` or
`--candidate-kind capacity` for its reference comparison. This argument is not
a scientific promotion decision.

Each report covers one matched training seed, the same 100 VAL rows per
condition selected with seed 2026, all five physical metrics and 500 paired
technology-stratified whole-meter bootstrap draws. Negative W1 differences favor
the candidate. Every condition and each seed's separate interval are retained;
these descriptive intervals are neither simultaneous nor selection-adjusted.

## Common-hourly utility

`scripts/evaluate_resolution_utility.py` accepts the same prepared data,
checkpoint, map and optional candidate paths as `scripts/compare_resolutions.py`.
It fits each predictor on all 606,060 matched hourly TRAIN profiles, using three
observed hourly lags, four clock features, 64 condition indicators and an
intercept. Ridge alpha remains 1. The target is one hour ahead, with 21 observed
windows per profile. Native96 next-quarter-hour RMSE is a different estimand.

Saved native96 maps are applied in bounded blocks before hourly aggregation.
Generated training profiles and coefficients are hash-bound. Paired utility
intervals use frozen errors on the same selected real rows; they do not refit on
VAL or quantify model-selection uncertainty. Real-trained, resampling-trained,
Gaussian-trained and persistence references are retained. Produce both primary
cap-100 and precision cap-1,000 reports for every seed.

Targeted synthetic verification covered 34 paired-contrast cases, 10 hourly
utility cases and 102 distinct unconditional-supplement cases before execution.
Independent reviews found and resolved provenance/alias guards. These checks
establish software behavior, not model performance.
