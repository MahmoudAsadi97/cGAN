# Flow and dropout development screen

All eight declared runs completed: two conditional flow-matching runs, four
cGAN dropout variants, and the two matching cGAN dropout-0.4 controls from the
earlier momentum screen. Each used training seeds 101 or 202, a batch size of
512, 6,000 training updates, and 3,071,016 real-profile presentations from the
same inner training data. The six cGAN runs share global standardization,
Adam beta1=0 and all settings except discriminator dropout. The original cGAN
topology is preserved in these variants. Flow matching is a separate comparator.

The declared update budgets and real-row presentations match. They are not equal
computational budgets: each cGAN update comprises a discriminator/generator
update pair, while flow matching uses one velocity-network optimizer update.
Sampling algorithms and validation costs also differ. Recorded elapsed times
reflect the particular concurrent CPU runs and are not controlled speed tests.

## Selected inner-validation results

All values below use the same 256 real days per condition, generation seed
20261022 and common physical-unit normalization. Each neural checkpoint was
selected by its minimum profile SW1 on these same validation curves. Lower SW1
and Wasserstein errors are better; the diversity ratio measures generated to
real variance, with 1 denoting equality of that aggregate statistic.

| Method | Training seed | Selected step | Normalized profile SW1 | Daily-energy W1 (kWh) | Ramp W1 (kW) | Diversity ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Conditional flow matching | 101 | 6,000 | 0.005768 | 2.434 | 0.191 | 1.376 |
| Conditional flow matching | 202 | 4,000 | 0.005716 | 2.427 | 0.196 | 1.298 |
| cGAN, dropout 0 | 101 | 6,000 | 0.011699 | 5.784 | 0.168 | 0.567 |
| cGAN, dropout 0 | 202 | 6,000 | 0.012027 | 5.744 | 0.169 | 0.648 |
| cGAN, dropout 0.2 | 101 | 6,000 | 0.010912 | 5.367 | 0.168 | 0.611 |
| cGAN, dropout 0.2 | 202 | 6,000 | 0.012238 | 5.879 | 0.164 | 0.615 |
| cGAN, dropout 0.4 control | 101 | 5,000 | 0.012346 | 5.742 | 0.174 | 0.627 |
| cGAN, dropout 0.4 control | 202 | 6,000 | 0.012232 | 5.559 | 0.154 | 0.665 |
| Conditional mean | — | — | 0.020827 | 9.508 | 0.283 | 0.000 |
| Training-day resampling | — | — | 0.005269 | 2.208 | 0.046 | 1.271 |
| Empirical Gaussian copula | — | — | 0.005414 | 2.365 | 0.123 | 1.245 |

Reducing dropout provides no consistent solution to the cGAN's distribution
errors at this budget. Dropout 0 improves selected SW1 slightly for both seeds,
while dropout 0.2 improves seed 101 and leaves seed 202 approximately unchanged.
All six cGANs retain substantially less variance than the validation sample and
have much larger profile and daily-energy distribution errors than the strong
controls. Five of the six cGAN checkpoints are selected at the budget boundary;
this screen does not establish convergence or rule out longer-run effects.

Basic flow matching has much smaller profile SW1 and daily-energy error than
these cGAN runs, but both flow seeds remain behind resampling and copula on
those endpoints. Its ramp W1 is approximately 0.19 kW, versus 0.046 for resampling
and 0.123 for copula. This temporal mismatch persists despite a competitive
profile SW1 and illustrates why one metric cannot establish profile realism.
Generated variance also exceeds the observed validation variance. The empirical
controls themselves have diversity ratios above 1 on this finite sample, so
these ratios alone do not isolate a model-specific source of excess variation.

These are two-seed, validation-selected development comparisons. They are not
independent final-validation results, significance tests, proof of superiority,
or evidence of publication readiness. The two dropout-0.4 controls are reused
historical trajectories, not new independent training replicates. No original
TEST data were opened or scored for this summary, and Phase 2 has not started.

## Saved record and reproduction

`checkpoints.csv`, `learning_curves.csv` and `manifest.json` are byte-identical
copies from `artifacts/phase1-development/flow_dropout_summary`. The manifest
binds all eight source run manifests, the matched baseline manifest, the renderer
source and the generated artifacts. Each bound run manifest records its source
hashes, commit and checkpoint hashes. Full per-condition metrics, all three
baseline sampling draws, summary JSON and both curve figures remain in that
ignored artifact directory. The copied receipt also lists hashes for those
larger artifacts; they are not duplicated here.

The renderer read saved reports and verified artifact hashes. It did not train,
deserialize checkpoints or recompute model scores. Reproduce into a new directory:

```bash
.venv/bin/python scripts/summarize_development.py \
  --runs \
  runs/phase1-development/flow_screen/flow_s101 \
  runs/phase1-development/flow_screen/flow_s202 \
  runs/phase1-development/dropout_screen/standard_b0_d0_s101 \
  runs/phase1-development/dropout_screen/standard_b0_d0_s202 \
  runs/phase1-development/dropout_screen/standard_b0_d2_s101 \
  runs/phase1-development/dropout_screen/standard_b0_d2_s202 \
  runs/phase1-development/momentum_screen/standard_b0_s101 \
  runs/phase1-development/momentum_screen/standard_b0_s202 \
  --baselines artifacts/phase1-development/baselines_screen \
  --output artifacts/phase1-development/flow_dropout_summary_rerender
```
