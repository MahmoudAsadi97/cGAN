# V2 research design

Goal: adapt malvin0130/cGAN_EB to the Fluvius release in this repository and provide
an executable, reproducible research baseline. All commits go to V2. No merge to main.

## Evidence

- Base main: af20fdc02f95f64d8b098d4271bafe12b6e0a314.
- Reference: 885f78052a91fbc755e6b17a354c68849b5556ca; MIT, Li et al. (2025),
  https://doi.org/10.1016/j.enbuild.2025.115584.
- Release data/raw.zip: 395085174 bytes, SHA256
  934cc871e9fe18dbe30530472816d711ece878ec7cd8725d9d8417bbdb389bd8.
- Eight residential groups by heat pump (HP), EV and PV, 2024 quarter-hour energy.
  Embedded legend says CET/CEST despite CSV strings ending in Z. Preserve the source
  wall-clock date/slot; do not reinterpret as UTC. Drop incomplete/duplicate days
  and local DST transition days, count exclusions, never silently impute.

## Decisions

Default target is signed net grid import = import minus injection, kWh per interval.
This is not gross household load, PV production, or EV charging demand. Also support
import and injection separately. Default resolution 96 quarter hours; optional 24
hourly sums provides a closer temporal comparison to upstream. No arbitrary
outlier clipping. Train-only scalar min/max transform, inverse transform before metrics.
Conditions: ((HP*4 + EV*2 + PV)*2 + weekend)*4 + season, with season winter=0,
spring=1, summer=2, autumn=3. One-hot dimension 64. No fabricated EV charger levels.

Split meters within technology strata 70/15/15 using seed 42 before fitting any
transform; every day of an EAN belongs to exactly one split. Use validation for
checkpoint selection and reserve test for final evaluation. Hold-out generalization
is to other meters within 2024, not future years. Record manifests and source hash.

Preserve upstream MLP widths (G 64/128/64, D 64/128/256), LeakyReLU .2,
D dropout .4, linear G output, Gaussian latent24, Adam G 1e-4 / D 2e-4,
default Adam betas. BCEWithLogitsLoss replaces sigmoid+BCE equivalently for stability.
Shuffle training data and detach fake samples for D updates. Include an unconditional
ablation. Export raw outputs without clipping.

Evaluate per condition and overall against conditional empirical-resampling and mean
profile baselines using Wasserstein distances of interval energy, daily net energy,
peak power, ramps, mean profile error, ACF differences, and nearest-neighbor diagnostics.
Nearest neighbors are diagnostics, not differential privacy or a privacy guarantee.
No image FID or invalid KL on raw time-series values. Multi-seed runs and meter-level
uncertainty remain required before publication claims; successful smoke tests are not
model quality evidence.

## Shared interfaces

Prepared directory: train.npz / val.npz / test.npz with x float32[N,T] in original
kWh/interval, y int64[N], meter_id unicode[N], date unicode[N]. metadata.json contains
resolution, target, units, timezone policy, source hash, label formula, split seed,
counts and exclusions. scaler.json has minimum and maximum fitted on train only.
The trainer loads only train and val; it records their file digests.

Model API: Generator(latent_dim=24, condition_dim=64, hidden_dim=64, output_dim=96),
Discriminator(input_dim=96, condition_dim=64, hidden_dim=64). forward(data, onehot).
Train API: train(prepared_dir, output_dir, config: dict) -> dict summary.
Generate API: generate(checkpoint, labels: ndarray, seed=42, batch_size=1024) -> ndarray
in original units. Checkpoints include config/scaler/data hashes/condition support.

Metrics API: evaluate_profiles(real, generated, interval_hours=.25) -> dict;
evaluate_conditional(real, real_labels, generated, generated_labels, interval_hours=.25)
-> dict; resample_baseline(train, train_labels, labels, seed=42) -> ndarray;
mean_baseline(train, train_labels, labels) -> ndarray;
nearest_neighbor_diagnostics(train, generated, heldout, max_samples=1000, seed=42) -> dict.
