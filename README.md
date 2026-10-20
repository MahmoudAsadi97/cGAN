# Conditional generation of Fluvius electricity profiles

A reproducible adaptation of [malvin0130/cGAN_EB](https://github.com/malvin0130/cGAN_EB)
for the **2024 Fluvius residential smart-meter data** in this repository's
[`data` release](https://github.com/MahmoudAsadi97/cGAN/releases/tag/data).

**Status:** 88 tests pass locally and in CI; a released-data smoke run completed
on 48 households and 17,472 daily profiles. Its two-epoch cGAN underperforms the
resampling baseline and generates negative imports. See the [executed validation
and limitations](docs/VALIDATION.md). A short smoke run checks execution, not generative quality. Full
multi-seed experiments and publication validation remain necessary. See the
[research protocol](docs/RESEARCH_PROTOCOL.md) and [upstream audit](docs/UPSTREAM_AUDIT.md).

## What is implemented

- Streaming ingestion of all eight CSVs inside the pinned ZIP, without extracting
  or loading the roughly 6.39 GB of raw CSV data into memory.
- Complete daily profiles at 15-minute resolution (96 values) or hourly
  resolution (24 values). Hourly aggregation **sums interval kWh**.
- Joint conditions from source PV/EV/heat-pump indicators, season, and weekday /
  weekend; optional simpler condition sets for ablations.
- Equipment-stratified, household-disjoint train/validation/test splits, default
  70/15/15, and a training-only label vocabulary and normalization.
- The upstream MLP/BCE architecture, with stable logits loss and training fixes;
  an explicitly separate nonnegative `softplus` output variant.
- Seeded minibatches, fixed validation noise, validation-only checkpoint selection,
  saved scaling, class dictionary, optimizer states, hashes, and environment.
- Original-unit metrics for distributions, energy, peak power, ramps, and
  autocorrelation; conditional empirical resampling baseline and nearest-neighbor
  diagnostics. No fabricated FID, privacy guarantee, or quality scores.

## Data decisions that matter

The release dictionary specifies **CET/CEST local clock time**, although the CSV
timestamp strings end in `Z`. The default follows the dictionary: preserve those
clock readings and exclude the two Brussels daylight-saving transition dates.
Incomplete, duplicate, invalid, or negative import/export days are counted and
excluded. The pipeline never interpolates or silently clips them.

The default target is **grid import**, `Volume_Afname_KWh`. This is not total
behind-the-meter household demand in homes with PV. Export and signed net exchange
are optional targets; nonnegative generation is not applicable to signed net data.
EV=0 means not detected, not confirmed absence. Saturday/Sunday define weekend;
public holidays are not included. See [release dictionary evidence](docs/release_schema.json).

## Setup

Python 3.11 or newer. The implementation is on branch `research/fluvius-cgan`.

```bash
git clone --branch research/fluvius-cgan https://github.com/MahmoudAsadi97/cGAN.git
cd cGAN
python -m venv .venv
source .venv/bin/activate
# CPU environment used by CI; for CUDA use a suitable PyTorch installation.
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'
python -m pytest -q
```

The download is approximately 395 MB. Keep several GB of free disk space for
streaming preparation and checkpoints. Full training also loads the compact
prepared profile matrix into RAM. Do not use the smoke subset for paper claims.

## Run on the released data

```bash
python -m energy_cgan fetch-data --output data/raw.zip
python -m energy_cgan prepare --archive data/raw.zip \
  --config configs/data_15min.json --output data/prepared_15min
python -m energy_cgan train --data data/prepared_15min \
  --config configs/upstream_15min.json --output runs/baseline_seed42
python -m energy_cgan evaluate --data data/prepared_15min \
  --checkpoint runs/baseline_seed42/best.pt --split val \
  --output runs/baseline_seed42/validation.json
```

Use `configs/data_smoke.json` with `configs/smoke.json` for a short test using the
first six meters from each source CSV and two training epochs. This subset is
deterministic and intentionally not representative.

The release SHA-256 is checked before use:
`934cc871e9fe18dbe30530472816d711ece878ec7cd8725d9d8417bbdb389bd8`.
`prepare` also accepts an already downloaded archive and records its hash.
Check that the recorded hash matches the pinned release for reported experiments.

## Compare models and conditions

For the hourly adaptation, prepare a second folder using
`configs/data_hourly.json`, then use the same training settings. For the
nonnegative variant, use `configs/nonnegative_15min.json` on import data.
The generator latent dimension remains 24 in both resolution variants.

To ablate conditioning, copy a data config and set `conditioning` to `calendar`
or `pv_ev_calendar`, then prepare a separate folder. With the same complete input
and split seed, equipment strata and household membership remain the same.
Do not reuse a checkpoint with another prepared dataset: evaluation checks both
the dataset and metadata hashes.

```bash
python scripts/run_experiments.py --data data/prepared_15min \
  --config configs/upstream_15min.json --output runs/baseline_multiseed \
  --seeds 42 43 44 45 46 --split val
```

The runner reports mean and sample standard deviation across seeds. These are
**not household-level confidence intervals**. Its evaluation cap is 200 profiles
per condition by default; counts and sampling seed are recorded. Complete the
research protocol's household-clustered analysis separately.

Freeze the method and hyperparameters using validation results, then explicitly
evaluate the saved checkpoints with `--split test`. The default evaluation split
is validation. Never use test results to choose epochs or change hyperparameters.

## Generate and inspect outputs

```bash
python -m energy_cgan generate --checkpoint runs/baseline_seed42/best.pt \
  --condition 0 --num-samples 100 --seed 123 --output runs/generated.npz
```

Read the integer-to-condition mapping from `data/prepared_15min/metadata.json`.
Generated NPZ files contain `x` in interval kWh, `y` as condition IDs, and a JSON
`provenance` string with checkpoint hash, seed, label dictionary, and units. Linear
baseline outputs can be negative; report the violation rate rather than clipping.

| Output | Meaning |
|---|---|
| Prepared `dataset.npz` | Original-unit profiles, labels, household IDs, dates, split |
| Prepared `metadata.json` | Class dictionary, source digest, policy, split counts, dropped-day audit |
| Run `best.pt` / `last.pt` | Selected / final weights, optimizers, scaler, metadata, history |
| Run `manifest.json` / `config.json` | Input hashes, git state, environment, seed and settings |
| Evaluation JSON | Global and condition-level cGAN and bootstrap diagnostics |

Checkpoints support generation but **exact training resume is not implemented**.
Commands refuse to overwrite existing results. Raw data, prepared arrays, and
model outputs are excluded from git.

## Publication checklist and attribution

A new dataset alone does not establish methodological novelty. The remaining
work includes five-seed full-data runs, matched baselines and ablations, a strong
temporal generative comparator, downstream utility, household-level uncertainty,
and a justified contribution based on an updated literature review. Current
nearest-neighbor diagnostics do not establish privacy protection.

Upstream: Li, Y., Dong, B., & Qiu, Y. (2025). *Conditional generative adversarial
network (cGAN) for generating building load profiles with photovoltaics and
electric vehicles*. Energy and Buildings, 335, 115584.
[DOI](https://doi.org/10.1016/j.enbuild.2025.115584).
The [upstream MIT notice](THIRD_PARTY_NOTICES.md) is retained. Fluvius dataset
reuse terms must be checked separately before redistribution or publication.
