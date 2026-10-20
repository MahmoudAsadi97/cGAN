# Fluvius conditional GAN — V3 Phase 1

Phase 1 adapts a reference conditional GAN to **24 hourly net grid-import energy
values per day** from Fluvius 2024. The five comparison methods are the
conditional GAN, a matched unconditional GAN, conditional mean, complete-day
training resampling, and a conditional empirical Gaussian copula.

**Status:** the frozen Phase 1 benchmark is complete. The reference cGAN
underperformed resampling and the Gaussian copula in all 64 conditions for
each of five training seeds on the primary profile-distribution metric.
Competitive fidelity, convergence and publication readiness are not established.
Read the [results and scientific assessment](docs/PHASE1_RESULTS.md) and
[complete review record](reports/phase1/README.md).
Development belongs to
[**`MahmoudAsadi97/cGAN`, branch `V3`**](https://github.com/MahmoudAsadi97/cGAN/tree/V3);
preserve `main`, `V1`, and `V2`. Architecture
improvements, auxiliary losses, calibration, and federated learning require the
user's separate approval after the Phase 1 review.

| Method | Primary normalized profile SW1; lower is better |
| --- | ---: |
| Reference conditional GAN, five-seed mean | 0.033563 |
| Matched unconditional GAN, five-seed mean | 0.036304 |
| Conditional mean | 0.021675 |
| Conditional training-day resampling | 0.003781 |
| Conditional empirical Gaussian copula | 0.004032 |

Each fitted model was evaluated with three generation draws. Neural training
used a frozen **100-epoch CPU budget**, rather than the upstream 10,000 epochs.
All ten runs and thirteen evaluations completed; all seeds and uncertainty
limitations are reported. The code's 220 pre-experiment software tests passed.
Phase 2 has not started.

## Method and provenance

The hourly architecture follows
[malvin0130/cGAN_EB](https://github.com/malvin0130/cGAN_EB) at commit
`885f78052a91fbc755e6b17a354c68849b5556ca`: Gaussian latent dimension 24,
64 one-hot conditions, generator hidden widths 64/128/64 and linear 24-value
output; discriminator widths 64/128/256 with dropout 0.4. Both use LeakyReLU
slope 0.2. Training uses stable binary cross-entropy with logits and the reference
Adam rates (generator `1e-4`, discriminator `2e-4`; betas `0.9, 0.999`).
Normalization fits retained TRAIN only; physical outputs are not clipped.
This is an adaptation to a different dataset, not a numerical reproduction of
the original paper or its unavailable training data.

Citation: Li, Y., Dong, B., and Qiu, Y. (2025). *Conditional generative adversarial
network (cGAN) for generating building load profiles with photovoltaics and
electric vehicles*. Energy and Buildings, 335, 115584.
[DOI](https://doi.org/10.1016/j.enbuild.2025.115584).
The [upstream MIT notice](docs/UPSTREAM_LICENSE.txt) is retained. See
[provenance and prior data access](docs/PROVENANCE.md),
[research design](docs/V3_DESIGN.md), and the
[frozen experiment protocol](docs/PHASE1_PROTOCOL.md).
The [methods supplement](docs/PHASE1_METHODS.md) defines the implemented
estimators, baselines, units and uncertainty calculations.

## CPU setup and tests

Run commands from the project root with Python **3.12** and `uv` available.
The documented cloud machine has **4 CPU cores, 32 GiB RAM, and no GPU**.

```bash
bash scripts/setup_environment.sh
.venv/bin/python -m pytest -W error
```

Setup installs [pinned CPU dependencies](requirements-cpu.txt) and the editable
package. It does not download data or start training. The test suite uses
synthetic fixtures without the real archive. See
[environment instructions](docs/ENVIRONMENT.md) and
[CPU CI](.github/workflows/tests.yml). Setup caches and `.venv` are ignored;
cloud environment publication is a separate operation.

## Prepare the audited hourly dataset

Source: [the `data/raw.zip` release](https://github.com/MahmoudAsadi97/cGAN/releases/download/data/raw.zip),
395,085,174 bytes, SHA-256:

```text
934cc871e9fe18dbe30530472816d711ece878ec7cd8725d9d8417bbdb389bd8
```

```bash
.venv/bin/python -m fluvius_cgan download --output data/raw.zip
.venv/bin/python -m fluvius_cgan prepare \
  --archive data/raw.zip --output data/original96 --seed 42
.venv/bin/python -m fluvius_cgan curate \
  --data data/original96 --output data/curated96
.venv/bin/python -m fluvius_cgan hourly \
  --data data/curated96 --output data/hourly24
```

Prepare **96 quarter-hours first**, remove exact complete-trace aliases, then
sum each adjacent group of four into hourly kWh. Source timestamps use the
source-defined Brussels local calendar despite their literal trailing `Z`.
Both DST transition dates and invalid days are excluded without imputation.
Net import is import minus injection; negative exports are retained.

Identity curation preserves original split priority and held-out membership;
it removes 33 aliases from 2,400 meters. The resulting TRAIN/VAL/TEST counts
are **1,665/353/349 meters** and **606,060/128,492/127,036 daily profiles**.
Each split contains `x`, `y`, `meter_id`, and `date`. Manifests bind split bytes,
canonical identities, source provenance, and train-only scalar bounds. See the
[data preparation record](reports/PHASE1_DATA_PREPARATION.json) and
[independent source identity audit](reports/V3_SOURCE_IDENTITY_AUDIT.json).
Exact float32 trace equality does not prove household independence or exclude
near-duplicates. Historical held-out access is documented in the provenance.

Use new output directories. Keep archives, prepared arrays, checkpoints, and
large generated outputs in ignored `data/`, `runs/`, `artifacts/`, or `.cache/`;
do not add them to Git.

## Run the frozen five-seed study

The tracked [machine-readable protocol](reports/PHASE1_FROZEN_PROTOCOL.json)
binds the prepared file hashes, settings, training seeds **11, 22, 33, 44, 55**,
and reporting seeds **31001, 31002, 31003**. The
[confirmatory config](configs/phase1_confirmatory.json) specifies **100 epochs**,
batch size 2,000, all training rows, and one CPU thread per run. Conditional
and unconditional arms each use all five training seeds: **ten neural runs**.
The other three methods fit TRAIN without neural training.

```bash
.venv/bin/python scripts/run_phase1.py \
  --data data/hourly24 \
  --protocol reports/PHASE1_FROZEN_PROTOCOL.json \
  --runs runs/phase1 --workers 4
```

Training has substantial CPU cost. Four workers share the four-core quota;
use fewer workers if other work needs CPU or memory. The launcher requires a
new study directory, verifies the frozen data hashes, and records configs,
source hashes, per-run logs, failures, and status in `runs/phase1/manifest.json`.
It does not silently retry or replace runs. Preserve failed attempts and all
five seed outcomes. If prepared hashes differ, investigate the source and
pinned environment before proceeding; do not replace the frozen protocol to
bypass the mismatch.

Checkpoint selection uses fixed validation rows/noise and conditional normalized
profile sliced Wasserstein-1 at epoch 1 and every fifth epoch. Final evaluation
requires each full-budget run's validation-selected `best.pt`. Selection draws
and reporting draws are separate. The 100-epoch budget does not guarantee
convergence; report learning curves and boundary-selected checkpoints.

## Evaluate and render the record

After all ten runs complete, evaluate the frozen study and render the saved
reports using fresh output directories:

```bash
.venv/bin/python scripts/evaluate_phase1.py \
  --data data/hourly24 \
  --protocol reports/PHASE1_FROZEN_PROTOCOL.json \
  --runs runs/phase1 --output artifacts/phase1-evaluation
.venv/bin/python scripts/render_phase1.py \
  --runs runs/phase1 --evaluations artifacts/phase1-evaluation \
  --output artifacts/phase1-figures
```

The evaluator saves thirteen study reports, a seed summary, clustered
comparison and memorization diagnostics, and access/provenance logs. The renderer
saves PNG/PDF figures and CSV tables from those records.

To package verified completed checkpoints, reports and figures for transfer:

```bash
.venv/bin/python scripts/archive_phase1.py \
  --runs runs/phase1 --evaluations artifacts/phase1-evaluation \
  --figures artifacts/phase1-figures --output artifacts/phase1-record.zip
```

The ZIP includes a SHA-256 inventory and excludes the source dataset. Keep the
large ZIP outside Git. After extraction, selected models are under
`runs/conditional/seed_<seed>/best.pt`; generation needs the checkpoint and
installed package, without the original data arrays.

The primary endpoint is condition-macro normalized profile sliced
Wasserstein-1. Secondary diagnostics cover daily energy, peak power, ramps,
slot distributions, ACF, diversity, and condition counts. Whole-meter clustered
uncertainty, training-seed variation, generation variation, and nearest-training
profile diagnostics have separate interpretations; see the
[protocol](docs/PHASE1_PROTOCOL.md). Test-performance accesses must remain in
the append-only access record. Do not tune on final test results.

Figures and numerical reports support review of fidelity, comparators,
convergence, and limitations. Favorable metrics would not establish household
privacy, population equipment prevalence, future-year generalization, or
publication readiness. Phase 2 waits for the user's review and approval.

## Generate hourly profiles

For a completed conditional run, choose a condition `0..63`:

```text
condition = 8*(4*HP + 2*EV + PV) + 4*weekend + season
```

`HP`, `EV`, `PV`, and `weekend` are binary; season is winter `0`, spring `1`,
summer `2`, autumn `3`. Condition `56` means HP/EV/PV present, a winter weekday.
Equipment flags are reported metadata, not proof that unreported equipment is
absent.

```bash
.venv/bin/python -m fluvius_cgan generate \
  --checkpoint runs/phase1/conditional/seed_11/best.pt \
  --condition 56 --count 100 --seed 31001 \
  --output artifacts/generated_winter_weekday.csv
```

The CSV contains `hour_00` through `hour_23` in physical **kWh per hourly
interval**, with negative values allowed. Its JSON sidecar records the
checkpoint hash, condition, count, seed, and units. Existing generation outputs
are rejected. These daily scenarios are not household forecasts or coherent
synthetic annual trajectories.
