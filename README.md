# Fluvius cGAN — V2

A reproducible adaptation of [malvin0130/cGAN_EB](https://github.com/malvin0130/cGAN_EB)
for the Fluvius residential electricity dataset in this repository's
[`data` release](https://github.com/MahmoudAsadi97/cGAN/releases/tag/data).
All development is on **V2**; `main` is unchanged. This is a research baseline;
passing tests or a short training run does not establish publication readiness.

## What it generates

Daily **net grid import** (`import − injection`) in kWh per interval, conditioned on
reported heat pump, EV and PV indicators, weekend and season. Negative net values
mean grid export. Choose 96 quarter-hour values (default) or 24 hourly energy sums.
Import-only and injection-only targets are also supported. This data does not reveal
gross household load, PV generation, or separate EV charging demand.

The MLP architecture and BCE objective follow the reference. V2 adds meter-disjoint
splits, train-only normalization, seeded training, validation-selected checkpoints,
physical-unit evaluation, empirical baselines, and provenance. Exact differences:
[UPSTREAM_ADAPTATION.md](docs/UPSTREAM_ADAPTATION.md).

## Install and run

Python 3.10 or later. Install a suitable PyTorch build for your CPU/GPU first if needed.
From the repository directory:

```bash
git switch V2
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q

# Download about 395 MB and verify the pinned SHA-256; no 6.4 GB extraction needed.
fluvius-cgan download --output data/raw.zip
fluvius-cgan audit --archive data/raw.zip --output reports/archive_audit.json

# Full dataset, all 8 equipment groups, all complete non-DST days.
fluvius-cgan prepare --archive data/raw.zip --output data/prepared96 --resolution 96

# First verify the entire training/generation/evaluation path with a small run.
fluvius-cgan train --prepared data/prepared96 --config configs/smoke.json --output runs/smoke
fluvius-cgan evaluate --prepared data/prepared96 --checkpoint runs/smoke/best.pt --output runs/smoke/validation.json

# Then run the declared full-data experiment budget; GPU is used when available.
fluvius-cgan train --prepared data/prepared96 --config configs/cgan96.json --output runs/cgan96_seed42
fluvius-cgan generate --checkpoint runs/cgan96_seed42/best.pt --labels 0 8 56 --samples-per-label 100 --output runs/generated.npz
```

Use `python -m fluvius_cgan` instead of the installed command if preferred.
PowerShell activation is `.venv\Scripts\Activate.ps1`. Run directories must be new
or empty; reuse is rejected to avoid mixing experiments. Checkpoints currently start
new runs only; optimizer states are saved for inspection, but resume is not implemented.
Raw data, prepared arrays, checkpoints and generated profiles are ignored by git.

For a faster ingestion check, use `--max-meters-per-group 3 --output data/prepared_smoke`.
That selects the earliest meters and is intentionally biased; it is only a software
smoke check. The training smoke config separately samples at most 4,096 training days
and runs 3 epochs. Neither shortcut is suitable for a manuscript's main results.

## Dataset handling

The supplied archive contains 2,400 residential meters and 84,326,400 quarter-hour
records for 2024. The legend says Belgian local CET/CEST even though strings end in
`Z`. Source-local days are preserved; the two DST transition days are excluded,
leaving 873,600 daily profiles. Do not parse these timestamps as ordinary UTC.
See [DATASET_AUDIT.md](docs/DATASET_AUDIT.md) for full-source checks, original legend
cell references and limitations. Meters are split within equipment groups:

| Split | Meters | Daily profiles |
|---|---:|---:|
| Train | 1,680 | 611,520 |
| Validation | 360 | 131,040 |
| Test | 360 | 131,040 |

The split seed defaults to 42. No EAN ID appears in more than one split. All scaler
statistics come from train. Prepared NPZ files contain `x`, `y`, `meter_id`, `date`;
`x` remains in original units. `metadata.json` records source hash, IDs, counts,
exclusions, resolution, target, timezone policy and labels. `scaler.json` records
training min/max. Metadata and data fingerprints bind checkpoints to evaluation.

The 64 V2 conditions are:

```text
technology = 4*HP + 2*EV + PV
label = (2*technology + weekend)*4 + season
season: winter=0, spring=1, summer=2, autumn=3
weekend: Saturday/Sunday=1, otherwise=0 (public holidays are not inferred)
```

For example 0 is HP0/EV0/PV 0, winter weekday; 8 is HP0/EV0/PV1,
winter weekday; 56 is HP1/EV1/PV1, winter weekday. V2's IDs do not mean the same thing
as upstream IDs. Equipment flags are reported/detected proxies, not perfect ground
truth. EV zero does not prove absence; small exports also occur in some PV 0 records.

## Research experiments

Run separate full-data conditional and unconditional experiments with five seeds:

```bash
python scripts/run_experiments.py --prepared data/prepared96 --config configs/cgan96.json --output runs/cgan96 --seeds 11 22 33 44 55
python scripts/run_experiments.py --prepared data/prepared96 --config configs/unconditional96.json --output runs/unconditional96 --seeds 11 22 33 44 55

# Hourly sensitivity: fit a new scaler and model after summing quarters.
fluvius-cgan prepare --archive data/raw.zip --output data/prepared24 --resolution 24
python scripts/run_experiments.py --prepared data/prepared24 --config configs/cgan96.json --output runs/cgan24 --seeds 11 22 33 44 55
```

The config filename describes the recommended primary experiment; output dimension
is read from the prepared metadata, so the same architecture settings also work at 24.
The experiment runner evaluates validation only. After freezing all decisions,
explicitly use the untouched test split for each final model:

```bash
fluvius-cgan evaluate --prepared data/prepared96 --checkpoint runs/cgan96/seed_11/best.pt --split test --output runs/cgan96/seed_11/final_test.json
```

Every evaluation compares generated profiles with training-only conditional resampling
and conditional means. It reports per-condition and overall energy, peak, ramp,
diversity, mean-profile and autocorrelation diagnostics. Negative outputs remain
visible; there is no hidden clipping. Nearest-neighbor diagnostics are bounded and
do not provide a privacy guarantee. The Python evaluation module also provides a
meter-cluster bootstrap with its uncertainty scope documented in its docstring.

Read [RESEARCH_PROTOCOL.md](docs/RESEARCH_PROTOCOL.md) for mandatory experiments,
uncertainty, leakage controls, final-test rules and manuscript completion criteria.
Read [VALIDATION.md](docs/VALIDATION.md) for what was actually executed in this revision.

## Attribution

Li, Y., Dong, B., and Qiu, Y. (2025). *Conditional generative adversarial network (cGAN)
for generating building load profiles with photovoltaics and electric vehicles.*
Energy and Buildings 335, 115584. [DOI](https://doi.org/10.1016/j.enbuild.2025.115584).
The upstream MIT notice is preserved in [UPSTREAM_LICENSE.txt](docs/UPSTREAM_LICENSE.txt).
Dataset licensing and provenance must be confirmed separately before redistribution
or publication; the upstream code license does not license this dataset.
