# Verified hourly development pipeline

This implements the preparation and physical-comparison parts of
[REFERENCE_COMPLETION.md](REFERENCE_COMPLETION.md). It does not supply trained
hourly results or authorize TEST access. The first actual development derivation
is recorded in [hourly_derivation_verification.json](../reports/hourly_derivation_verification.json).

## Actual preparation record

On 20 October 2026, clean published source
`ec10d83359eb1d5a3fa965a882687569f0a86e39` created and independently verified the
published child directory. The integrated fixture suite passed **982 tests**
before publication; its source hashes are in
[hourly_software_verification_20261020.json](../reports/hourly_software_verification_20261020.json).
The actual derivation took 63.46 seconds and preserved 606,060 TRAIN profiles
from 1,665 canonical meters and 128,492 VAL profiles from 353 canonical meters.
An open-audit guard observed zero attempted TEST content accesses. No TEST24
artifact was created. The hourly TRAIN scaler bounds are
−10.769000053405762 and 22.850000381469727 kWh per interval.

The hourly training queue uses the same clean source pin, the separately
published reference protocol/configs at
`64a2e9aee98b0e3552ad3775df99f9cee35b2f8c`, and all five fixed seeds. It waits for
all three capacity pilot training/raw-validation receipts before using up to
three one-thread training slots. This scheduling condition uses job completion,
not model outcomes. Preparation success is not a training or performance result.

## Preparation and lineage

```bash
python scripts/derive_hourly.py --parent data/curated96 --output data/curated24
```

The output must be a new directory. The command streams compressed NumPy array
members in bounded chunks, verifies the direct curated96 parent, and creates only
TRAIN and VAL. Each hourly energy is the float64 sum of four consecutive float32
quarter-hour energies, cast back to float32. Labels, meter IDs, dates and row
order are preserved byte for byte. The hourly scaler is fitted to the complete
TRAIN partition. Negative net exports remain in the data.

The derivation manifest binds every parent development artifact, the parent
curation manifest and every child development artifact. Verification checks
every transformed row, its canonical membership, calendar and condition label.
The parent path is relative to the child directory, so moving the whole prepared
data tree preserves its hashes and checkpoint bindings. A different or modified
parent is rejected. Keep the relative parent/child layout when distributing data.

The child has an explicit derived-data qualification, rather than a copied claim
that a new curation audit was performed. Quarter-hour canonical representatives
remain fixed. Hourly uniqueness and independent households are not established.
Parent TEST expectations are retained from existing metadata only. The pipeline
does not read, hash, aggregate or copy parent TEST contents; child TEST must be
absent, and known symlink/hard-link aliases are rejected before development reads.

Training and validation verify this lineage and bind the derivation manifest to
checkpoints. Calibration, utility and paired comparison retain the qualification;
their saved calibrator contract continues to bind the four development files.
Collectors check report/checkpoint bindings without claiming a new array audit.
The ordinary evaluation CLI rejects derived-data TEST evaluation. A separately
sealed reference TEST implementation is still required.

## Common-hourly physical comparison

```bash
python scripts/compare_resolutions.py \
  --prepared96 data/curated96 --prepared24 data/curated24 \
  --checkpoint96 runs/curated_reference_bce96/seed_11/best.pt \
  --checkpoint24 runs/curated_reference_bce24/seed_11/best.pt \
  --map96 runs/curated_reference_bce96/seed_11/calibration.npz \
  --map24 runs/curated_reference_bce24/seed_11/calibration.npz \
  --output runs/reference_hourly_seed11.json
```

The runner verifies identical development rows, full TRAIN usage, matching seeds
and reference configurations. It selects the same VAL rows once. Native96 saved
maps are applied before sum4; no new map is fitted to the hourly sums. Both
pipelines are then evaluated in 24 dimensions with one-hour intervals, shared
projection directions and the same battery scenario. TRAIN-only conditional
mean, resampling, Gaussian-copula and independent-marginal controls are retained.
An optional same-seed candidate96 checkpoint/map can be included.

Native quarter-hour metrics are separate endpoints. This runner does not provide
the common-hourly ridge task or bootstrap comparisons; those stages require
their own verified scripts and reports. Five-seed aggregation must retain every
seed and use sample standard deviations. Per-seed confidence interval endpoints
must not be averaged.

Software checks use synthetic fixtures, including relocation, exact aggregation,
TRAIN-only scaling, row-order preservation, tampered lineage and forbidden TEST
access. Passing them establishes implementation behavior, not model quality.
