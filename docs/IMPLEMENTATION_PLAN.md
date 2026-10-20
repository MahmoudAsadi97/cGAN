# cGAN electricity adaptation — implementation plan

Goal: implement the architecture in malvin0130/cGAN_EB against the dataset
released by this repository, with reproducible data preparation and evaluation.
This is a research implementation, not evidence of publication readiness.

## Evidence and constraints

- Destination starts at `af20fdc02f95f64d8b098d4271bafe12b6e0a314`.
- Upstream inspected revision: `885f78052a91fbc755e6b17a354c68849b5556ca`.
- Dataset: release `data`, asset `raw.zip`, 395085174 bytes;
  SHA-256 `934cc871e9fe18dbe30530472816d711ece878ec7cd8725d9d8417bbdb389bd8`.
- The dataset schema must be audited before assigning column semantics.
- Keep upstream MLP/BCE as an identifiable baseline. Any new loss or constraint
  is an explicit adaptation and must be evaluated separately.
- Split by meter before fitting transformations; do not randomly split daily
  windows from the same meter into train and test.
- Report results only from executed experiments. Synthetic test fixtures are
  software tests and cannot establish real-data quality.

## Work units

1. Audit release members, columns, units, timestamps and metadata. Preserve the
   release digest. Confirm which conditional attributes are actually available.
2. Implement strict long-format ingestion, complete daily profiles, deterministic
   disjoint meter splits, and train-only scaling with tests for leakage, missing
   slots, daylight saving transitions and invalid values.
3. Implement configurable MLP conditional GAN, stable BCE logits, seeded
   minibatches, validation-only model selection and checkpointed provenance.
   Test shapes, one training step and save/load generation.
4. Implement conditional empirical resampling baseline and evaluation in original
   units: pointwise distributions, energy, peak, ramps, temporal dependence,
   conditional support and nearest-neighbor diagnostics. Test analytically known
   metric cases and invalid inputs.
5. Add CLI commands/configs, multi-seed experiment instructions, attribution,
   CI smoke tests and a research protocol describing remaining publication gates.
6. Independently review and push the tested research branch. Preserve main.

## Review focus

Ambiguous timestamp/units, meter IDs reused across years, non-96-slot days,
conditions absent from train, and accidental test-driven checkpoint selection.
