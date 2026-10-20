# V3 research design

Status: proposed for review. No V3 model has been implemented or trained.

## Objective

Adapt the conditional GAN in `malvin0130/cGAN_EB` to the supplied Fluvius
2024 electricity data and establish a reproducible experimental basis for a
research paper. The initial question is whether a reference-style conditional
GAN can generate realistic daily net grid-import profiles for held-out meter
traces, conditioned on reported heat-pump, EV and PV status, season and day type.

This is daily scenario generation. It is not individual-household forecasting,
appliance disaggregation, or proof of coherent synthetic annual trajectories.
Implementation correctness and favorable scientific results are separate
requirements. A failed comparison must remain visible.

All new work belongs to `V3` in `MahmoudAsadi97/cGAN`. The branch starts from
main commit `af20fdc02f95f64d8b098d4271bafe12b6e0a314`. Do not modify or merge
`main`, `V1`, `V2`, other branches, or unrelated projects.

## Evidence inspected

- Reference source: `https://github.com/malvin0130/cGAN_EB`, commit
  `885f78052a91fbc755e6b17a354c68849b5556ca`.
- Dataset: `https://github.com/MahmoudAsadi97/cGAN/releases/download/data/raw.zip`,
  395,085,174 bytes, SHA-256
  `934cc871e9fe18dbe30530472816d711ece878ec7cd8725d9d8417bbdb389bd8`.
  The downloaded bytes match the digest published in the release.
- The archive contains eight CSVs and an embedded source-definition workbook.
  Its legend explicitly defines local CET/CEST timestamps and interval energy
  in kWh. The literal `Z` in CSV timestamp strings must not override that
  definition.
- Existing V2 work was inspected at
  `1ca62d0fbe1d77f916f8904e812db7d720f8cf53`. It documents exact annual trace
  aliases and a corrected split. Its numerical experiment claims are historical
  records, not verified V3 results; some status documents disagree.
- V1's tracked tree lacks modules imported by its commands. V3 must be checked
  as an installed package from a clean checkout, not only from a working tree.

The reference repository lacks its original real training dataset. Its original
paper's full text has not been verified during this inspection. The supplied
reference code and its documented citation are the available method evidence.

## Approach

Recommended: implement a compact, faithful reference adaptation with a robust
data pipeline and controlled comparisons. Reuse reviewed ingestion and identity
curation logic from V2 where appropriate, with explicit provenance and fresh
tests. Do not import its entire experimental history or inherit trained models.

Two alternatives were considered:

1. Continue V2's spectral-normalized hinge GAN, distribution losses, and output
   calibration. This could address known weaknesses, but it changes several
   elements of the reference method and complicates attribution of improvements.
2. Start with a new convolutional or attention-based generator. This requires
   a separate research hypothesis and larger comparison budget.

Both alternatives remain possible later experiments. The first V3 milestone
establishes a trustworthy reference baseline and the evidence needed to justify
an extension.

## Data contract and split

Use the nine named source columns, never positional column assumptions.
Validate numeric fields, binary metadata, contract category, dates, within-day
slots and stable meter metadata. Stream ZIP members in bounded chunks.
Keep raw data outside Git, verify its checksum, and never rewrite the archive.

The primary target is signed net grid import:

`net_kWh[t] = Volume_Afname_KWh[t] - Volume_Injectie_KWh[t]`.

Retain negative export, genuine zero days and measured outliers. Do not copy
the reference's 0-to-100 filter. Exclude the two local daylight-saving dates,
2024-03-31 and 2024-10-27, from the fixed-length daily experiment. Reject or
explicitly account for other incomplete or malformed days; never interpolate
them silently. Report exclusions and metadata discrepancies.

Primary resolution: 96 quarter-hour energy values per day. The hourly
sensitivity sums consecutive groups of four, producing 24 hourly kWh values.
Quarter-hour interval-average kW equals four times the interval kWh. Net import
does not identify gross consumption or PV generation.

Conditions use the explicit mapping:

`condition = ((4*HP + 2*EV + PV)*2 + weekend)*4 + season`,

with weekend = Saturday/Sunday and season = winter 0, spring 1, summer 2,
autumn 3 using meteorological months. There are 64 possible conditions. Their
technology semantics differ from the reference's EV-charging-level categories.
Meter identifiers are grouping keys only, never model inputs.

Preserve V2's original 70/15/15, technology-stratified, seed-42 meter allocation
and its subsequent deterministic identity curation. Do not generate a new test
set from previously exposed development traces. Fingerprint date-aligned
complete retained float32 net traces, canonicalizing signed zero only for
identity comparison. Confirm matching hashes by direct equality and require
matching condition metadata. Keep one canonical representative using original
split priority train, validation, test and V2's within-split representative rule.

V2 reports 33 aliases removed, leaving 1,665 training, 353 validation and
349 test meters. These are expected reproduction counts, not grounds to force
the audit to pass. Disagreement must be investigated. Apply the same retained
meter membership to the hourly data.

Test data may be accessed for deterministic schema, identity and provenance
checks. Model-performance and distribution summaries on test remain prohibited
during development. Carry forward known prior access history; opening a new
branch does not reset it. Exact trace deduplication does not prove household
independence or eliminate near-duplicate dependence.

Fit every scaler and other learned transform on the retained training set only.
Persist hashes, split membership, label definitions, units, exclusions and
scaler parameters. Preserve original metadata and curation provenance.

## Model and training

Implement the reference generator with Gaussian latent dimension 24,
one-hot conditions, hidden widths 64/128/64 and LeakyReLU slope 0.2. Its output
is linear with 96 or 24 values. Use the reference discriminator widths
64/128/256, LeakyReLU 0.2 and dropout 0.4.

Use non-saturating binary cross-entropy with logits, equivalent to the
reference's sigmoid plus BCE but numerically stable. Preserve the reference
Adam learning rates: generator 0.0001 and discriminator 0.0002. Record all Adam
parameters explicitly. Average real/fake discriminator terms as in the
reference. Detach generated samples during discriminator updates, freeze
discriminator gradients during generator updates, and shuffle training batches.

Use train-only global min/max normalization for the reference arm. Any
standardization, alternative loss, temporal regularization, resampling policy,
or output calibration is a separately named experiment. No output clipping.
Handle degenerate normalization and unsupported labels with clear errors.

Separate configurations for fast correctness checks, validation pilots and
confirmatory experiments. The reference's 10,000-epoch budget is documented,
not adopted blindly for this much larger dataset. Benchmark actual throughput
before freezing the full budget. Compare neural arms under a declared matched
budget and report actual steps, epochs, best checkpoints and runtime.

Save model, optimizer, configuration, scaler, dataset hash and random states.
Validate resume compatibility and use separate run directories. Repeated runs
must not overwrite earlier attempts. Validation checkpoint selection uses a
fixed condition-stratified subset and fixed generation noise. Adversarial loss
alone must not select a checkpoint.

## Comparators and evaluation

Required comparisons:

- Conditional training-profile resampling: a strong empirical distribution
  baseline, explicitly unsuitable as a privacy guarantee.
- Conditional mean: a deliberately low-diversity control.
- Conditional Gaussian copula: a non-neural joint-distribution baseline.
- Unconditional GAN: matched architecture and budget without conditioning.
- Conditional reference GAN at quarter-hour and hourly resolutions.

Use the same curated membership for all methods. Define the primary selection
metric before confirmatory runs: equally weighted conditional sliced
Wasserstein-1 distance on complete normalized profiles, using a fixed set of
128 unit projection directions. Report the exact estimator, sampling counts,
seed and projection checksum. Fit normalization on train only.

Evaluate fidelity in physical units: per-slot Wasserstein distances,
daily net-energy distributions, maximum interval-average power, signed
within-day power differences, autocorrelation distributions, mean and quantile
profiles, and diversity. Report condition-level sample counts and both
condition-macro and sample-weighted summaries. Compare resolutions on a common
hourly representation as well as at their native resolutions.

Do not pair independent generated and real days and interpret their pointwise
RMSE as forecasting error. Do not port the reference's invalid raw-value
"KL divergence", or label scalar moment distances as image FID.

Declare five neural seeds (11, 22, 33, 44, 55); retain all outcomes, including
failed or collapsed runs. Use paired comparisons against matched baselines.
Separate training-seed variability from uncertainty over real meter traces.
Use stratified meter-cluster resampling for uncertainty, preserving all sampled
days within each selected meter. Synthetic sampling variability must also be
reported. Daily rows are not independent confidence-interval units.

Check memorization by comparing generated-to-training proximity with
held-out-real-to-training proximity under the same metric and conditioning.
This is a diagnostic, not differential privacy or a proof of privacy.

Freeze the protocol, hyperparameters, selection rule and candidate list before
final test evaluation. Record test access and do not retune after seeing final
results. A new research claim needs a literature-grounded contribution and
credible uncertainty, not merely a different dataset or a functioning model.

## Software layout and validation

Use a small installable Python package under `src/`, explicit dependency
constraints, reproducible configurations and a command-line workflow:
download, audit, prepare, train, generate, evaluate and summarize.
Keep ingestion, identity curation, modeling, training and evaluation separate.
Generate headless figures with a noninteractive plotting backend.

Tests must exercise malformed inputs, local-calendar/DST handling, units,
negative exports, alias removal, split separation, train-only scaling, condition
round trips, model gradients, checkpoint restoration and known metric behavior.
Run an end-to-end synthetic fixture without network access, plus a clearly
labeled real-data pilot. Neither substitutes for full scientific evaluation.
Build and install the package into an isolated environment to catch missing
tracked modules. CI must not download the full release or run long experiments.

Persist compact manifests and measured reports in Git. Keep raw data,
large arrays and model checkpoints in ignored artifact directories. Preserve
required upstream MIT notices and the Li, Dong and Qiu citation. Dataset access
does not establish redistribution rights; refer readers to the source release
instead of copying the dataset into the code repository.

## Resources and deliverables

The inspected cloud machine exposes a four-CPU quota, a 32-GiB memory limit,
and no NVIDIA device. Use bounded data loading and serialize memory-heavy
evaluation. Choose full training parallelism from measured pilot resources.
An unavailable GPU does not prevent correctness tests or CPU training, but
runtime and completed experiment scope must be reported accurately.

Deliverables are the runnable V3 package, tests, configurations, source and split
manifests, reproduction instructions, measured validation reports and a clear
record of outstanding publication requirements. Push all project commits only
to `V3`, using the repository owner's Git identity. No merge to main is part
of this work.

## Reference

Li, Y., Dong, B., and Qiu, Y. (2025). Conditional generative adversarial
network (cGAN) for generating building load profiles with photovoltaics and
electric vehicles. Energy and Buildings, 335, 115584.
DOI: 10.1016/j.enbuild.2025.115584.
