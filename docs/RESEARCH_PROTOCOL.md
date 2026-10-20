# Research protocol: conditional electricity-profile generation

This is a protocol for a reproducible adaptation and evaluation, not a report of completed publication experiments. The initial implementation covers the upstream-style MLP/BCE cGAN, conditional bootstrap, and diagnostic evaluation. No quality, superiority, privacy, or novelty claim follows from a smoke test or a single training run.

## Question and scope

**Research question:** How well does the daily conditional generation method of Li, Dong, and Qiu transfer, after retraining, to Belgian residential electricity profiles across recorded PV, EV, and heat-pump groups, and what changes when retaining 15-minute resolution?

The upstream training data are confidential and pretrained weights are not supplied. This is method transfer to another dataset, not pretrained-weight transfer or reproduction of the original paper's numerical results. Adding heat-pump conditioning, retaining quarter-hour resolution, and constraining nonnegative outputs are proposed extensions. Their novelty has not been established; a comprehensive related-work search remains necessary.

## Data and interpretation

The source is `raw.zip` in the [data release of MahmoudAsadi97/cGAN](https://github.com/MahmoudAsadi97/cGAN/releases/tag/data), inspected in [release_schema.json](release_schema.json). It contains eight Fluvius 2024 electricity CSVs, totaling 6,389,855,650 uncompressed bytes, plus a legend workbook and preparation log. The legend describes 2,400 residential meter IDs in eight equipment groups of 300; verify these counts against the complete ingestion rather than treating the legend as an ingestion result.

| Source field | Analysis use and qualification |
|---|---|
| `EAN_ID` | Group all records for this meter into one split. A residential meter is the available household proxy; do not claim independently verified household identity or cross-meter household linkage. |
| `Datum`, `Datum_Startuur` | Source date and interval start. The included legend identifies CET/CEST wall time; see the timestamp policy below. |
| `Volume_Afname_KWh` | Electricity imported from the grid during the following 15 minutes, in kWh. This is the primary target. |
| `Volume_Injectie_KWh` | Electricity injected into the grid during the following 15 minutes, in kWh. Export is a separate target; it is not total PV generation. |
| `PV_Installatie_Indicator` | Supplied PV flag; a positive flag records a reported installation. Preserve the source's evidence level. |
| `Elektrisch_Voertuig_Indicator` | Supplied EV flag. The legend warns that zero means not detected and does not establish absence of an EV. |
| `Warmtepomp_Indicator` | Supplied heat-pump flag. Its legend description appears to contain copied EV wording; document this source defect and verify the intended interpretation before strong heat-pump-specific claims. |
| `Contract_Categorie` | Verify residential scope in the full input; do not silently mix contract categories. |

Grid import is not whole-building demand when self-consumption exists. Net grid exchange, if explicitly requested in a separate experiment, is import minus export and may be negative. None of these measurements isolates appliance consumption or identifies a causal effect of equipment ownership. Dataset reuse and redistribution terms remain to be verified with the original provider; the upstream code's MIT license does not establish a license for these measurements.

### Timestamp and completeness policy

The archive legend states that `Datum_Startuur` uses CET/CEST (winter/summer time), although inspected values end in `Z`. The default therefore preserves the displayed source wall-clock date and time rather than converting the apparent `Z` values from UTC to Europe/Brussels. Record this dictionary-based decision and the contradictory serialization in every preprocessing manifest. A timezone-aware parser must not silently shift these records.

For fixed-length daily vectors, require 96 unique, aligned quarter-hour intervals on a source date, consistent equipment metadata, finite numeric volumes, and a valid meter ID. Reject and count duplicate or incomplete days; do not average duplicate intervals or silently fill gaps. Exclude Europe/Brussels DST-transition dates (2024-03-31 and 2024-10-27), even if serialization happens to provide 96 rows. Report exclusions by reason and equipment group. Check `Datum` against the preserved timestamp date and report disagreements. Any alternative UTC interpretation requires an explicit, separately reported sensitivity analysis; it must not replace the documented default without additional provider evidence.

Construct 24 hourly kWh values by **summing** each consecutive four valid quarter-hour volumes. Keep the 96-point representation as 15-minute kWh. For resolution-comparable power diagnostics, divide interval energy by its duration in hours: hourly kWh/1 h and quarter-hour kWh/0.25 h. Preserve daily energy under aggregation. Do not transfer upstream's arbitrary 0–100 filter or silently clip generated samples; report invalid/negative outputs explicitly.

## Splits, scaling, and conditions

Use deterministic, equipment-stratified **70/15/15 percent train/validation/test splits by `EAN_ID`**, keeping every date of each meter together. Save the actual assignments and their hash. Stratify on the supplied PV/EV/heat-pump tuple, validate constant metadata within meters, and fail or explicitly document how inconsistent or undersized strata are handled. Assert pairwise-disjoint ID sets. Fit scaling and label mappings on training data only; persist them and apply unchanged to validation, test, and generated outputs. Never normalize each test day using its own extrema.

This split measures generalization to unseen residential meters within the observed 2024 setting, with shared dates and weather context across splits. It does not establish future-year, geographic, causal, or zero-shot equipment generalization. A chronological holdout is a separate study if future-period claims are pursued.

Compare nested condition sets with identical data splits:

1. Calendar: season and weekday/weekend.
2. Calendar plus supplied PV/EV flags.
3. Calendar plus supplied PV/EV/heat-pump flags.

Map observed training combinations to contiguous class IDs and save the vocabulary. Treat unobserved conditions explicitly; do not generate them and call the output validated. The upstream weekend category includes holidays; a weekday/weekend-only implementation does not. Any Belgian holiday treatment must be explicitly sourced, configured, and applied consistently. Do not infer charger level, adoption date, or equipment absence from load shape.

## Comparisons and current status

| Experiment or requirement | Current scope | Required before associated publication claim |
|---|---|---|
| Upstream-style hourly MLP/BCE cGAN | Implemented baseline scope | Train and evaluate on the complete audited data with the frozen protocol. |
| Conditional training-profile bootstrap | Implemented comparison scope | Use the same conditions and sample counts; include real-versus-real reference variation. Bootstrap is a fidelity reference, not a privacy-preserving release method. |
| 24 hourly versus 96 quarter-hour outputs | Configurable adaptation | Match training budgets and report parameter counts, wall time, and energy/power units. Compare common hourly diagnostics after aggregation and quarter-hour diagnostics separately. |
| Calendar versus equipment conditions | Configurable conditioning study | Run all nested conditions with the same split and training budget; examine sparse strata and worst-performing groups. |
| Linear versus softplus output | Explicit optional extension | Use linear output for source fidelity. Evaluate softplus only for nonnegative targets; it changes the model and cannot represent exact zero or signed net exchange. |
| Distribution and temporal diagnostics | Implemented diagnostic scope | Inspect metric definitions and degenerate-condition handling; report full test results without selective omission. |
| Five independent training seeds | Required experiment | Run at least five seeds per comparison; a smoke run is not this experiment. |
| Stronger time-series generative baseline | Pending | Add at least one established temporal generator with comparable conditioning, tuning effort, and training budget. Select it after the related-work review. |
| Downstream utility | Pending | Predefine and run a useful task, such as train-on-synthetic/test-on-real forecasting with real-trained and bootstrap controls. Use separate task splits and avoid leakage. |
| Meter-clustered confidence intervals | Required analysis | Resample complete held-out meters, retaining their daily dependence; report seed variation separately. |
| Privacy assessment | Diagnostic only | Nearest-neighbor distances do not establish privacy. Formal privacy or attack-resistance claims need a defined threat model and additional evaluation. |

Changing representation, condition size, or output activation changes the model; label each comparison accurately. Preserve source hidden widths, dropout, BCE objective, and Adam learning rates in the baseline. See [UPSTREAM_AUDIT.md](UPSTREAM_AUDIT.md) for discrepancies between the paper and released code.

## Training, selection, and evaluation

Before final experiments, freeze data exclusions, the seed list, hyperparameter search space, stopping/selection rule, metrics, and all baselines. Use the same split, maximum optimizer-update budget, validation opportunities, and comparable tuning budget for model comparisons. Record batch size, actual updates, parameter count, runtime, and any divergence. Width and latent dimension changes must be named ablations rather than unreported tuning advantages.

Use validation data to choose hyperparameters and checkpoints. Predeclare one primary validation criterion or an explicit composite; do not choose whichever test metric improves. Use the test set only after this procedure is frozen. Draw repeated synthetic samples at documented counts for each evaluated condition and separate sampling randomness from training seeds. Report seed-level results and all runs, including failures.

The final evaluation should cover:

- **Distribution:** conditional marginal discrepancies; daily energy, peak power, near-base/near-peak quantiles, load factor, and peak-time distributions. Define histogram bins and smoothing before test evaluation; valid probability histograms are required for KL or JS calculations.
- **Temporal structure:** mean profile, dispersion, ramp distributions, within-day autocorrelation, and cross-hour dependence. Assess tails as well as averages. Independently generated daily profiles do not demonstrate continuity or dependence across days.
- **Conditional fidelity and coverage:** group-level metrics, actual meter/day counts, equal-weight condition summaries, population-weighted summaries when justified, and worst-group performance. Keep absent or degenerate groups visible.
- **Validity and diversity:** negative/nonfinite fraction, energy-scale validity, duplicate rate, and distances to training and held-out real profiles under a specified, train-fitted distance metric. Compare against real-to-real distances; do not interpret distance alone as a privacy guarantee.
- **Uncertainty:** at least five training seeds, paired comparisons on common test meters, and 95% confidence intervals with whole-meter cluster resampling. Document the number of resamples and how generative Monte Carlo uncertainty is handled. Do not treat thousands of correlated daily profiles as thousands of independent households.

Report Gaussian moment distances by their actual feature space, not as Inception-network FID. Peak time, ramps, and threshold durations require declared definitions and interval units. Improved summary statistics alone do not prove downstream utility, privacy, physical interpretability, or superiority over other generators.

## Reproducibility and release gate

Each result must trace to a manifest containing source release URL and asset name, raw SHA-256, archive-member inventory, dataset/legend version, preprocessing settings and exclusion counts, timestamp policy, input schema, split assignments/hash, training-only scaler, class vocabulary, configuration, seeds, dependency/runtime versions, hardware, source commit, checkpoint hash, evaluation settings, and output hashes. Store commands and machine-readable metrics beside each run. Preserve source data separately from generated data.

The inspected release reports `raw.zip` SHA-256 `934cc871e9fe18dbe30530472816d711ece878ec7cd8725d9d8417bbdb389bd8`; verify the downloaded bytes against it before training. A manifest must distinguish complete-data runs from limited-member, limited-row, or synthetic-fixture checks.

Before submission: complete ingestion and rights checks; resolve or qualify equipment metadata defects; complete the literature review; freeze the protocol; run the full comparison matrix and stronger baseline; establish downstream utility; compute clustered uncertainty; retain failed runs; and independently verify tables/figures against saved outputs. Do not populate missing experiments with invented numbers or describe planned work as completed.

## Sources

- Li, Y., Dong, B., & Qiu, Y. (2025). Conditional generative adversarial network (cGAN) for generating building load profiles with photovoltaics and electric vehicles. *Energy and Buildings*, 335, 115584. [DOI](https://doi.org/10.1016/j.enbuild.2025.115584), [NSF-hosted paper](https://par.nsf.gov/servlets/purl/10599218).
- [Upstream code, pinned commit](https://github.com/malvin0130/cGAN_EB/tree/885f78052a91fbc755e6b17a354c68849b5556ca), [MIT license](https://github.com/malvin0130/cGAN_EB/blob/885f78052a91fbc755e6b17a354c68849b5556ca/LICENSE).
- [Dataset release](https://github.com/MahmoudAsadi97/cGAN/releases/tag/data), its included `1.50 Aanleveirng DMK Elek Fluvius Legende 25200929.xlsx`, and [inspected schema](release_schema.json).
- [Original Fluvius dataset catalog](https://opendata.fluvius.be/explore/assets/1_50-verbruiksprofielen-dm-elek-kwartierwaarden-voor-een-volledig-jaar/). Timestamp interpretation above is supported by the included legend; the public catalog page alone did not verify that convention during this audit.
