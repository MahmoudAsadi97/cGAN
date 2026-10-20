# Implementation validation — 20 October 2026

Tested implementation commit: `35cd9cbbc27267af23da44b93b619de2645cbf23`.

## Executed checks

- **88 tests passed locally** with Python 3.12 / PyTorch 2.14.1 CPU execution.
- **88 tests passed in GitHub Actions** with Python 3.11 / PyTorch 2.5.1 CPU.
- A separate local CLI integration check prepared a synthetic fixture, trained,
  evaluated validation data, and generated profiles. Fixture results are not
  evidence of real-data quality.
- The actual pinned release passed checksum verification and the workflow's
  preparation, training, checkpoint loading, generation, and evaluation steps.
- Independent review prompted fixes for absent-condition reporting, generated
  sample provenance, and equipment-group exclusion counts. Regression tests pass.

[Successful workflow and downloadable evidence](https://github.com/MahmoudAsadi97/cGAN/actions/runs/39354154531).
The workflow artifact retains checkpoints, manifests, metadata and the full
condition-level report for 30 days. The permanent [summary JSON](smoke_validation.json)
records global results and the full report's SHA-256.

## Real-data smoke sample

This was the first six meters in each of eight source CSVs, **not a random or
representative sample** of all 2,400 meters. It used `configs/data_smoke.json`
and `configs/smoke.json` (two epochs, linear generator output).

| Item | Observed |
|---|---:|
| Households | 48 |
| Raw rows processed | 1,686,528 |
| Candidate household-days | 17,568 |
| Retained household-days | 17,472 |
| Excluded household-days | 96 |
| Conditions | 64 |
| Train / validation / test households | 32 / 8 / 8 |
| Train / validation / test profiles | 11,648 / 2,912 / 2,912 |
| Validation profiles in final diagnostic report | 1,280 (20 per condition) |

All 96 excluded days are daylight-saving transition days. Of these, 48 also have
duplicate intervals and 48 incomplete intervals; exclusion reasons overlap.
The trainer selected epoch 1 using 256 validation profiles and fixed validation
noise. Its selection metric (macro conditional per-slot Wasserstein) was
0.4188314255 kWh. The report below uses a different, explicitly recorded
validation sample and the global distribution metric. Test values were not used
for fitting, checkpoint selection, or reported evaluation.

## Quality observations — insufficient for publication

| Global validation metric | Two-epoch cGAN run, selected checkpoint | Conditional resampling |
|---|---:|---:|
| Mean per-slot Wasserstein, kWh | 0.394225 | 0.042442 |
| Daily-energy Wasserstein, kWh | 9.156562 | 1.097108 |
| Peak-power Wasserstein, kW | 2.423369 | 0.190931 |
| Negative generated interval fraction | 0.332324 | 0 |

Lower is better for these distances. The selected smoke checkpoint underperforms
the empirical baseline and generates physically invalid negative grid imports.
These are deliberately reported, with no clipping or selective omission. The
run demonstrates functioning software, **not a valid final generative model**.

## Work still required

Run and inspect preparation on the full release. Train the hourly and
quarter-hour upstream adaptations and the nonnegative variant with the specified
matched budgets and five seeds. Complete the stronger temporal baseline,
conditioning ablations, downstream-utility study, household-clustered uncertainty,
and held-out test evaluation after freezing choices. Establish and substantiate
the research contribution, check dataset reuse terms, and follow the
[research protocol](RESEARCH_PROTOCOL.md). Full-data quality and publication
readiness have not been established.
