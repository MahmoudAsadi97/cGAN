# Supplied Fluvius 2024 dataset audit

The audit covers every row of the supplied `raw.zip`, whose independently
recomputed SHA256 is
`934cc871e9fe18dbe30530472816d711ece878ec7cd8725d9d8417bbdb389bd8`.
The eight CSVs contain **84,326,400 observations** and occupy 6,389,855,650 bytes
uncompressed. A separate complete scan checks the local calendar slot sequence.
The compact executed evidence is [`reports/dataset_audit.json`](../reports/dataset_audit.json).
This establishes the contents of this archive; it does not independently verify
its download origin, licensing, or population representativeness.

## Reproduce the checks

Run from the repository root with its Python dependencies installed. The scripts
read ZIP members in chunks and do not extract the large CSV files.

```bash
python scripts/audit_calendar.py \
  --archive /path/to/raw.zip \
  --output /tmp/fluvius-calendar-audit.json
python scripts/audit_source.py \
  --archive /path/to/raw.zip \
  --calendar-report /tmp/fluvius-calendar-audit.json \
  --output reports/dataset_audit.json
```

Both commands accept `--chunk-size` and `--help`. They explicitly audit the 2024
schema and CET/CEST source-local calendar, including the archive's sorted autumn
clock labels. They are not general timezone converters. The source audit checks
the attached calendar report's SHA256 and per-file row/meter counts. JSON reports
contain findings, including nonzero anomaly counts; successful command execution
means the audit completed, not that every finding is zero. Missing columns,
invalid meter IDs, unreadable inputs, and mismatched calendar reports fail the
command.

## Sample and schema

Each CSV has 10,540,800 rows and 300 meters. Global meter identifiers are exactly
1–2400, with no overlap across files. Each meter has 35,136 quarter-hour records
across all 366 dates of 2024 and stable metadata. All contract values are
`Residentieel`.

| File suffix | EAN_ID range | Heat pump | EV | PV |
|---|---:|---:|---:|---:|
| enkel_ZP | 1–300 | 0 | 0 | 1 |
| geen_ZP | 301–600 | 0 | 0 | 0 |
| WP_met_ZP | 601–900 | 1 | 0 | 1 |
| WP_geen_ZP | 901–1200 | 1 | 0 | 0 |
| EV_met_ZP | 1201–1500 | 0 | 1 | 1 |
| EV_geen_ZP | 1501–1800 | 0 | 1 | 0 |
| WP_EV_met_ZP | 1801–2100 | 1 | 1 | 1 |
| WP_EV_geen_ZP | 2101–2400 | 1 | 1 | 0 |

The exact columns are `EAN_ID`, `Datum`, `Datum_Startuur`,
`Volume_Afname_KWh`, `Volume_Injectie_KWh`, `Warmtepomp_Indicator`,
`Elektrisch_Voertuig_Indicator`, `PV_Installatie_Indicator`, and
`Contract_Categorie`.

The full scan found no missing cells, nonfinite volume values, negative raw
import/export values, unknown or off-quarter-hour timestamp strings, source
date/timestamp mismatches, sorting violations, invalid binary indicators,
nonresidential rows, or metadata conflicts.

## Timezone and daylight saving time

The embedded workbook
`1.50 Aanleveirng DMK Elek Fluvius Legende 25200929.xlsx`, sheet `LEGENDE`, cell
`A18`, explicitly defines the readings as CET/CEST. CSV timestamp strings
nevertheless end in literal `.000Z`. Treating this suffix as UTC would contradict
both the legend and the observed calendar.

Every meter follows the same source-local calendar template, verified row by
row for all eight CSVs:

- **364 dates have 96 ordered, distinct quarter-hour slots.**
- **2024-03-31 has 92 rows.** Local 02:00, 02:15, 02:30, and 02:45 are absent.
- **2024-10-27 has 100 rows.** Each local 02:00, 02:15, 02:30, and 02:45 label
  appears twice, with identical clock labels adjacent in the file. There are
  1,200 duplicate timestamp rows per CSV, or four per meter. Row order alone
  does not establish which repeated value belongs to which UTC fold.

Excluding both DST dates yields **873,600 complete meter-days** (2,400 × 364)
and 83,865,600 quarter-hour observations. It removes 4,800 meter-days containing
460,800 rows. No interpolation is required for retained dates. The spring clock
gaps are expected DST behavior, not unexplained missing measurements.

## Volume meaning and units

`LEGENDE!A21` and `A24` define kWh during the quarter-hour following
`Datum_Startuur`. These values are interval energy, not cumulative registers
or instantaneous power. Net grid import is
`Volume_Afname_KWh - Volume_Injectie_KWh`; negative values represent net export.
Four consecutive quarter-hour values sum to hourly kWh. Multiplying a
quarter-hour kWh value by four gives interval-average kW.

Full-year source import totals **13,076,523.788958 kWh** and export totals
**3,126,393.891 kWh**, including DST dates. Maximum individual quarter-hour
import is 7.095 kWh; maximum export is 3.007 kWh. Machine totals use floating-point
summation; the displayed precision follows the source values.

Non-PV export is not identically zero despite the source scope wording:

| File suffix | Positive export rows | Export kWh | Affected meters |
|---|---:|---:|---:|
| geen_ZP | 1 | 0.001 | 1 |
| WP_geen_ZP | 339 | 4.722 | 1 |
| EV_geen_ZP | 0 | 0 | 0 |
| WP_EV_geen_ZP | 12 | 0.012 | 1 |
| Total | 352 | 4.735 | 3 |

Preserve these measured values and document the discrepancy. The audit does
not establish why those small exports occur.

## Source scope and limitations

The following definitions come from the embedded legend, not assumptions about
the wider residential population:

- `LEGENDE!A8`: customers requested quarter-hour readings and complete-year
  data were available. All selected EANs were residential throughout the year.
  Equal sampling of 300 meters per category does not estimate equipment
  prevalence in the population.
- `A8`: PV scope is described as injection in all 12 months; non-PV scope is
  described as no injection during the year. The observed non-PV exports above
  are a discrepancy. This audit does not independently test the all-12-month
  injection criterion for each PV meter.
- `A27`: only heat pumps reported as primary heating are considered. The next
  two Boolean explanatory lines discuss charging poles and EV detection,
  apparently a copy/paste error. They do not reliably explain heat-pump
  ascertainment.
- `A30`: EV includes battery EVs and plug-in hybrids. Zero means not detected
  and explicitly does not establish absence.
- `A33`: PV means photovoltaic installations, not solar thermal. One means a
  reported installation; zero is described as unreported with no injection.
- `A39`: known storage-system EANs were excluded, but unreported batteries can
  remain.
- No household demographics, coordinates, tariff, weather, holidays,
  occupancy, or equipment-capacity fields are present.

Source dates support deterministic weekday/weekend and season labels when
their definitions are recorded. Assign household-disjoint train/validation/test
splits within equipment categories before creating daily examples. EAN IDs
support grouping and split checks; they should not become GAN condition inputs.

The archive also contains a 555-byte `_prepare.log` referencing an earlier
conversion. It was ignored as nonmeasurement content. The scripts do not infer
data quality or completeness from that log.
