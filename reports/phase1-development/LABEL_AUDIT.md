# Fluvius conditioning-label audit

Audit date: 2026-10-21. The user clarified that the raw data and available
labels are suitable for the current task; the missing information is building
type, heat-pump subtype, and separate appliance components. This audit documents
field semantics and their limits, and does not claim the dataset is defective.
The current goal is to generate aggregate whole-house net-energy profiles using
the available equipment flags and calendar variables. The release does not
provide independently verified appliance states, appliance-level energy targets,
or paired counterfactual household profiles. Binary completeness and consistency
do not establish the indicators' accuracy.

## Sources and access scope

This audit inspected the release ZIP directory, the embedded source-definition
workbook, and CSV header lines only. It also read existing preparation metadata,
the historical structural identity audit, and ingestion code. It did not read
CSV data records, any original prepared TRAIN/VAL/TEST profiles, or model
performance metrics. Existing aggregate structural metadata was reused rather
than rescanning household records.

Release: `.cache/data/raw.zip`, with the previously verified pinned SHA-256
`934cc871e9fe18dbe30530472816d711ece878ec7cd8725d9d8417bbdb389bd8`.

Workbook member: `1.50 Aanleveirng DMK Elek Fluvius Legende 25200929.xlsx`.
SHA-256 of the workbook bytes read in this audit:
`344ac3a69adab39bb0e6cdf490b63380ea5ebb0c83cc26ab964c73430c1b0acf`.
Cell references below refer to `xl/worksheets/sheet1.xml` in that workbook.

Additional inspectable sources:

- [Ingestion and label encoding](../../src/fluvius_cgan/data.py).
- [Existing source identity audit](../V3_SOURCE_IDENTITY_AUDIT.json).
- [Existing methods and interpretation](../../docs/PHASE1_METHODS.md).
- `.cache/phase1/data/original96/metadata.json` (structural metadata only).

## What the fields mean

| Field | Source definition | What can be claimed |
| --- | --- | --- |
| `Warmtepomp_Indicator` | A26–A27: considers heat pumps reported as the primary heating system. The following 1/0 sentences instead describe charging-station reporting and EV algorithmic detection. | Intended heat-pump reporting proxy; the detailed source definition is internally inconsistent. Do not silently reinterpret the copied EV text as a verified HP detection mechanism. |
| `Elektrisch_Voertuig_Indicator` | A29–A30: includes battery-electric and plug-in hybrid vehicles. 1 means an EV was detected; 0 means none was detected and explicitly does **not** mean none can be present. | EV detection indicator, with unknown false-negative/false-positive rates. |
| `PV_Installatie_Indicator` | A32–A33: photovoltaic installations, excluding solar boilers. 1 means an installation was reported; 0 means none was reported and no injection volume was registered. | Reported-PV indicator combined with release selection based on injection; not independent equipment verification. |
| `Datum`, `Datum_Startuur` | A14–A18: reading date and interval start, in local CET/CEST. | Recorded calendar information; the trailing textual `Z` is not interpreted as UTC in preparation. |
| `Volume_Afname_KWh`, `Volume_Injectie_KWh` | A20–A24: measured import/injection energy over the quarter-hour following the timestamp. | Meter-level energy observations, not separate HP/EV/PV consumption or generation measurements. |
| `Contract_Categorie` | A35–A36: only EANs residentieel throughout the year were retained. | Source population-selection metadata. |

The exact problematic HP passage in A27 is:

> We nemen enkel warmtepompen in beschouwing die als hoofdverwarming gemeld werden.
> Dit wordt weergegeven als boolean 0 of 1.
> Indien 1: Laadpaal gemeld door de klant of EV gedetecteerd door het algoritme.
> Indien 0: Laadpaal niet gemeld door de klant en EV niet gedetecteerd door het algoritme.

The first sentence refers to reported main-heating heat pumps; the final two
refer to charging stations and EV detection. This appears to be a source
copy/paste error, but its cause cannot be established from the release. No
corrected official definition was found inside the archive.

A8 also states that the PV scope requires injection in **all 12 months**;
the non-PV scope requires **no injection during the entire year**. It describes
eight groups of 300 meters, selected from customers who requested quarter-hour
reading and had all year values available. Thus PV grouping partly depends on
observed energy behavior and is not a simple random sample of installed PV
status. Equal group sizes are a sampling design, not equipment prevalence.
A39 says meters with **known** storage systems were excluded, while unknown
storage systems may remain.

## Coverage, missingness and time variation

All eight release CSV headers contain all three equipment columns. This was
checked directly without reading their measurement records.

Existing full-ingestion metadata reports 2,400 meters seen and retained before
identity curation, no sampled-out meters, and full archive validation of
84,326,400 rows. The ingestion function `_meter_metadata` requires literal
non-null `0`/`1` values in every equipment field and one constant value per
meter. It raises an error for missing, nonbinary, or changing values; it does
**not** turn missing values into zero.

The separate existing source audit records one metadata profile per meter for
all 2,400 meters and `metadata_constant_within_each_meter: true`. These prior
records support complete, binary, year-constant **stored indicators**. This
audit did not repeat the full raw-data check. The release does not provide
installation dates or within-year equipment-state transitions. Constant stored
indicators therefore cannot establish constant real-world ownership or usage
throughout 2024.

Semantic missingness remains possible even when storage missingness is absent:
unreported/undetected devices can have a zero indicator. EV=0 explicitly has
this meaning; PV=0 follows reporting/injection criteria; HP=0 requires caution
because of the contradictory documentation. No independent verification panel,
label confidence, error rates, or true-device crosswalk was found among the
archive members.

## Derived conditions and modeling implications

Weekend and meteorological season are derived deterministically from local
dates, rather than reported equipment attributes. In `data.py`, weekend means
Saturday/Sunday; season is `(month % 12) // 3`, with winter/spring/summer/autumn
coded 0/1/2/3. These are calendar classes, not measured weather, occupancy,
holidays, operating states, or independent human annotations.

The combined condition is `8*(4*HP + 2*EV + PV) + 4*weekend + season`.
Its 64-way label is a derived index mixing imperfect equipment proxies with
calendar categories. It should be described as conditioning on source-reported
or source-detected equipment groups, not verified appliance ground truth.

Conditional generative learning can learn distributions of observed meter
profiles using these indicators without paired supervised output targets.
Flow matching can similarly construct its noise-to-data interpolation target
from observed training samples; external class labels are not required for an
unconditional model. A calendar-conditioned model can use date-derived
variables. If equipment indicators are excluded or judged unreliable, the new
experiment should state that choice and avoid claims about accurate true-device
conditioning or equipment classification. Merely replacing the generator does
not resolve uncertain source labels.

There are no paired observations for the same household with and without an
EV, HP, or PV system. Differences between source groups therefore support
associational generation comparisons, not causal equipment effects. The
source-defined groups also cannot validate an equipment detector without
additional independently verified labels.

For the clarified scope, the available aggregate measurements and equipment
flags support whole-house profile generation. Building-type conditioning,
heat-pump-subtype conditioning, and appliance disaggregation cannot be validated
from this release because those attributes and separate component targets are
absent. These missing attributes do not prevent the current aggregate-profile
modeling task.
