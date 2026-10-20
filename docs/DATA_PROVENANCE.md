# Official dataset provenance and licence

**Verified on 20 October 2026:** all eight CSVs and the legend in `data/raw.zip`
match the current official Fluvius attachments in uncompressed byte length and
SHA-256. The local combined ZIP is a separate package, not an official attachment.
Exact URLs, retrieval times, HTTP metadata and hashes are retained in
[`reports/official_source_binding.json`](../reports/official_source_binding.json).

## Official source and content binding

The [Fluvius dataset page](https://opendata.fluvius.be/explore/dataset/1_50-verbruiksprofielen-dm-elek-kwartierwaarden-voor-een-volledig-jaar/?flg=en-gb) describes 2,400 residential digital
meters in eight technology categories of 300. Its
[catalog API](https://opendata.fluvius.be/api/explore/v2.1/catalog/datasets/1_50-verbruiksprofielen-dm-elek-kwartierwaarden-voor-een-volledig-jaar) lists two ZIP parts and one legend workbook. The publisher
field is Dataroom Fluvius; attachment filenames identify the 2024 data.

The verified release container is 395,085,174 bytes, SHA-256
`934cc871e9fe18dbe30530472816d711ece878ec7cd8725d9d8417bbdb389bd8`. All eight CSVs, totaling
6,389,855,650 uncompressed bytes, match:

| CSV basename | Uncompressed bytes | SHA-256, identical in release and official ZIP |
|---|---:|---|
| `P6269_Open_Data_EV_geen_ZP.csv` | 800,926,499 | `6a993a4d1778343ba356c55aa9c9783c7d50719f64825aa76b762dcbe610ecc8` |
| `P6269_Open_Data_EV_met_ZP.csv` | 804,175,121 | `4925fe8aeeb7c032e62658904295212608e0590ca22c5b072720456bcf485fb7` |
| `P6269_Open_Data_WP_EV_geen_ZP.csv` | 800,697,761 | `d39727ea98c23255568366d2e15371f13960874ea8ea1ea519a00c7c34c2c826` |
| `P6269_Open_Data_WP_EV_met_ZP.csv` | 808,132,319 | `0c0c96bc2de8de54f43717c9532353287e300a750917ee699c8089bf9fbc7e77` |
| `P6269_Open_Data_WP_geen_ZP.csv` | 797,294,675 | `ef3cbf1e441a96879987eee63d4dc0fb842142a2a9e3381e006372c034020967` |
| `P6269_Open_Data_WP_met_ZP.csv` | 797,786,309 | `8ec22469ce1674bc14fc43eb89a79d78db53d7047c932c21f23af3eabba8b0ab` |
| `P6269_Open_Data_enkel_ZP.csv` | 790,497,785 | `16d55c99fc6ea47a2bd0efd1d908a857b9ded1fc08e2be38001cbb28002670b6` |
| `P6269_Open_Data_geen_ZP.csv` | 790,345,181 | `f4e31d559ac46faf94f629cce35e3edc520c2bb210c44b2fbc16f19a6f567692` |

The separate official legend and the release workbook are both 20,436
bytes, SHA-256 `344ac3a69adab39bb0e6cdf490b63380ea5ebb0c83cc26ab964c73430c1b0acf`.
The official downloads are:

- [P6269_1_50_DMK_Sample_Elek_2024_Deel_1.zip](https://opendata.fluvius.be/api/explore/v2.1/catalog/datasets/1_50-verbruiksprofielen-dm-elek-kwartierwaarden-voor-een-volledig-jaar/attachments/p6269_1_50_dmk_sample_elek_2024_deel_1_zip): 197,375,122 bytes; SHA-256 `0f8addd57bd9f7bc0fd0c28b019a4b6c8753a19e3fd301de758f5a5858c9d86c`.
- [P6269_1_50_DMK_Sample_Elek_2024_Deel_2.zip](https://opendata.fluvius.be/api/explore/v2.1/catalog/datasets/1_50-verbruiksprofielen-dm-elek-kwartierwaarden-voor-een-volledig-jaar/attachments/p6269_1_50_dmk_sample_elek_2024_deel_2_zip): 199,377,167 bytes; SHA-256 `cfef37973c40e503fe53afa4966bbdcd7a12525fb8b20881f365d0ee8616a3d6`.
- [1.50 Aanleveirng DMK Elek Fluvius Legende 25200929.xlsx](https://opendata.fluvius.be/api/explore/v2.1/catalog/datasets/1_50-verbruiksprofielen-dm-elek-kwartierwaarden-voor-een-volledig-jaar/attachments/1_50_aanleveirng_dmk_elek_fluvius_legende_25200929_xlsx): 20,436 bytes; SHA-256 `344ac3a69adab39bb0e6cdf490b63380ea5ebb0c83cc26ab964c73430c1b0acf`.

Official ZIPs contain CSVs at their root; the release places them inside two part
folders. The release also contains `_prepare.log` (555 bytes), which was hashed
but not interpreted or attributed to Fluvius. Directory entries are not data.

## Licence and attribution

The catalog assigns **Open data license - FLUVIUS**, linking to
[Licentie open data Fluvius](https://opendata.fluvius.be/p/licentieopendatafluvius/). The terms permit copying, publication,
redistribution, adaptation, derived data and commercial reuse. Every reuse must
name Fluvius and give the last-update date; effective data links may supply
attribution. Do not imply official status or endorsement, mislead about content,
source or update date, or breach applicable law, including personal-data
protection. Data carry no guarantee of defect-free or continuous provision;
reuse remains the reuser's responsibility. Noncompliance ends the licence
automatically. No numbered licence version is stated. These are the dataset's
custom terms, not a claim that the repository's software licence covers the data.
The linked terms control; this summary is not blanket legal or privacy clearance.

Suggested attribution for this project's derived outputs:

> Source: Fluvius (Dataroom Fluvius), 2024 digital electricity-meter profiles.
> Catalogue modified: 29 September 2025. Retrieved: 20 October 2026.
> [Dataset](https://opendata.fluvius.be/explore/dataset/1_50-verbruiksprofielen-dm-elek-kwartierwaarden-voor-een-volledig-jaar/?flg=en-gb); [Fluvius licence](https://opendata.fluvius.be/p/licentieopendatafluvius/).
> Transformed by this project into net-import daily profiles and curated trace
> groups; no Fluvius endorsement implied.

The official API's `modified` field is `2025-09-29T08:41:01.980000+00:00`. Its separate
`metadata_processed` field is `2026-09-21T09:29:07.238000+00:00`; `data_processed`
is null. The CSV ZIP HTTP Last-Modified dates are 30 July 2025; the legend's is
29 September 2025. The attribution above explicitly identifies the catalog field
rather than treating processing time as a data-update date. The licence page's
JSON-LD modification time, `2026-06-02T08:48:12.019000+00:00`, is neither a numbered
licence version nor the dataset-update date.

## Verification scope and reproducibility

The API, page and licence were fetched from `opendata.fluvius.be`. Licence text
was decoded from embedded JSON using `json.JSONDecoder.raw_decode`, without
executing JavaScript; the exact JSON paths and decoded-text SHA-256 are in the
report. Each official attachment was downloaded once, its advertised byte count
checked, and its SHA-256 recorded. Both copies of every ZIP member were then
streamed through SHA-256 in 8,388,608-byte chunks, without parsing
CSV records, changing encoding/line endings, or extracting CSVs to disk. Python
`zipfile` also checked each complete member's CRC. The single-process check
peaked at 64,620 KiB resident memory.

To reproduce, obtain the three attachment URLs from the recorded official API,
verify their container hashes, then stream each `ZipFile.open(member)` into
`hashlib.sha256` and compare both byte count and digest with the table/report.
Hash the whole official XLSX and the uncompressed release XLSX in the same way.
Use the pinned release-container hash above to identify the input.

This confirms current official content identity and the catalog's licence
reference. It does not reconstruct the earlier hosting/repackaging chain or
establish accuracy, representativeness, household independence, anonymization
quality or absence of near-duplicates. Raw source bytes were read only for
hashing; **no prepared split file was opened or hashed, no profile statistics
were computed, and no TEST performance was evaluated**. Prior source audits and
curation have their own separately documented access scopes.
