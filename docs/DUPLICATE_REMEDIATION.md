# Exact source-trace duplicates: remediation decision

The preparation and identity curation were regenerated after a workspace reset.
The execution record below describes the regenerated files, with independent
fingerprint verification and an exact alias ledger. Original training/validation
arrays and scaler reproduce their published byte hashes; metadata values also
reproduce exactly, with a JSON key-order difference documented below.

The extended audit found five train/validation meter pairs with different EAN
identifiers but byte-identical, date-aligned, nonconstant 364-day net-energy
traces. EAN-disjointness therefore did not establish independent traces.

All earlier runs using `data/prepared96` are **exploratory results from a split
with detected cross-split duplicates**. Their numerical values remain recorded
for audit, but they cannot establish independent-household validation or paper
readiness. This includes the pilot, normalization, discriminator, residual,
conditional-loss and copula screens. Their code tests remain useful software
checks; the results must be re-established on the curated data.

## Prospective correction

1. Fingerprint each meter's complete retained trace by sorting ISO dates, then
   hashing the concatenation of fixed-width ASCII dates (`S10`) and contiguous
   little-endian float32 profile values. Canonicalize signed zeros to positive
   zero for identity keys only, without rounding other values or changing output
   profiles. Confirm hash matches by direct key-byte equality. Fail closed if
   duplicate members disagree on aligned condition labels or equipment groups.
2. Inspect the original test array only for this deterministic identity audit
   and preparation. Do not compute test performance, distribution summaries or
   model-selection metrics. Record this access explicitly; it is no longer
   accurate to say the file has never been opened.
3. Preserve original split priority **train, then validation, then test**. Keep
   one representative per identical full-trace group in its earliest original
   split, choosing the lexicographically smallest string meter ID within that split. Remove every other
   alias. Never move a previously used development trace into a nominal test set.
4. Write a separate curated prepared directory, with the alias/exclusion manifest,
   original hashes, new counts, new split hashes and newly fitted training scaler.
   Preserve the original prepared data and historical run directories.
5. Train all confirmatory candidates anew. Final evaluation uses retained
   original-test traces with no exact full-trace match in development under this
   fingerprint. Test values are accessed for identity auditing and preparation,
   but no test performance or distribution summaries inform model selection.
   Describe this set as identity-audited and performance-unseen. Validation remains
   exploratory; curation does not make previous tuning disappear.

This deterministic removal can alter the original equal meter counts by
technology; report the resulting counts rather than claiming the original
70/15/15 proportions still hold exactly. Bootstrap real-meter uncertainty within
the surviving technology strata. Daily-profile coincidences alone are not grounds
for dropping a meter: repeated zero or low-use days can occur naturally.
Exact deduplication does not establish household independence or exclude
near-duplicates and other shared-source dependence. The estimand changes from
identifier-weighted to one observation per retained unique trace; neither the
original nor curated group counts estimate household prevalence.

The identity criterion concerns the modeled net-energy target at quarter-hour
resolution. For subsequent hourly sensitivity experiments, retain the same
canonical meter membership and aggregate those curated profiles; do not create a
new split that reintroduces aliases.

## Regenerated execution and access record

The reviewed curation was executed from `data/prepared96` into the separate
`data/curated96` directory. All five original prepared files retained their
recorded SHA-256 hashes throughout curation. Command output is recorded in
`runs/curation96.log`; compact counts, the complete duplicate-group/alias ledger,
input/output hash manifest and independent verification are recorded in
`reports/curation96.json`. The curated directory contains
`curation_manifest.json`, binding every NPZ, scaler and metadata file, and full
identity-group membership in its metadata.

The regenerated original `train.npz`, `val.npz` and `scaler.json` exactly match
the SHA-256 values in the published `reports/copula_screen.json`. The metadata
file's byte hash differs because its split `sha256` keys occur earlier in each
JSON object. Moving each such key to the end, without changing any values, and
serializing with the existing two-space indentation and final newline reproduces
the published metadata hash
`36b40313730d206ba3195c228807e184044aa98f35c21a05a4b4fa144539fc63`
exactly. This check also binds the unchanged original test split hash recorded
in that metadata. No source file was rewritten for this comparison. The current
manifest records the actual regenerated bytes, and the report records both
hashes and the exact key-order replay procedure.

| Original split | Original meters | Retained meters | Removed aliases | Retained profiles |
| --- | ---: | ---: | ---: | ---: |
| Train | 1,680 | 1,665 | 15 | 606,060 |
| Validation | 360 | 353 | 7 | 128,492 |
| Test | 360 | 349 | 11 | 127,036 |
| Total | 2,400 | 2,367 | 33 | 861,588 |

All retained meters had the same complete 364-date audited calendar. The 33
duplicate groups each contained two meter IDs: 15 train-only, two validation-only,
one test-only, five train/validation, nine train/test, and one validation/test.
Removing aliases discarded 12,012 profiles. No retained meter changed splits.
The signed-zero-canonicalized identity rule is fingerprint version 2; the earlier
strict-byte audit remains a valid lower bound on identities under this rule.

The original preparation had 210 training and 45 validation/test meters in each
technology group. The verified retained counts are:

| Technology code (`HP*4 + EV*2 + PV`) | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| 0 | 210 | 45 | 45 |
| 1 | 210 | 45 | 45 |
| 2 | 207 | 44 | 41 |
| 3 | 210 | 45 | 45 |
| 4 | 207 | 43 | 41 |
| 5 | 210 | 45 | 45 |
| 6 | 201 | 41 | 42 |
| 7 | 210 | 45 | 45 |

The original test file was opened solely for deterministic complete-trace
fingerprinting, structural/date/condition validation, and copying retained
original-test rows. Its file bytes were also rehashed to verify immutability.
The curated test was subsequently opened for an independent implementation of
the fingerprint and retained-count checks. Neither operation loaded a model or
computed test performance or energy-distribution summaries. Only TRAIN min/max
bounds were inspected for scaler verification and refitting. Thus the test files
were accessed for preparation; their retained original-test subset was
performance-unseen in the experiment record, rather than a new random test split.

The regenerated independent verification found 2,367 distinct retained fingerprints
and zero repeated fingerprints within or across the three splits. Every retained
fingerprint matched its audited canonical representative, every retained ID
belonged to the same original split, and all input/output manifest hashes matched.
That established no exact full-trace overlap under the declared fingerprint;
it did not establish independent households or exclude near-duplicates.
All models used for subsequent confirmation must be trained anew on the curated preparation.
Historical checkpoints and validation results remain exploratory.

The metadata explicitly separates counter scopes: original source-preprocessing
counters and exclusions remain recorded, while `meters_kept` and `profiles_kept`
refer to the curated copy. Original counters and alias exclusions are retained
under `curation`; downstream analyses should use the current split manifests for
sample sizes.
