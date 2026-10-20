# Exact source-trace duplicates: remediation decision

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
