# Empirical quantile interpolation correction

The shared inverse-quantile helper in `src/fluvius_cgan/copula.py` had a
floating-point defect on repeated arbitrary float64 values. It affected both
copula marginal interpolation and fixed marginal-calibration maps. The correction
preserves the declared linear empirical-quantile convention, endpoint bounds and
absence of tail extrapolation; it changes the arithmetic used to evaluate it.

## Reproduced edge case

With `default_rng(7312)`, draw eight normal rows of width 24, repeat each row 512
times, then compress the 4,096 sorted observations to 1,025 uniformly spaced
quantile knots. The former expression, `(1-f)*a + f*b`, can round two weighted
copies of the same float64 value differently as `f` changes. Consequently, a
constant plateau can acquire tiny ascending and descending steps.

The synthetic reproduction produced **931 descending steps**, with a minimum
increment of **−4.440892098500626e−16**. Calibration rejected the resulting table
with `invalid source quantile table`. A fitted copula could likewise save those
knots and then reject them on reload. The validation checks were correct; the
interpolation arithmetic needed correction.

## Corrected arithmetic

For ordered finite endpoints `a ≤ b` and fraction `0 ≤ f ≤ 1`, use one fixed
formula for each sign case:

| Interval | Formula |
|---|---|
| Nonnegative | `a + f*(b-a)` |
| Nonpositive | `a + f*(b-a)` |
| Strictly crossing zero | `(1-f)*a + f*b` |

Same-sign open intervals use the left endpoint. Equal endpoints then have an exact
zero difference, so plateaus remain constant. This avoids a midpoint anchor
switch and the negative lower-endpoint defect described below, at the cost of
relative accuracy near right-endpoint cancellation. The crossing-zero case
avoids subtracting opposite float64 extremes. Masks are applied before arithmetic,
so an unused branch cannot overflow. Fractions exactly zero or one copy the
corresponding endpoint directly.

Input probabilities retain their existing endpoint clamping. There is **no output
clipping, cumulative maximum, tolerance-based monotonicity repair or added noise**.
The reproduced table now has zero descending steps and calibration accepts it.

## First correction: synthetic verification

The new regression file first produced three failures and three passes under the
old helper. After correction, the six new regressions and existing copula and
calibration tests produced **126 passes**. The recorded pytest durations were
11.13 seconds before correction and 3.73 seconds for the combined suite afterward;
these different test selections and run conditions are not a speed comparison.
These timings and the replay below describe the first correction, whose
same-sign anchor was the endpoint nearest zero; they are retained as historical
evidence, not measurements of the subsequent lower-endpoint correction.

A separate seeded replay used 8,192 signed float32 profiles with 96 intervals,
two conditions, 1,025 quantile knots and 6,400 generated profiles. Its synthetic
load/export construction, seeds, settings, source hashes and scalar results are
retained in [the verification report](../reports/quantile_interpolation_verification.json).

| Compared quantities | Maximum absolute old/new difference | Identical array bytes? |
|---|---:|---|
| Float64 tied-source compressed knots | 4.440892098500626e−16 | No |
| Float32-input compressed copula knots | 0 | Yes |
| Copula generated profiles | 4.440892098500626e−16 | No |
| Direct inverse evaluation on float32-input margins | 4.440892098500626e−16 | No |
| Float32 calibration source and target knots | 0 | Yes |
| Fixed-map transformed profiles | 4.440892098500626e−16 | No |

The corrected helper also passed synthetic checks for finite opposite-sign
float64 extremes, exact ties/endpoints, monotonicity across random finite knots
and segment boundaries, and a manually computed linear-quantile example.

## Second correction: negative lower endpoint

The first correction still failed for `a = -(1.5 + 2**-52)` and
`b = -(3 * 2**-53)`. At probabilities zero and `1e-30`, the negative-interval
formula returned `-1.5000000000000002` followed by `-1.5000000000000004`:
a one-ULP decrease below the lower knot. This was observed independently in
the differentiable decoder and the NumPy helper before either source changed.

The current helper uses the fixed left anchor shown above. Exact endpoint
copies, crossing-zero arithmetic and probability bounds are unchanged. The
new regression retains the exact failing values and asserts equality, support
and monotonicity without a tolerance. The original tied-source, extreme-value,
calibration and save/load regressions remain unchanged. Before/after traces,
source hashes and focused verification results are recorded in the
[boundary verification receipt](../reports/quantile_interpolation_boundary_verification.json).
The new lower-endpoint test first failed on the previous helper; after correction,
the interpolation, copula, calibration and decoder suites passed **158 tests in
6.31 seconds**. The seeded plateau construction that originally had 931
descending steps still has zero. These are synthetic correctness checks, not a
performance benchmark or a proof for every floating-point input.

## Actual-study provenance and limits

Validation jobs in retained pinned worktrees use the implementation saved in
their respective source pin and retain their original source and result hashes.
Those worktrees were not modified.
The **final freeze must use the corrected source**, with its exact source and
sealed-array hashes recorded. Existing fixed maps may be loaded, but applying
them uses the corrected shared interpolation helper. The old and new source
SHA256 values for each correction are recorded in its respective verification
report. Historical reports and retained clean worktrees remain bound to their
original interpolation sources; the upcoming decoder comparison must evaluate
both arms with the same newly pinned helper.

The observed maximum synthetic difference of 4.44e−16 is **not evidence of
actual-study equality**, nor a bound on differences for other inputs. Actual
arrays, metrics, byte hashes and scientific conclusions were not compared in
this correction task. Even the tiny observed synthetic changes alter array
hashes. No real prepared arrays or TEST data were read, and no actual-study
reevaluation or statistical equivalence claim is made.
