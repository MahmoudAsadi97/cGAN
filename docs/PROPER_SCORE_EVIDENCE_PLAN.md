# Prospective proper-score evidence supplement

This document specifies a possible later Phase 1 evaluation. It does not
execute an experiment, select a model, replace the primary sliced-Wasserstein
endpoint, or amend any frozen benchmark, calibration, baseline, or reflection
study. An execution plan must bind the final models, datasets, seeds, sample
counts, comparisons, and source revision before this supplement is run.

Only the existing Fluvius 2024 release is available. The inner validation
meters have already informed checkpoint, parameter, and method selection.
**Intervals calculated on those reused meters are postselection descriptive
intervals that ignore selection. Conditioning on fitted models or generated
ensembles does not restore independent evaluation data, and these intervals
cannot claim nominal coverage for performance on new meters.** A later locked
original-VAL audit also retains its historical prior-use disclosure. Neither a
new split nor fresh generation noise creates an untouched confirmation cohort.

The design below distinguishes an ideal fixed-model, independent-meter
estimand from what the actual reused data can support. Synthetic coverage
experiments can validate the interval algorithm under their declared sampling
models; they cannot remove selection bias or establish coverage for the real
postselection analysis. No new external data are assumed or required to
complete this descriptive supplement.

## Purpose and scope

Energy and variogram scores assess the quality of a predictive distribution
against individual held-out profiles. Their paired differences can be
aggregated within meters, avoiding the nonsmooth empirical-Wasserstein
functional in a meter-bootstrap calculation. This is a change of estimand,
not a correction to the historical W1 intervals or to data reuse.

Retain the declared SW1/W1 metrics, all unfavorable results, and their existing
limitations. Add these two scores prospectively, rather than choosing a score
because its ranking is favorable. Lower scores are better; a negative paired
difference, method A minus method B, favors A. Neither score supplies an
application adequacy threshold, a privacy guarantee, or a publication gate.

The target remains a signed 24-hour whole-house net-import energy profile,
conditioned on reported HP/EV/PV, season, and weekday/weekend. Scores use
physical interval-kWh values, including negative values, without clipping.
They do not assess appliance disaggregation, population equipment prevalence,
future weather, or coherent household-year trajectories.

## Proposed generation and comparison design

The proposed main ensemble size is **1,024 IID profiles per condition**, with
**three independent generation replicates per fitted model**. The first 512
members of each ordered ensemble provide a prespecified numerical-sensitivity
comparison; the 1,024-member result remains the main result regardless of
ranking. These choices and all actual seeds must be fixed in the later
machine-readable execution plan before generation or score inspection.

Each ensemble has one fixed model and one of the 64 conditions. Members must
be independent identically distributed generator draws within that ensemble;
hours within a profile are allowed to be dependent. Antithetic sampling,
quasi-Monte Carlo sampling, and other dependent-member designs require a
separate justification and cannot inherit the IID unbiasedness claims below.

Evaluate the same eligible real profiles for all methods. Archive generated
arrays or reproducibly recoverable artifacts, their hashes, model hashes,
condition order, random seeds, sampler settings, and ensemble sizes. Generation
replicates are separate from training seeds and from historical checkpoint
selection noise. A common random-number design across methods is optional but
must be explicit; identical integer seeds alone do not establish a matched
noise construction across different generators.

Use all eligible inner-VAL profiles only after the model recipe and reporting
plan have been frozen. The preparation records describe complete 364-day
meter histories after the two DST exclusions. Verify calendar completeness,
condition counts, and meter membership at execution; do not infer these
properties from the total row count alone. Any subsequent original-VAL audit
needs its own locked plan and prior-use disclosure.

Report every declared comparison. Any nominal 95% intervals are marginal
intervals, not simultaneous coverage over models, scores, or conditions. No
unadjusted collection of favorable intervals may be presented as familywise
evidence. The primary comparison and the role of each supplementary comparison
must be fixed before this evaluation.

## Energy score and finite-ensemble correction

For physical profiles, define the distance

\[
d(x,y)=\frac{\lVert x-y\rVert_2}{\sqrt{24}}.
\]

The positive scaling preserves the energy score's propriety and gives the
score kWh units. For a predictive distribution \(Q\), independent
\(Z,Z'\sim Q\), and observation \(y\), the population energy score is

\[
ES(Q,y)=E[d(Z,y)]-\frac12E[d(Z,Z')].
\]

For IID generated profiles \(z_1,\ldots,z_M\), \(M\ge2\), use

\[
\widehat{ES}_U(y)=
\frac1M\sum_{i=1}^M d(z_i,y)
-\frac1{2M(M-1)}\sum_{i\ne j}d(z_i,z_j).
\]

All ordered off-diagonal generated pairs enter the second term. An equivalent
implementation may sum unordered pairs with the corresponding factor. Real
profiles are not paired with other real profiles in this score. Its expectation
over independent ensemble generation equals \(ES(Q,y)\), conditional on a fixed
fitted generator and fixed observation, under the required finite moments.

## Variogram score and finite-ensemble correction

Fix \(p=1/2\) and use all within-day pairs \(0\le h<k<24\), with normalized
inverse-lag weights

\[
w_{hk}=\frac{1/(k-h)}{\sum_{0\le a<b<24}1/(b-a)}.
\]

These weights emphasize nearby hours while retaining all within-day lags.
There is no cyclic connection from hour 23 to hour 0 of another day. No lag
weight is selected after inspecting model results.

Define

\[
a_{hk}(y)=|y_h-y_k|^{1/2},\qquad
\mu_{hk}=E_Q[|Z_h-Z_k|^{1/2}].
\]

The population score is

\[
VS(Q,y)=\sum_{h<k}w_{hk}[a_{hk}(y)-\mu_{hk}]^2.
\]

For each pair let \(A_{i,hk}=|z_{i,h}-z_{i,k}|^{1/2}\), let
\(\bar A_{hk}\) be its ensemble mean, and let \(s^2_{hk}\) be its sample
variance with denominator \(M-1\). Use

\[
\widehat{VS}_U(y)=\sum_{h<k}w_{hk}
\left([a_{hk}(y)-\bar A_{hk}]^2-\frac{s^2_{hk}}M\right).
\]

Equivalently, the unbiased estimate of \(\mu_{hk}^2\) is

\[
\frac{(\sum_i A_{i,hk})^2-\sum_i A_{i,hk}^2}{M(M-1)}.
\]

This correction removes finite-IID-ensemble bias in the squared expectation.
Negative estimates are possible and must be retained; clipping them would
change the estimator and reintroduce bias. With \(p=1/2\), the score has kWh
units. The variogram score is proper but not strictly proper: it does not
distinguish every possible predictive distribution. It supplements the energy
score and the existing ramp/ACF diagnostics rather than proving complete
temporal realism.

## What conditioning on a generated ensemble means

The corrected energy score is not exactly the energy score of the discrete
empirical distribution \(\widehat Q_M\) that gives mass \(1/M\) to each saved
ensemble member. Write

\[
D_U=\frac1{M(M-1)}\sum_{i\ne j}d(z_i,z_j).
\]

Then

\[
\widehat{ES}_U(y)=ES(\widehat Q_M,y)-\frac{D_U}{2M}.
\]

The correction is a model-specific random offset common to every observation
scored with that ensemble. The corrected variogram score similarly differs
from the empirical-distribution variogram score by the fixed offset
\(\sum_{h<k}w_{hk}s^2_{hk}/M\).

Therefore, conditioning on saved ensembles defines a risk difference between
**fixed finite-ensemble score functions**. It is not exactly the canonical
proper-score risk difference of the empirical distributions, and a meter-only
interval does not include the ensemble-generation uncertainty in the risk
difference of the underlying generator laws. Unbiasedness over fresh IID
generation does not make the common random offset disappear from one saved
ensemble. A larger ensemble reduces Monte Carlo variation but is not a reason
to omit its interpretation or measurement.

This distinction is separate from selection bias. Even a correct interval
algorithm for fixed score functions and independent evaluation meters does
not justify nominal new-meter coverage on the reused inner validation sample.

## Ideal meter-level target and actual descriptive estimate

For a declared pair of methods and a given fixed fitted-model/ensemble pair,
let

\[
\delta(y)=\widehat S_A(y)-\widehat S_B(y).
\]

Let \(g\) denote one of eight equipment strata, \(m\) a meter within that
stratum, and \(c\) one of eight season/day-type bins within the stratum. With
\(n_{mc}\) eligible days in that bin, calculate

\[
A_{gm}=\frac18\sum_{c\in g}
\frac1{n_{mc}}\sum_{d\in(m,c)}\delta(y_{md}),
\qquad
\widehat\Delta=\frac18\sum_g\frac1{M_g}\sum_{m\in g}A_{gm}.
\]

The ideal population target first chooses an equipment stratum uniformly, a
meter from its target population, a calendar bin uniformly, and a retained
2024 day uniformly within that bin. The fixed fitted models and generated
ensembles are not reselected or refitted in this target. In an ideal evaluation
design, sampled meters would be independent of fitting and selection and
independent across meter clusters conditional on the shared calendar.

The actual estimate uses the already selected validation meters and models.
It describes their paired score differences under these weights. It does not
erase their use in model development or estimate the performance of the full
adaptive search procedure without selection bias.

With identical complete calendars for all meters, this estimator also equals
the equal-weight mean of the 64 condition-specific daily means. Verify that
equivalence. If calendars or condition counts differ, the specified estimator
remains meter-balanced and can differ from a within-condition pooled-day
estimator; show that distinction rather than silently changing weights.
Missing calendar bins or inconsistent equipment metadata require an explicit
failure or a separately declared estimand, not selective renormalization.

Save per-profile paired contributions, meter/condition score sums and counts,
and the resulting meter summaries. These permit independent reconstruction
of the aggregation without regenerating profiles or recomputing distances.

## Equipment-stratified paired meter bootstrap

Resample \(M_g\) complete meter histories with replacement within each
equipment stratum. Use identical meter multiplicities for both methods, both
scores, all conditions, and every fixed fitted-model/generation replicate.
Do not resample individual days as independent households.

For the stratified mean above, define

\[
\widehat{SE}^2=\frac1{64}\sum_g\frac{s_g^2}{M_g},
\]

where \(s_g^2\) is the sample variance of the paired meter summaries in
stratum \(g\). The proposed studentized bootstrap uses 1,999 resamples:

\[
T^*=\frac{\widehat\Delta^*-\widehat\Delta}{\widehat{SE}^*}.
\]

For its quantiles \(q_{.025},q_{.975}\), the nominal 95% construction is

\[
[\widehat\Delta-q_{.975}\widehat{SE},\;
 \widehat\Delta-q_{.025}\widehat{SE}].
\]

The interval algorithm, its quantile convention, bootstrap seed, and handling
of zero variance or insufficient stratum sizes must be fixed and tested
before execution. Do not discard problematic resamples or switch interval
methods after inspecting results. Exactly identical score functions should
produce an explicit degenerate case; a failed studentization must not be
silently converted into a zero-width claim.

Multinomial meter multiplicities can compute bootstrap means and variances
from weighted meter sums and squared sums. This avoids expanding bootstrap
samples into daily rows and retains all within-meter dependence.

For genuinely independent evaluation meters, this is a regular mean problem
under the required sampling and moment assumptions. Meter-cluster independence
can still fail through shared feeder effects or residual related identities.
The calendar is conditioned on, so the calculation does not estimate future
weather or future-year variation. **For the actual reused inner-VAL data, label
the resulting intervals postselection descriptive, conditional on saved
models/ensembles and ignoring selection; do not claim actual 95% new-meter
coverage.** The same historical-exposure caveat accompanies a later original-VAL
audit even if further choices are locked before that audit.

## Training, generation, and meter variation

Keep the following quantities separate:

- Per-training-seed, per-generation-replicate paired score differences.
- Training-seed mean and SD of generation-averaged differences, retaining every
  declared seed and failure. These describe a finite set of fitted models;
  the seeds are not independent household cohorts.
- Within-model generation SD over the three independent ensembles. Three draws
  give a coarse Monte Carlo sensitivity estimate, not a precise interval for
  the underlying generator-law score.
- Meter-bootstrap descriptive intervals for fixed models and fixed ensembles,
  with the selection qualification above.

A summary meter interval may use per-profile contributions averaged over the
declared finite collection of fitted models and generation ensembles, with
that averaging fixed in advance. It then conditions on that entire collection.
Averaging model-specific scores evaluates the average score of those fitted
models. Concatenating ensembles across training seeds instead scores a mixture
generator and changes the estimand; do not interchange these procedures.

Do not add training, generation, and meter SDs ad hoc, or interpret a
meter-only interval as unconditional generator-law inference. If differences
are comparable to generation variability, report that limitation alongside
the comparison. Independent generation replication does not repair reuse of
validation meters. Claims about the expected performance of an entire adaptive
training/search procedure would require a different design.

## Verification before any real-data scoring

Verification should have independent numerical targets, not merely agreement
between two functions implementing the same formula. It does not require
reading Fluvius profiles.

1. **Exact finite-support calculations.** Enumerate all size-two and size-three
   ensembles from a distribution on two fixed profiles. Compute population
   scores directly from their probabilities and compare them with expectations
   of the implemented estimators. Independently verify ordered/unordered pair
   factors, diagonal exclusion, unequal distribution probabilities, both
   finite-ensemble corrections, and valid negative variogram estimates.
2. **Normal energy-score calculation.** Embed a scalar normal variable in a
   fixed profile direction. Use the closed forms for \(E|X-y|\) and
   \(E|X-X'|\) to check mean corrected scores and the known finite-ensemble
   bias of the uncorrected empirical score. Do not derive the expected result
   by calling the score implementation being tested.
3. **Clustered coverage simulation.** Generate independent meter random effects
   plus daily noise under a fixed 364-day calendar, with eight unequal equipment
   strata and varying season/day-type scales. Use profiles \(vY\) with fixed
   \(v=(0,1,0,1,\ldots)\). Energy-score expectations then reduce to absolute
   scalar moments; variogram expectations reduce to \(E|Y|\) and
   \(E|Y|^{1/2}\). Fixed-ensemble risk differences can be computed independently
   from known normal moments, mixture-component calculations, or one-dimensional
   quadrature. Hold archived generation ensembles fixed when checking coverage
   of this conditional target.
4. **Meaningful scenarios.** Include weak and strong within-meter dependence,
   asymmetric or mixture distributions, unequal stratum sizes, small effects,
   and identical score functions. Retain all scenarios and failures rather
   than selecting only those with favorable coverage. Independent sampled
   clusters are an explicit assumption of these simulation scenarios.
5. **Monte Carlo reporting.** Approximately 1,000 simulated cohorts per chosen
   scenario with 999 bootstrap resamples each provide a feasible initial check.
   Report mean bias with Monte Carlo standard error, empirical coverage with
   a binomial interval, interval widths, and undefined cases. At nominal 95%
   coverage and 1,000 independent cohorts, coverage's Monte Carlo standard
   error is approximately 0.0069. Report the uncertainty rather than declaring
   exact calibration from a rounded coverage percentage. Material failures
   require investigation before interval claims; do not tune the method on
   real model rankings.
6. **Omitted generation-error demonstration.** Repeat an appropriate synthetic
   scenario with a small independently regenerated ensemble for each simulated
   cohort. Contrast meter-only interval coverage for that repetition's
   conditional-ensemble target with coverage for the underlying generator-law
   target. This makes the shared ensemble error visible and checks that scope
   labels are substantively necessary.

Run simulations sequentially or in small vectorized chunks on one CPU. Scalar
profile constructions and meter-summary bootstrap weights avoid unnecessary
24-dimensional simulation arrays or bootstrap copies of daily records. Exact
fixtures establish algebraic correctness; clustered simulations assess the
interval procedure only within their tested designs. Neither validates
postselection coverage for the reused Fluvius cohort. Any real-data intervals
retain the descriptive label even if all synthetic checks behave well.

## Reporting and claim limits

The eventual report should state the frozen methods and comparisons, ensemble
sizes and seeds, score definitions and units, condition/meter weighting,
per-seed effects, generation sensitivity, and the precise interval scope. Keep
selection, shared-calendar, and marginal-coverage qualifications beside the
affected numbers, including in exported tables.

Do not replace historical W1 intervals, recenter them after seeing results,
or describe this supplement as restoring an untouched test. Non-rejection does
not establish equivalence; a favorable interval does not establish practical
adequacy, neural superiority on every diagnostic, or publication readiness.
The frozen original cGAN benchmark remains intact. This is Phase 1 methodology;
Phase 2 and federated work still require the user's explicit approval.

## Primary references and retrieval limits

The following foundational references motivate proper scoring and finite
ensemble correction. Their bibliographic details are supplied from established
literature knowledge. A Crossref metadata request for each DOI was attempted
on 21 October 2026, but the environment proxy returned HTTP 403 before retrieval.
No primary text or metadata was successfully retrieved in this verification
attempt. These entries are **not newly verified source receipts**; obtain and
record primary-source verification before making a stronger literature-access
claim in a manuscript.

- Gneiting, T., and Raftery, A. E. (2007). *Strictly Proper Scoring Rules,
  Prediction, and Estimation*. Journal of the American Statistical Association.
  [DOI: 10.1198/016214506000001437](https://doi.org/10.1198/016214506000001437).
- Scheuerer, M., and Hamill, T. M. (2015). *Variogram-Based Proper Scoring Rules
  for Probabilistic Forecasts of Multivariate Quantities*. Monthly Weather Review.
  [DOI: 10.1175/MWR-D-14-00269.1](https://doi.org/10.1175/MWR-D-14-00269.1).
- Ferro, C. A. T. (2014). *Fair Scores for Ensemble Forecasts*. Quarterly Journal
  of the Royal Meteorological Society.
  [DOI: 10.1002/qj.2270](https://doi.org/10.1002/qj.2270).

The exact variogram finite-ensemble correction above is given with its algebraic
derivation and must be independently verified as specified. No unverified claim
is made that one of these papers validates the complete proposed clustered,
postselection workflow.
