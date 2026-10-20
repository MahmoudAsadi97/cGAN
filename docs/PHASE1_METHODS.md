# Phase 1 methods and interpretation

This document describes the implemented experiment. Results and their
scientific assessment belong in the final results record; the equations here
are not evidence that the fitted model is accurate.

## Population, target and partition

The source release contains 2,400 meter identifiers in eight reported equipment
groups for 2024. A group is the binary combination of heat pump (HP), electric
vehicle (EV) and photovoltaic installation (PV) indicators. The sample's equal
original group sizes do not represent equipment prevalence in the population.

Quarter-hour net grid-import energy is

\[
x_{d,q}=E^{\rm import}_{d,q}-E^{\rm injection}_{d,q}.
\]

The source workbook specifies local CET/CEST clock labels, even though the
timestamp text ends in `Z`. No UTC conversion is applied. Both local daylight
saving transition dates are excluded. A retained day has exactly one finite,
valid observation for each of the 96 quarter-hour slots. Negative net energy
is retained: it represents net export, rather than a negative source reading.
No imputation, clipping or outcome-dependent outlier removal is used.

The original seed-42 equipment-stratified meter allocation is preserved. Exact
complete-trace identities are audited at quarter-hour resolution, with signed
zero canonicalized for identity comparison only. Alias removal keeps an
identity in its earliest original partition, in TRAIN → VAL → TEST order;
ties use the smallest lexicographic identifier. This prevents an already
exposed development trace from entering the retained test set. The procedure
removes 33 aliases and retains 1,665/353/349 meters, respectively. Each meter
contributes 364 days; the corresponding profile counts are
606,060/128,492/127,036.

An hourly target sums four adjacent quarter-hour energies. Identity decisions
and split membership precede this aggregation. The 24-value output therefore
has units of kWh per hourly interval. The network produces independent daily
scenarios, not complete synthetic household trajectories.

All partitions share the same 2024 weather/calendar. Exact trace deduplication
does not establish independence of different homes, exclude near-duplicates,
or undo historical access. See [PROVENANCE.md](PROVENANCE.md) for the inspected
V2 access record and its limits.

## Conditioning and normalization

For equipment group \(g=4\,HP+2\,EV+PV\), weekend indicator \(w\), and
meteorological season \(s\), the class index is

\[
c=8g+4w+s,\qquad c\in\{0,\ldots,63\}.
\]

Season codes are winter 0, spring 1, summer 2 and autumn 3. The network receives
a 64-dimensional one-hot class. Equipment conditions describe reported
indicators, not verified absence of unreported equipment.

One global affine transform fits the retained training values only:

\[
u=(x-a)/(b-a),\quad
a=\min(X_{\rm train}),\quad b=\max(X_{\rm train}).
\]

The code uses denominator 1 for a constant training range. The actual retained
hourly bounds are approximately −10.769001 and 22.850000 kWh. Validation and
test values may fall outside the training range. The linear generator output
is inverse-transformed without clipping.

## Reference network and optimization

The reference is `malvin0130/cGAN_EB` at commit
`885f78052a91fbc755e6b17a354c68849b5556ca`. Its hourly architecture is preserved:

| Component | Input | Hidden widths | Output |
| --- | --- | --- | --- |
| Generator | 24 standard-normal latent values + 64 conditions | 64, 128, 64 | 24 linear values |
| Discriminator | 24 profile values + 64 conditions | 64, 128, 256 | 1 logit |

Hidden activations are LeakyReLU with slope 0.2. The discriminator applies
dropout 0.4 before its output layer. There is no added auxiliary loss,
normalization layer, attention module, calibration or federated training.

Let \(D\) denote the probability corresponding to the discriminator logit.
Each shuffled batch receives one discriminator update followed by one
generator update, using fresh Gaussian noise:

\[
\mathcal L_D=-\tfrac12\left[
\mathbb E_{x,c}\log D(x,c)+
\mathbb E_{z,c}\log(1-D(G(z,c),c))\right],
\qquad
\mathcal L_G=-\mathbb E_{z,c}\log D(G(z,c),c).
\]

The implementation evaluates these losses with stable `BCEWithLogitsLoss`.
Fake samples are detached during the discriminator update. Discriminator
parameters are frozen during the generator update while dropout remains in
training mode; a regression test verifies the same updates and random states
as the unfrozen calculation. Adam uses learning rates \(10^{-4}\) and
\(2\times10^{-4}\) for G and D, with betas (0.9, 0.999).

The frozen study uses batch size 2,000, 100 epochs and the full training set.
Each run performs 30,400 updates of each network. The ten runs comprise five
declared training seeds for each neural arm. The unconditional control removes
both conditioning inputs while preserving the hidden widths and optimizer
settings. Each run keeps its own complete outcome; no favorable seed is chosen.

The upstream 10,000-epoch budget is not reproduced. This is a different-data
adaptation with an explicitly declared computational budget. The upstream real
training data are unavailable, and no numerical reproduction of the original
paper is claimed. The development pilot and budget rationale are recorded in
[PHASE1_PROTOCOL.md](PHASE1_PROTOCOL.md).

## Selection and comparators

Validation selection uses fixed rows and fixed generation noise for every
training seed and both neural arms. Up to 1,024 rows are sampled without
replacement in each condition, with an equal generated count. Epoch 1 and each
fifth epoch are scored; the earliest minimum is retained. The final evaluator
requires that selected checkpoint from a completed full-budget run.

Three additional comparators fit the training partition only:

* **Conditional mean:** repeat the training mean curve within each condition.
  This is a deliberately low-diversity control.
* **Complete-day resampling:** independently sample full training days with
  replacement within each condition. It preserves observed within-day
  relationships but copies training profiles.
* **Conditional empirical Gaussian copula:** within each condition, map average
  ranks to normal scores using \((r-0.5)/n\), estimate their correlation, and
  shrink it as \(0.99R+0.01I\). An eigenvalue floor of \(10^{-8}\) and diagonal
  renormalization ensure a valid correlation matrix. Constant coordinates
  remain fixed. Correlated normal samples map back through empirical marginal
  quantiles using index \(\lceil n\,u\rceil-1\), bounded to the observed support.
  No jitter, Gaussian marginal assumption or test-fitted calibration is added.

Each method generates as many days in a condition as that condition has real
evaluation days. Reporting uses all eligible real evaluation rows and three
new generation seeds, separate from selection noise.

## Endpoints

For a condition, profile sliced Wasserstein-1 is

\[
\widehat{SW}_1(P,Q)=\frac1{128}\sum_{\ell=1}^{128}
 W_1\!\left(\{\theta_\ell^\top x\},\{\theta_\ell^\top\tilde x\}\right),
\]

where each \(\theta_\ell\) is a unit Gaussian direction in 24 dimensions.
Directions are fixed with seed 2026 and shared across methods. One-dimensional
distances use exact empirical CDF integration, including unequal sample counts.
The primary endpoint divides physical profile SW1 by the frozen training range
and then gives each of the 64 conditions equal weight. Smaller is better.

Secondary endpoints diagnose different forms of mismatch:

| Endpoint | Quantity compared |
| --- | --- |
| Slot W1 | Hourly marginal energy distributions, averaged across slots |
| Daily-energy W1 | Distributions of signed sums over 24 hours |
| Peak-power W1 | Distributions of each day's maximum hourly mean power |
| Ramp W1 | Distributions of signed adjacent-hour power differences |
| Mean-profile RMSE | Conditional hourly mean curves |
| Variance MAE / diversity ratio | Conditional slot variances and total variance |
| ACF lag-1 and lag-6 W1 | Distributions of within-day biased autocorrelations |
| Mean and quantile curves | Hourly means and pointwise 1/5/50/95/99 percentiles |

For one-hour intervals, mean kW has the same numerical value as interval kWh.
Constant profiles receive autocorrelation zero and are counted explicitly.
An undefined diversity ratio is retained as undefined, with valid-condition
counts, instead of silently replacing it with a favorable value. Both macro
and real-count-weighted summaries are saved. These measure different
condition mixtures; neither estimates population equipment prevalence.

The reported quantities are distribution diagnostics, not formal proof that
each generated day is physically plausible. No invalid raw-energy KL
divergence or image-feature FID is used.

## Separate sources of uncertainty

For each trained model, the three reporting draws give a mean and generation
sampling SD. Across five trained models, the five draw-means give a separate
mean and sample training-seed SD. Paired differences match conditional and
unconditional runs by declared training seed; baseline estimates remain fixed
across those comparisons. These SDs are not confidence intervals.

The predeclared training-seed-11, generation-seed-31001 exemplar receives 499
paired bootstrap comparisons against resampling and the copula, using seed
20261020. Each bootstrap samples complete real meter histories with replacement
within equipment strata. Both comparators use the same resampled real meters;
their fitted models and generated arrays remain fixed. Percentile 95% intervals
describe the conditional-minus-comparator difference in primary profile SW1
and daily-energy W1. Positive differences mean the cGAN has greater error.
Physical profile intervals are divided by the same training range as the
primary endpoint. Intervals are marginal for each declared comparison, not
simultaneous intervals for the whole set of comparisons.

These intervals describe uncertainty under real-meter resampling for one
specified fitted model and synthetic draw. They are not intervals for the
five-model mean and do not include training variation, generation variation
or future-weather uncertainty. The complete seed results make those limits
visible. If a bootstrap draw loses an original condition, the implementation
withholds intervals rather than selectively retaining complete draws.

Conditional nearest-neighbor diagnostics compare generated-to-training and
held-out-to-training profile RMSE. They use all training references and at most
256 query days per condition, with fixed seed 41001. Quantiles, exact matches
and matches within \(10^{-8}\) interval-kWh RMSE are reported. Resampling is
expected to copy training profiles. A lack of exact copies in a GAN is not a
privacy guarantee.

## What the experiment can establish

Software tests establish specified implementation properties. Held-out metrics
estimate fidelity for retained meters under this dataset, calendar,
conditioning system and training budget. Comparators reveal whether the neural
method adds value over simpler generators. Learning curves indicate remaining
training instability or a budget boundary; they cannot prove global
convergence.

Publication assessment must consider these findings together with novelty,
data rights, historical access and external validity. Neither positive plots
nor small distances guarantee publication. No Phase 2 modification follows
from test inspection without the user's separate review and approval.
