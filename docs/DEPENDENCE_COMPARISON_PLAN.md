# Stronger dependence models: declared Phase 1 screens

Declared 21 October 2026, before these experiments. The user has authorized
continued Phase 1 work, stronger baselines and controlled combinations. The
reference cGAN benchmark remains preserved. Phase 2 and federated learning are
deferred. These screens use only the existing inner-TRAIN/inner-VAL development
split; original VAL/TEST performance is not accessed.

The [completed calibration study](../reports/phase1-development/marginal_calibration/README.md)
exposes the current gap: conditional hourly marginals can be repaired, but
calibrated flows still lose ramp fidelity to day resampling in every condition.
The next experiments isolate dependence while holding the empirical marginal
inverse fixed. No method is presumed superior because it is recent or neural.

## Conditional reflected-flow comparator

The [reflection screen](../reports/phase1-development/REFLECTION_SCREEN_PLAN.json)
adapts the inspected Huk–Damoulas reflection construction; the
[source review](SUPPORT_AWARE_GENERATORS.md) states the paper-access limits and
pins the author code. This is an independently implemented conditional adapter,
not a reproduction of the author's larger unconditional network or experiments.

Fit separate empirical distributions on inner TRAIN for all 64 conditions and
24 hours. For each sampled TRAIN value, store lower/upper probability bounds
`L = count(TRAIN < x)/n` and `H = count(TRAIN <= x)/n`. Every presentation draws
new independent coordinatewise jitter and uses `u0 = L + W*(H-L)`. Under the
empirical mixture and continuous randomness this makes each conditional marginal
uniform, including atoms. A constant physical coordinate has interval [0,1]
and decodes to its constant. It chooses a continuous extension of tied data;
it does not identify a unique population copula or appliance decomposition.

Bounds and jitter use float64 before the trajectory is cast to float32.
Finite RNG precision and rounding can land on jump boundaries. The canonical
inverse remains `sorted_TRAIN[clip(ceil(n*u)-1,0,n-1)]`: an exact lower jump
boundary can decode to the preceding value. No hidden probability/physical
clipping or alteration of this convention is allowed.

Draw independent Gaussian velocities and one time per row, `t = 1.5*A**4`.
The forward state is `R(u0+t*v)`, where `R(y)=1-abs(1-(y mod 2))`. The velocity
target is v in even unfolded unit intervals and -v in odd intervals. At exact
collisions use the forward-right derivative: `abs(v)` at reflected zero and
`-abs(v)` at reflected one. This explicit finite-precision convention differs
from the author's floor-only boundary convention on a probability-zero set.

Use the existing basic-flow architecture: inputs 24 probabilities, scalar time
and 64 one-hot conditions; hidden widths 64/128/64, SiLU, output 24; 23,896
parameters. The four declared runs cross learning rates **1e-3 and 1e-4** with
training seeds **101 and 202**, each with 6,000 updates, batch size 512, Adam
betas 0.9/0.999 and gradient-norm clipping at 1. The higher rate matches basic
flow; the lower rate and clipping reflect the author training settings. Keep
all four outcomes. This is not an equal-compute comparison with the cGAN.

Use independent streams for initialization, shuffled row order, Gaussian
velocity, times and tie jitter. Training visits every inner-TRAIN row per pass,
including the final partial batch. Save all settings, states, data hashes,
fitted marginal arrays, source hashes, time-bin losses, presentations and timing.

For sampling, initialize iid uniform coordinates and integrate the learned
forward velocity backward from 1.5 to zero with reflected Euler:
`u_next = R(u + (t_next-t)*velocity(u,t,c))`. Use **64 intervals and 64 NFE**,
matching basic flow's 32-step Heun NFE. Select the earliest minimum SW1
checkpoint using the existing saved selection rule and seed 20261022. Inspect
128 Euler intervals on that checkpoint with common initial uniforms, without
reselection. The author sampler's 50 nodes correspond to 49 intervals; this
pilot deliberately uses a different, compute-matched count. Uniform terminal
initialization approximates the exact finite-time forward distribution; neither
reflection nor empirical decoding guarantees correct learned marginal frequencies.

Before real training, validate geometry, negative/multiple reflections, exact
collision derivatives, zero-field invariance and an analytic dependent-copula
oracle. For `p0(u,v)=1+0.8*cos(pi*u)*cos(pi*v)`, the exact forward amplitude is
`0.8*exp(-pi**2*t**2)` and velocity is `-t*grad(log(p_t))`. Reverse sampling
should recover mixed cosine moment 0.2 and uniform individual marginals.
This checks signs and numerical integration without relying on neural fitting.

Compare against the same-budget 6,000-update basic-flow runs, the stronger
classical baselines below, and independent marginals `Q_TRAIN(U)`. Existing
30,000-update neural models are contextual comparisons with unequal budgets.
Initially report direct reflection outputs. Any additional generated-sample
calibration is a separately declared combination and must retain direct outputs.

## Stronger classical controls

The [strong-baseline plan](../reports/phase1-development/STRONG_BASELINES_PLAN.json)
retains conditional mean, full-day resampling and the frozen Gaussian copula,
and adds four declared controls:

- **Empirical beta copula:** for each condition/hour, create full ranks 1..n
  using independent, reproducible random tie breaking, fit seed 20261026.
  Select one common TRAIN row I, then independently draw each coordinate from
  `Beta(rank[I,h], n+1-rank[I,h])`. Decode with the canonical empirical inverse.
  Average tied ranks are unsuitable: full rank permutations make the beta
  mixture exactly uniform marginally in law. Constant physical hours are fixed.
  No extra bandwidth parameter is introduced.
- **Student-t copulas with df 4, 8 and 16:** transform average TRAIN ranks via
  `t_ppf((rank-.5)/n,df)`, estimate their Pearson correlation, and use the same
  0.01 identity shrinkage/eigenvalue floor/renormalization as the Gaussian
  comparator. Sample correlated normals divided by one shared chi-square scale
  per row, apply the t CDF, then the same inverse. This is fixed-df t-score
  moment fitting, not maximum likelihood; ties make it an approximation to
  the continuous model. Report every df rather than only the best outcome.

The beta copula follows Segers, Sibuya and Tsukahara (2017), *The Empirical Beta
Copula*, Journal of Multivariate Analysis 155, 35–51,
[arXiv:1607.04430](https://arxiv.org/abs/1607.04430). The manuscript endpoint was
blocked; [CRAN copula documentation](https://raw.githubusercontent.com/cran/copula/master/man/empCopula.Rd)
and [package implementation](https://raw.githubusercontent.com/cran/copula/master/R/empCopula.R)
were inspected. The original continuous-data construction assumes no ties;
our declared random tie handling is an extension for this measured dataset.
[Student-t package code](https://raw.githubusercontent.com/cran/copula/master/R/tCopula.R)
and installed SciPy source corroborate the shared chi-square construction.
Retrieved-source identities and hashes are recorded with the plan.

## Common evaluation and proximity checks

Reuse the same 16,384 inner-validation rows, 256 per condition, and paired
generation seeds 20261023/20261024. Preserve all physical metrics, conditional
tails, covariance/ACF, zero/export frequencies, endpoint frequencies and exact
TRAIN-row matches. Reflection additionally reports generated probability
histograms, means, variances, boundary mass and per-step reflection rates.
Training time-bin MSE is descriptive and does not replace physical validation.

Add exact nearest-neighbor distances to same-condition inner TRAIN for both
generated and fixed real validation profiles. Use Euclidean distance divided
by sqrt(24), giving profile RMS distance in kWh, and additionally divide by the
fixed TRAIN range for normalized distances. Report per-condition means and
1/5/50/95/99% quantiles, plus the generated fractions below the corresponding
real 5th-percentile and median distances. Aggregate condition summaries with
equal weights and state this explicitly. These are descriptive proximity
diagnostics, not privacy guarantees or tests of population novelty.

Beta-copula draws may be particularly close to selected TRAIN days: their
coordinate probability noise is roughly O(n^-1/2). Strong fidelity must be
reported alongside this proximity, and resampling's nearest-TRAIN distance
must be exactly zero. Do not confuse zero exact-row matches with meaningful
separation from training profiles.

For the stronger-baseline comparison, regenerate the existing four raw and
four calibrated neural arms, verify their saved profile-array hashes, and reuse
their preserved physical scores while adding proximity. Frozen calibrations
must not be refitted. Verify existing baseline arrays and scores against their
reference study as well. All input and output artifacts, including partial
failed runs, receive hash receipts. Keep large fitted arrays and model outputs
in ignored project-local directories.

## Evidence needed after the screens

The next replication will freeze a useful recipe and test multiple fresh
training seeds, then isolate conditional dependence from pooled dependence
under identical conditional empirical decoders and budgets. Additional
calibration, boundary constraints or temporal structure must be explicit
ablations, with the standalone method retained. Different validation sample
sizes, all eligible inner-validation rows and meter/season summaries will
check whether apparent gains depend on the small screening sample.

W1/SW1 comparisons remain descriptive; the historical invalid percentile-W1
intervals are not reused or recentered. A prospectively specified paired proper
score with additive per-meter contributions is a candidate for cluster-aware
uncertainty, after synthetic validation of its estimator. New training seeds
or resplits cannot turn previously used data into untouched confirmation.
With only the current release, a carefully scoped internal study remains
possible, but future-year, appliance-level and privacy claims remain unsupported.
