# Conditional mixture baseline: declared Phase 1 screen

Declared before fitting on 21 October 2026. The completed stronger-baseline
screen shows why another dependence model is useful. The empirical beta copula
matches resampling closely on profile SW1, but 93.2% of its generated days lie
below the corresponding real-validation fifth-percentile nearest-TRAIN
distance. The df4 Student-t copula improves several Gaussian-copula metrics
while worsening daily-energy W1. Neither finding establishes a universally
better generator. All earlier outcomes and controls remain preserved.

This screen adds **normal-score Gaussian-mixture fitting followed by analytic
mixture-marginal probability transforms**. It is a classical comparator,
not a new method or a reproduction of Gaussian-mixture copula maximum
likelihood. Multiple components can represent different whole-day shapes;
within-component full covariance represents correlated hourly variation.
Components have no appliance, building-type or causal interpretation.

## Fixed construction

Use all inner-TRAIN rows for both marginals and dependence fitting. For every
condition and nonconstant hour, independently break ties with seeded random
permutations to give full ranks 1 through n. Transform ranks to finite normal
scores `z = Phi_inverse((rank - 0.5) / n)`. The rank stream is identical across
component counts. This is an explicit continuous extension of measured atoms,
not identification of a unique population copula. Constant physical hours
are excluded from the latent fit and restored by the empirical decoder.

Fit a separate full-covariance Gaussian mixture per condition using
scikit-learn 1.7.2. The declared component counts are **1, 2, 4 and 8**. All
use fit seed 20261027, two independent initializations, k-means++ initialization,
at most 100 EM iterations, convergence tolerance 0.001, and `reg_covar=0.001`.
Regularization adds 0.001 to covariance diagonals in normal-score units; it is
not an eigenvalue-clipping estimator. Use separate recorded random streams for
rank ties and initialization. Preserve each initialization's final TRAIN log
density, lower bound, iterations, convergence and warnings. Select the largest
final TRAIN log density, with earliest initialization breaking exact ties.
Record nonconvergence rather than silently retrying or selecting with VAL.

For one generated day, draw one component K from its mixture weights and a
correlated vector `Z ~ Normal(mu_K, Sigma_K)`. Then, for each active hour h,

```
U_h = sum_k weight_k * Phi((Z_h - mu_kh) / sqrt(Sigma_khh))
X_h = Q_TRAIN,condition,h(U_h)
```

The CDF sums over the **entire mixture**, including components other than the
sampled one. The standard deviation is the square root of the covariance
diagonal, not the Cholesky diagonal. With positive variances, the analytic
transform makes each U marginal uniform in law, so the same empirical inverse
preserves TRAIN hourly atom frequencies and support in law. Finite simulation
and numerical precision remain visible in diagnostics. The analytic CDF is
bounded to [0,1] to correct floating-point roundoff; physical profiles are not
clipped. No empirical mapping
is fitted on generated evaluation batches or validation values.

The one-component arm controls for the changed tie handling and fitting
pipeline. Keep the original Gaussian copula separately: its average ranks and
correlation shrinkage differ. A finite nonsingular Gaussian mixture remains
asymptotically tail independent, so Student-t controls remain relevant.

EM fits density in fixed normal-score space. It does not optimize the copula
likelihood, whose parameter-dependent marginal inverse and Jacobian require a
different fitting procedure. Report this surrogate estimator precisely.

## Evaluation and preserved evidence

The machine-readable [plan](../reports/phase1-development/MIXTURE_BASELINES_PLAN.json)
binds the unchanged inner-TRAIN/inner-VAL data, completed stronger-baseline
reference, all four arms and paired generation seeds 20261023/20261024. Use the
same 16,384 validation profiles, condition weights, projections and physical
metrics. Include ramps, daily energy, peaks, covariance/ACF, diversity,
zero/export frequencies, endpoints, exact TRAIN-row matches and exact
same-condition nearest-TRAIN RMS distances. A favorable SW1 alone is not a
promotion criterion. Retain every component count and report tradeoffs.

Save fitted mixture parameters and empirical marginal arrays without pickle;
hash and reload each artifact before generating evaluation profiles. Record
source/library versions, per-condition occupancies, covariance eigenvalues,
initialization results, runtimes and failed partial artifacts. Verify and reuse
unchanged reference results rather than refitting earlier models.

The screen is exploratory post-test development on reused validation meters.
It does not create independent confirmation, establish privacy, validate
future-year performance, or produce appliance-level targets. No original
VAL/TEST performance access or Phase 2 work is authorized by this plan.

## Sources

Bilgrau et al. (2016), *GMCM: Unsupervised Clustering and Meta-Analysis Using
Gaussian Mixture Copula Models*, Journal of Statistical Software 70(2), 1–23,
DOI [10.18637/jss.v070.i02](https://doi.org/10.18637/jss.v070.i02).
The author package's simulation code corroborates sampling a Gaussian mixture
then applying its full marginal CDFs; its fitting documentation discusses
optimization difficulties and different fitting methods. Source URLs and
retrieved-byte hashes are recorded in the accompanying source receipt. This
screen does not claim to reproduce that package's copula-likelihood estimator.
