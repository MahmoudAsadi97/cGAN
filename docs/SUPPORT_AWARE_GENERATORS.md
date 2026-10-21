# Support-aware copula generators: targeted review

Reviewed 21 October 2026. This was a bounded primary-source search, not an
exhaustive literature review. No implementation, experiments, or dataset reads
were performed. Retrieved files and failures have URL, timestamp and SHA-256
receipts in the [retrieval record](../reports/phase1-development/SUPPORT_METHOD_RETRIEVALS.json);
[paper identities and inspected code excerpts](../reports/phase1-development/SUPPORT_METHOD_SOURCE_REVIEW.json)
record the evidence and exact source commits.

## Most relevant verified recent paper

**David Huk and Theodoros Damoulas, _Diffusion and Flow-based Copulas:
Forgetting and Remembering Dependencies_.** The official repository cites
**ICLR 2026**, [OpenReview YrX77XRgku](https://openreview.net/forum?id=YrX77XRgku),
and [arXiv:2509.19707](https://arxiv.org/abs/2509.19707), a September 2025
identifier. The conference attribution is verified against author materials,
not independently against proceedings. Both paper endpoints were blocked by
the network proxy. No full manuscript was retrieved.

[Official repository at inspected commit](https://github.com/Huk-David/Diffusion-and-Flow-based-Copulas/tree/5014c0f35179b38803f3634c0b0acd4a6f6e2e57).
The full README, reflection model/training/sampler, and CDC objective/training
and sampling code were inspected. The repository's only root subdirectory,
`Figures`, contained illustration PDFs, not a manuscript. The author profile
links a Warwick page; no full paper was available in the inspected repository.

This is a close match to the newly diagnosed failure: a dependence model can
use explicit fitted marginal distributions instead of learning both components
solely from an unconstrained physical-output loss. The paper introduces:

- **Classification-Diffusion Copula (CDC):** rank Gaussianization followed by
  an Ornstein–Uhlenbeck dependence-destruction process. A classifier predicts
  the noising time, classifier density ratios yield a copula density estimate,
  and input gradients provide a denoiser. The inspected trainer combines
  classification cross-entropy and denoising MSE.
- **Reflection Copula:** reflected linear trajectories in the unit hypercube,
  trained by velocity regression. The released sampler starts uniformly on
  the cube and numerically integrates backward, reflecting after each step.
  The inspected trainer samples `t = 1.5 * Uniform(0,1)^4`, Gaussian initial
  velocities, and fits their reflected velocities.

These mechanisms go beyond simply putting a Gaussian correlation matrix in
the starting noise. The earlier failed correlated-source experiment therefore
does not test either of these methods.

### What the code supports, and what it does not establish

The README says the ideal forward processes preserve one-dimensional
marginals and recover dependence under the stated theory. Its convergence
claims have not been checked against the full paper here. **Reflection keeps
numerical samples inside the unit cube; it does not, by itself, guarantee that
a finitely trained neural sampler has exactly uniform marginal distributions.**
Finite integration, finite terminal noising time and velocity approximation
can all matter. The same distinction applies to approximately standard-normal
marginals from the learned CDC sampler.

The official model shown is an **unconditional** dependence model; adding our
64 observed conditions would be an adaptation, not an exact reproduction of
the released experiment. Its example scripts load precomputed ECDF CSVs and
then split rows. Those files' transformation-fitting provenance was not
available. This is not evidence of leakage in the paper, but the scripts do not
establish the meter-separated, TRAIN-only transformation contract required here.

Practical code details also need independent decisions before any adaptation:

- `CDC.py` assigns distinct ranks using double `argsort`, even to ties. This is
  not a validated treatment of zero atoms or repeated meter readings.
- CDC training clamps input probabilities to `[1e-5, 1-1e-5]`; its sampler
  clamps Gaussian-scale states to `[-6,6]`. These settings affect tail behavior
  and must not be copied silently into physical-energy validation.
- `CDC.py` imports `ddpm_unet`, absent from the inspected root tree. The README
  reflection-training filename differs from the actual `train_Refcop.py`.
  No claim is made that the upstream checkout runs unmodified.

Key pinned code:
[reflection construction](https://github.com/Huk-David/Diffusion-and-Flow-based-Copulas/blob/5014c0f35179b38803f3634c0b0acd4a6f6e2e57/Ref_copula.py),
[reflection training](https://github.com/Huk-David/Diffusion-and-Flow-based-Copulas/blob/5014c0f35179b38803f3634c0b0acd4a6f6e2e57/train_Refcop.py),
[reflection sampler](https://github.com/Huk-David/Diffusion-and-Flow-based-Copulas/blob/5014c0f35179b38803f3634c0b0acd4a6f6e2e57/Sample_Ref_copula.py),
[CDC](https://github.com/Huk-David/Diffusion-and-Flow-based-Copulas/blob/5014c0f35179b38803f3634c0b0acd4a6f6e2e57/CDC.py).

## Classical control and the interpretation needed for Fluvius

Empirical CDFs, probability-integral transforms, Gaussian scores, inverse
quantile calibration and Gaussian copulas are **classical techniques**, not
2026 inventions. Our existing `src/fluvius_cgan/baselines.py` already implements
a conditional Gaussian-copula comparator with average ranks, empirical step
inverse quantiles, constant-hour handling, and regularized Gaussian dependence.
Its conditional empirical marginals should remain the canonical control.

For hour `h` and observed condition `c`, a learned Gaussian-scale dependence
model could decode with `x_h = Q_train,h,c(Phi(z_h))`; a cube model would decode
with `Q_train,h,c(u_h)`. This construction is an adaptation proposal, not a
result. The empirical inverse maps into that condition's observed support,
even when learned `u_h` is not uniform. **Correct marginal frequencies require
uniform `u_h`; support compliance alone does not imply marginal calibration.**

This distinction is directly relevant to the [recorded fresh-draw development
diagnostic](../reports/phase1-development/fresh_diagnostics/README.md): generated PV=0 export days were roughly 36–47% for the 30k cGAN,
75–76% for basic flow, and 79–80% for correlated-source flow, versus zero in
the sampled real diagnostic subset. These numbers summarize the saved diagnostic artifacts; the literature review
does not rerun that evaluation or provide independent held-out evidence.
Existing empirical/copying baselines almost avoid this failure by construction.
Any calibrated neural generator must therefore beat those baselines in
dependence and whole-day behavior, rather than claim neural improvement solely
from restoring their marginal support constraint.

Required interpretive boundaries for a possible later experiment:

1. **Fit only on fitting meters.** Fit each transform and any pooling/shrinkage
   rule inside the existing inner-TRAIN partition; freeze them for validation,
   checkpoint selection and generation. Recomputing ranks on validation or on
   a generated batch would change the evaluation procedure. Save fitted
   quantiles, condition counts, transform seeds and source hashes.
2. **Retain conditional signed support.** Preserve negative net load for PV
   groups. Do not infer a universal nonnegativity law from PV=0 alone: use
   verified TRAIN support and explicitly account for any small observed
   negative readings and measurement resolution. A pooled marginal across
   PV groups would reintroduce the failure. Sparse 64-way cells may need a
   declared pooling strategy selected only within development data.
3. **Handle zero atoms and ties explicitly.** For mixed distributions, ordinary
   midranks are not a continuous uniform sample and the copula is not uniquely
   identified. A randomized distributional transform
   `F(x-) + v * (F(x)-F(x-))`, with its own reproducible RNG, or an explicit
   atom/hurdle model is a defensible option to study. Neither is automatically
   validated by a continuous-copula paper. Step inverse quantiles preserve
   measured zero mass when the generated probabilities are calibrated;
   interpolating through a jump can destroy that mass.
4. **Disclose empirical tail truncation.** Empirical inverse quantiles cannot
   exceed observed per-condition/hour extrema. This enforces sampled support
   but cannot establish correct unseen peak tails. Use the same inverse rule
   for the Gaussian-copula control and neural model; assess peaks, exports,
   daily energy, ramps, temporal dependence and diversity together. Parametric
   tail extrapolation would be a separate model choice and separate ablation.
5. **Check the learned copula as well as physical outputs.** Assess generated
   marginal uniformity/rank calibration, zero frequencies, export-energy
   magnitude, and dependence. An arbitrarily small negative reading can make
   an export-day indicator large, so include a predeclared measurement-aware
   threshold and negative-energy magnitude alongside the raw sign statistic.
6. **Preserve the reference cGAN.** Any transformed-output comparator must be
   clearly separate from the reference architecture. No hidden post-generation
   clipping or modification of its reported samples should manufacture success.

## Secondary recent source

**Houman Safaai, _Amortized Vine Copulas for High-Dimensional Density and
Information Estimation_**, [arXiv:2604.20568](https://arxiv.org/abs/2604.20568),
**2026 preprint** in the author repository. No conference acceptance claim was
found or made. [Official VDC code](https://github.com/KempnerInstitute/vine-denoising-copula/tree/24e11afacc0872570730bc70d4610152d2530abc).
The README and user guide were retrieved; the README and relevant transformation,
density-estimation and sampling sections were inspected. No full paper was read.

It trains a shared bivariate denoising estimator on synthetic copulas, projects
estimated grids with iterative proportional fitting to enforce copula density
marginals, and reuses the estimator across vine edges. This is a useful
dependence comparator, with a more explicit grid-normalization step than the
reflection sampler. The released model assumes continuous marginals, uses a
64-by-64 grid, and adopts the simplifying assumption of constant conditional
pair copulas. A 24-dimensional full vine has 276 pair edges; zero atoms,
conditional groups and cumulative recursion errors need examination. Author
claims about numerical validity or information-estimation accuracy were not
independently reproduced. Lower priority than the directly matched Huk–Damoulas
study for an initial investigation.

## Screened out and decision

[Discrete Copula Diffusion](https://github.com/liuanji/Copula-Diffusion/tree/c35c7dbc12e7121f3245d730edb377a6c69da891)
is identified as ICLR 2025 by its official README. It combines token-wise
diffusion with an autoregressive copula model for language sampling. This is
not a direct continuous signed-energy generator. Its complete paper identity
was not established in this short search, so it is not promoted as a primary
candidate.

The useful recent direction is **dependence learning with explicit TRAIN-fitted
conditional marginals**, supported by the verified 2025/2026 copula paper.
The lowest-complexity scientific control is classical marginal calibration
matched to the existing Gaussian copula. A recent neural dependence model is
justified only if it improves the joint profiles after that control is fixed.
No support-aware neural method has demonstrated an improvement on Fluvius in
this review. None recovers unobserved building type, heat-pump subtype, causal
equipment effects, or appliance-level load from aggregate profiles alone.
