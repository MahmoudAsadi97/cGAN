# Generative methods without building or appliance-level labels

Assessment: 21 October 2026. This review supports continued Phase 1 development;
it does not replace the preserved reference-cGAN benchmark or start federated
learning. The recommendation is to test a small flow-matching comparator after
the current cGAN optimization controls, then retain methods according to measured
performance. Flow matching is a plausible alternative to adversarial training,
not evidence that the Fluvius task has already been solved.

## What “without labels” means here

The user clarified that the raw data and available equipment flags are usable.
The missing information is finer building/heat-pump type and separate heat-pump
versus other household consumption. The primary issue is therefore unobserved
heterogeneity and missing component measurements, not corrupted equipment flags.

The current release contains reported heat-pump, electric-vehicle and photovoltaic
indicators. The existing code combines those indicators with weekend and season
into 64 conditions. These are source-reported equipment categories, not verified
appliance inventories, appliance-level power traces, or observations of equipment
operation. The methods supplement already states that a negative indicator does
not verify absence of unreported equipment.

These tasks need different treatment:

| Available information | Defensible modeling task | Limit on interpretation |
| --- | --- | --- |
| Daily profiles and usable reported equipment indicators | Generate profiles conditional on the reported indicators and calendar | Results concern reported groups; they do not identify causal equipment effects. |
| Aggregate profiles with equipment flags, but no building/heat-pump subtype | Generate the mixture of household behaviors within each available condition | Cannot claim controllable generation for an unobserved subtype or explain which subtype produced a profile. |
| Aggregate profiles without separate appliance measurements | Generate whole-house net-load profiles | Cannot validate separate HP, EV or other-household consumption from the aggregate measurements alone. |
| Daily profiles with no paired input–output targets | Ordinary generative modeling remains possible | Paired real target profiles are not required for GAN, flow-matching or diffusion training. |

A learned cluster can describe a recurring load shape; naming it a heat-pump
type or building category requires separate ground truth. Holding calendar fixed
while changing a model's equipment condition measures learned conditional
association, not the effect of installing that equipment. Multiple appliance
decompositions can produce the same observed net-load curve. A more powerful
generator does not resolve that identification problem by itself.

## Why flow matching is worth a controlled test

Flow matching trains a vector field that transports a simple distribution, such
as Gaussian noise, into the data distribution. A useful first construction draws
a standardized training profile \(x\), independent Gaussian noise \(z\), and
\(t\sim U(0,1)\), then sets

\[
x_t=(1-t)z+tx,\qquad u_t=x-z.
\]

Train \(v_\theta(x_t,t,c)\) to predict \(u_t\) with squared error. The optional
\(c\) contains available conditioning variables. Noise/profile pairings are
created during training; they are not external supervision labels. “Conditional
flow matching” can refer to conditioning the mathematical path on sampled
endpoints, which is different from supplying household equipment labels.

At generation time, draw \(x_0=z\) and integrate
\(dx_t/dt=v_\theta(x_t,t,c)\) to time 1, then invert the training-only affine
transform. The training objective requires no discriminator and no ODE solve
inside each training update. Sampling still requires numerical integration.
Straight interpolation targets do not guarantee a perfectly straight learned
marginal flow or accurate one-step generation. Rectification, distillation and
minibatch optimal-transport coupling are additional techniques with their own
costs and ablations, rather than free consequences of the basic objective.

Flow matching does not need building-type or appliance-consumption labels for
whole-house generation. Stochastic sampling can represent different behaviors
within the same HP/EV/PV condition, although adequate mixture coverage must be
demonstrated. Without component targets, do not interpret parts of the latent
space or generated outputs as validated appliance disaggregation.

This is relevant to the observed cGAN instability: regression of a velocity field
removes the two-player optimization process. It does not remove model-capacity,
normalization, tail-fitting or data-quality problems, and it does not guarantee
better diversity or fidelity than the copula baseline.

## Prioritized alternatives

| Method | Reason to consider it | First useful comparison |
| --- | --- | --- |
| Stabilized cGAN training | Preserves the reference topology and directly addresses the observed scaling and optimization failures | Continue isolated scaling and optimizer controls, retaining all seeds and budgets. |
| Direct flow matching / basic rectified-flow objective | Non-adversarial training; feasible for 24 real-valued hourly coordinates and optional metadata | A small time-conditioned MLP with the same train-only scale, same meter split and matched evaluation rows. |
| Score-based diffusion | Another non-adversarial density model; can support conditional generation and masked observations | Add only if it answers an unresolved comparison; disclose noise schedule, denoising/sampling steps and runtime. CSDI is specifically an imputation method, not a drop-in validation of whole-day generation. |
| Self-supervised representations and behavioral clustering | Uses unlabeled histories to discover recurring behavior within the available equipment groups | Fit representations and clustering on fitting meters only; assess stability on other meters. Keep clusters distinct from building or device-subtype labels. |
| Learned latent heterogeneity | Represents several behaviors within one observed equipment/calendar condition | Test conditional multimodality, tails and diversity; do not name latent factors as physical appliances without additional evidence. |

The 24-dimensional hourly problem does not initially require a large transformer,
a latent autoencoder, or a foundation model. Those additions introduce capacity
and preprocessing changes that can obscure the comparison. Daily models also
cannot establish multi-day continuity; that requires a separately defined
sequence-generation task and evaluation.

## Smallest useful flow-matching experiment

This is a proposed diagnostic, not a frozen execution plan or a reported result.
Keep the original cGAN record immutable and name a new flow arm explicitly.

1. Use the fixed inner fitting/validation meter split, global standardization
   fitted on inner TRAIN, signed physical outputs and no clipping. Begin with
   the current reported 64-class conditions so the first comparison changes the
   generative method. Calendar-only and unconditional variants may then test the
   value of the available conditions under a separately declared ablation plan.
2. Use a modest time-conditioned MLP with direct 24-value velocity output and the
   simple independent-pair, linear-path objective above. Record all trainable
   parameters and compare training wall time and data exposures; one flow update
   is not computationally equivalent to one discriminator/generator update pair.
3. Predeclare a small common screen, for example the existing training seeds
   101 and 202 and a bounded update budget. Record failed and interrupted attempts.
   Promising methods require longer budgets and more seeds before final claims.
4. Fix a solver and step count before checkpoint selection. A reasonable numerical
   check compares a moderate fixed-step Heun solve with a doubled-resolution
   solve at identical latent draws. Report the number of vector-field evaluations
   (NFE), integration sensitivity and generation time. Avoid selecting a solver
   solely because it happens to improve the validation metric.
5. Use the same physical-unit distribution metrics and selection rows as the
   cGAN and conditional resampling/copula baselines. Report conditional means,
   hourly and daily-energy distributions, ramps, temporal dependence, export
   behavior, tails, diversity and copying/proximity diagnostics. A favorable
   mean profile or lower velocity-regression loss is insufficient.

If conditions are dropped for an ablation, every comparator must receive the same available
conditioning information. Use matched marginal/calendar endpoints, and report
equipment subgroup discrepancies only as descriptive diagnostics;
an unconditional model is not expected to produce an arbitrary requested true
equipment group. Do not silently score its unconditional output as though it
were correctly conditioned on all 64 classes.

Representation learning, clustering, imputation and normalization all belong
inside each fitting partition. Clustering the full cohort before a meter split
would leak evaluation information. No method can turn the already observed
2024 release into a newly independent confirmation dataset.

## Sources and limits of the review

The following foundational papers and author repositories ground this shortlist:

- Lipman et al., *Flow Matching for Generative Modeling*, ICLR 2023,
  [paper](https://arxiv.org/abs/2210.02747).
- Liu, Gong and Liu, *Flow Straight and Fast: Learning to Generate and Transfer
  Data with Rectified Flow*, ICLR 2023,
  [paper](https://arxiv.org/abs/2209.03003) and
  [official implementation](https://github.com/gnobitab/RectifiedFlow).
- Lipman et al., *Flow Matching Guide and Code*, 2024,
  [guide](https://arxiv.org/abs/2412.06264) and
  [official codebase](https://github.com/facebookresearch/flow_matching).
- Tong et al., *Improving and Generalizing Flow-Based Generative Models with
  Minibatch Optimal Transport*, 2023 preprint;
  [TorchCFM author repository](https://github.com/atong01/conditional-flow-matching)
  describes independent-pair, optimal-transport and related variants.
- Tashiro et al., *CSDI: Conditional Score-based Diffusion Models for
  Probabilistic Time Series Imputation*, NeurIPS 2021,
  [paper](https://arxiv.org/abs/2107.03502) and
  [author implementation](https://github.com/ermongroup/CSDI).
- Yue et al., *TS2Vec: Towards Universal Representation of Time Series*, AAAI 2022,
  [paper](https://arxiv.org/abs/2106.10466) and
  [author implementation](https://github.com/zhihanyue/ts2vec).

Author/official GitHub READMEs were retrieved and read on 21 October 2026. The
arXiv paper endpoints returned a network-proxy error, so this is not a newly
verified full-paper review or an exhaustive search of the latest 2026 literature.
[The source receipt](../reports/phase1-development/METHOD_SOURCE_REVIEW.json)
records successful and failed retrievals, content hashes and repository commits.
Published results on other datasets do not establish performance on Fluvius.

For implementation, the basic equations can be implemented directly with paper
citation. Any copied library code retains its own license: the fetched official
`facebookresearch/flow_matching` README identifies CC BY-NC, which differs from
the preserved reference-cGAN MIT notice. This review has not copied that code.
