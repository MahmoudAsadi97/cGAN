# Recent hybrid generators relevant to Fluvius profiles

Assessment date: 21 October 2026. This review compares structured temporal
sources, diffusion bridges, latent generators and energy-based refinement.
The named paper methods are research candidates; their reported performance
has not been independently reproduced on Fluvius.

**Current priority after diagnostics:** the completed source studies and
[fresh-draw diagnostics](../reports/phase1-development/fresh_diagnostics/README.md)
now motivate explicit conditional marginal modeling plus learned dependence.
The targeted [support-aware copula review](SUPPORT_AWARE_GENERATORS.md) examines
Huk and Damoulas's 2025 preprint, author-listed at ICLR 2026, including its
reflection-flow and classification-diffusion code. This is a new research
candidate; it has not been implemented or validated on Fluvius. The earlier
structured-source recommendation below records the hypothesis that was tested.

**Development follow-up:** the first controlled TRAIN-correlation source
ablation has now completed at two seeds and 6,000 updates. It worsened profile
SW1 and ramp W1 compared with the independent source, despite slightly better
daily-energy W1. See the [complete comparison](../reports/phase1-development/correlated_flow_screen/README.md).
This specific combination has not earned adoption. A declared common longer
budget tests whether the result depends on optimization time; it cannot erase
the unfavorable short-budget result or establish the full TSFlow method's quality.

Our target is a signed 24-hour whole-house net-energy profile, conditioned on
the available HP/EV/PV flags and calendar. Building type, heat-pump subtype and
individual appliance consumption are unobserved. A distribution model can
represent variation caused by those unknown factors without identifying them.
Missing household attributes are different from missing entries in an observed
time series, so advances in imputation do not automatically solve this task.

## Four relevant combinations

| Work and source status | What is combined | Relevance and first useful test |
| --- | --- | --- |
| **TSFlow**, *Flow Matching with Gaussian Process Priors for Probabilistic Time Series Forecasting*, Kollovieh et al.; author repository identifies ICLR 2025 | Flow matching with a Gaussian-process source and past-observation conditioning | Temporal structure enters the starting distribution. For whole-day generation, compare an independent Gaussian source with a TRAIN-fitted correlated Gaussian source; this is an adaptation, since TSFlow's forecasting model uses history unavailable in our present task. |
| **TimeBridge**, *Better Diffusion Prior Design with Bridge Models for Time Series Generation*; author repository identifies KDD 2026 and links the earlier preprint 2408.06672 | Diffusion bridges with alternative source priors, including GP and trend priors, using ODE or SDE sampling | Directly addresses whole-series generation and source-distribution choice. A GP-source bridge is a relevant comparator after the simpler flow control. Trend-conditioned experiments require a specified trend at inference and are a different conditioning task. |
| **Energy Matching**, *Unifying Flow Matching and Energy-Based Models for Generative Modeling*, Balcerak et al.; NeurIPS 2025 identified in author materials | Optimal-transport alignment of a scalar potential, followed by contrastive energy learning and Langevin refinement | Could combine a transport route with a density-based refinement stage. Test only if simpler methods leave a specific mixture/tail failure; input gradients, transport coupling and finite-chain sampling add cost and mixing questions. |
| **Diffusion-TS**, *Interpretable Diffusion for General Time Series Generation*, Yuan and Qiao; official repository identifies ICLR 2024 | Diffusion with temporal decomposition, direct sample reconstruction and a Fourier loss | Offers a time-structure-aware diffusion comparator and a source of isolated spectral-loss ablations. Smooth trends/seasonality are statistical components, not measured appliances. A large transformer is not required merely because this paper uses one. |

Sources:
[TSFlow](https://github.com/marcelkollovieh/TSFlow),
[TimeBridge](https://github.com/JinseongP/TimeBridge)
([paper identifier](https://arxiv.org/abs/2408.06672)),
[Energy Matching](https://github.com/m1balcerak/EnergyMatching)
([paper](https://arxiv.org/abs/2504.10612)), and
[Diffusion-TS](https://github.com/Y-debug-sys/Diffusion-TS)
([paper](https://openreview.net/forum?id=4h1apFjO99)).
Venue descriptions above are verified against author/official materials; the
conference proceedings themselves were inaccessible in this environment.

## Why the structured-source comparison comes first

The inspected TSFlow implementation supports isotropic, squared-exponential,
Ornstein–Uhlenbeck and periodic Gaussian-process kernels. Its conditional model
forms a GP posterior from past observations and transports a sample of that
posterior toward a future trajectory. This is a concrete source-design method,
not just a larger velocity network. Its observed history must not be replaced
with the held-out day's actual values in our generation experiment.

For our 24-dimensional task, a simpler analogue is feasible:

1. Fit the existing global affine scale on the current fitting meters only.
2. Compare basic conditional flow matching from independent Gaussian noise
   with the same model starting from a correlated Gaussian. Fit any covariance,
   kernel parameters, mean or equipment-group prior on fitting meters only.
3. Include the source distribution by itself as a baseline. A full-covariance
   Gaussian and a regularized empirical covariance offer useful controls
   alongside a smooth GP kernel and the existing copula.
4. Hold the velocity network, training budget, seeds, solver and evaluation
   samples fixed when comparing source choices. Report prior fitting cost and
   sampling cost as well as neural updates.

The hypothesis is reduced transport difficulty and better temporal dependence,
not guaranteed accuracy. A smooth kernel can suppress sharp EV/household ramps
or rare export/load extremes. Compare ramps, peaks, tails and temporal covariance
in addition to SW1. Condition-specific mean/covariance priors change several
features at once; start with a common zero-mean correlated source before adding
those as separate ablations. Calling this adaptation an exact TSFlow reproduction
would be inaccurate.

TimeBridge is the next relevant step because its author code explicitly exposes
`normal`, `uniform`, trend and GP priors, `vp`/`ve` prediction choices, and
ODE/SDE samplers. It builds on Diffusion-TS and denoising diffusion bridge models.
This flexibility is useful, but selecting many priors and samplers simultaneously
would obscure which change helps. The GP-source flow and GP-source bridge need
their own declared settings and numerical-solver checks. A source derived from
a requested trend changes the information supplied to the model and must be
compared against baselines given the same information.

## What energy-based refinement would add

The Energy Matching author's July 2026 technical slides describe a
time-independent scalar potential whose negative spatial gradient supplies a
transport field. An initial objective aligns that field with a minibatch
optimal-transport coupling. A later contrastive objective compares data with
finite-chain Langevin samples, and a time-dependent temperature shifts sampling
from directed transport toward stochastic exploration.

This is a substantive flow/energy-model hybrid. It is not electrical energy
balance: the model's scalar potential is a statistical energy. At equilibrium
it defines a density up to an unknown normalizing constant; relative density
information does not give a freely available normalized likelihood. Finite-chain
negatives are biased, and a fixed number of Langevin steps does not prove mixing.

For Fluvius, conditional mixture coverage and tails would be the relevant
hypotheses. A conservative, time-independent gradient field is also a modeling
restriction. Compare transport alone, refinement alone and the combined method,
with matched wall time and reported gradient evaluations. The published
image/protein examples do not establish an advantage for household profiles.
This is lower priority than a direct flow and a structured-source ablation on
the current CPU resources.

## Latent representations and mixed attributes

Two verified author codebases provide useful related designs:

- **TabSyn**, Zhang et al., ICLR 2024: a variational autoencoder maps mixed-type
  tables into continuous embeddings, then score-based diffusion models those
  embeddings. [Official source](https://github.com/amazon-science/tabsyn),
  [paper](https://arxiv.org/abs/2310.09656).
- **TabDiff**, Shi et al., ICLR 2025: continuous-time mixed-type diffusion with
  feature-wise learned noise schedules and conditional imputation guidance.
  [Official source](https://github.com/MinkaiXu/TabDiff),
  [paper](https://arxiv.org/abs/2410.20626).

A learned latent representation plus diffusion or flow can represent behavioral
mixtures without naming the missing building/HP types. However, our profile is
only 24 continuous values and the equipment flags are already available as
conditions. Mixed-type joint generation is not required to generate these fixed
conditions, and an autoencoder adds a reconstruction bottleneck. Establish its
reconstruction error on peaks, exports and ramps before expecting a latent
generator to improve them. Count encoder/decoder training and capacity in the
comparison.

An embedding inferred from a complete held-out profile cannot be used to claim
unconditional generation of that same profile. Generation must sample its latent
state from a TRAIN-fitted prior, or explicitly condition on information available
at deployment. Likewise, missing-value imputation assumes the relevant attribute
was observed in training examples: it cannot validate a building-type column
that is absent from every household in this release. Learned clusters remain
behavioral strata until independently labeled.

## Newer 2026 items checked, with narrower applicability

**MissPath-FM**, Zhang, *Flow Matching with Structured Priors for Partially
Observed Time Series*, has an accessible author-hosted full manuscript. Its
repository and PDF identify NeurIPS 2026; independent proceedings verification
was not available. The method combines an observation-aware interpolation/GP
prior, a mask-dependent path and a context-conditioned velocity network. The
paper's method, experimental design and limitations were inspected.
[Author repository](https://github.com/zgp110/misspath-fm),
[manuscript in that repository](https://github.com/zgp110/misspath-fm/blob/main/docs/static/MissPath-FM_NeurIPS2026.pdf).

This is relevant if we later generate missing intervals conditional on observed
ones. The present profiles are complete after curation; missing appliance or
building labels are not time-entry masks. Its imputation MAE and averaging across
generated completions also answer a different question from distribution fidelity
of independent whole-day samples. Its reported comparison wins are author
results on other datasets, not evidence for Fluvius or an independently verified
ranking here.

**SDFlow**, Li et al., *Similarity-Driven Flow Matching for Time Series
Generation*, combines a VQ-VAE tokenizer, low-rank latent structure and flow
matching. The repository identifies NeurIPS 2026, but neither a linked full
paper nor independent acceptance verification was obtained.
[Author repository](https://github.com/William-Liwei/SDFlow).
Its README includes a two-stage GPU recipe and numerical speed/quality claims;
those claims have not been verified. Tokenization may help long sequences, but
can quantize rare continuous extremes and adds substantial machinery for 24
hourly values. Keep it on the literature watchlist rather than importing its
reported speedup or superiority into the project.

## Controlled decision rule and source coverage

Retain the preserved reference-cGAN results and continue the current training
controls. For advanced comparators, first establish a simple conditional flow,
then change one mechanism at a time: source covariance, bridge objective,
temporal penalty, latent representation or energy refinement. Prefer a small
number of falsifiable hypotheses over a large combined architecture. A mixture
of individually promising components is not evidence that the combination works.

Use the same meter-separated fitting/development cohorts, train-only transforms,
generation sample counts and physical metrics. Report all training seeds,
parameter counts, training time, generation time and solver/gradient evaluations.
Evaluate conditional fidelity, diversity, energy totals, covariance/ACF, ramps,
export behavior, tails and exact-copy diagnostics against the resampling and
copula controls. No technique creates appliance-level validation or a newly
independent dataset from the existing release.

Do not import an upstream split unchanged: for example, Diffusion-TS documents
using all data for its default unconditional evaluation. Our adaptation must
retain fitting/evaluation meter separation and TRAIN-only preprocessing, even
when reproducing that architecture or objective.

The [source receipt](../reports/phase1-development/HYBRID_METHOD_SOURCE_REVIEW.json)
records successful retrievals, failures, content hashes and repository commits.
Primary evidence includes the full MissPath-FM author manuscript, the 28-page
Energy Matching author talk, official READMEs, and inspected TSFlow source code.
The arXiv/OpenReview/NeurIPS proceedings endpoints attempted here were blocked
by the network proxy. Full papers for the other shortlisted works were not
retrieved. This is a targeted, explicitly bounded 2024–2026 review, not an
exhaustive frontier survey or independent reproduction of their reported results.
