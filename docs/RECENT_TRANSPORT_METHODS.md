# Recent one-step and few-step transport methods

Assessment date: 21 October 2026. This review follows the request to examine
recent methods beyond basic flow matching, including useful combinations. It is
a research shortlist for Phase 1 comparisons, not a decision to replace the
reference cGAN or start federated learning.

The most relevant recent transport direction is **improved MeanFlow or flow-map
self-distillation**, provided an ordinary flow model first achieves adequate
profile fidelity. Their main additional goal is to learn a large transport step
so that sampling needs fewer network evaluations. That advantage is primarily
over iterative flow/diffusion sampling: the existing GAN already generates a
profile in one network pass. Fewer sampling steps do not establish better
distribution fitting, tail coverage or recovery of missing appliance labels.

## What was verified

Official or author GitHub repositories, pinned source files and an author's
publication page source were retrieved and read. The complete official
`meanflow.py`, `imf.py`, and `targets_shortcut.py` objective implementations were
read, beyond their READMEs. The flow-map self-distillation project equations and
official loss implementation, and F2D2's joint loss and likelihood code, were
also inspected. Full arXiv, proceedings and author-web endpoints
attempted from this environment were blocked by the network proxy. The author's
website Git repository was inspected for PDFs; it did not contain these recent
papers. Therefore this document distinguishes implementation inspection and
author-reported publication metadata from a full-paper review. It does not claim
an exhaustive search of all 2026 work.

The [retrieval receipt](../reports/phase1-development/RECENT_TRANSPORT_SOURCE_REVIEW.json)
records URLs, retrieval outcomes, content hashes and available source commits.
The dates below distinguish initial preprint year from later venue year.

| Work | Verified identity/date | What it adds | Relevance to this study |
| --- | --- | --- | --- |
| **One-Step Diffusion via Shortcut Models** | Kevin Frans, Danijar Hafner, Sergey Levine and Pieter Abbeel; arXiv:2410.12557, October 2024. Official repository verifies title, authors and objective; conference venue was not independently retrieved here. | Conditions the model on both time and requested step size; combines local flow regression with bootstrap targets built from two half steps. | A relatively interpretable way to compare one, few and many sampling steps using one model. Sampling acceleration is not a missing-label solution. |
| **Mean Flows for One-step Generative Modeling** | Zhengyang Geng, Mingyang Deng, Xingjian Bai, J. Zico Kolter and Kaiming He; arXiv:2505.13447, May 2025. Author publication source lists NeurIPS 2025 Oral. | Learns average velocity over a time interval, using a consistency identity involving a directional derivative; can sample with one learned transport step. | A direct modern alternative to repeated velocity integration, with extra derivative computation and training choices. |
| **Improved Mean Flows: On the Challenges of Fastforward Generative Models** | Zhengyang Geng, Yiyang Lu, Zongze Wu, Eli Shechtman, J. Zico Kolter and Kaiming He; arXiv:2512.02012, December 2025. Author publication source lists CVPR 2026. | The inspected implementation adds an instantaneous-velocity head, changes the derivative direction to predicted velocity, and trains a compound velocity objective. It also supports guidance conditioning. | Stronger candidate than blindly porting the first MeanFlow recipe if a few-step comparison becomes useful. Its image-specific losses and guidance need separate justification for energy data. |
| **How to build a consistency model: Learning flow maps via self-distillation** | Nicholas M. Boffi, Michael S. Albergo and Eric Vanden-Eijnden; arXiv:2505.18825, May 2025; NeurIPS 2025 according to the author metadata and official repository. | Combines diagonal flow matching with finite-jump consistency objectives: Lagrangian, progressive or Eulerian self-distillation. | Lagrangian self-distillation is a useful alternative to improved MeanFlow for a small controlled experiment; it avoids the map's spatial Jacobian. |
| **Joint Distillation for Fast Likelihood Evaluation and Sampling in Flow-based Models (F2D2)** | Xinyue Ai, Yutong He, Albert Gu, Ruslan Salakhutdinov, J. Zico Kolter, Nicholas M. Boffi and Max Simchowitz; arXiv:2512.02636, December 2025; ICLR 2026 in the author repository and publication metadata. | Jointly distills transport and divergence-related information for fast sampling and approximate likelihood evaluation. | Conditional priority: needs a good teacher and an actual likelihood/large-volume sampling use case. Additional training stages may not pay off for 24 coordinates. |
| **One-step Latent-free Image Generation with Pixel Mean Flows** | Yiyang Lu, Susie Lu, Qiao Sun, Hanhong Zhao, Zhicheng Jiang, Xianbang Wang, Tianhong Li, Zhengyang Geng and Kaiming He; arXiv:2601.22158, January 2026. Author source lists a technical report; official JAX/PyTorch code was released in February 2026. | Applies one-step mean-flow generation directly in image pixel space, avoiding a pretrained image latent autoencoder. | Verified recent work, but low priority here: our generator already operates directly on 24 hourly values and needs no image autoencoder. |

Official source commits inspected:

- [MeanFlow](https://github.com/Gsunshine/meanflow/tree/d70cb55d298ee03c53bf6da67bec281082e4e2d9):
  `d70cb55d298ee03c53bf6da67bec281082e4e2d9`.
- [Improved MeanFlow](https://github.com/Lyy-iiis/imeanflow/tree/bf60cd7cb653f6628e59d48034b333c5eba445e2):
  `bf60cd7cb653f6628e59d48034b333c5eba445e2`.
- [Shortcut Models](https://github.com/kvfrans/shortcut-models/tree/601004348667094e1b71f30942199759412d4432):
  `601004348667094e1b71f30942199759412d4432`.
- [Author publication metadata](https://github.com/Gsunshine/Gsunshine.github.io/blob/cc320fef373e039c7c79c474684123306466bfad/_pages/about.md):
  `cc320fef373e039c7c79c474684123306466bfad`.
- [Flow-map self-distillation](https://github.com/nmboffi/flow-maps/tree/2f115a07fa9073553193e4b265dfc303827af2b0):
  `2f115a07fa9073553193e4b265dfc303827af2b0`.
- [F2D2](https://github.com/Keely-Ai/F2D2/tree/391ea389b9fc0ee980406dce73e1d320852465c8):
  `391ea389b9fc0ee980406dce73e1d320852465c8`.

## The methodological differences that matter

Ordinary flow matching learns an instantaneous velocity, then uses a numerical
solver at generation time. MeanFlow instead models average velocity between two
times. The inspected official implementation uses a Jacobian-vector product
(JVP) to form a target of the form

\[
u_{\rm target}=v-(t-r)\,\frac{d u}{dt},
\]

with the target detached during the parameter update. The derivative is along
a transport direction; it is not merely a partial derivative with respect to
the time input. A single generated step then uses the predicted average velocity
over the whole interval. This is a different training problem from taking one
large Euler step through an ordinary flow model.

In the inspected improved-MeanFlow implementation, the JVP direction uses a
predicted instantaneous velocity, and the compound quantity
\(V=u+(t-r)\,\mathrm{stopgrad}(du/dt)\) is compared with a velocity target. An
auxiliary instantaneous-velocity head is trained too. These are concrete changes
to the target, architecture and gradient computation; they should not be bundled
with unrelated conditioning or physical-loss changes in one unexplained model.

Shortcut training uses a different construction. Local training examples teach
the small-step flow direction; larger-step targets combine two model-predicted
half steps. The requested step size is an explicit input. Bootstrapping allows a
single model to support several inference budgets, but inaccurate shorter steps
can contaminate the larger-step targets. It adds training computation even when
inference becomes cheaper.

Flow-map self-distillation provides a broader way to combine these ideas. A map
\(X_{s,t}(x)=x+(t-s)v_{s,t}(x)\) learns finite jumps; its diagonal is tied to the
ordinary velocity by flow matching. Lagrangian self-distillation matches the
terminal-time derivative of the map to the diagonal velocity at its output.
Progressive self-distillation enforces composition of shorter maps, with Shortcut
Models as a special case. Eulerian self-distillation uses a transport-PDE
residual involving the map's spatial derivative. The first is an attractive
candidate here because it avoids that spatial Jacobian, although time derivatives
and extra evaluations still cost training work. The official experiments concern
images and a two-dimensional synthetic task, not residential load data.

F2D2 addresses a different combination: fast transport plus fast likelihood
evaluation. Its inspected pipeline first trains a flow teacher and a shortcut
student, then jointly distills velocity and a divergence-related output. Small
steps use teacher targets, including a stochastic divergence estimator; longer
steps bootstrap two shorter ones. Evaluation integrates the predicted divergence
alongside the state. A distilled likelihood estimate must be checked against a
resolved teacher calculation; it is not automatically the exact density of the
finite-step sampler. F2D2's image dequantization, input scaling and divergence
scaling cannot be transplanted to signed kWh values unchanged.

The verified repositories primarily report image-generation benchmarks. Their
FID results cannot be transferred to hourly load profiles, and image encoders
should not be used as electricity-quality metrics without an appropriate
validation argument.

## Adaptation risks visible in the inspected code

The official Shortcut target code clips intermediate states and target velocities
to an image-oriented range of `[-4,4]`. Copying that behavior into this study would
change the signed-load and tail distributions. Our existing physical outputs
remain unclipped; any newly proposed bounds would require a physical basis and
a separate reported ablation.

MeanFlow and improved MeanFlow include per-example adaptive loss weighting.
In the inspected defaults this reduces the influence of large residuals. That
can have different consequences for energy data, where infrequent high-import,
high-export and ramp events are scientifically important. Average fidelity must
therefore be assessed alongside tail and subgroup coverage.

Classifier-free guidance can trade diversity for sharper conditional samples.
For a load-distribution model, changing probabilities to improve an image-style
quality score would be undesirable. Start with the unguided target distribution;
if guidance is later studied, measure the distributional distortion explicitly.

Neither average-velocity targets, self-distillation nor a latent behavioral
mixture provides true building-type or heat-pump subtype labels. The usable
HP/EV/PV indicators support whole-house conditional generation. Separate appliance
consumption remains unobserved and cannot be validated through aggregate-only
training metrics.

## A useful combination and its controlled comparison

A defensible combination is an ordinary conditional flow model plus a learned
few-step transport map, evaluated against the original flow as its accuracy
reference. This can test whether generation becomes cheaper while retaining
diversity, multimodal behavior within the observed equipment groups, temporal
dependence and extreme-event frequencies. It should not be presented as an
automatic improvement in all those properties.

For the present 24-value problem, prioritize improving fidelity before reducing
an already small sampling cost. If the ordinary flow comparator is competitive,
choose one recent family, such as improved MeanFlow, and compare a small fixed
set of inference budgets under a declared seed study. Record trainable parameters,
training wall time, data presentations, JVP/teacher costs, sampling NFE and total
generation time. A one-step model may need more training work or capacity than
the multi-step control.

Keep the fitting and validation meters, train-only normalization, physical
metrics and resampling/copula controls fixed. Compare all selected training
seeds, not a favorable image-recipe checkpoint. Add a controlled latent-mixture
or behavioral-conditioning change only after that comparison, so any gain can
be attributed to the relevant change. No architecture can create a fresh
independent cohort from the already observed release.

Primary links:

- [Shortcut Models paper](https://arxiv.org/abs/2410.12557)
- [Mean Flows paper](https://arxiv.org/abs/2505.13447)
- [Improved Mean Flows paper](https://arxiv.org/abs/2512.02012)
- [Flow-map self-distillation paper](https://arxiv.org/abs/2505.18825) and
  [official project equations](https://github.com/flow-maps/flow-maps.github.io/blob/cbee36be41b9cc95f10d585f551f960cb24ab62b/index.html)
- [F2D2 paper](https://arxiv.org/abs/2512.02636) and
  [conference record](https://openreview.net/forum?id=8uZ5UdIul2)
- [Pixel Mean Flows paper](https://arxiv.org/abs/2601.22158) and
  [official implementation](https://github.com/Lyy-iiis/pMF)

Two date/scope distinctions prevent overstating novelty. *Flow map matching with
stochastic interpolants: A mathematical framework for consistency models*
([arXiv:2406.07507](https://arxiv.org/abs/2406.07507)) originated in 2024 and is
listed as TMLR 2025; the author's minimal repository explicitly does not reproduce
the exact paper experiments. *Generalised Flow Maps for Few-Step Generative
Modelling on Riemannian Manifolds* ([arXiv:2510.21608](https://arxiv.org/abs/2510.21608),
ICLR 2026 in author metadata) is verified recent work but lower priority: no
Riemannian manifold constraint has been established for the present hourly
net-load vectors. Only its author metadata and official README were reviewed.
