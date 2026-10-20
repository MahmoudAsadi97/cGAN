# Literature context and claim boundaries

Checked against primary publications and the upstream repository on 20 October
2026. This is a targeted attribution check, not an exhaustive systematic review
or evidence of priority. It does not change the frozen experiment settings.

## What is established work

Li, Dong and Qiu [1] already study conditional generation of building load
profiles with PV and EV adoption. Their linked implementation [2] is the source
of this project's reference MLP. V2 changes the population, data preparation,
condition semantics and output resolution; its current calibrated candidate
also changes training. Consequently, it is an **adaptation**, not a numerical
reproduction of the published experiment. The exact preserved architecture and
departures are recorded in [UPSTREAM_ADAPTATION.md](UPSTREAM_ADAPTATION.md).
The upstream generated examples do not substitute for the missing original
training data and trained weights.

Janke, Ghanmi and Steinke [3] separate neural dependence modeling from marginal
modeling by using probability-integral transforms. Their implicit generative
copula is trained in copula space with energy distance and a differentiable
softrank approximation. V2 instead trains an adversarial profile generator and
then fits a fixed conditional marginal map. It therefore borrows a related
construction, **not that paper's training algorithm or theoretical guarantee**.
Finite reference samples, quantile interpolation, ties and endpoint saturation
mean V2's marginal alignment is approximate. Better marginal scores after this
map cannot, by themselves, demonstrate better learned dependence.

Deshpande, Zhang and Schwing [4] establish generative distribution matching
through random one-dimensional projections and sliced Wasserstein distances.
V2's condition-specific squared-W2 auxiliary loss uses this established family
of objectives; neither adding it to a GAN nor using random projections warrants
a new-algorithm claim. Miyato and colleagues [5] establish spectral normalization
as discriminator stabilization. Their image-generation results motivate the
technique but do not prove that it improves this electricity dataset.

Conditional load generation also extends beyond GANs. Xia and colleagues [6]
describe a flow model for conditional and unconditional electricity profiles.
This is a relevant alternative family, not an implemented V2 comparison. Its
published comparisons concern its own settings; the current study cannot claim
state-of-the-art performance over flows or other untested methods.

## Why test profile and adjacent-difference losses?

The following is this project's methodological reasoning, not a result proved
by the cited load-generation papers. Whole-profile projections involve several
time slots together, so their distribution discrepancy can supply training
information that separate per-slot marginal discrepancies miss. However, finite
random projections, minibatches, limited model capacity and optimization error
leave room for particular temporal features to be represented poorly.

Applying the same discrepancy to the 95 signed adjacent differences emphasizes
within-day changes. This is distribution matching of a fixed linear transform,
not smoothing each generated trace or penalizing large ramps indiscriminately.
The difference vector does not identify a profile's level, which is one reason
to retain the profile objective. Exact agreement of full profile distributions
would already imply agreement after this transform; the extra term changes the
practical emphasis of finite-sample training rather than adding a fundamentally
new target distribution.

The derivative loss operates in TRAIN-standardized space. Slot-specific nonlinear
calibration changes differences, so better raw derivative matching does **not**
guarantee better calibrated ramp errors. Daily energy, peaks, autocorrelation,
whole-profile projections and utility can also trade off. The unchanged
[temporal pilot protocol](TEMPORAL_REFINEMENT.md) tests that possibility using
matched seeds and budgets, with all attempts retained. A failed pilot must remain
a failed pilot; it cannot be rescued by substituting a favorable metric or seed.

## Defensible empirical contribution

A bounded study question is: **On identity-curated Fluvius 2024 profiles, how much
does conditional distribution matching improve a calibrated cGAN's temporal
fidelity, and which joint and downstream properties remain weaker than simple
TRAIN-only controls?** The contribution can be a reproducible, provenance-bound
empirical assessment with explicit failures and tradeoffs. Applying an existing
model to another dataset, combining established ingredients or passing software
tests does not independently establish methodological novelty.

The completed original arm supports a narrower development observation:
calibrated ramp errors improve relative to the matched Gaussian copula, while
conditional resampling remains stronger and the five-seed primary ramp mean
misses the declared absolute target. The existing observed-history ridge task
meets its descriptive tolerance but is not day-ahead or future-year validation.
See [the completed validation evidence](CURATED_COMPLETED_VALIDATION.md) and
[matched SWD attribution](CURATED_SWD_ATTRIBUTION.md); these statements concern
reused validation data, not an unseen-test result.

Any final claim must retain all five seeds, adverse metrics and the resampling
reference. The sealed TEST assessment can assess the frozen method on this
internal meter holdout; it cannot establish external generalization, household
independence, privacy or publication acceptance. New decisions after TEST
inspection would make that partition development data. The reference BCE,
unconditional and common-membership hourly experiments remain separate
manuscript requirements in [RESEARCH_PROTOCOL.md](RESEARCH_PROTOCOL.md).

## Primary references

1. Li, Y., Dong, B., and Qiu, Y. (2025). *Conditional generative adversarial
   network (cGAN) for generating building load profiles with photovoltaics and
   electric vehicles.* **Energy and Buildings, 335**, 115584.
   [Publisher / DOI](https://doi.org/10.1016/j.enbuild.2025.115584).
2. Li et al., upstream implementation, **malvin0130/cGAN_EB**.
   [Pinned commit 885f78052a91fbc755e6b17a354c68849b5556ca](https://github.com/malvin0130/cGAN_EB/tree/885f78052a91fbc755e6b17a354c68849b5556ca).
   The repository README identifies reference [1]; preserve its MIT notice.
3. Janke, T., Ghanmi, M., and Steinke, F. (2021). *Implicit Generative Copulas.*
   **NeurIPS 34**, 26028–26039.
   [Proceedings](https://proceedings.neurips.cc/paper/2021/hash/dac4a67bdc4a800113b0f1ad67ed696f-Abstract.html).
   Sections 2–3 distinguish the construction from its copula-space training.
4. Deshpande, I., Zhang, Z., and Schwing, A. G. (2018). *Generative Modeling
   Using the Sliced Wasserstein Distance.* **CVPR**, 3483–3491.
   [Conference proceedings](https://openaccess.thecvf.com/content_cvpr_2018/html/Deshpande_Generative_Modeling_Using_CVPR_2018_paper.html);
   [author preprint](https://arxiv.org/abs/1803.11188).
5. Miyato, T., Kataoka, T., Koyama, M., and Yoshida, Y. (2018).
   *Spectral Normalization for Generative Adversarial Networks.* **ICLR**.
   [Author preprint](https://arxiv.org/abs/1802.05957);
   [conference record](https://openreview.net/forum?id=B1QRgziT-).
6. Xia, W., Wang, C., Palensky, P., and Vergara, P. P. (2025). *A flow-based
   model for conditional and probabilistic electricity consumption profile
   generation.* **Energy and AI, 21**, 100586.
   [Publisher / DOI](https://doi.org/10.1016/j.egyai.2025.100586);
   [authors' institutional record and published paper](https://research.tudelft.nl/en/publications/a-flow-based-model-for-conditional-and-probabilistic-electricity-/).

The checked Li et al. title, authors, year, volume and article number, and the
Janke et al. NeurIPS attribution agree with the current V2 documents. This check
found no bibliographic correction needed in those two existing attributions.
