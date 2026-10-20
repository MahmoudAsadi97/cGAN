# Related work and claim boundaries

This is targeted positioning for the implemented study, not a systematic review
or a state-of-the-art claim. The method components below are established. Their
combination and application to another dataset do not by themselves establish
methodological novelty. The experimental decisions and current evidence belong
in [CALIBRATION_STUDY.md](CALIBRATION_STUDY.md), not in this literature summary.

| Component | Primary precedent and relationship to this repository |
|---|---|
| Reference conditional MLP | Li, Dong and Qiu [1] provide the upstream `cGAN_EB` implementation for building profiles with PV/EV conditions. The reference architecture and BCE objective are adapted here to Fluvius profiles and reported HP/EV/PV, day-type and season labels. Quarter-hour outputs, signed net import, preprocessing, splits and evaluation differ. The upstream training data and weights are unavailable here, so this is not a reproduction of its published numerical results. See [UPSTREAM_ADAPTATION.md](UPSTREAM_ADAPTATION.md). |
| Fixed marginal calibration | Janke, Ghanmi and Steinke's implicit generative copulas (IGC) [2, Section 3] estimate a frozen generator's marginal CDFs from generated reference samples, interpolate them for new draws, and apply inverse data marginals. Our post-training, conditional per-slot map follows that principle. IGC trains in copula space with energy distance and a differentiable rank approximation; our adversarial/SWD training does not reproduce that algorithm. Finite references, compressed quantiles, ties and saturation make our marginal calibration approximate; empirical endpoints bound the output tails. |
| Sliced-Wasserstein auxiliary loss | Deshpande, Zhang and Schwing [3] study generative learning through projected one-dimensional transport. Our auxiliary term averages squared sorted projection differences, i.e. empirical squared sliced W2 without a square root, within conditions. Balanced condition sampling and its weight are experimental choices, not new transport theory. Lezama, Chen and Qiu [4] demonstrate why limited batch populations can miss distributional structure; more projection directions do not replace adequate samples per condition. We do not implement their Run-Sort-ReRun algorithm. |
| Spectral normalization and hinge loss | Miyato et al. [5, Section 4, equations 16-17] evaluate spectral normalization with hinge adversarial losses. Our implementation uses a conditional MLP. The study's combined configuration also changes discriminator width, dropout and Adam settings relative to the reference arm. Its results cannot isolate spectral normalization's contribution without further controls. |
| Statistical energy baselines | Xia et al. [6] compare GAN/VAE/WGAN variants with Gaussian mixtures and Gaussian-mixture copulas for transformer- and consumer-level profiles at several temporal resolutions. This motivates retaining statistical comparators. Our rank-Gaussian copula with empirical quantile margins is not their Gaussian-mixture copula. Independent-margin and whole-profile conditional resampling controls further distinguish marginal information from within-day dependence; they are not claimed as new methods. |
| Application evaluation | Claeys et al. [7] combine annual daily-consumption generation with intraday wavelet recombination and evaluate PV/battery applications. This supports assessing temporal behavior and use-specific utility alongside distributional similarity. Our independent daily profiles and within-day next-interval forecasting diagnostic do not establish annual, cross-household or battery-system realism. Signed net import alone does not identify separate demand and PV generation. |

A potential contribution is an audited, reproducible application benchmark and a
tested pipeline for conditional daily net-import profiles: trace-identity
curation, explicit training-data provenance, matched statistical controls,
component ablations, condition-level failures, joint diagnostics and downstream
utility. Whether those provide a useful scientific finding remains an empirical
question. Strong results from simpler controls, or a carefully explained failure
of the GAN, must remain reportable outcomes.

Calibration-versus-raw comparisons measure the complete mapping's effect; they
do not show that the GAN learned additional dependence during calibration.
Matched curated SWD0 and reference-cGAN comparisons are needed for their
respective attribution claims. Validation remains exploratory after development
reuse. Neither exact-trace curation nor nearest-neighbor diagnostics establishes
household independence or privacy. Publication readiness requires a frozen final
evaluation, adequate robustness evidence and a contribution justified against
related work; it is not implied by software completion or marginal improvements.

Primary references (paraphrased above; no quoted passages):

1. Yuewei Li, Bing Dong and Yueming Qiu (2025). *Conditional generative adversarial network (cGAN) for generating building load profiles with photovoltaics and electric vehicles*. **Energy and Buildings 335**, 115584. [Publisher DOI](https://doi.org/10.1016/j.enbuild.2025.115584); [author code](https://github.com/malvin0130/cGAN_EB), reference commit `885f78052a91fbc755e6b17a354c68849b5556ca`.
2. Tim Janke, Mohamed Ghanmi and Florian Steinke (2021). *Implicit Generative Copulas*. **NeurIPS 34**. [Proceedings paper](https://papers.nips.cc/paper/2021/file/dac4a67bdc4a800113b0f1ad67ed696f-Paper.pdf).
3. Ishan Deshpande, Ziyu Zhang and Alexander G. Schwing (2018). *Generative Modeling Using the Sliced Wasserstein Distance*. **CVPR**. [Proceedings](https://openaccess.thecvf.com/content_cvpr_2018/html/Deshpande_Generative_Modeling_Using_CVPR_2018_paper.html); [author preprint](https://arxiv.org/abs/1803.11188).
4. Jose Lezama, Wei Chen and Qiang Qiu (2021). *Run-Sort-ReRun: Escaping Batch Size Limitations in Sliced Wasserstein Generative Models*. **ICML, PMLR 139**, 6275-6285. [Proceedings](https://proceedings.mlr.press/v139/lezama21a.html).
5. Takeru Miyato, Toshiki Kataoka, Masanori Koyama and Yuichi Yoshida (2018). *Spectral Normalization for Generative Adversarial Networks*. **ICLR**. [Author paper](https://arxiv.org/pdf/1802.05957); [conference record](https://openreview.net/forum?id=B1QRgziT-).
6. Weijie Xia, Hanyue Huang, Edgar Mauricio Salazar Duque, Shengren Hou, Peter Palensky and Pedro P. Vergara (2024). *Comparative assessment of generative models for transformer- and consumer-level load profiles generation*. **Sustainable Energy, Grids and Networks 38**, 101338. [Publisher DOI](https://doi.org/10.1016/j.segan.2024.101338); [author-institution record](https://research.tudelft.nl/en/publications/comparative-assessment-of-generative-models-for-transformer-and-c/). The comparison's scope is verified from the authors' abstract; no numerical ranking from this paper is asserted here.
7. Robbert Claeys, Rémy Cleenwerck, Jos Knockaert and Jan Desmet (2024). *Capturing multiscale temporal dynamics in synthetic residential load profiles through Generative Adversarial Networks (GANs)*. **Applied Energy 360**, 122831. [Publisher DOI](https://doi.org/10.1016/j.apenergy.2024.122831); [author-institution record](https://researchportal.vub.be/en/publications/capturing-multiscale-temporal-dynamics-in-synthetic-residential-l/).
