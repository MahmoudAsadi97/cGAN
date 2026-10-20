# Prospective improvement study

**Audit update:** the original prepared split used by these initial screens has
exact cross-split full-year trace duplicates. Retain results as exploratory
diagnostics only. Follow [DUPLICATE_REMEDIATION.md](DUPLICATE_REMEDIATION.md) and
retrain on curated data before making independent-household performance claims.

This exploratory study follows the unsuccessful ten-epoch full-data pilot. It is
not a confirmatory publication result. Final test performance remains unseen until the method, budgets, metrics and
seed list are frozen. Curation reads test values only for deterministic identity
auditing and retained-row preparation.

## Diagnosis and hypotheses

On 100 validation profiles per condition, mean absolute conditional slot bias
provides a lower bound equal to 94.5% of the pilot's interval Wasserstein distance.
Repeated generated class
means account for 97.6% of squared ramp energy. Replacing these means with training
means in a diagnostic counterfactual improves interval W1 from 0.5375 to 0.1227
kWh, but does not repair deficient diversity. This is a diagnosis, not a fitted
or reportable model. Current global min-max scaling also compresses training
standard deviation to 0.03525; its effect on optimization must be tested.

## First controlled screen (declared before execution)

Compare `screen_minmax.json` and `screen_standard.json`: identical seed 42,
65,536 uniformly selected training profiles, 120 epochs, batch 512, reference
MLP/BCE architecture and learning rates. Only normalization differs. Standard
normalization uses the complete training split's scalar mean and population
standard deviation. The fitted transform is saved in the checkpoint. All
transforms remain invertible and outputs unbounded, preserving signed net energy.
Both arms perform 15,360 generator and discriminator steps. The subset is used
only for affordable method screening; final experiments must use the full data.

Checkpoint selection initially retains the original fixed 2,048-profile pooled
validation W1, to isolate normalization. Raw normalized scores cannot be compared
between transforms; report physical-unit metrics using identical validation
labels and seed 2026 (100 profiles per condition). Resampling and conditional-mean
baselines use each run's exact training subset. Every attempted arm is recorded,
including failures. These runs do not reuse the pilot's smaller training budget
as a supposedly fair comparator.

## Subsequent decisions

If the dominant mean bias persists, test train-only conditional residual
centering. If adversarial optimization remains unstable, test spectral
normalization with hinge loss, then gradient-penalty Wasserstein loss. Temporal
architecture is a separate ablation, not silently bundled with loss changes.
Any change in checkpoint selection must be versioned and applied equally to
its baseline. Monitor equal-condition metrics as well as pooled selection.

Exploratory screening targets, set before improved runs: condition-macro interval
W1 <= 0.10 kWh; daily-energy W1 <= 5 kWh; peak W1 <= 1 kW; ramp W1 <= 0.10 kW;
lag-one ACF W1 <= 0.10; diversity ratio 0.75--1.5. These are practical project
thresholds, not statistical significance or publication criteria. Review every
condition for collapse, support violations and copying. Require consistent
direction on at least three seeds, then the five declared seeds (11,22,33,44,55)
with full-data baselines and uncertainty before a promising-result claim.

## Second screen (declared before execution)

A read-only review found that adapting 24 outputs to 96 while retaining a
64-unit discriminator first layer introduces an at-least-32-dimensional linear
nullspace. At pilot epoch nine, 23.4% of squared conditional mean error lies in
that subspace. This is a demonstrable limitation, not proof of dominant causation:
most error remains visible and the nullspace changes during optimization.

After the two normalization arms, retain standard normalization and compare:
(a) the same BCE generator with discriminator width 128, and (b) width-128
discriminator with spectral normalization on all linear layers, no dropout,
hinge adversarial losses and Adam betas (0.5,0.999). Generator width stays 64.
Use the same 120 epochs, 65,536 profiles, batch 512 and seed 42. Arm (a) isolates
capacity; arm (b) tests a declared stabilization package rather than claiming to
isolate spectral normalization from loss, dropout or optimizer changes. The
reference BCE arms keep the original Adam defaults. Future attribution of the
stabilization package requires separate component ablations.

## Conditional mean residual screen (declared before execution)

The completed normalization comparison reduced macro interval W1 from 0.37745
to 0.10940 kWh and ramp W1 from 1.54375 to 0.15496 kW. Diversity remained low
(0.37607), daily-energy W1 was 7.92715 kWh and lag-one ACF W1 was 0.18011.
Standardization helped, but did not meet the screening gates.

Test a generator skip connection containing the normalized training conditional
mean profile. Fit the means only from the actual selected training rows; preserve
the learned generator's original initialization, latent inputs and unbounded
residual outputs. This is a learned residual model, distinct from the diagnostic
post-hoc mean replacement. Both adversaries still see reconstructed normalized
profiles and use the same pooled checkpoint score. For an unconditional ablation,
use a single global mean so no label information enters the model.

Run one residual arm paired with standard BCE/width64, and one paired with the
standard spectral-hinge/width128 package. Keep the same seed42, 65,536 profiles,
120 epochs and batch512. These comparisons isolate residual centering within
each objective. All arms, including ineffective changes, remain in the ledger.

## Conditional distribution screen (declared before execution)

The non-residual spectral-hinge screen's selected model had low pooled normalized
W1 (0.16321), but macro physical W1 0.13468 kWh, daily-energy W1 8.39869 kWh,
ramp W1 0.28793 kW and macro diversity 2.18827. Thus pooled fidelity concealed
important conditional errors. This arm does not pass the gates.

For the next comparison, keep the standardized spectral-hinge model with D width
128 and G width64, without the residual mean offset. Change all three arms to
the same condition-balanced sampler: four distinct uniformly chosen conditions
per update, with 128 actual training rows drawn with replacement per condition.
Each epoch is now 128 updates (65,536 draws), not a pass over all unique rows.
Keep 120 epochs and seed42. Record actual exposures and update counts.

Compare auxiliary conditional sliced-Wasserstein weights 0, 1 and 10. The auxiliary
loss is the condition-macro mean squared difference of sorted whole-profile
projections, using 128 fresh unit directions per update and a separate seeded RNG.
It uses TRAIN only and augments, rather than replaces, the adversarial objective.
Weight zero is the identical-sampler control. At 128 observations per condition,
finite-minibatch transport bias still exists; more projection directions do not
remove it. This is an established-method ablation, not a novelty claim.

Prospectively change checkpoint selection for all three arms to condition-macro
interval W1 on a fixed balanced validation sample of 64 rows per condition
(4,096 total). The 100-per-condition external evaluation remains unchanged.
Do not compare these selected checkpoints with earlier pooled-selected runs as
if selection were controlled. Any final reference comparison must rerun with
the same declared selection rule and budget.

Foundations: Deshpande et al., CVPR2018,
https://openaccess.thecvf.com/content_cvpr_2018/html/Deshpande_Generative_Modeling_Using_CVPR_2018_paper.html;
Lezama et al., ICML2021, https://proceedings.mlr.press/v139/lezama21a.html;
Fatras et al., AISTATS2020, https://proceedings.mlr.press/v108/fatras20a.html;
Jang and Noh, AISTATS2026, https://proceedings.mlr.press/v300/jang26b.html.

Published foundations: Gulrajani et al., *Improved Training of Wasserstein GANs*,
NeurIPS 2017 (https://papers.nips.cc/paper/7159-improved-training-of-wasserstein-gans.pdf);
Miyato et al., *Spectral Normalization for Generative Adversarial Networks*, ICLR
2018 (https://arxiv.org/abs/1802.05957). These methods are established; adapting
them alone does not establish methodological novelty.
