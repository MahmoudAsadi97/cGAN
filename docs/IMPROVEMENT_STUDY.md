# Prospective improvement study

This exploratory study follows the unsuccessful ten-epoch full-data pilot. It is
not a confirmatory publication result. The real test split remains sealed until
the method, budgets, metrics and seed list are frozen.

## Diagnosis and hypotheses

On 100 validation profiles per condition, conditional slot mean bias explains at
least 94.5% of the pilot's interval Wasserstein distance. Repeated generated class
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

Published foundations: Gulrajani et al., *Improved Training of Wasserstein GANs*,
NeurIPS 2017 (https://papers.nips.cc/paper/7159-improved-training-of-wasserstein-gans.pdf);
Miyato et al., *Spectral Normalization for Generative Adversarial Networks*, ICLR
2018 (https://arxiv.org/abs/1802.05957). These methods are established; adapting
them alone does not establish methodological novelty.
