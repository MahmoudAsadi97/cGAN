# Relationship to cGAN_EB

Reference commit: `885f78052a91fbc755e6b17a354c68849b5556ca` in
https://github.com/malvin0130/cGAN_EB.

Li, Y., Dong, B., and Qiu, Y. (2025). Conditional generative adversarial network
(cGAN) for generating building load profiles with photovoltaics and electric
vehicles. *Energy and Buildings*,335,115584.
https://doi.org/10.1016/j.enbuild.2025.115584

Preserved: fully connected generator64/128/64 and discriminator64/128/256;
LeakyReLU(.2); discriminator dropout(.4); linear generator output; latent24 Gaussian
noise; alternating non-saturating BCE GAN objective; D loss average real/fake;
Adam generator1e-4 and discriminator2e-4 with default betas. Sigmoid followed by
BCE becomes mathematically equivalent BCEWithLogitsLoss for numerical stability.

Explicit adaptations:

| Upstream implementation | V2 |
|---|---|
| Hourly24 output, 64 PV/EV charger/day/season labels | Quarter-hour96 or hourly24, 64 HP/EV/PV/day/season labels |
| PV/EV level1, level2, both groups | Only actual binary metadata; no fabricated charger levels |
| Seasons spring/summer/fall/winter; upstream label ordering | Winter/spring/summer/autumn and versioned V2 label formula |
| 0–100 row filter and global min/max over supplied table | No arbitrary cutoff; train-only min/max; preserve negative net grid import |
| Training CSV assumed available | Pinned real ZIP ingestion, audited timezone/DST policy |
| No held-out splits, checkpoint or seeds | Meter-disjoint split, saved provenance, seeded validation checkpoint selection |
| Fixed input order, cumulative printed loss | Shuffled epochs, epoch-specific losses |
| Gradients propagated through fake examples during D update | Detached fake examples, frozen D parameters for G update |
| KL computed on raw values; scalar feature Fréchet | Explicit Wasserstein/temporal metrics and baselines |

This is a reference-method adaptation. The source's training CSV and weights are
not supplied, so this repository does not reproduce its published numeric results.
The upstream code license is retained in UPSTREAM_LICENSE.txt. Data license is a
separate matter and is not granted by that code license.
