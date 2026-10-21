# Fixed conditional marginal calibration result

All **22 declared draws** completed on 21 October 2026. Calibration substantially
reduces excess PV=0 negative readings, but does not consistently improve joint
profile quality. The calibrated flows improve several metrics over the Gaussian
copula; complete-day resampling still has substantially better ramps and peaks.
This supports a further dependence-model experiment, not publication readiness.

The [fixed plan](../../../docs/MARGINAL_CALIBRATION_PLAN.md) and code were committed
before execution at `a272bb033a9f8e97dd1eaa40ed54ca43baa28251`.
Four previously selected checkpoints use 16,384 independent calibration profiles
per condition (1,048,576 per model), seed 20261025, and the exact empirical step
inverse used by the Gaussian copula. Target marginals fit only inner TRAIN.
No retraining, checkpoint reselection, physical clipping or evaluation-batch
ranking was performed. Every regenerated raw and baseline draw matched its
preserved profile-array hash and metric record exactly.

Evaluation reuses 16,384 inner-validation days and two previously used sampling
seeds, 20261023/20261024. These are paired development comparisons, not independent
validation or additional training replicates. No original VAL/TEST performance
was accessed. The original benchmark and unsuccessful experiments remain intact.

## Paired results

Values average the two generation draws within each training seed. All distance
metrics are smaller-is-better; diversity is the macro conditional generated/real
hourly variance ratio. It is not an error distance. Resampling's ratio is also
above one, so all variance mismatch cannot be attributed to the neural method.

| Method | Training seed | Profile SW1, normalized | Energy W1, kWh/day | Peak W1, kW | Ramp W1, kW | Diversity ratio | PV=0 export days, % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cGAN, raw | 101 | 0.006955 | 3.4008 | 0.5682 | 0.1068 | 0.640 | 46.7773 |
| cGAN, calibrated | 101 | 0.005656 | 2.3259 | 0.6075 | 0.1170 | 1.308 | 0.0061 |
| cGAN, raw | 202 | 0.006698 | 3.1755 | 0.5574 | 0.0947 | 0.689 | 36.1267 |
| cGAN, calibrated | 202 | 0.005536 | 2.4340 | 0.5288 | 0.0921 | 1.304 | 0 |
| Isotropic flow, raw | 101 | 0.005535 | 2.2609 | 0.4390 | 0.1379 | 1.235 | 76.3916 |
| Isotropic flow, calibrated | 101 | 0.005361 | 2.1823 | 0.4622 | 0.1208 | 1.293 | 0 |
| Isotropic flow, raw | 202 | 0.005348 | 2.1052 | 0.4352 | 0.1563 | 1.259 | 74.9207 |
| Isotropic flow, calibrated | 202 | 0.005399 | 2.2211 | 0.4460 | 0.1241 | 1.303 | 0.0061 |
| Conditional mean | — | 0.020827 | 9.5077 | 1.4332 | 0.2828 | 0 | 0 |
| Complete-day resampling | — | 0.005272 | 2.1529 | 0.3420 | 0.0464 | 1.250 | 0.0061 |
| Conditional Gaussian copula | — | 0.005509 | 2.4310 | 0.5313 | 0.1279 | 1.279 | 0.0061 |

Both cGAN seeds improve profile SW1 by roughly 17–19% and daily-energy error by
23–32%. However, seed 101's peak, ramp and lag-one ACF errors worsen. Its ramp
error remains below the Gaussian copula's, but substantially above resampling.
The second seed's covariance error also worsens. Calibration therefore does not
repair every dependence statistic simply by fixing marginals.

Flow seed 101 improves SW1 by about 3.1%; seed 202 worsens by about 1.0%.
Both improve ramps relative to raw flow, while peak and covariance errors worsen.
The calibrated flows have better mean SW1, energy, peak, ramp and lag-one ACF
scores than the Gaussian copula on these draws. This is descriptive evidence
that the learned dependence contains useful structure; it is not a significance
claim. Their ramp errors remain **2.60–2.67 times** resampling's, and peak errors
remain about 30–35% higher. Lag-one ACF errors are 0.0527/0.0548 versus 0.0305
for resampling. Both calibrated flows lose on ramp W1 to resampling in all
64 conditions in each draw. Temporal structure remains a material limitation.

## Calibration and support diagnostics

The fixed PV=0 diagnostic population contains 8,192 profiles per draw and no
negative real readings. Calibrated cGAN seed 101 and flow seed 202 each generate
one such day across their two draws; the other two calibrated models generate
none. Rare negative readings exist in TRAIN and remain possible. These small
counts cannot establish the correct rare-event probability.

Across all conditions/hours and the eight calibrated draws, about 0.0041–0.0081%
of generated values fall below their fitted source range and 0.0046–0.0084%
above it. These map to the observed target endpoints. The macro absolute
generated-versus-TRAIN zero-frequency difference is 0.00372–0.00394; the negative
frequency difference is 0.00709–0.00755. These are fractions, measured on 256
profiles per condition, with no sampling-noise correction or inferential claim.
Probability-decile L1 differences from uniform are 0.148–0.152; source atoms and
finite sampling mean that zero discrepancy is not a valid universal criterion.

Empirical quantiles restrict each hourly value to observed conditional TRAIN
support. They do not validate unseen tails or ensure realistic whole-day
combinations. The same support advantage belongs to the Gaussian comparator.
All four calibrated models have zero full-row float32-quantized TRAIN matches
on these draws; this establishes neither privacy nor novelty.

## Records and next experiment

[Metrics](metrics.csv), [per-condition diagnostics](condition_diagnostics.csv),
[summary](summary.json), [manifest](manifest.json),
[mapping diagnostics](calibration_diagnostics.csv) and
[fit timing/provenance](calibration_fits.csv) are extracted from the preserved
artifacts. The [extraction script](derive_summary.py) verifies all artifact
hashes; its [receipt](DERIVATION.json) records byte-identical copies and derived
tables. Large fitted arrays and full draw diagnostics remain under ignored
`artifacts/phase1-development/marginal_calibration`.

The run took about 201 seconds on the existing CPU environment. Software
verification recorded **364 passed tests and two CUDA hardware skips** before
execution; no GPU or paid cloud resource was provisioned.

The next declared comparator will learn conditional dependence directly in
TRAIN-transformed rank space using reflected flow trajectories. It must retain
the same empirical inverse and strong controls, disclose zero/tie handling,
and test numerical integration sensitivity. Additional seeds and robustness
checks are needed before a publication-quality claim. Phase 2 and federated
learning remain deferred.
