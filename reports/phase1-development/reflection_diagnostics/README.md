# Reflection comparator: paired generation diagnostics

All 22 declared draws completed: four reflection arms, two matched 6k basic
flows, an isotropic source control, independent empirical marginals, and three
classical baselines, each with generation seeds 20261023 and 20261024. These
reuse the existing 16,384 inner-validation rows and previously used generation
seeds. They are paired development diagnostics, not unseen validation or
independent training replications. All reported means average two sampling
draws; the saved sampling SD is not a confidence interval.

Energy W1 is in kWh; peak and ramp W1 are in kW. SW1 is divided by the fixed
TRAIN range, 33.619001388549805 kWh; ACF errors and diversity are dimensionless.

| Method | SW1 | Energy W1 | Peak W1 | Ramp W1 | Lag-1 ACF W1 | Diversity |
| --- | --- | --- | --- | --- | --- | --- |
| Reflection lr1e3 s101 | 0.0070236 | 2.80443 | 0.909383 | 0.180363 | 0.100601 | 1.80197 |
| Reflection lr1e3 s202 | 0.00870968 | 4.29745 | 1.01186 | 0.192059 | 0.0843935 | 2.15629 |
| Reflection lr1e4 s101 | 0.0141707 | 3.98796 | 2.26685 | 0.529605 | 0.2309 | 3.42808 |
| Reflection lr1e4 s202 | 0.0141175 | 4.78436 | 2.26077 | 0.463355 | 0.169101 | 3.57444 |
| Basic flow 6k seed101 | 0.00585451 | 2.35736 | 0.473894 | 0.19369 | 0.0572086 | 1.42869 |
| Basic flow 6k seed202 | 0.00581636 | 2.40039 | 0.428037 | 0.198088 | 0.0506816 | 1.35003 |
| Isotropic source | 0.0217044 | 10.1172 | 1.66109 | 1.15827 | 0.525369 | 3.25512 |
| Independent marginals | 0.00847657 | 5.50539 | 1.45979 | 0.53626 | 0.421544 | 1.27193 |
| Conditional mean | 0.0208267 | 9.50766 | 1.43322 | 0.28281 | 0.334423 | 0 |
| Day resampling | 0.00527199 | 2.15286 | 0.34205 | 0.046413 | 0.0304821 | 1.25026 |
| Gaussian copula | 0.00550871 | 2.43102 | 0.531283 | 0.127889 | 0.05926 | 1.27913 |

The best reflection arm has SW1 0.007024, versus 0.005816–0.005855 for the
matched-budget basic flows, 0.005509 for Gaussian copula and 0.005272 for
day resampling. The two lower-rate reflection arms (about 0.0141) are also worse
than independent marginals (0.008477). The higher-rate reflection models improve
ramps, ACF and energy relative to independent marginals, showing some learned
dependence; this does not make the complete generator competitive.

Reflection's average ramp W1 at the higher rate (0.180–0.192) is slightly below
basic flow (0.194–0.198), but absolute-ramp quantile errors are 0.666–0.747,
versus 0.194–0.206 for basic flow and 0.126 for resampling. Peak quantile errors
are also much worse. The limited aggregate ramp advantage therefore does not
establish good ramp tails. Reflection lag-6 ACF is comparatively close to the
strong baselines at the higher rate; lag-1 ACF and covariance remain worse.

Basic flow still has excess PV0 days with any negative hourly import (84.1–88.0%). Reflection
reduces this to 0.0183–0.0732%, compared with 0.00610% for independent marginals,
Gaussian copula and resampling. Rare negative TRAIN values are retained by the
decoder; neither zero errors nor exact rare-event frequencies are established.
The selected real PV0 sample has no negative hours. Reflection's empirical
hourly support therefore improves this defect while leaving joint and marginal
frequency errors. Physical support alone is not marginal calibration.

Reflection diversity ratios are 1.80/2.16 (higher rate) and 3.43/3.57 (lower
rate), far above the empirical controls' 1.25–1.28. Those controls themselves
exceed one on this cohort; one is not an automatic failure threshold. Here
reflection's excess spread is accompanied by much worse peaks and covariance.

The [selection and solver report](../reflection_screen/README.md) records the
64-to-128-step sensitivity and probability-marginal defects. The present screen
supports targeted numerical/marginal diagnosis before broad longer training.
It does not establish that reflection methods generally fail, nor superiority,
privacy, appliance disaggregation or publication readiness. Zero exact TRAIN
row matches for the learned models are only a precision-specific copy check;
resampling has exact match fraction one by construction.

`manifest.json`, `summary.json`, `metrics.csv` and `condition_diagnostics.csv`
are byte-identical copies of completed evaluator outputs. `joint_summary.csv`
and `pv0_diagnostics.csv` are deterministic extracts. `DERIVATION.json` records
their source hashes and the shared derivation script. No dataset/model access
or new evaluation is needed to reproduce these reports.
