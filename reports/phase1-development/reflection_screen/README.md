# Conditional reflection screen: all four declared arms

The direct conditional reflection adapter did not improve the existing strong
baselines. All four 6,000-update arms and selected checkpoints are retained.
The larger learning rate produced better physical distributions than the smaller rate,
but its two seeds differ substantially. This is a small conditional adaptation
of the inspected construction, not a reproduction or refutation of the author
model. See [design](../../../docs/DEPENDENCE_COMPARISON_PLAN.md).

These are previously used inner-TRAIN/inner-VAL development data. Selection uses
generation seed 20261022; paired sampling diagnostics use 20261023/20261024 on
the same 16,384 real rows. No new meters, independent confirmation or original
TEST performance are involved. The 128-step solver was not used for selection.

| Run | Selected update | Selection SW1 | Paired-draw SW1 | 128-step selection-draw SW1 |
| --- | --- | --- | --- | --- |
| reflection_lr1e3_s101 | 4000 | 0.00697964 | 0.0070236 | 0.00759685 |
| reflection_lr1e3_s202 | 6000 | 0.00885441 | 0.00870968 | 0.00992326 |
| reflection_lr1e4_s101 | 5000 | 0.0136765 | 0.0141707 | 0.0158535 |
| reflection_lr1e4_s202 | 6000 | 0.0138471 | 0.0141175 | 0.0160441 |

Doubling reflected Euler from 64 to 128 steps worsens SW1 in every arm by
8.8–15.9%, increases diversity and worsens peaks/ramps. Common-noise probability
RMSE changes are only 0.00048–0.00196, while decoded physical RMSE is
0.217–0.339 kWh and maximum coordinate changes are 11.19–13.81 kWh. The physical
sampler cannot be treated as numerically converged. A finer solver is not a
validated improvement in distribution fidelity here.

At the selected 64-step higher-rate checkpoints, pooled upper probability
deciles contain only 5.50%, against 10% under uniform marginals; middle bins
contain about 12–13%. The lower-rate arms have less conspicuous decile errors
but much worse physical peaks and spread. Coarse deciles do not diagnose tiny
tail intervals. Exact zero/one probabilities are mostly absent, yet up to
3.2–6.9% of coordinate proposals require reflection at a solver step; similar
fractions persist at 128 steps. These counts concern proposals, not unique
profiles or wall-crossing multiplicity. They suggest checking near-wall mass
and decoder tail amplification, without identifying the cause by themselves.

Loss falls toward 0.95 (higher rate) or 0.985 (lower rate); this MSE includes
irreducible velocity variation and is not a fidelity score. Selected steps are
4k/6k at the higher rate and 5k/6k at the lower rate. The learning curves and
solver diagnostics do not justify extending only the favorable seed.

The next defensible diagnostic is near-wall probability/decoded-endpoint
behavior at fixed checkpoints. Any calibration combination or zero-normal-
velocity boundary constraint must be a separately declared comparison, retain
these direct outputs, and include both seeds. It cannot retroactively establish
solver convergence. No result here establishes publication readiness.

`summary.json`, `selection_and_sampling.csv`, `solver_diagnostics.csv` and
`training_curves.csv` are deterministic extracts. `plan.json` and
`execution.json` are byte-identical copies. `DERIVATION.json` binds all inputs
and generated files. Run `python reports/phase1-development/reflection_screen/derive_reports.py`
from the repository to reproduce both reflection report directories without
loading datasets, checkpoints or generating samples.
