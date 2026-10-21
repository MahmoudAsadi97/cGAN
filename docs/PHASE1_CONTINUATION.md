# Phase 1 development after the initial benchmark

On 21 October 2026, the project scope was extended to continue Phase 1
development without requesting a review at the end of each unsuccessful study.
Training parameters and techniques may be changed, retained or removed according
to evidence. Federated learning remains deferred. All work stays on V3.

The original frozen protocol, ten runs, thirteen final evaluations and negative
results remain historical evidence. They are not replaced, retroactively tuned,
or relabeled as a successful replication. The original architecture benchmark
and any adapted training method must be distinguishable in a paper.
Supporting-file hashes in the historical benchmark receipt refer to commit
`72ade44dfa2f8215b11d218d16086f65461c6f94`; the live README now reflects this
continued scope. The historical receipt is not rewritten when documentation evolves.

## Diagnosis and first controlled comparison

The source audit found no material mismatch in adversarial optimizer updates.
Real hourly variation occupies a narrow part of the global min–max range.
TRAIN-only checkpoint diagnostics found that a common hourly error vector
accounts for approximately 83–93% of the squared conditional mean error in the
five cGAN models. This vector changes sharply between checkpoints. These facts
support unstable optimization and input scaling as hypotheses; they do not yet
establish a cause.

The first development comparison changes only the affine representation:

1. Global min–max, as the existing implementation's control.
2. Global mean and population standard deviation.
3. Per-hour mean and population standard deviation, kept as a separate arm
   because it also changes the relative weighting of coordinates.

All transforms fit the current inner training subset only and are inverted
without clipping. Networks, dropout, Adam learning rates/betas, latent size and
one-hot conditions remain fixed. Every arm is scored in physical kWh and then
divided by the same inner-training range for the primary endpoint. Representation
space losses are not compared as physical fidelity metrics.

## Development data and budgets

Only the original TRAIN NPZ supplies the new development split. Within each
reported equipment stratum, seed 20261021 assigns a rounded 20% of meter IDs to
inner validation. This produces 1,333 fitting meters / 485,212 daily profiles
and 332 validation meters / 120,848 profiles, with all 64 conditions in both.
Fresh split hashes, exact assignments and training-only extrema are recorded.

The initial screen uses seed 101, 6,000 discriminator/generator update pairs,
batch size 512, and one CPU thread per run. It shuffles every inner-training
pass and includes the last partial batch. Actual update counts and row
presentations are saved; changing batch size later must not be called equal
compute merely because epoch counts match. This small screen diagnoses an
effect and is not a final estimate across training seeds.

Selection uses fixed samples of up to 256 inner-validation days per condition,
index seed 20261021, generation seed 20261022, and the original 128 directions
with seed 2026. Checkpoints are scored at step 1, each 1,000 steps and the final
step. The earliest minimum selects the checkpoint. Conditional mean, training-day
resampling and copula controls use the same real selection rows. This is a
different development sample from the earlier final-test benchmark.

Promising changes must be replicated across training seeds, evaluated at a
longer common budget, and checked with controlled ablations. Rejected and failed
attempts remain in the experiment ledger. Broad searches should follow evidence
from these initial comparisons, rather than accumulating untested techniques.

Both standardized transforms improved over min–max at seeds 101 and 202.
Global standardization is the simpler transform, so the next small screen
isolates Adam momentum: beta1 values 0 and 0.5 against the existing 0.9 controls
at both seeds, with all other settings and the 6,000-update budget fixed.
The [momentum plan](../reports/phase1-development/MOMENTUM_SCREEN_PLAN.json)
was written before these runs. Longer-budget comparisons follow this screen;
the short runs do not establish convergence or superiority to strong controls.

The development trainer supports continuation of a completed run at a scheduled
validation boundary into a new directory. It verifies artifacts, data and
runtime and restores optimizer/sampler/random states. A CPU regression confirms
the same trajectory as uninterrupted training. CUDA support has a separate
[notebook pilot](GPU_NOTEBOOK.md); it has not yet been executed on GPU hardware.
Arbitrary interrupted-run recovery is not yet supported.

## Generative-method comparator

The literature review motivates a separate conditional flow baseline before
combining newer mechanisms. The [declared flow screen](../reports/phase1-development/FLOW_SCREEN_PLAN.json)
uses seeds 101/202, 6,000 updates, global training-only standardization, and the
same inner-validation rows and physical metrics as the cGAN screens. Independent
Gaussian/data pairs define a linear path with velocity target `data - noise`.
The velocity MLP has 23,896 parameters, scalar time input, widths 64/128/64 and
SiLU activations; Adam uses learning rate 0.001 and betas 0.9/0.999.

Heun integration uses 32 steps / 64 field evaluations for checkpoint selection.
The selected checkpoint is also scored at 64 steps with identical initial noise
to check numerical sensitivity, without reselection. Model, optimizer and
sampling cost differ from the cGAN: equal update counts do not mean equal compute.
All runs retain parameter counts, row presentations and elapsed time. This is a
Phase 1 comparator for whole-house generation, with no appliance-level targets.
Any structured-prior, bridge or few-step extension needs its own controlled
comparison after this basic baseline is assessed.

The six 30,000-update momentum trajectories completed. Beta1=0 gave the lowest
primary score at both seeds (approximately 0.00698 and 0.00687), but variance,
peak and ramp discrepancies remain too large relative to the descriptive
real-cohort comparisons. The next
[dropout screen](../reports/phase1-development/DROPOUT_SCREEN_PLAN.json)
changes only the existing discriminator dropout probability to 0 or 0.2, at
seeds 101/202 and 6,000 updates, against the saved beta1=0/dropout=0.4 controls.
It tests whether heavy discriminator regularization impairs distribution fitting.
The topology and adversarial objective remain fixed. The short screen must be
followed by a common longer budget if promising; it does not replace the
five-seed validation requirement or establish publication readiness.

The [completed flow/dropout screen](../reports/phase1-development/flow_dropout_screen/README.md)
shows why temporal diagnostics matter: basic flow reaches profile SW1 near
0.0057 but ramp W1 near 0.19 kW, compared with about 0.046 for day resampling.
Doubling flow integration resolution changes profiles by only about 0.002 kWh
RMS. Reduced dropout has no consistent short-budget effect across the two seeds.
Three further comparisons are declared separately:

- [Longer dropout study](../reports/phase1-development/DROPOUT_LONG_PLAN.json):
  continue both reduced-dropout settings at both seeds to 30,000 total updates,
  matching the existing dropout-0.4 controls. These are continuations, not new seeds.
- [Temporal source ablation](../reports/phase1-development/CORRELATED_FLOW_PLAN.json):
  replace independent source noise with a common TRAIN-fitted Pearson correlation,
  retaining zero means and unit marginal variances. Fixed 0.01 identity shrinkage
  gives a positive-definite source. Network, optimizer, base random draws, budget
  and solver stay fixed. Include the source alone to distinguish prior quality
  from learned transport. This is TSFlow-inspired source design, not an exact
  TSFlow reproduction or a fitted Gaussian-process kernel.
- [cGAN auxiliary-loss study](../reports/phase1-development/ENERGY_SCREEN_PLAN.json):
  add one declared weight, lambda=1, for conditional Euclidean energy distance
  on the same generator batch, in globally standardized coordinates. Average
  eligible conditions equally, exclude within-sample diagonals, and retain finite
  negative U-statistic estimates. This is an adapted training objective with the
  reference network topology, not the original method or the Energy Matching
  paper. The default zero weight preserves the existing training trajectory.

The auxiliary loss measures distribution discrepancy; its name does not refer
to electrical energy. Its U-statistic form is unbiased for independent conditional
samples, but minibatches of repeated meters do not supply independent household
observations. It is an optimization term, not a calibrated significance test.
The recorded synthetic CPU benchmark estimates roughly twice the cost per update
when enabled; real training records retain the actual elapsed time.

Fresh generation diagnostics use seeds 20261023/20261024, without checkpoint
reselection, on the same inner-validation rows. They add signed export behavior,
quantile errors, covariance and float32-quantized full-row copying to the physical
metrics. Fresh random draws assess sampling sensitivity; they do not create new
validation meters. The evaluator verifies checkpoint, source-prior and dataset
provenance and preserves failed attempts.

The correlated source worsened profile and ramp errors in both short-screen
seeds. Its ramp error was still falling with training, so the
[common 30,000-update flow comparison](../reports/phase1-development/FLOW_LONG_PLAN.json)
retains both sources and both seeds. Flow currently has no continuation interface;
new runs start from initialization and their first 6,000-update histories are
checked against the saved screen. Reusing an identical trajectory prefix does
not create an additional independent training seed.

## Scientific acceptance and independent validation

A useful foundation needs competitive distribution fidelity against strong
controls, meaningful diversity, conditional equipment/calendar behavior,
reasonable ramps and temporal dependence, stable seed outcomes, and learning
curves that support the chosen budget. A reduction in one mean-profile or SW1
score alone does not meet these requirements. Exact copying is assessed alongside
fidelity; proximity diagnostics do not establish privacy.

Further parameter or technique selection uses inner development data. Original
VAL may provide a later locked development audit with its prior-use disclosure.
The earlier TEST performance has already been observed. No resplit of these
meters can manufacture a new untouched confirmatory cohort. A genuinely new
year or independent meters would strengthen eventual confirmation; if unavailable,
claims must remain explicit about internal validation and post-test development.
The user confirmed that only the current release is available. The present
study therefore cannot claim validation on newly acquired independent data.

The previous empirical-W1 percentile intervals were displaced from their point
estimates. Coverage is not established by numerical correctness. Future interval
or hypothesis-test claims require a justified method and appropriate validation;
the old interval estimates are preserved without post-hoc recentering.

Publication readiness is assessed from this body of evidence and the strength
of the scientific contribution. It is not guaranteed by a favorable seed,
software tests, or a chosen number of training epochs.
