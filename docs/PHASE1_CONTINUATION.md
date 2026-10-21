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
