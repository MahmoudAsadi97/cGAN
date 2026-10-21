# Reference training audit: shared hourly drift

The audit found no material implementation mismatch explaining the reference
cGAN's poor quality. Across five historical conditional checkpoints, **83–93%
of squared TRAIN condition-mean error is explained by a shared hourly error
vector**. That vector moves sharply between checkpoints. Unstable adversarial
optimization and unfavorable input scaling are plausible explanations; neither
a particular optimizer defect nor an effective remedy is established.

The complete upstream `train.py`, `cGAN.py` and generation path were reviewed at
commit `885f78052a91fbc755e6b17a354c68849b5556ca`, against V3 commit
`72ade44dfa2f8215b11d218d16086f65461c6f94`. All ten historical run histories and
manifests were inspected. New numerical diagnostics used original TRAIN and
saved conditional checkpoints only. No TEST profiles were opened, no new TEST
metrics were calculated, and no real-data training was run for this audit.

## Implementation comparison

The network widths, linear G output, LeakyReLU slope 0.2, D dropout 0.4,
24-dimensional standard-normal latent input, one-hot labels, Adam learning
rates 1e-4/2e-4 and betas (0.9, 0.999), update ordering and adversarial objectives
match upstream. Upstream sigmoid plus BCE corresponds to local logits plus
BCEWithLogits. Local detachment during D updates and frozen D parameters during
G updates discard gradients that upstream calculates but never applies. D
dropout remains active during G updates; there is no BatchNorm state.

A five-batch synthetic replay using identical initial weights, minibatches,
noise and dropout draws produced maximum loss difference **5.96e-8**, G weight
difference **2.98e-8** and D weight difference **3.73e-8**, with identical final
dropout RNG state. This verifies the compared update path at ordinary logit
magnitudes; it does not compare behavior under extreme sigmoid saturation.

Material declared differences remain. Local training shuffles every epoch;
upstream slices a fixed, unavailable input ordering. Local training separates
RNG streams, uses validation selection and runs 100 rather than 10,000 epochs.
Upstream filters all input coordinates to [0, 100] before global min/max
normalization; this adaptation retains signed net load and fits normalization
on TRAIN only. Upstream's printed losses are cumulative over training, whereas
local losses are per-epoch averages. The unavailable upstream training data
prevent numerical reproduction of the paper.

## TRAIN-only checkpoint evidence

For each condition, compare its full TRAIN mean with the generated mean of 256
draws using fixed Torch normal stream seed 1781. Write that hourly error vector
as e_c and its equal-condition mean as b. The common-error fraction below is
64 × ||b||² / sum_c ||e_c||². RMSE pools squared errors across condition/hour
pairs before taking a square root. These are exploratory diagnostics.

| Training seed | Selected epoch | Mean-profile RMSE, kWh | Common-error fraction |
| --- | ---: | ---: | ---: |
| 11 | 100 | 1.2624 | 85.03% |
| 22 | 90 | 1.2203 | 88.75% |
| 33 | 65 | 1.4004 | 88.68% |
| 44 | 100 | 1.3729 | 93.47% |
| 55 | 95 | 1.0592 | 83.42% |

Using identical latent draws at each checkpoint, the shared error changes
substantially between the selected and final model:

| Seed | Epochs compared | Error-vector cosine | Error-vector change RMSE, kWh |
| --- | --- | ---: | ---: |
| 22 | 90 → 100 | 0.1815 | 1.6610 |
| 33 | 65 → 100 | -0.1349 | 2.7275 |
| 55 | 95 → 100 | -0.1260 | 1.4692 |

Seeds 11 and 44 select the final epoch and therefore have zero change. The
nearly orthogonal or reversing vectors in the other seeds weaken an explanation
based solely on fixed output initialization or an immutable hour permutation.
They do not by themselves identify the source of the instability. Averaged D
losses near log(2) do not establish equilibrium or rule out batch-level failure.

## Falsifiable development priorities

1. **Scaling:** the historical global range is about 33.619 kWh, and physical
   zero maps to 0.3203. Compare min/max with one global mean/SD transform fitted
   only to innerTRAIN, retaining networks, objectives, optimizer and sampling.
   Keep per-hour standardization separate because it also changes relative
   coordinate weighting. Inverse all outputs to physical units before scoring,
   then use one shared TRAIN range for normalized SW1.
2. **Cycling:** hold the chosen representation fixed and separately test lower
   Adam beta1 or reduced learning rates at equal update exposure. Betas 0.9
   and dropout 0.4 are hypotheses, not established causes. Trace shared hourly
   errors and fixed validation scores; an isolated favorable checkpoint is
   insufficient evidence of stable optimization.
3. **Duration:** compare declared longer budgets across predetermined seeds;
   four of five historical conditional models select epoch 90 or later.
   Continued improvement is possible, but longer identical training may also
   prolong cycling. The upstream 10,000 epochs are not evidence of sufficiency.

Smaller batches, condition-balanced sampling and reduced dropout are secondary
probes. Weight averaging may reduce moving error but need not average nonlinear
network outputs; evaluate it explicitly. Additional moment/temporal losses
change the objective and must be identified as method changes. Preserve the
frozen 100-epoch study and identify subsequent work as development after
historical TEST observation. Select changes using TRAIN/innerVAL only.

## Subsequent normalization screen

Subsequent development follows the [declared normalization screen](NORMALIZATION_SCREEN_PLAN.json)
and [second-seed replication plan](NORMALIZATION_REPLICATION_PLAN.json), with
the architecture and optimizer fixed while the representation varies. Their
results and matched baselines belong to their own development reports. This
audit preserves the earlier diagnosis and its original numerical output; it
does not promote later screen findings to confirmation of a specific mechanism.

## Reproduction and preserved provenance

[REFERENCE_AUDIT_NUMERICS.json](REFERENCE_AUDIT_NUMERICS.json) is a byte-for-byte
copy of the original captured audit output, with SHA-256
`8b5587dfe11cee728bd102e87176e7ca1fc0841e5d85d53c186ec1ae44a78f2a`.
It records all ten consumed best/last checkpoint hashes, the historical TRAIN
hash, upstream source hash, runtime versions and complete shared hourly error
vectors. Its paths identify the original local artifacts; it contains aggregate
diagnostics and file digests, with no meter identities or source profiles.

That output was generated by the original local `audit_numerics.py`, SHA-256
`f11694d6e81f41d4b6fdd28275f0b9ea4515f48745f30a16fb03f836919895d4`.
The tracked [reproduction script](../../scripts/audit_reference_training.py)
preserves its calculations while replacing machine-specific paths and import
side effects with explicit CLI arguments. It is a later wrapper, so its source
digest differs from the script used for the preserved output. It verifies the
pinned upstream source digest before importing it and does not discover TEST or
VAL files. From an installed repository environment, run:

```sh
python scripts/audit_reference_training.py \
  --data /path/to/historical/hourly24/train.npz \
  --runs /path/to/confirmatory/conditional \
  --reference /path/to/cGAN_EB/cGAN.py > /path/to/new-audit.json
```

Large checkpoints, prepared TRAIN and the upstream checkout are local inputs;
the script does not fetch them. Use the input hashes in the preserved JSON to
identify the original artifacts. Different path strings can change JSON bytes
without changing calculated values. The tracked wrapper was syntax-checked and
its CLI checked without rerunning the real-data diagnostic.
