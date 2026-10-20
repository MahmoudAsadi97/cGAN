# V2 implementation plan

Goal: a reproducible cGAN adaptation with defensible held-out evaluation.
Spec: DESIGN.md. Python 3.10+, NumPy, pandas, SciPy, PyTorch, pytest.
User authorizes implementation and pushes exclusively to V2; main remains unchanged.

1. Data: implement pinned download, archive audit, streaming CSV preparation, explicit
   source wall-clock/DST rules, disjoint stratified meter split, train-only scaling.
   Test chunk boundaries, missing/duplicate slots, invalid metadata, DST, leakage.
2. Model/training: implement documented BCE MLP, seeded training, validation-selected
   checkpoints, exact generation replay. Test shapes, gradients, train-only scaler,
   held-out isolation, checkpoint round trip and unsupported labels.
3. Evaluation: implement physical-unit diagnostics and train-only baselines. Test
   identity distributions, known energy/peak differences, per-condition coverage,
   invalid arrays and nearest-neighbor duplication detection.
4. Integration: CLI download/audit/prepare/train/generate/evaluate, example configs,
   multi-seed runner, CPU CI, attribution and research protocol. Run full suite,
   real release preparation and bounded CPU training, record exact observed results.
5. Independent review; address important findings, commit and push V2. Verify main SHA.

Review focus: ambiguous time zone; cross-chunk meter/day boundaries; incomplete days;
conditions absent from training; unit conversions; validation/test contamination;
checkpoint provenance; falsely treating a short CPU run as paper evidence.
