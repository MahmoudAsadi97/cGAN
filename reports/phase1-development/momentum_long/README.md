# Common 30,000-update momentum study

All six declared continuations completed: Adam beta1=0, 0.5 and 0.9 at training seeds 101 and 202. These continue the corresponding 6,000-update trajectories; they are not additional independent seeds. The selected beta1=0 scores were 0.006982 and 0.006873, still behind the matched resampling and copula controls. Generated variance, peak and ramp distributions remain inadequate relative to the descriptive real-cohort reference.

The two CSV tables and saved-artifact receipt are copied from `artifacts/phase1-development/momentum_long_summary`. That ignored directory retains complete condition metrics, baseline draws, summary JSON and plots. Regenerate with `scripts/summarize_development.py` using all six run directories in `runs/phase1-development/momentum_long`, and `artifacts/phase1-development/baselines_screen` as comparator.

These are validation-selected development scores, not independent final performance estimates or evidence of publication readiness. All parent checkpoints, full histories and failed launch records are preserved.
