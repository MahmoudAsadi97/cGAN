# Short development screens

These saved records cover all ten completed 6,000-update runs: three scaling
choices at seeds 101/202, followed by two lower-momentum choices with global
standardization at the same seeds. No run is omitted. Checkpoint scores selected
on inner validation are development evidence.

`manifest.json` records the complete saved artifact set in
`artifacts/phase1-development/screens_summary`; only the two compact CSV tables
and receipt are copied here. Full condition metrics, baseline draws, summary
JSON and plots remain in that ignored artifact directory. Recreate them with
`scripts/summarize_development.py` using every run in both normalization plans
and the momentum screen plan, with
`artifacts/phase1-development/baselines_screen` as the comparator directory.

Global-standard scaling scores are 0.012641/0.011866 at 6,000 updates. Matched
resampling and copula selection-draw scores are 0.005269 and 0.005414. These short
runs do not establish adequate fidelity or diversity. The common 30,000-update
continuation is declared separately.
