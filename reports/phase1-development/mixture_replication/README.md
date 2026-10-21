# TRAIN-only mixture replication

Six fixed K8 fits use all 485,212 inner-TRAIN profiles, 64 conditions and two initializations per condition. The cap is 300 iterations for every fit, with tolerance .001 and covariance diagonal regularization .001. No validation/test profiles, generated draws or held-out scores were loaded for this report.

| Fit seed | Role | Converged starts | Converged selected | Iterations min/median/max | Minimum effective rows (condition/component) | Components <25 rows | Fit seconds |
|---|---|---:|---:|---|---:|---:|---:|
| 20261027 | numerical followup | 128/128 | 64/64 | 30/54/117 | 2.000 (38/3) | 2 | 97.56 |
| 20261101 | fresh replication | 128/128 | 64/64 | 29/56/132 | 21.000 (39/3) | 1 | 97.83 |
| 20261102 | fresh replication | 128/128 | 64/64 | 24/54/110 | 19.000 (54/0) | 1 | 94.27 |
| 20261103 | fresh replication | 128/128 | 64/64 | 31/54/146 | 10.000 (54/3) | 4 | 97.34 |
| 20261104 | fresh replication | 128/128 | 64/64 | 26/56/132 | 5.000 (13/0) | 2 | 101.21 |
| 20261105 | fresh replication | 128/128 | 64/64 | 26/56.5/168 | 3.000 (63/0) | 2 | 99.48 |

The original-seed numerical follow-up changes selected parameter arrays in conditions **23, 36, 37**; 61/64 conditions retain byte-identical weights, means and covariances. No selected initialization changed. The five previously capped starts now stop as follows: condition 23/initialization 1: 100 to 117; condition 27/initialization 1: 100 to 109; condition 36/initialization 1: 100 to 108; condition 37/initialization 1: 100 to 105; condition 44/initialization 0: 100 to 109. All 123 previously converged starts retained identical iterations, final scores and lower bounds. TRAIN arrays, empirical targets, rank permutations, normal scores and estimator source hashes are unchanged.

Across the five fresh fits, 640/640 starts converge with 0 recorded warnings. There are 10 components with fewer than 25 effective rows and 10 components whose minimum covariance eigenvalue is at the .001 regularizer (within the documented numerical tolerance). The largest fresh-fit covariance eigenvalue is 49.802485, at seed 20261102, condition 51, component 7. Covariances are in latent normal-score units. Floor detection uses `isclose(eigenvalue, .001, rtol=1e-8, atol=1e-12)`.

Seed 20261027 is a numerical follow-up and is excluded from the five-fresh-fit aggregate. Fresh seeds change tie permutations and EM initialization; these fits reuse the same TRAIN observations and are not independent household samples. Convergence satisfies the declared numerical stopping rule; it does not establish a global optimum or better held-out performance. Small components and covariance regularization remain visible in the tables.

The estimator fits an ordinary Gaussian mixture to normal scores and selects by final TRAIN log density. It does not optimize copula likelihood. The reused validation screen informed the fixed K8 choice; these numerical results remain postselection development and establish no held-out quality improvement. Fit times are descriptive recorded wall times, not a controlled speed benchmark.

[Fit summaries](fit_summary.csv), [all 768 initialization records](initializations.csv), [component diagnostics](components.csv), and [condition diagnostics](conditions.csv) preserve every fit. The `initializations/` directory also provides one 128-row table per fit. [Numerical initialization comparison](numerical_followup_initializations.csv) and [selected-parameter comparison](numerical_followup_conditions.csv) retain the original 100-iteration evidence. Direct array differences use stored component indices without component alignment.

All six models remain in `runs/phase1-development/mixture_replication`; model hashes are recorded in the summary and derivation receipt. Root and child inventories were verified. For the original screen, only its manifest, K8 fit snapshot and model were inspected; its earlier fit snapshot has a running-stage status, while its enclosing manifest is completed. No original validation metric report was parsed.

Rebuild from the repository root with:

```bash
.venv/bin/python reports/phase1-development/mixture_replication/derive_summary.py
```

[DERIVATION.json](DERIVATION.json) binds input/output hashes and the derivation script. [summary.json](summary.json) contains full-precision summaries. The script performs no fitting, sampling or dataset loading.
