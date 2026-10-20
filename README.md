# Conditional GANs for residential net-load profiles with PV, heat pumps and EVs

This repository adapts the conditional GAN of Li, Dong & Qiu (2025),
*Conditional generative adversarial network (cGAN) for generating building load
profiles with photovoltaics and electric vehicles*, Energy and Buildings 335,
115584 ([code](https://github.com/malvin0130/cGAN_EB)), to the Fluvius 2024
smart-meter sample of 2,400 Flemish households, and evaluates it with a
held-out-household protocol.

It contains

* a **faithful reproduction** of the cGAN_EB model (`configs/cgan_eb.yaml`),
* an **improved conditional WGAN-GP** with factorised condition embeddings and a
  projection critic (`configs/cwgan_gp.yaml`) plus single-factor ablations,
* **non-adversarial baselines** (per-label Gaussian, Gaussian copula, GMM,
  resampling of training days) and a **real-data reference**,
* an **evaluation suite** (distributional, temporal, load-shape KPI,
  conditional-fidelity, utility and memorisation metrics) that fixes the
  defects of the reference evaluation code (see [Evaluation](#evaluation)).

## Dataset

Fluvius, *Verbruiksprofielen digitale elektriciteitsmeters: kwartierwaarden
voor een volledig jaar* (2024 sample), provided as `raw.zip` in the
[`data` release](https://github.com/MahmoudAsadi97/cGAN/releases/tag/data) of
this repository.

| | |
|---|---|
| Households | 2,400 residential digital meters (contract category *residential* all year) |
| Resolution | 15 min, 1 Jan - 31 Dec 2024 (35,136 readings per meter, no gaps) |
| Variables | grid import (`Volume_Afname_KWh`) and export (`Volume_Injectie_KWh`) per quarter hour |
| Groups | 300 meters for each combination of PV (*ZP*), heat pump as main heating (*WP*) and home-charged EV |

Group naming used throughout: `none`, `PV`, `HP`, `HP+PV`, `EV`, `EV+PV`,
`HP+EV`, `HP+EV+PV` (group index = 4·EV + 2·HP + PV).  PV meters are those that
injected in all 12 months; non-PV meters never injected (a few heat-pump meters
show injections up to 0.18 kWh per quarter hour).  The flags are registry
indicators of the distribution system operator: the EV flag comes from
customer declarations or Fluvius' detection algorithm, a heat pump is flagged
only when declared as main heating, and meters with known storage were removed,
but undeclared batteries cannot be excluded (a share of PV homes shows a
battery-like signature).  The measured load therefore does not always reflect
the flags, which bounds how well any model — or a classifier on real data — can
recover them; factor accuracies are always reported next to the real-data
reference.

Seasons and clock time: spring and fall each mix CET and CEST days, so within
these labels real PV timing is a mixture of two clock offsets one hour apart;
winter combines January-February and December of 2024.

### Preprocessing (`cgan/data`)

* **Timestamps** are local wall-clock time (CET/CEST) despite the `Z` suffix:
  31 March has 92 quarter hours and 27 October 100.  Readings are indexed by
  (local date, quarter of day); the two DST days are excluded (`drop_dst_days`).
* **No-flow days**: meter-days with zero import and zero export in every
  quarter hour (vacancy, disconnection or zero-filled data; 881 of 873,600,
  almost all from a few training households) are excluded in all splits
  (`drop_zero_days`).
* **Target**: daily profiles of hourly **net load** = import − export (kWh/h),
  i.e. what the distribution grid sees; negative values are PV export.
  `target: import | export | import_export` and `resolution: quarter` (96
  values per day) are also supported.
* **Conditions** (64 labels, same layout as cGAN_EB):
  `label = daytype·32 + group·4 + season`, day type = weekday / weekend (Belgian
  public holidays counted as weekend, as in the reference paper), meteorological
  seasons (MAM, JJA, SON, DJF; winter = Jan, Feb and Dec 2024).
* **Split by household**: meters are split 70/15/15 into train / validation /
  test, stratified by group (210/45/45 meters per group; `split_seed: 2024`).
  No household contributes days to more than one split.  The validation
  households are used only for checkpoint selection and GMM order selection; all
  reported numbers are on the test households.

Resulting sizes: 610,745 training, 130,935 validation and 131,039 test
day-profiles; 5,635 - 13,440 training days and 1,215 - 2,880 test days per label.

## Models

### Faithful reproduction: `cgan_eb`

Identical to `cGAN.py` / `train.py` of the reference: 24-d Gaussian noise
concatenated with the 64-d one-hot label; generator 64-128-64 with LeakyReLU(0.2)
and linear output; discriminator 64-128-256, Dropout(0.4), sigmoid output;
binary cross-entropy (D loss = mean of real and fake terms, non-saturating G
loss); Adam with default betas, lr 1e-4 (G) and 2e-4 (D); batch 2,000; one D
and one G step per batch, the G step reusing the batch labels; global min-max
scaling.

Deliberate deviations, each required for a valid study on this data:

| Reference | Here | Reason |
|---|---|---|
| minibatches in file order | reshuffled every epoch | unshuffled batches are not i.i.d. (data are ordered by meter and day) |
| drops days with any value < 0 or > 100 kWh | no value filter | the target is net load; the `< 0` rule would remove PV-export days and bias the PV labels |
| min / max over all data | min / max over training households | no test information in preprocessing |
| 10,000 epochs on 206,009 profiles | 1,040,000 generator updates (≈ 3,400 epochs here) | budget matched in updates, since our training set is 3× larger |
| final iterate | best validation checkpoint and final iterate, both with raw weights | standard model selection, test households untouched |
| no seeds | 5 seeds, deterministic algorithms | variance estimates |

`D(fake)` is detached in the D step and `retain_graph` is not used; both are
numerically identical to the reference.  An exponential moving average of the
generator weights is tracked as a read-out only (it does not influence
training); the reproduction results use the raw weights (`cgan_eb[best_raw]`,
`cgan_eb[last_raw]`), and `cgan_eb` with EMA weights is reported for a
like-for-like comparison with the improved model.  Training steps are captured in a CUDA
graph for speed; eager and graph execution give identical results.

### Improved model: `cwgan_gp`

* **Loss**: WGAN-GP (λ = 10, 5 critic steps per generator step, Adam
  lr 1e-4, betas (0.5, 0.9)), batch 1,024, 50,000 generator iterations.
* **Conditioning**: factorised embeddings for PV, heat pump, EV, day type and
  season (8-d each) plus a 16-d joint-label embedding, fed to the generator
  input; the critic uses input concatenation and a projection term
  ⟨embed(y), φ(x)⟩ (Miyato & Koyama, 2018).  Labels that share a technology or a
  season share parameters.
* **Networks**: 3×256 LeakyReLU MLPs (no normalisation in the critic).
* **Value transform**: asinh(x / c) followed by standardisation, with
  c = 1.5 kWh; it is linear for ordinary loads, mildly compresses the EV /
  heat-pump tail and handles negative net load.
* **Bounded output**: the generator output passes through a sigmoid onto the
  scaled training range widened by 5 % on each side.
* **EMA** of generator weights (0.999); checkpoint selected on validation.

The transform knee and the output bound were chosen on the validation
households.  With a strongly compressive transform (c = median |x| = 0.32 kWh)
and a linear output, the fit in the transformed space kept improving while a
few hundred samples per million fell far outside the observed range (up to
≈ 370 kWh/h against a training maximum of 24.4) after the inverse transform,
inflating the dispersion of the kWh profiles by ≈ 40 %; the validation score
in kWh then deteriorated after ≈ 4k iterations.  Bounding the output and
using c = 1.5 kWh removed the effect (dispersion ratio 0.99, 28 ppm of values
beyond the training range).  `abl_unbounded` documents this configuration.

### Ablations (3 seeds each)

| config | change w.r.t. `cwgan_gp` |
|---|---|
| `abl_onehot` | 64-way one-hot conditioning instead of factorised embeddings |
| `abl_noproj` | no projection term in the critic |
| `abl_minmax` | global min-max scaling instead of asinh |
| `abl_refnet_wgan` | the exact cGAN_EB networks and scaling trained with WGAN-GP (isolates the loss) |
| `abl_unbounded` | linear (unbounded) generator output and the strong asinh transform (c = median \|x\|) |
| `cwgan_gp[best_raw]` | raw instead of EMA generator weights (no retraining needed) |

### Baselines (`cgan/baselines.py`)

Fitted per label on the training households: multivariate **Gaussian**,
**Gaussian copula** with empirical hourly marginals, full-covariance **GMM**
(1-16 components chosen by validation log-likelihood) and **resampling** of
training days without replacement.  Resampling is the *train-empirical
reference*: it is the fidelity floor a generator of new households can hope to
reach on held-out households, and the positive control for the memorisation
test (it copies training days).  The **between-household reference** compares
real days of the validation households with the test households; since both are
finite sets of 45 households per group, it measures household heterogeneity
and is not a floor (a generator can score below it).

## Evaluation

For every label the number of synthetic days equals the number of real test
days.  Profile-level metrics use the hourly mean and standard deviation of the
training data for standardisation.  All metrics are computed per label and then
averaged over the 64 labels; tables report mean ± standard deviation over
seeds.  Because the labels of a group share the same test households, labels
are not treated as independent units: model comparisons report the share of
labels on which a model is better (descriptive), Welch tests over training
seeds (Holm-corrected), and a household-level cluster bootstrap
(`cgan/bootstrap.py`: the 45 test households of each group are resampled with
replacement, 1,000 replicates) that gives paired confidence intervals for
differences in energy distance and MMD².  Precision / recall / density /
coverage treat days as exchangeable and are used descriptively.

| family | metrics |
|---|---|
| whole profile | Fréchet distance of Gaussian fits (24-d; also relative to the trace of the real covariance), energy distance (also relative to the real E\|X−X′\|), MMD² (RBF, median heuristic, 3 bandwidths), precision / recall / density / coverage (k = 5), classifier two-sample test (gradient boosting, ROC-AUC with 5 folds grouped by household) |
| temporal | RMSE of the mean, standard-deviation and 90th-percentile profiles, mean absolute difference of hour-to-hour correlations, autocorrelation error (lags 1-6), hourly Wasserstein-1 and KS, Wasserstein-1 of hourly ramps |
| load-shape KPIs | Price (2010) near-peak (97.5th pct), near-base (2.5th pct), high-load duration, rise and fall time; daily energy, peak hour, ramps, exported energy — each compared by W1 (also in units of the real standard deviation), KS, Jensen-Shannon divergence and KL on shared smoothed histograms |
| conditional fidelity | a label classifier trained on the *validation* households (unseen by every generator) recovers the intended label / PV / HP / EV / day type / season from synthetic days; compared with its accuracy on the test households (balanced accuracy) |
| utility | train-on-synthetic, test-on-real (TSTR) vs train-on-real training households (TRTR), same classifier, budget and days per label |
| aggregate load | feeders of 20 distinct households (one random day each): Wasserstein-1 of the aggregate peak, relative error of its 95th percentile, coincidence-factor difference, read against the real reference; evening (17-22 h) energy on EV labels, exported energy on PV labels |
| memorisation | distance of each synthetic day to its closest training day vs the same distance for real days of unseen households: share below the holdout 5th percentile (0.05 expected, 1 for a copy model), median ratio, exact copies |
| physical | negative net load for non-PV labels, values beyond the training maximum |

**Reference-metric defects.**  The reference `utils.py` is reproduced for
comparability (`kpi.ref_*`), but its derived statistics are not used as
evidence: (i) its "KLD" applies `rel_entr` to two sorted pandas Series, whose
index alignment undoes the sort, so real day *i* is compared with generated day
*i* and the value is not a divergence; here KL and JS divergence are computed on
shared, smoothed histograms; (ii) its "FID" is (Δμ)² + (Δσ)² per feature in raw
units (reported as `fd1`); (iii) its peak / base are the 2nd-highest / lowest
hour and its rise / fall times measure distances between extreme hours rather
than Price's definitions, so both versions are reported; (iv) all of its
comparisons are against the training data.

## Reproducing

```bash
python -m venv .venv && .venv\Scripts\activate          # Windows
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
# put raw.zip from the `data` release into data/raw/
python -m cgan.data.prepare --raw data/raw/raw.zip --out data/processed
python -m pytest
python -m cgan.experiments --plan configs/experiments.yaml   # train + evaluate everything (resumable)
python -m cgan.report --runs runs --out results
python -m cgan.bootstrap --runs runs --out results --models cwgan_gp "cgan_eb[best_raw]" baseline_copula baseline_gmm baseline_gaussian baseline_resample
python -m cgan.figures --runs runs --out results/figures --models cgan_eb cwgan_gp baseline_copula
```

Single runs: `python -m cgan.train --config configs/cwgan_gp.yaml --seed 0`,
`python -m cgan.evaluate --run runs/cwgan_gp/seed0 --ckpt best`,
`python -m cgan.sample --run runs/cwgan_gp/seed0 --n-per-label 2000 --out samples.npz`.

Each run directory stores the resolved config, git revision, library versions,
loss history, validation curve and checkpoints.

## Layout

```
cgan/data/       prepare.py (raw zip -> arrays), profiles.py (labels, splits), scaling.py
cgan/models/     nets.py (reference and improved networks), losses.py
cgan/eval/       kpis.py, metrics.py, conditional.py, fast.py (validation score)
cgan/train.py    training loop          cgan/sample.py       generation
cgan/evaluate.py evaluation             cgan/baselines.py    non-adversarial baselines
cgan/experiments.py  experiment matrix  cgan/report.py / figures.py  tables and figures
configs/         YAML configs           tests/               unit tests
```

## References

* Li, Y., Dong, B., Qiu, Y. (2025). Conditional generative adversarial network (cGAN) for generating building load profiles with photovoltaics and electric vehicles. *Energy and Buildings* 335, 115584. https://doi.org/10.1016/j.enbuild.2025.115584
* Price, P. N. (2010). Methods for analyzing electric load shape and its variability. LBNL-3713E.
* Gulrajani, I. et al. (2017). Improved training of Wasserstein GANs. NeurIPS.
* Miyato, T., Koyama, M. (2018). cGANs with projection discriminator. ICLR.
* Naeem, M. F. et al. (2020). Reliable fidelity and diversity metrics for generative models. ICML.
* Gretton, A. et al. (2012). A kernel two-sample test. JMLR 13.
* Platzer, M., Reutterer, T. (2021). Holdout-based empirical assessment of mixed-type synthetic data. Frontiers in Big Data 4.
* Ravuri, S., Vinyals, O. (2019). Classification accuracy score for conditional generative models. NeurIPS.
