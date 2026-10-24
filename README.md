# cGAN load profiles with PV, heat pumps and EVs — Fluvius 2024

This branch applies the conditional GAN of Li, Dong & Qiu (2025),
*Conditional generative adversarial network (cGAN) for generating building load
profiles with photovoltaics and electric vehicles*, Energy and Buildings 335,
115584 (code: [malvin0130/cGAN_EB](https://github.com/malvin0130/cGAN_EB)), to
the Fluvius 2024 smart-meter sample in the
[`data` release](https://github.com/MahmoudAsadi97/cGAN/releases/tag/data) of
this repository.

It contains the reference model with its original hyper-parameters
(`faithful`, the GitHub code) and the delivered model (`final`), trained on the
same households and evaluated on households that neither model has seen,
together with the model-selection candidates and an ablation ladder that leads
from one to the other.

## Results

Held-out **test households** (354 households, 128,826 days); `final`: five
training seeds, validation-selected checkpoints (`best`) and final iterates
(`last`); `faithful` (the GitHub code): three training seeds.  Accuracy in
percent; ± is the standard deviation over seeds.

| model | statistical accuracy | worst label | labels ≥ 85 % | rFD ↓ | C2ST AUC (0.5 ideal) | coverage ↑ | DCR ratio (≈ 1) | negative load, no PV ↓ |
|---|---|---|---|---|---|---|---|---|
| GitHub cGAN [last] (as the reference) | 48.3 ± 1.3 | 27.4 ± 5.5 | 0 % | 1.40 | 1.00 | 0.04 | 2.00 | 21.9 % |
| GitHub cGAN [best] | 55.1 ± 0.6 | 39.8 ± 4.1 | 0 % | 0.91 | 1.00 | 0.06 | 1.51 | 18.2 % |
| **final [best]** | **87.9 ± 0.2** | 74.6 ± 0.6 | 82.5 % | **0.117** | 0.61 | 0.81 | 1.01 | **0** |
| final [last] | 87.6 ± 0.2 | 70.8 ± 1.1 | 82.2 % | 0.129 | 0.59 | 0.82 | 1.00 | 0 |
| real: validation households | 86.1 | 76.9 | 59.4 % | 0.167 | 0.50 | 0.79 | 1.02 | 0 |
| real: training households | 87.9 | 70.6 | 82.8 % | 0.132 | 0.49 | 0.82 | – | 0 |

* **The 85 % target is met** by every seed (87.72, 88.00, 88.12, 87.61,
  87.88 %).  With the household-level uncertainty (`results/uncertainty.md`)
  the accuracy is 87.9 % (95 % interval 86.4–89.3 %, one-sided 95 % lower
  bound 86.7 %).  The final model matches the test households 1.8 points
  better than real days of the 356 validation households (paired 95 %
  interval +0.6 to +3.1) and as well as real days of the 1,657 training
  households (−0.1, interval −0.8 to +0.7); it is 39.5 points above the
  GitHub model [last] (38.3 to 40.7).
* **Better than the GitHub model on every ranked metric** (the reference
  code's KLD is descriptive only, see *Comparison with the GitHub model*),
  each with complete separation: every final seed beats every GitHub value
  (`results/comparison.md`).  The GitHub code trained on these data
  oscillates and does not converge (`results/figures/validation_curves.png`;
  the dotted line there is the training-household benchmark scored on the
  validation households); its average days zig-zag from hour to hour
  (`results/figures/compare/`).
* **What makes the difference** (`results/ablation.md`): two changes in how
  the data are fed to the networks — minibatches reshuffled every epoch
  (+11.2 points) and the per-hour value transform (+21.1) — lift the
  reference networks with their BCE training from 54.8 % (seed 0; 55.1 ± 0.6
  over three seeds) to 87.2 % (one seed, validation-selected checkpoint).  The
  WGAN-GP recipe with wider networks, EMA and the hour-to-hour critic input
  adds 0.5–0.7 points of accuracy (seed 0: +0.55, within the seed spread of
  the BCE runs) and mainly makes single days more realistic: C2ST AUC 0.70 →
  0.61, rFD 0.137 → 0.117, coverage 0.80 → 0.81.  With the EMA read-out the
  validation accuracy drifts down by at most 0.8 points after its peak,
  against 2.0 points for the BCE model (88.0 % at epoch 1,450, 86.0 % at
  epoch 10,000).  The no-PV floor removes negative load without PV (0.15 % of
  the hours with BCE, 0.39–0.87 % in the WGAN steps before the floor).  The
  BCE model is better on the worst label (78.9 % against 74.6 %), the
  key-parameter FID (0.69 against 0.85), the near-base KL (0.070 against
  0.080) and the key-parameter std (90.7 % against 90.1 %).
* **Acceptance levels of the paper** (`results/paper_criteria.md`, test
  households, share of label × key-parameter pairs, with the paper's
  definitions and a proper KL; the reference-code definitions are in the same
  table; GitHub averaged over its three seeds): |difference of mean| < 1 —
  GitHub [last] 54 %, final 99 %;
  |difference of std| < 1 — 78 % / 100 %; KL < 0.3 — 7 % / 100 %; FID < 5 —
  71 % / 100 %.  With the paper's own protocol (generated days against the
  training households, `results/paper_protocol.md`) the final model reaches
  93.2 % accuracy and meets every level for 100 % of the pairs with the
  paper's definitions (99 % for KL < 0.3 with the reference code's KLD).

Figures (`results/figures/`, seed 0): the paper's Figs. 3–12 for the final
model (`final_best/`; files with `_code` use the definitions of the reference
code) and for the GitHub model (`faithful_last/`), plus `temp_dist_<key>.png`
with the distribution of each key parameter in all 64 labels; both models in
one plot (`compare/`); and the validation curves of every finished seed
(`validation_curves.png`).
`results/figures/final_best/*_gen.csv` are 2,000 generated days per label (kWh
per hour), as published by the reference repository.

## Dataset (`data_prep.py`)

`raw.zip` holds eight CSV files of 300 residential meters each, one per
combination of PV (*ZP*), heat pump as main heating (*WP*) and home-charged
electric vehicle (*EV*), with quarter-hourly grid import
(`Volume_Afname_KWh`) and export (`Volume_Injectie_KWh`) for all of 2024 and
the three indicators per meter.  `data_prep.py` aggregates the quarter hours to
hourly energy and builds daily profiles as in the paper (Sec. 2.1–2.2):

| | |
|---|---|
| Target | hourly **net load** = import − export (kWh); negative while PV exports |
| Day type | weekday / weekend, Belgian public holidays counted as weekend |
| Season | spring MAM, summer JJA, fall SON, winter DJF |
| Group | PV, heat-pump and EV indicators: `No PV HP EV`, `PV`, `HP`, `PV+HP`, `EV`, `PV+EV`, `HP+EV`, `PV+HP+EV` |
| Label | `daytype · 32 + group · 4 + season` → 64 labels, the layout of the reference |

**EV labels.** The reference distinguishes EV households by charger level
(Level 1, Level 2, both).  The Fluvius release has no charger information,
only a yes/no EV indicator (an EV declared by the customer or detected by
Fluvius; 0 does not exclude an EV).  The three charger classes are therefore
replaced by the heat-pump indicator, which keeps eight technology groups and
64 labels.

**Cleaning** (the paper removes "daily profiles with missing or incorrect
recordings"):

* timestamps are local time (CET/CEST, see the legend in the archive): the DST
  days 31 March (92 quarter hours) and 27 October (100) do not form a 24-hour
  day and are removed — every kept hour consists of exactly four quarter hours;
* 33 meters have a full-year import and export series identical to another
  meter of the same group under a different `EAN_ID`; only the first copy is
  kept, so that no household can appear in two splits;
* 866 meter-days without any import or export in all 24 hours (vacancy,
  disconnection or zero-filled data) are removed;
* the reference drops days with values < 0 or > 100 kWh.  The > 100 kWh rule is
  kept (it removes nothing); the < 0 rule is not applied because negative net
  load is genuine PV export.

**Households are split** 70 / 15 / 15 % into train / validation / test,
stratified by group (1,657 / 356 / 354 households; 602,356 / 129,540 /
128,826 daily profiles; 598,249 weekday and 262,473 weekend days in total).
The scaler and the models see only the training households, the validation
households select checkpoints and models, and the test households give the
reported numbers.  `results/table1_labels.md` lists the days and households of
every label and split (the counterpart of the paper's Table 1): 5,355–13,440
training days and 199–210 training households per label.  Unlike the paper's
data (316–9,325 days per label), the labels are nearly balanced, so the
paper's explanation of its weakest labels by class imbalance does not arise
here.

## Models (`cGAN.py`, `train.py`)

### `faithful` — the GitHub code

`Generator`, `Discriminator` and `cGAN` are the reference classes: 24-d noise
concatenated with the 64-d one-hot label; generator 64-128-64 with LeakyReLU
and linear output; discriminator 64-128-256 with Dropout(0.4) and sigmoid;
binary cross-entropy; Adam with lr 1e-4 (G) and 2e-4 (D); batch 2,000;
10,000 epochs; global min-max scaling; minibatches in data order (by meter and
day), as in the reference.  Additions that do not change the training
mathematics: seeds, a CUDA graph that replays the training step (the same
operations, ~6× faster), periodic validation, checkpointing and resuming.
Both the final iterate (`last`, what the reference uses) and the
validation-selected checkpoint (`best`) are evaluated.

### `final` — the delivered model

The same conditional set-up as the reference — noise and the one-hot label at
the input of both networks, 64 labels — with these changes:

| change | reference | final | why |
|---|---|---|---|
| minibatches | data order, every epoch the same | reshuffled every epoch | ordered batches hold one technology group at a time; the networks then chase one group after another |
| value transform | global min-max over all values | per-hour Gaussianisation (empirical CDF of the training data, then the standard-normal quantile) | net load is heavy-tailed (EV and heat-pump peaks up to 24 kWh/h); min-max squeezes ordinary loads of 0.1–1 kWh into a sliver of [0, 1]; the inverse maps every generated value back into the observed range of its hour |
| loss and training recipe | BCE, sigmoid, dropout; Adam lr 1e-4 (G) / 2e-4 (D), default betas; batch 2,000; 10,000 epochs | Wasserstein loss with gradient penalty (λ = 10), critic without sigmoid and dropout, 5 critic steps per generator step; Adam lr 1e-4 for both, betas (0.5, 0.9); batch 1,024; 400 critic epochs | stable training without mode collapse; the BCE training oscillates strongly (`results/figures/validation_curves.png`) |
| networks | G 64-128-64, D 64-128-256, noise 24 | G and critic 4 × 512 LeakyReLU, noise 64; the critic also receives the hour-to-hour changes x[t+1] − x[t] | capacity for 64 labels with very different shapes; generated days of the smaller models were rougher than real days |
| read-out | final weights | exponential moving average (0.999) of the generator weights; checkpoint with the best validation accuracy | averaging removes the oscillation of the iterates; selection uses only validation households |
| physics | — | net load of labels without PV floored at 0 kWh | meters without PV register no injection; without the floor 0.40 % of the generated hourly values of these labels were negative (validation households) |

The hour-to-hour changes are linear in the profile, so giving them to the
critic adds no information: it reparameterises the critic's first layer, which
gives the changes their own weights and changes how the critic is initialised
and optimised.

**Model selection.**  `final` was chosen on the validation households among
four candidates trained with seed 0 (`results/selection.md`):
`dev_improved_nofloor` (the `improved` configuration — 3 × 256 networks,
noise 32 — trained before the no-PV floor existed; the floor changes its
validation accuracy by about 0.02 point), `improved_diff` (+ hour-to-hour
changes for the critic), `improved_big` (4 × 512, noise 64) and
`improved_bigdiff` (both).  Rule, fixed before the comparison: highest
validation accuracy, the simpler model if two are within 0.1 percentage point.
`improved_bigdiff` won (88.70 % against 88.48 % and 88.38 %), and `final` is its
configuration trained with five seeds (seed 0 is the selection run itself:
training is deterministic, so both runs are identical); `selection.md` also
lists the validation accuracy of the five `final` seeds.  The selection-draw
value of `dev_improved_nofloor` (87.88 %) was recorded by the development
version of the validation code; the published code scores the same checkpoint
with the same draw at 88.07 %, which leaves the ranking unchanged (the other
three candidates re-score exactly).  The rule ranked the
best of 40 validation checkpoints scored with one fixed noise draw; scored
again with five fresh noise draws (`results/selection.md`), the two 4 × 512
candidates are within noise of each other for seed 0.  Seeds 1–4 of
`improved_big`, trained after the selection for the ablation and listed in
`selection.md` together with the selection run (seed 0), give the
hour-to-hour changes a small validation advantage (+0.14 ± 0.16 points over
the five seeds with fresh noise draws, seed-matched; higher for 4 of 5
seeds), and the ablation finds +0.09 ± 0.15 points on the test households:
the change is kept, but its effect is within seed noise.

**Ablation ladder** (`results/ablation.md`, validation-selected checkpoints;
seeds 0–4 for `improved`, `improved_big` and `final`, seeds 0–2 for
`faithful` and the other WGAN steps, seed 0 for `abl_shuffle` and
`abl_quantile`).  Each step adds one change to the previous row, and the table
gives the seed-matched difference of test accuracy to the previous step;
`abl_wgan_refnets` switches to the WGAN-GP training recipe as a whole (loss,
critic without sigmoid and dropout, 5 critic steps, optimiser settings, batch
1,024, 400 critic epochs) with the reference networks:

`faithful` → `abl_shuffle` (+ reshuffled batches) → `abl_quantile` (+ value
transform) → `abl_wgan_refnets` (+ WGAN-GP recipe) → `abl_wide` (+ 3 × 256
networks, noise 32) → `abl_ema` (+ EMA read-out) → `improved` (+ no-PV floor)
→ `improved_big` (+ 4 × 512 networks, noise 64) → `final` (+ hour-to-hour
changes for the critic).

| step | seeds | test accuracy | Δ to previous step (pt) | worst label | C2ST AUC | rFD | negative load, no PV |
|---|---|---|---|---|---|---|---|
| `faithful` [best] | 3 | 55.1 ± 0.6 | | 39.8 | 1.00 | 0.913 | 18.2 % |
| `abl_shuffle` | 1 | 66.1 | +11.2 | 55.3 | 1.00 | 0.469 | 13.6 % |
| `abl_quantile` | 1 | 87.2 | +21.1 | 78.9 | 0.70 | 0.137 | 0.15 % |
| `abl_wgan_refnets` | 3 | 83.7 ± 0.3 | −3.8 (seed 0) | 72.2 | 0.76 | 0.200 | 0.87 % |
| `abl_wide` | 3 | 87.1 ± 0.2 | +3.43 ± 0.54 | 74.8 | 0.68 | 0.119 | 0.48 % |
| `abl_ema` | 3 | 87.5 ± 0.3 | +0.36 ± 0.41 | 75.1 | 0.65 | 0.117 | 0.39 % |
| `improved` | 5 | 87.5 ± 0.2 | −0.00 ± 0.01 (seeds 0–2) | 76.0 | 0.65 | 0.117 | 0 |
| `improved_big` | 5 | 87.8 ± 0.1 | +0.26 ± 0.20 | 75.2 | 0.62 | 0.117 | 0 |
| `final` | 5 | 87.9 ± 0.2 | +0.09 ± 0.15 | 74.6 | 0.61 | 0.117 | 0 |

The data-handling changes carry most of the gain; `abl_shuffle` and
`abl_quantile` have one seed each, and the BCE runs of `faithful` have seed
SDs of 0.6 (best) and 1.3 (last) points (ranges 1.1 and 2.5), so their single
values are less certain than the WGAN rows.  With the reference networks the
WGAN-GP recipe alone scores lower than BCE (within its 400 critic epochs;
seeds 0 and 1 are still at their maximum at epoch 400); with the wider
networks it reaches the BCE level (87.1 ± 0.2 against 87.2 %) and passes it
only with the EMA read-out and the 4 × 512 networks (87.5–87.9 %).  Without
EMA its validation accuracy fluctuates about as much as BCE's and ends
0.5–1.1 points below its peak; with EMA it drifts down by at most 0.8 points
after its peak, against 2.0 points for the BCE model.  The EMA and the
no-PV floor change only how generated days are read out and which checkpoint
is selected: with the same seed, `abl_wide`, `abl_ema` and `improved` follow
the same training trajectory, so their differences are not affected by
training noise.

## Paper, reference code and this implementation

| item | paper (text) | reference code | `faithful` | `final` |
|---|---|---|---|---|
| data | 110 US households with PV and EV, 2013–2019, hourly | — | Fluvius 2024, 2,367 Flemish households | same |
| labels | day type × 8 PV / EV-charger statuses × season | same | day type × 8 PV / HP / EV groups × season | same |
| value filter | "missing or incorrect recordings" removed | days with values < 0 or > 100 kWh dropped | > 100 kWh rule only (net load) | same |
| normalisation | divided by the maximum of the data set | min-max with global min and max | min-max, training households | per-hour Gaussianisation, training households |
| data used for validation | — (same data as training) | training data | held-out test households | same |
| minibatch order | — | file order | data order (meter, day) | reshuffled every epoch |
| loss | BCE (Eqs. 1–3) | BCE, D loss = (real + fake) / 2 | as the code | WGAN-GP |
| networks | Table 2 (the 88 units of "layer 1" are the 24 + 64 inputs) | G 64-128-64, D 64-128-256, dropout 0.4 | as the code | 4 × 512, critic with hour-to-hour changes |
| training | Adam 1e-4 / 2e-4, 10,000 epochs, batch 2,000 | same | same | Adam 1e-4, betas (0.5, 0.9), 400 critic epochs, batch 1,024, EMA |
| weights used | — | after the last epoch | `last` and `best` | `best` (EMA) |
| generated days per label | 2,000 | 2,000 | test-day count for the metrics; 2,000 for the figures | same |
| key parameters | near-peak / near-base = 97.5th / 2.5th percentile; rise and fall time after Price (2010) | 2nd-highest / 2nd-lowest hour; rise and fall as \|argmax − argmin\| of hour indices | both reported | both reported |
| KL | KL(P ‖ Q), Eq. 4 | `rel_entr` on two sorted, index-aligned pandas Series (pairs day *i* with day *i*) | both reported | both reported |
| FID | Eq. 5 on the five key parameters; figures per parameter | per parameter, 1-D | both reported | both reported |

"Both reported": the figures and `paper_criteria.md` come in two versions, with
the definitions of the paper's text (and a proper KL) and exactly as the
reference code computes them (`metrics.reference_key_parameters`,
`metrics.reference_kld`); the second is the like-for-like counterpart of the
published Figs. 5–12.

## Evaluation (`metrics.py`, `evaluate.py`, `report.py`)

For every label the generator produces as many days as the test households
have for that label; all metrics are computed per label and averaged over the
64 labels.

**Statistical accuracy** (the target, ≥ 85 %).  The paper validates with the
average day per label (Figs. 3–4), the mean and standard deviation of five key
parameters (Figs. 5–6) and their distributions (KL, FID; Figs. 9–12).  The
accuracy turns these comparisons into bounded scores (1 = perfect), each
against the held-out real days of the label:

| term | definition |
|---|---|
| mean profile | 1 − Σₜ\|μ_gen(t) − μ_real(t)\| / Σₜ\|μ_real(t)\| |
| std profile | 1 − Σₜ\|σ_gen(t) − σ_real(t)\| / Σₜ σ_real(t) |
| KP mean | 1 − \|mean_gen(k) − mean_real(k)\| / mean_real(\|k\|), averaged over the key parameters k |
| KP std | 1 − \|σ_gen(k) − σ_real(k)\| / σ_real(k), averaged over k |
| hourly 1−KS | 1 − Kolmogorov–Smirnov distance of the hourly values, averaged over the 24 hours |
| KP 1−KS | 1 − KS distance of each key parameter, averaged over k |

accuracy = mean(mean profile, std profile, (KP mean + KP std)/2, (hourly 1−KS + KP 1−KS)/2),
each term clipped to [0, 1].  Level, spread, key-parameter statistics and full
distributions weigh equally, so a model that reproduces only the average day
scores below 50 %.  Key parameters (paper Sec. 2.4, after Price 2010):
near-peak load (97.5th percentile of the day), near-base load (2.5th
percentile), peak load duration (hours above the mean of the two), rise time
(from the last near-base hour to the start of the high-load period), fall time
(from its end to the next near-base hour) and, in addition, daily energy.  The
high-load period is the contiguous run of hours above the midpoint that
contains the daily maximum, and a near-base hour lies within 10 % of the daily
range above the near-base load.  (With the first and last high hour of the
whole day, rise and fall time would be 0 on most PV days, whose night and
evening lie above the midpoint set by the midday export.)

**Paper criteria** (`results/paper_criteria.md`).  The paper accepts its model
because the mean and standard-deviation differences of the key parameters are
below 1 kWh (1 h for the time parameters) for most labels, the KL divergences
are about 0.3 and below and most FIDs are below 5.  The table gives the share
of (label, key parameter) pairs that meet each level, for both definition sets.
`results/paper_protocol.md` repeats it with the paper's own protocol —
generated days against days of the *training* households (2,000 per label) —
to separate the model from the stricter held-out protocol used everywhere
else.

**Other metrics.**  The relative Fréchet distance of the 24-h profiles (rFD),
the error of the hour-to-hour correlation matrix, a classifier two-sample test
(C2ST: ROC-AUC of a gradient-boosting classifier separating real from
generated days, with cross-validation folds grouped by household so that it
cannot win by recognising households; 0.5 = indistinguishable), precision /
recall / coverage (Naeem et al. 2020), a memorisation check (DCR ratio:
distance of generated days to their nearest training day relative to the same
distance for real held-out days; ≈ 1 expected, ≪ 1 would mean copying) and the
share of negative values for households without PV (physically impossible).

**Real-data benchmarks.**  Two rows score real days of other households
against the test households with the same metrics: days of the 356
*validation* households and days of the 1,657 *training* households (what a
generator that copied its training data would score).  They show how close
real homes of the same population come to the test homes.  They are not upper
bounds: they depend on the size and composition of the pool, and a generator
that represents the population well can exceed them on some metrics.  They are
single deterministic evaluations (no seeds); the DCR ratio of the
training-household benchmark is not meaningful (those households are the
training data).

**Uncertainty** (`uncertainty.py`, `results/uncertainty.md`).  The ± in the
tables is the standard deviation over training seeds.  The larger source of
uncertainty is which households are in the test set, shared by all rows.  It
is estimated by stratified half-sampling: the test households of every group
are split at random into two halves, every model and benchmark is scored on
both halves, and the standard error of the full-sample accuracy is
SD(half A − half B) / 2 over 100 splits (the same splits for all rows, so
differences between rows are paired).  The statistical accuracy depends on how
many real days a label has — fewer days give a lower score — so it is
comparable only between rows evaluated on the same test set, as all rows here
are, and the 85 % target refers to this test split.  All results are
conditional on the 1,657 training households.

**Comparison with the GitHub model** (`results/comparison.md`).  Every metric
has a direction (higher, lower, or closer to its ideal value: 0.5 for the C2ST
AUC, 1 for the DCR ratio); the table marks a metric as better for the final
model when every final seed beats every value of the GitHub model (all seeds,
both checkpoints).  The reference code's "KLD" is shown for comparison with the
published figures only and is not used to rank models.

### Protocol notes

* One development run — the first `improved` run, before the no-PV floor
  existed (`runs/dev_improved_nofloor`) — was scored on the test households
  during development (accuracy 87.2 %, 21 October 2026).  Its test numbers are
  not reported.  The floor added afterwards rests on the physical argument and
  on the validation households, where the same run produced 0.40 % negative
  hourly values for labels without PV.  The three other selection candidates
  (`improved_diff`, `improved_big`, `improved_bigdiff`) were defined after
  this test scoring; the comparison among all four used validation households
  only, with the selection rule fixed beforehand.  Every run in the summary,
  comparison and ablation tables was trained after these decisions; the
  selection table lists the development run itself, scored on validation
  households.
* The metric definitions were finalised after two code reviews and before the
  reported runs: C2ST folds grouped by household, KL bins at the quantiles of
  the real sample, and the high-load period of the rise and fall times.
* Training runs that were interrupted before checkpoints with optimiser state
  existed were restarted from scratch; since then every run writes
  `resume.pt` and continues after an interruption (`run_jobs.py`).

## Reproducing

```bash
python -m venv .venv
source .venv/bin/activate                             # Linux/macOS; Windows: .venv\Scripts\activate
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
mkdir -p data/raw
curl -L -o data/raw/raw.zip https://github.com/MahmoudAsadi97/cGAN/releases/download/data/raw.zip
python data_prep.py                                   # -> data/processed/fluvius_hourly.npz (~1 min)
python -m pytest

python run_jobs.py jobs/variants.txt --parallel 3     # model-selection candidates and faithful seed 0
python run_jobs.py jobs/final.txt --parallel 3        # final model, seeds 0-4
python run_jobs.py jobs/ablation.txt --parallel 3     # faithful seeds 1-2 and the ablation ladder
python evaluate.py --real train --split val           # real benchmark on the validation households (curves, dashboard)
python report.py --models faithful final --figures faithful:last final:best \
    --ablation faithful abl_shuffle abl_quantile abl_wgan_refnets abl_wide abl_ema improved improved_big final \
    --selection dev_improved_nofloor improved_diff improved_big improved_bigdiff final \
    --paper-protocol faithful:last final:best --curves faithful final
python uncertainty.py --reps 100                      # household-level uncertainty (needs report.py first)
python dashboard.py                                   # live training monitor, http://127.0.0.1:8050
```

`run_jobs.py` starts every job with `train.py --resume`: running a job file
again after an interruption continues all unfinished runs from their last
checkpoint and skips finished ones.  Single runs:
`python train.py --config final --seed 0`, `python evaluate.py --run runs/final/seed0 --ckpt best`.

## Files

```
data_prep.py   raw.zip -> hourly arrays; labels, cleaning, household split
cGAN.py        reference Generator / Discriminator / cGAN; cWGAN (improved and final models)
train.py       configurations, value transforms, validation, checkpoints and resuming
run_jobs.py    runs a job file a few jobs at a time, resuming interrupted runs
metrics.py     key parameters (paper text and reference code), accuracy, KL / FID, other metrics
evaluate.py    evaluation of one run (or of the real-data benchmarks)
report.py      result tables and figures for all finished runs
uncertainty.py household-level standard errors and paired differences (half-sampling)
utils.py       reference result analysis: per-label CSVs and the paper's Figs. 3-12
dashboard.py   live training monitor
jobs/          job files of all experiments
results/       tables and figures (generated days of the final model in results/figures/final_best)
tests/         unit tests
LICENSE-cGAN_EB  license of the reference code adapted in cGAN.py and utils.py
```

## References

* Li, Y., Dong, B., Qiu, Y. (2025). Conditional generative adversarial network (cGAN) for generating building load profiles with photovoltaics and electric vehicles. *Energy and Buildings* 335, 115584. https://doi.org/10.1016/j.enbuild.2025.115584
* Price, P. N. (2010). Methods for analyzing electric load shape and its variability. LBNL-3713E.
* Wang, Z., Hong, T. (2020). Generating realistic building electrical load profiles through the Generative Adversarial Network (GAN). *Energy and Buildings* 224, 110299.
* Gulrajani, I., Ahmed, F., Arjovsky, M., Dumoulin, V., Courville, A. (2017). Improved training of Wasserstein GANs. NeurIPS.
* Yazıcı, Y. et al. (2019). The unusual effectiveness of averaging in GAN training. ICLR.
* Naeem, M. F., Oh, S. J., Uh, Y., Choi, Y., Yoo, J. (2020). Reliable fidelity and diversity metrics for generative models. ICML.
