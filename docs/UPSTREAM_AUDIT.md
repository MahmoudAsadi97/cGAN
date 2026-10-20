# Upstream method and implementation audit

Audited source: [`malvin0130/cGAN_EB`](https://github.com/malvin0130/cGAN_EB), pinned commit **`885f78052a91fbc755e6b17a354c68849b5556ca`**. The linked paper was inspected through its [NSF-hosted PDF](https://par.nsf.gov/servlets/purl/10599218). Findings about code below come from the released files, not an assumption that every detail matches the article.

The repository implements a conventional conditional GAN with binary cross-entropy. The repository suffix `EB` is not evidence of an energy-based GAN objective. This adaptation retrains that method on another dataset; the original confidential training data and pretrained checkpoints are not provided.

## Released model

| Component | Released implementation |
|---|---|
| Sample | One independent 24-hour daily load vector |
| Noise | 24-dimensional standard Gaussian vector |
| Generator | Concatenate noise and one-hot label; dense 64 → 128 → 64 → 24 |
| Discriminator | Concatenate profile and one-hot label; dense 64 → 128 → 256 → dropout 0.4 → 1, followed by sigmoid |
| Hidden activations | LeakyReLU, negative slope 0.2 |
| Generator output | Linear, without a nonnegative constraint |
| Labels | 64 categories: 8 PV/EV/charger combinations × 4 seasons × 2 day types |
| Discriminator loss | Half the sum of real-target-one and fake-target-zero BCE losses |
| Generator loss | Non-saturating BCE with fake predictions targeted to one |
| Optimizers | Adam defaults; generator learning rate 0.0001 and discriminator 0.0002 |
| Schedule | One discriminator and one generator update per batch; 10,000 epochs; batch size 2,000 |
| Preprocessing | Remove daily profiles outside [0, 100]; apply global min–max scaling to the supplied frame |
| Generation | 2,000 profiles per fixed condition; inverse scaling |

Sources: [`cGAN.py`](https://github.com/malvin0130/cGAN_EB/blob/885f78052a91fbc755e6b17a354c68849b5556ca/cGAN.py), [`train.py`](https://github.com/malvin0130/cGAN_EB/blob/885f78052a91fbc755e6b17a354c68849b5556ca/train.py), and [`utils.py`](https://github.com/malvin0130/cGAN_EB/blob/885f78052a91fbc755e6b17a354c68849b5556ca/utils.py). A batch uses the observed real labels for both real and generated examples. Labels are concatenated as one-hot vectors; there is no learned embedding, temporal convolution, recurrent network, physics loss, or Wasserstein loss.

The paper's original measurements cover 110 residential customers in the southwest United States during 2013–2019. It reports 147,245 weekday and 58,764 weekend daily profiles; holidays are included with weekends. PV/EV adoption dates and charger-level metadata support the original condition labels. These are source-study facts, not statistics of the Belgian release. The article states that the original data are confidential and identifies sparse classes and geographic/configuration generalization as limitations.

## Paper–code differences and evaluation defects

| Location | Observation | Treatment in the adaptation |
|---|---|---|
| Paper §2.2; `train.py:17–21` | Paper describes scaling by a fixed dataset maximum; code subtracts a global minimum and divides by its range. | Name the actual scaling method. Fit only on the training split, save it, and guard a zero range. |
| Paper Eq. 2; `cGAN.py:95–97` | Paper writes the sum of two BCE terms; code divides that sum by two. | Preserve the released code's factor in the source-style baseline and document it. |
| Paper §2.4; `utils.py:157–158,173–174` | Paper specifies 97.5th and 2.5th percentiles; code takes the second-highest and second-lowest entries of a 24-point day. | Use explicitly defined numerical quantiles for corrected diagnostics; do not claim exact reproduction of the upstream statistic. |
| Paper §2.4; `utils.py:162–165,178–181` | Directional rise/fall descriptions become absolute distances between indexes nearest/farthest from selected amplitudes, without chronological direction or midnight handling. | Prefer clearly defined ramp and peak-time diagnostics. Any rise/fall metric needs an explicit directional/circular convention. |
| `utils.py:281–290` | The KL routine passes magnitude-normalized sorted observations to `rel_entr`, removes infinite terms, and takes the absolute mean. These are not shared-support probability distributions. | Do not reproduce this value as valid KL. Use common-bin probability estimates with documented smoothing or an appropriate distribution distance. |
| `utils.py:96–113,459–469` | The function named FID flattens each individual scalar statistic and computes a Gaussian mean/variance distance separately. It does not use Inception features or jointly assess temporal dependence. | Report the actual Gaussian feature/moment distance and feature space. Add distribution and temporal diagnostics; no universal quality threshold is justified. |
| `train.py:45–48` | Real comparison data are the same frame supplied for training; no holdout is constructed. | Introduce meter-disjoint train/validation/test partitions and train-only preprocessing. |

These are concrete limitations of the released implementation. They are not evidence that all original study conclusions are false, nor are numerical thresholds from the paper transferable to different data, scales, or metrics.

## Engineering issues

- **Unavailable input and positional schema:** `train.py:8` requires `new_train_all.csv`, which is absent. Repeated `iloc[:, :-2]` slicing assumes two trailing metadata columns. The Belgian input requires a named-column adapter and validation.
- **Class assumptions:** `len(df_train.Label.unique())` and `F.one_hot` assume contiguous zero-based labels. The utilities hard-code 64 source conditions even when other class counts are supplied. Save a training-fitted mapping and validate requested conditions.
- **No split or reproducibility machinery:** released files contain no seed control, epoch shuffling, validation selection, checkpoint/resume support, persisted scaler, environment specification, or dataset manifest. Sequential batches may retain input ordering. Add these features and identify them as implementation improvements.
- **Unnecessary gradient work:** fake samples are not detached for the discriminator step; discriminator parameters remain trainable during the generator step; generator backward retains a graph unnecessarily. Correcting these avoids waste. Since only the corresponding optimizer steps and gradients are cleared, these facts alone do not establish that the wrong network is updated.
- **Loss logging:** loss lists are initialized outside the epoch loop, so printed means describe the entire training history rather than the current epoch. Report epoch-local sample-weighted losses and actual update counts.
- **Output validity:** the linear generator can emit negative or out-of-range values. Do not conceal this by post hoc clipping. Softplus is a named architectural extension for nonnegative import/export targets and is unsuitable for signed net exchange.
- **Portability:** plotting forces `TkAgg`, which can fail on headless systems. Use a noninteractive backend for batch artifacts.

Using logits with `BCEWithLogitsLoss` in place of explicit sigmoid plus BCE preserves the mathematical BCE objective while improving numerical stability. Document the implementation choice; it is not a new generative objective. Removing unnecessary gradients, validating schemas, and recording reproducibility metadata likewise do not establish methodological novelty.

## What transfers and what changes

The source-style baseline retains Gaussian noise, concatenated one-hot conditioning, the hidden MLP widths, LeakyReLU slope, discriminator dropout, BCE losses, and Adam learning-rate ratio. Its 24 output values can represent hourly energy after summing four 15-minute observations. The Belgian source contains measured grid import and export plus supplied PV/EV/heat-pump indicators; it does not supply the original US adoption dates or three charger-level categories. Do not invent equivalent charger labels or infer missing attributes from the consumption curve.

The following must be described as adaptations or experiments: 96-point output, heat-pump conditioning, changed condition vocabulary, softplus output, train-only normalization, grouped holdouts, corrected evaluation, and conditional bootstrap comparison. Stronger temporal models, additional objectives, physical constraints, downstream utility experiments, and privacy attack studies are further work. The initial repository scope is an executable baseline and diagnostic framework; publication-quality comparative evidence still requires the experiments in [RESEARCH_PROTOCOL.md](RESEARCH_PROTOCOL.md).

The source paper discusses privacy benefits, but the released repository provides no privacy mechanism or attack evaluation. Synthetic generation and large nearest-neighbor distances alone do not prove privacy. Similarly, independent daily sampling does not establish realistic inter-day or annual coherence, and differences across observed equipment groups do not estimate causal equipment effects.

## Attribution and license

Upstream is licensed under the [MIT License](https://github.com/malvin0130/cGAN_EB/blob/885f78052a91fbc755e6b17a354c68849b5556ca/LICENSE), with the notice **Copyright (c) 2024 [malvin0130]**. Preserve its full copyright and permission notice with copied or substantially adapted source. The upstream README also requests citation of the associated article. This audit does not assign the MIT license to independently supplied dataset files; verify the Fluvius data terms before redistribution.

> Li, Y., Dong, B., & Qiu, Y. (2025). Conditional generative adversarial network (cGAN) for generating building load profiles with photovoltaics and electric vehicles. *Energy and Buildings*, 335, 115584. https://doi.org/10.1016/j.enbuild.2025.115584

Useful primary references: [paper DOI](https://doi.org/10.1016/j.enbuild.2025.115584), [open paper at NSF](https://par.nsf.gov/servlets/purl/10599218), [pinned upstream README](https://github.com/malvin0130/cGAN_EB/blob/885f78052a91fbc755e6b17a354c68849b5556ca/README.md), and [complete pinned source tree](https://github.com/malvin0130/cGAN_EB/tree/885f78052a91fbc755e6b17a354c68849b5556ca).
