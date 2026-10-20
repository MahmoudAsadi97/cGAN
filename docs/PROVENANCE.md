# Source and experiment provenance

The reference method is `malvin0130/cGAN_EB` at commit
`885f78052a91fbc755e6b17a354c68849b5556ca`. Phase 1 preserves its hourly MLP
topologies, latent dimension 24, one-hot conditions, LeakyReLU slope 0.2,
discriminator dropout 0.4, adversarial BCE and Adam learning rates/betas.
The upstream MIT notice is retained in [UPSTREAM_LICENSE.txt](UPSTREAM_LICENSE.txt).

The new training implementation uses stable BCE-with-logits, detached fake
samples for discriminator updates, shuffled batches, train-only normalization,
reproducible seeds, checkpoints and held-out selection. These are documented
implementation/validation changes. They do not constitute an improved
architecture. The original real training dataset is absent upstream, so this
is an adaptation to Fluvius, not a numerical reproduction of the original paper.

The archive is the repository release `data/raw.zip`, SHA-256
`934cc871e9fe18dbe30530472816d711ece878ec7cd8725d9d8417bbdb389bd8`.
The source-definition workbook identifies CET/CEST local clock labels and
quarter-hour interval energy. Input data and large research outputs stay outside
Git. This repository makes no new claim about the source dataset's license.

`release.py`, `curation.py` and the initial `data.py`, together with their
corresponding tests, were reused from this repository's V2 commit
`1ca62d0fbe1d77f916f8904e812db7d720f8cf53`. Subsequent V3 ingestion validation
changes and the new hourly-aggregation step have dedicated tests. The original
split-allocation logic and identity-curation priority are preserved.

The V2 records inspected during design document substantial validation
exploration, exact-trace identity audits involving the original test arrays,
and no final test-performance evaluation. Those records are historical
statements; V3 cannot establish every prior user's data-access history. V3
therefore preserves the retained original test membership and records its own
test-performance accesses, rather than calling the test data entirely untouched.

All source-split and prepared-output hashes are stored in dataset manifests.
Runs record consumed input hashes, exact settings, seeds, runtime versions,
selection rows/noise/projection digest, optimizer state and completed epochs.
Reporting uses different noise seeds from checkpoint selection. Software tests,
resource pilots, validation experiments and final test experiments are separate
evidence categories.

Citation: Li, Y., Dong, B., and Qiu, Y. (2025). Conditional generative adversarial
network (cGAN) for generating building load profiles with photovoltaics and
electric vehicles. *Energy and Buildings*, 335, 115584.
https://doi.org/10.1016/j.enbuild.2025.115584
