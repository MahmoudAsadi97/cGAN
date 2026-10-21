# Running the Phase 1 GPU pilot

The [notebook](../notebooks/phase1_development_gpu.ipynb) runs a single
1,000-update global-standardization pilot on an available Colab or Kaggle GPU.
It measures the total training-process time, including validation, and preserves
the development metrics and provenance. This is a hardware trial, not a final
seed study or a publication-readiness assessment.

[Open in Colab](https://colab.research.google.com/github/MahmoudAsadi97/cGAN/blob/V3/notebooks/phase1_development_gpu.ipynb).
For Kaggle, import the notebook from GitHub or upload the `.ipynb` file, enable
internet, and select an available GPU accelerator. Both providers have variable
free availability and account limits; no paid resource is created by this
notebook. See [compute options](COMPUTE_OPTIONS.md) for the Azure comparison.

## Inputs

Use the prepared `inner24-transfer.zip` archive. It is 33,752,550 bytes and its
SHA-256 is:

```text
3fd420cdb8c611f03f3252f7a10f14cae66e8018dcdf92b23e354bbdc2df618f
```

The local copy is `artifacts/phase1-development/inner24-transfer.zip`, excluded
from Git. It contains only `train.npz`, `val.npz`, `scaler.json`, and
`metadata.json` from the fixed inner split: 1,333 fitting meters and 332
validation meters. Each member must match the hashes in
[`NORMALIZATION_SCREEN_PLAN.json`](../reports/phase1-development/NORMALIZATION_SCREEN_PLAN.json).
The notebook verifies every member before training and does not read the
original benchmark VAL or TEST.

Upload the archive to the notebook runtime, or attach it as a private Kaggle
dataset if appropriate. Set `DATA_ARCHIVE` in the first code cell to its actual
location, then run the cells in order. Publicly publishing a notebook does not
require publicly publishing this transfer archive or the output checkpoints.

To recreate the transfer archive on the original development machine from its
already prepared data, run the following in the repository root. It refuses to
replace an existing archive and verifies the source files first:

```python
from pathlib import Path
import hashlib
import json
import zipfile

plan = json.loads(Path("reports/phase1-development/NORMALIZATION_SCREEN_PLAN.json").read_text())
source = Path(plan["data"])
output = Path("artifacts/phase1-development/inner24-transfer.zip")
output.parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_STORED) as archive:
    for name, expected in plan["data_hashes"].items():
        content = (source / name).read_bytes()
        assert hashlib.sha256(content).hexdigest() == expected, name
        member = zipfile.ZipInfo(name, date_time=(2026, 10, 21, 0, 0, 0))
        member.external_attr = 0o100644 << 16
        archive.writestr(member, content)
print(hashlib.sha256(output.read_bytes()).hexdigest())
```

The public release can be reconstructed with the audited preparation commands
in the [README](../README.md). That is a separate data-preparation workflow.
The GPU pilot deliberately accepts the fixed transfer artifact so a different
preparation path, package version, or metadata path cannot silently replace the
declared inputs. Do not bypass an input mismatch by changing expected hashes.

## Environment and outputs

The notebook clones only `V3` into its own `cGAN` directory and records the exact
commit. Existing checkouts are verified and left on their current contents; the
notebook does not pull, reset, commit, or push. It creates an isolated project
environment that reuses the provider's CUDA PyTorch, requiring PyTorch
`>=2.8,<3`, Python `>=3.11`, and an available GPU. Numerical package versions
match the CPU experiment. The recorded package list and GPU/runtime details
remain necessary because provider CUDA builds can differ.

The pinned CPU setup remains unchanged. Installing `requirements-cpu.txt` in a
GPU notebook would select the CPU PyTorch build and is not part of this workflow.
If the provider's PyTorch is incompatible, choose a compatible runtime; this
notebook does not replace the provider's GPU driver or install a different
PyTorch CUDA distribution automatically.

Run directories include the timestamped attempt ID. Export the verified ZIP
from the final cells before the session ends. It contains configs, source/data
receipts, runtime versions, logs, histories and checkpoints. It excludes the
dataset and virtual environment. Download from Colab's file browser, or save a
Kaggle notebook version with outputs and download through its Output panel.
Keep failed attempts and their logs too.

Optional continuation is disabled by default. When enabled, it extends the
completed 1,000-update pilot to a total of 6,000 updates in a new directory,
preserving the first run. The current continuation path requires a completed
run at a scheduled validation boundary; recovery from an arbitrary interrupted
session or partially written checkpoint is not established. A changed device
or runtime must not be treated as an exact continuation of the same trajectory.

## Validation status

This environment has no GPU or Colab/Kaggle session. Notebook structure, Python
syntax and the fixed input archive can be checked locally; end-to-end CUDA
execution and the provider interface must still be checked on an actual GPU.
The notebook is saved without invented execution outputs.

For that verification, select a GPU, upload the archive, set its path, and run
all cells with the default 1,000-update budget and continuation disabled. The
pilot must report `completed`, record `device: cuda`, and produce an export ZIP
whose member hashes verify. Preserve any deterministic-kernel or compatibility
failure rather than changing the training settings without a recorded new plan.
CPU/GPU outputs are not assumed to match bit for bit. Compare identical budgets
and validation schedules when measuring speedup; do not extrapolate a short
pilot directly into a promised publication-study cost.
