# CPU development environment

The Phase 1 cloud machine has a quota of **4 CPU cores**, **32 GiB RAM**, and
**no GPU**. The CPU build of PyTorch is intentional. Parallel CPU training must
respect this quota; the default setup does not launch experiments.

Python **3.12** is the reproducibility target for this environment. The pinned
package versions are in [requirements-cpu.txt](../requirements-cpu.txt), including
`torch==2.8.0+cpu`. The environment used `uv 0.12.23`; the CI workflow pins that
version. `uv` must already be available on `PATH` when running local setup.

From the project checkout:

```bash
bash scripts/setup_environment.sh
.venv/bin/python -m pytest -W error
```

The script locates the checkout from its own path, creates `.venv` with
`uv venv --python 3.12` if needed, synchronizes the pinned CPU requirements, and
installs the project in editable mode with `--no-deps --no-build-isolation`.
The requirements synchronization uses `--index-strategy unsafe-best-match` to
compare the PyPI and PyTorch CPU indexes: the CPU index also contains package
names used by other dependencies. Every requested version is pinned, and the
explicit `torch==2.8.0+cpu` requirement selects the CPU build without GPU backend
autodetection. The build backend is supplied by the pinned requirements. It then checks
installed dependency compatibility and verifies Python, package import, and the
CPU-only PyTorch build. An existing Python 3.12 virtual environment is reused;
an existing virtual environment with another interpreter is rejected without
being deleted. Synchronizing removes packages absent from the pinned list
before reinstalling the editable project.

The script sets absolute paths inside the checkout:

| State | Path |
| --- | --- |
| Package/download cache (`UV_CACHE_DIR`) | `.cache/uv` |
| Managed Python installations (`UV_PYTHON_INSTALL_DIR`) | `.cache/uv-python` |
| Managed Python executable links (`UV_PYTHON_BIN_DIR`) | `.cache/uv-bin` |
| Build temporary files (`TMPDIR`) | `.cache/tmp` |
| Virtual environment | `.venv` |

Existing `.gitignore` rules exclude `.cache`, `.venv`, bytecode, test caches,
build outputs, and package metadata. The script does not change Git branches or
repository configuration. Setup requires package downloads from PyPI and the
PyTorch CPU wheel index. If Python 3.12 is unavailable, uv may also download a
managed interpreter. TLS and artifact verification remain enabled.

No dataset, prepared array, model checkpoint, training run, evaluation, or
publication is produced by setup. Source data and large experiment outputs
belong in the ignored `.cache`, `data`, `runs`, or `artifacts` directories.
The actual data preparation and experimental commands are separate from setup;
see [PHASE1_PROTOCOL.md](PHASE1_PROTOCOL.md) and [V3_DESIGN.md](V3_DESIGN.md).

The [CPU tests workflow](../.github/workflows/tests.yml) runs on Ubuntu with
Python 3.12, installs the same pinned requirements and editable package, then
runs `pytest -W error`. Tests use synthetic fixtures and require no real source
data or GPU. The workflow runs for pushes to `V3`, pull requests, or manual
invocation. A passing CI run establishes software checks for that revision;
scientific results require the separate Phase 1 experiment and review.

These repository instructions are a reusable cloud setup draft. Adding them
does not publish a cloud environment or establish validation in a fresh cloud
machine. Publication, current-machine verification, and a later fresh-machine
verification are separate steps. The setup script was executed successfully in the current cloud checkout;
package compatibility, CPU-only PyTorch, and package import checks passed. A
standalone wheel was also installed into a separate virtual environment and
its imports and command entry point were checked outside the source tree.
