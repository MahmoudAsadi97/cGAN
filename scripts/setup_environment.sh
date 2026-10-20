#!/usr/bin/env bash
# Recreate the pinned CPU development environment without running experiments.
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd -- "$PROJECT_ROOT"

if ! command -v uv >/dev/null 2>&1; then
    echo 'uv is required on PATH; see docs/ENVIRONMENT.md.' >&2
    exit 1
fi

# Keep installation state inside this checkout, including Python downloads and
# temporary build files. These paths are already covered by .gitignore.
export UV_CACHE_DIR="$PROJECT_ROOT/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$PROJECT_ROOT/.cache/uv-python"
export UV_PYTHON_BIN_DIR="$PROJECT_ROOT/.cache/uv-bin"
export UV_LINK_MODE=copy
export TMPDIR="$PROJECT_ROOT/.cache/tmp"
mkdir -p -- "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$UV_PYTHON_BIN_DIR" "$TMPDIR"

VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"
if [[ -e "$PROJECT_ROOT/.venv" ]]; then
    if [[ ! -x "$VENV_PYTHON" ]]; then
        echo 'Existing .venv has no Python executable; inspect it before retrying.' >&2
        exit 1
    fi
    "$VENV_PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Existing .venv must use Python 3.12"'
else
    uv venv --python 3.12 "$PROJECT_ROOT/.venv"
fi

# The requirements contain exact pins, including torch's +cpu build. Search
# both package indexes because the CPU index also hosts other package names.
uv pip sync --python "$VENV_PYTHON" --index-strategy unsafe-best-match --strict "$PROJECT_ROOT/requirements-cpu.txt"
uv pip install --python "$VENV_PYTHON" --no-deps --no-build-isolation --editable "$PROJECT_ROOT"
uv pip check --python "$VENV_PYTHON"
"$VENV_PYTHON" - <<'PY'
import sys
import torch
import fluvius_cgan

assert sys.version_info[:2] == (3, 12), 'Python 3.12 is required for this environment'
assert torch.version.cuda is None, 'Expected the pinned CPU-only PyTorch build'
print(f'Python {sys.version.split()[0]}; PyTorch {torch.__version__}; package import succeeded')
PY

echo 'Setup finished. Run .venv/bin/python -m pytest -W error for the synthetic test suite.'
