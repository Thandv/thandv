#!/usr/bin/env bash
# Build a single-file `thandv` binary with PyInstaller.
# The binary still needs Ollama installed on the target machine; it bundles
# only the Python runtime + thandv package.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

python3 -m pip install --upgrade ".[build]"

rm -rf build dist
python3 -m PyInstaller \
    --onefile \
    --name thandv \
    --hidden-import thandv.cli \
    --collect-submodules thandv \
    --add-data "thandv/skills:thandv/skills" \
    -p . \
    thandv/__main__.py

echo
echo "binary: $ROOT/dist/thandv"
file "$ROOT/dist/thandv" || true
