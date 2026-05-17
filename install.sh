#!/usr/bin/env bash
# Thandv installer.
# Installs Ollama (if missing), pulls the best model for this host, and
# installs the `thandv` CLI from source.
set -euo pipefail

say()  { printf "\033[1;36m[thandv]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[thandv]\033[0m %s\n" "$*"; }
die()  { printf "\033[1;31m[thandv]\033[0m %s\n" "$*" >&2; exit 1; }

UNAME_S="$(uname -s)"
UNAME_M="$(uname -m)"
say "host: $UNAME_S/$UNAME_M"

# --- Python -----------------------------------------------------------------
command -v python3 >/dev/null || die "python3 not found. Install Python 3.10+ first."
PY_VER="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
say "python: $PY_VER"

# --- Ollama -----------------------------------------------------------------
if ! command -v ollama >/dev/null; then
    say "installing ollama..."
    if [[ "$UNAME_S" == "Darwin" ]]; then
        if command -v brew >/dev/null; then
            brew install ollama
        else
            die "homebrew not found. Install from https://brew.sh or grab ollama from https://ollama.com/download"
        fi
    elif [[ "$UNAME_S" == "Linux" ]]; then
        curl -fsSL https://ollama.com/install.sh | sh
    else
        die "unsupported OS for auto-install: $UNAME_S. See https://ollama.com/download"
    fi
else
    say "ollama: $(ollama --version 2>&1 | head -1)"
fi

# --- Thandv CLI -------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
say "installing thandv from $SCRIPT_DIR"
python3 -m pip install --user --upgrade "$SCRIPT_DIR"

# --- Pick & pull model ------------------------------------------------------
MODEL="$(python3 -c 'from thandv.runtime import detect_host, pick_model; print(pick_model(detect_host()))')"
say "selected model: $MODEL"

if ! pgrep -x ollama >/dev/null && ! pgrep -f "ollama serve" >/dev/null; then
    warn "ollama is not running. Start it with: ollama serve"
fi

say "pulling $MODEL (this can take a while on first run)..."
ollama pull "$MODEL" || warn "model pull failed; you can retry with: ollama pull $MODEL"

say "pulling nomic-embed-text (used by RAG; ~270MB)..."
ollama pull nomic-embed-text || warn "embed model pull failed; RAG will be disabled until you run: ollama pull nomic-embed-text"

say "done. Try: thandv doctor && thandv chat"
