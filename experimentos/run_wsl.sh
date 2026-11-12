#!/usr/bin/env bash
# Roda o script no WSL2 (Ubuntu) com GPU. Uso: bash experimentos/run_wsl.sh [args...]
VENV=${VENV:-$HOME/venvs/tcc}
SP=$VENV/lib/python3.11/site-packages/nvidia
# O TF 2.21 nao encontra libcusolver.so.11 sozinho; expor todas as libs CUDA do pip.
export LD_LIBRARY_PATH=$(ls -d "$SP"/*/lib | paste -sd:):$LD_LIBRARY_PATH
export TF_CPP_MIN_LOG_LEVEL=2
cd "$(dirname "$0")/.."
exec "$VENV/bin/python" -u experimentos/aeon_dl_baseline.py "$@"
