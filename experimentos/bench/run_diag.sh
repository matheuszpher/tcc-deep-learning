#!/usr/bin/env bash
SP=$HOME/venvs/tcc/lib/python3.11/site-packages/nvidia
export LD_LIBRARY_PATH=$(ls -d "$SP"/*/lib | paste -sd:)
export TF_CPP_MIN_LOG_LEVEL=2
timeout 600 "$HOME/venvs/tcc/bin/python" /mnt/d/dev/tcc/tcc-agrupamento-series/experimentos/bench/diag_cudnn.py 2>/dev/null
echo "codigo $?"
