#!/usr/bin/env bash
# Benchmark TensorFlow x PyTorch + teste de compatibilidade aeon (Keras 3) x pilha do VMM.
# Roda no WSL2 com GPU. Saida: results/bench_frameworks/
set -u
ROOT=/mnt/d/dev/tcc/tcc-agrupamento-series
OUT=$ROOT/results/bench_frameworks
mkdir -p "$OUT"
UV=$HOME/.local/bin/uv
TFPY=$HOME/venvs/tcc/bin/python
TORCHPY=$HOME/venvs/torch/bin/python
SP=$HOME/venvs/tcc/lib/python3.11/site-packages/nvidia
TF_LD=$(ls -d "$SP"/*/lib | paste -sd:)
export TF_CPP_MIN_LOG_LEVEL=2
log() { echo "[$(date -Is)] $*"; }

# 1) ambiente PyTorch separado (nao mexe no ambiente da linha de base)
if [ ! -x "$TORCHPY" ]; then
  log "criando ~/venvs/torch"
  $UV venv -q --python 3.11 "$HOME/venvs/torch"
  $UV pip install -q --python "$TORCHPY" numpy
  $UV pip install -q --python "$TORCHPY" torch --index-url https://download.pytorch.org/whl/cu126
fi
$UV pip freeze --python "$TORCHPY" > "$OUT/requirements-lock-torch.txt" 2>/dev/null

# 2) casos: mesmo modelo, mesmos dados, mesma GPU, um processo por caso
RES=$OUT/bench.jsonl
for L in 286 1024; do
  for model in fcn_ae bigru_tanh bigru_relu; do
    log "tf $model L=$L"
    LD_LIBRARY_PATH=$TF_LD:${LD_LIBRARY_PATH:-} timeout 1800 "$TFPY" "$ROOT/experimentos/bench/bench_tf.py" \
      --model $model --length $L --out "$RES" 2>>"$OUT/bench_stderr.log" | tail -1
    [ ${PIPESTATUS[0]} -ne 0 ] && echo "{\"framework\":\"tensorflow\",\"model\":\"$model\",\"length\":$L,\"status\":\"erro_processo\"}" >> "$RES"
    log "torch $model L=$L"
    timeout 1800 "$TORCHPY" "$ROOT/experimentos/bench/bench_torch.py" \
      --model $model --length $L --out "$RES" 2>>"$OUT/bench_stderr.log" | tail -1
    [ ${PIPESTATUS[0]} -ne 0 ] && echo "{\"framework\":\"pytorch\",\"model\":\"$model\",\"length\":$L,\"status\":\"erro_processo\"}" >> "$RES"
  done
done

# 3) compatibilidade
log "compatibilidade"
bash "$ROOT/experimentos/bench/compat.sh" > "$OUT/compat.log" 2>&1
log "FIM bench"
