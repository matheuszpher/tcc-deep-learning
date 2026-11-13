#!/usr/bin/env bash
# Valida o notebook em WSL: executa de ponta a ponta com TESTE_RAPIDO=1 numa pasta limpa.
set -e
VENV=~/venvs/tcc
LIBS=$(ls -d $VENV/lib/python3*/site-packages/nvidia/*/lib 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="$LIBS$LD_LIBRARY_PATH"
$VENV/bin/python -c "import nbclient, ipykernel" 2>/dev/null || ~/.local/bin/uv pip install -q --python $VENV/bin/python nbclient nbformat ipykernel
W=/tmp/valida_nb; [ "$MANTER" = 1 ] || rm -rf $W; mkdir -p $W
cp "$1" $W/nb.ipynb
cd $W
export TESTE_RAPIDO=1 AEON_HPC_DIR=$W/trabalho MODELOS_CPU="${MODELOS_CPU:-}"
$VENV/bin/python - <<'PY'
import nbformat, time
from nbclient import NotebookClient
nb = nbformat.read("nb.ipynb", as_version=4)
t = time.time()
NotebookClient(nb, timeout=None, kernel_name="python3", resources={"metadata": {"path": "."}}).execute()
nbformat.write(nb, "nb_executado.ipynb")
print(f"NOTEBOOK OK em {time.time()-t:.0f}s")
PY
