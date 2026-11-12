#!/usr/bin/env bash
# Pode o aeon 1.6.0 (Keras 3) conviver com a pilha do VMM oficial (TF 2.15 + tensorflow-probability 0.23)?
set -u
ROOT=/mnt/d/dev/tcc/tcc-agrupamento-series
OUT=$ROOT/results/bench_frameworks
UV=$HOME/.local/bin/uv
export TF_CPP_MIN_LOG_LEVEL=2

echo "== Teste A: resolver aeon==1.6.0 + tensorflow 2.15 + tensorflow-probability 0.23 (pilha do VMM)"
printf 'aeon==1.6.0\ntensorflow==2.15.*\ntensorflow-probability==0.23.0\n' > /tmp/compat_a.in
$UV pip compile -q --python-version 3.11 /tmp/compat_a.in -o /tmp/compat_a.txt 2>&1 | tail -15
echo "codigo: ${PIPESTATUS[0]}"

echo; echo "== Teste B: ambiente novo com aeon 1.6.0 + TF atual + tensorflow-probability + tf-keras"
rm -rf "$HOME/venvs/compat"
$UV venv -q --python 3.11 "$HOME/venvs/compat"
PY=$HOME/venvs/compat/bin/python
$UV pip install -q --python "$PY" "aeon==1.6.0" tensorflow tensorflow-probability tf-keras 2>&1 | tail -5
$UV pip freeze --python "$PY" | grep -iE '^(aeon|tensorflow|tensorflow-probability|tf-keras|keras)=='

cat > /tmp/compat_b.py <<'PY'
import os, json, sys
mode = sys.argv[1]
res = {"modo": mode}
try:
    import tensorflow as tf, keras
    res["tf"], res["keras"] = tf.__version__, keras.__version__
    res["tf.keras"] = tf.keras.__version__ if hasattr(tf.keras, "__version__") else str(tf.keras)
except Exception as e:
    res["tf_erro"] = repr(e)[:300]
try:
    from aeon.networks import AEFCNNetwork
    enc, dec = AEFCNNetwork(latent_space_dim=8).build_network((32, 1))
    res["aeon_rede"] = "ok"
except Exception as e:
    res["aeon_rede"] = "erro: " + " ".join(repr(e).split())[:300]
try:
    import tensorflow_probability as tfp
    tfpl = tfp.layers
    d = 4
    inp = tf.keras.layers.Input((16,))
    h = tf.keras.layers.Dense(tfpl.MultivariateNormalTriL.params_size(d))(inp)
    q = tfpl.MultivariateNormalTriL(d)(h)
    m = tf.keras.Model(inp, q)
    res["tfp"] = tfp.__version__
    res["tfp_layers_em_tf_keras"] = "ok"
except Exception as e:
    res["tfp_layers_em_tf_keras"] = "erro: " + " ".join(repr(e).split())[:300]
print(json.dumps(res))
PY
echo; echo "-- B1: padrao (tf.keras = Keras 3)"
$PY /tmp/compat_b.py keras3 2>/dev/null | tail -1 | tee -a "$OUT/compat.jsonl"
echo; echo "-- B2: TF_USE_LEGACY_KERAS=1 (tf.keras = tf-keras 2.x, exigido pelo tfp.layers)"
TF_USE_LEGACY_KERAS=1 $PY /tmp/compat_b.py legacy_keras2 2>/dev/null | tail -1 | tee -a "$OUT/compat.jsonl"
