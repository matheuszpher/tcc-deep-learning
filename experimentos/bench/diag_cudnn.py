"""Diagnostico: a GRU do Keras 3 (backend TF) usa o kernel cuDNN nesta maquina?

Testa backend.cudnn_ok e chama backend.gru diretamente (tanh e relu). Se o caminho cuDNN
levanta NotImplementedError, o Keras cai silenciosamente no laco generico (lento).
"""

import json
import os
import time
import traceback

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from keras.src import backend  # noqa: E402
import importlib  # noqa: E402

tfrnn = importlib.import_module("keras.src.backend.tensorflow.rnn")

res = {"tf": tf.__version__, "gpus": [d.name for d in tf.config.list_physical_devices("GPU")],
       "cuda_build": tf.test.is_built_with_cuda()}
for act in ("tanh", "relu"):
    r = {}
    try:
        r["cudnn_ok"] = bool(backend.cudnn_ok(act, "sigmoid", False, True, reset_after=True))
    except Exception as e:  # noqa: BLE001
        r["cudnn_ok"] = "erro: " + repr(e)[:200]
    layer = tf.keras.layers.GRU(64, activation=act, return_sequences=True)
    x = tf.random.normal((32, 286, 8))
    layer(x)  # build
    c = layer.cell
    try:
        t = time.perf_counter()
        out = tfrnn.gru(x, tf.zeros((32, 64)), None, kernel=c.kernel, recurrent_kernel=c.recurrent_kernel,
                        bias=c.bias, activation=c.activation, recurrent_activation=c.recurrent_activation,
                        return_sequences=True, go_backwards=False, unroll=False, reset_after=True)
        r["backend_gru"] = f"ok ({time.perf_counter() - t:.3f}s)"
    except NotImplementedError as e:
        r["backend_gru"] = "NotImplementedError (cai no laco generico): " + str(e)[:200]
    except Exception as e:  # noqa: BLE001
        r["backend_gru"] = "erro: " + " ".join(traceback.format_exception_only(e)[-1].split())[:300]
    r["layer.supports_jit"] = getattr(layer, "supports_jit", None)
    res[act] = r
print(json.dumps(res, indent=1))
