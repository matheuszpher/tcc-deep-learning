"""Benchmark TensorFlow/Keras: mesma carga do bench_torch.py, um caso por processo.

Modelos:
  fcn_ae      AEFCNNetwork do proprio aeon (latente 128), como na linha de base
  bigru_tanh  BiGRU(64) -> Dense(128) -> RepeatVector -> GRU(128) -> Dense(1), tanh (elegivel a cuDNN)
  bigru_relu  igual, com ativacao ReLU, como nos autoencoders recorrentes do aeon (sem cuDNN)
Saida: uma linha JSON anexada a --out.
"""

import argparse
import json
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np  # noqa: E402


def build(model, L, tf):
    if model == "fcn_ae":
        from aeon.networks import AEFCNNetwork

        enc, dec = AEFCNNetwork(latent_space_dim=128).build_network((L, 1))
        inp = tf.keras.layers.Input((L, 1))
        return tf.keras.Model(inp, dec(enc(inp)))
    act = "tanh" if model == "bigru_tanh" else "relu"
    inp = tf.keras.layers.Input((L, 1))
    h = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, activation=act))(inp)
    z = tf.keras.layers.Dense(128)(h)
    r = tf.keras.layers.RepeatVector(L)(z)
    r = tf.keras.layers.GRU(128, activation=act, return_sequences=True)(r)
    out = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1))(r)
    return tf.keras.Model(inp, out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--length", type=int, required=True)
    p.add_argument("--n", type=int, default=512)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--out", required=True)
    p.add_argument("--no-jit", action="store_true",
                   help="compile(jit_compile=False); o padrao do Keras 3 (e do aeon) e 'auto' (XLA)")
    a = p.parse_args()

    import tensorflow as tf
    import keras

    keras.utils.disable_interactive_logging()
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
    tf.keras.utils.set_random_seed(0)
    X = np.random.default_rng(0).normal(size=(a.n, a.length, 1)).astype(np.float32)

    rec = {"framework": "tensorflow", "version": tf.__version__, "keras": keras.__version__,
           "device": "GPU" if gpus else "CPU", "model": a.model, "length": a.length,
           "n": a.n, "epochs": a.epochs, "batch": 32,
           "jit_compile": False if a.no_jit else "auto"}
    try:
        t0 = time.perf_counter()
        m = build(a.model, a.length, tf)
        kw = {"jit_compile": False} if a.no_jit else {}
        m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse", **kw)
        rec["build_s"] = time.perf_counter() - t0
        rec["params_trainable"] = int(sum(np.prod(w.shape) for w in m.trainable_weights))

        class T(tf.keras.callbacks.Callback):
            def on_train_begin(self, logs=None):
                self.t = []

            def on_epoch_begin(self, e, logs=None):
                self.s = time.perf_counter()

            def on_epoch_end(self, e, logs=None):
                self.t.append(time.perf_counter() - self.s)

        cb = T()
        h = m.fit(X, X, batch_size=32, epochs=a.epochs, verbose=0, callbacks=[cb], shuffle=True)
        rec["first_epoch_s"] = cb.t[0]
        rec["epoch_median_s"] = float(np.median(cb.t[1:]))
        rec["total_s"] = time.perf_counter() - t0
        rec["final_loss"] = float(h.history["loss"][-1])
        if gpus:
            rec["peak_gpu_mb"] = tf.config.experimental.get_memory_info("GPU:0")["peak"] / 2**20
        rec["status"] = "ok"
    except Exception as e:  # noqa: BLE001
        rec["status"] = "erro"
        rec["error"] = " ".join(f"{type(e).__name__}: {e}".split())[:500]
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
