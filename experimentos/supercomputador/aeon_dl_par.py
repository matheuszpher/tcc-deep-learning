"""Treina UM par (conjunto, modelo) e grava tudo na pasta do par.

Chamado pelo agendador (aeon_dl_agendador.py), um subprocesso por par:
    python aeon_dl_par.py --config <out>/config.json --dataset Coffee --model AEFCNClusterer

Protocolo (CLAUDE.md, secao 4): treino+teste concatenados (rotulos so nas metricas),
NaN -> interpolacao linear, comprimentos diferentes -> Resizer("max"), z-norm por serie.
O autoencoder e treinado uma vez; o latente (model_.layers[1]) e salvo; para cada K roda
TimeSeriesKMeans euclidiano sobre o latente, igual ao aeon.

Saidas em <out>/pares/<modelo>/<conjunto>/:
    metricas.csv     formato longo (uma linha por K e metrica)
    curva_perda.csv  perda, taxa de aprendizado e tempo de cada epoca
    latente.npy      espaco latente (n_casos x dim)
    status.json      status final + informacoes de treino (escrito por ultimo)
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import platform
import shutil
import socket
import tempfile
import time
import traceback
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np  # noqa: E402

MODELS = [
    "AEFCNClusterer",
    "AEResNetClusterer",
    "AEDCNNClusterer",
    "AEBiGRUClusterer",
    "AEAttentionBiGRUClusterer",
    "AEDRNNClusterer",
]

LONG_COLS = ["dataset", "model", "k", "k_mode", "metric", "value", "seed", "n_epochs",
             "n_cases", "n_timepoints", "n_classes", "preprocessing"]

# Metricas gravadas para cada K (ver o dicionario no notebook).
METRICS = [
    "silhouette", "davies_bouldin", "calinski_harabasz",
    "ri", "ari", "nmi", "ami",
    "ri_train", "ari_train", "nmi_train", "ri_test", "ari_test", "nmi_test",
    "n_clusters_produced", "n_empty_components", "degenerate",
    "cluster_size_min", "cluster_size_max", "kmeans_inertia", "kmeans_n_iter",
    "recon_loss", "recon_mse", "elbo",
]


# ----------------------------------------------------------------------- dados
def _interp_nan(v: np.ndarray) -> tuple[np.ndarray, str]:
    mask = np.isnan(v)
    if not mask.any():
        return v, ""
    if mask.all():
        return np.zeros_like(v), "all_nan"
    idx = np.arange(v.size)
    v = v.copy()
    v[mask] = np.interp(idx[mask], idx[~mask], v[~mask])  # extremos: valor valido mais proximo
    return v, "nan"


def preprocess(series: list[np.ndarray]) -> tuple[np.ndarray, str]:
    """NaN -> interpolacao linear; comprimentos diferentes -> Resizer("max"); z-norm."""
    from aeon.transformations.collection.unequal_length import Resizer

    if series[0].ndim != 2 or series[0].shape[0] != 1:
        raise ValueError(f"formato inesperado: {series[0].shape}")
    n_nan = n_all_nan = 0
    clean = []
    for s in series:
        ch, flag = [], ""
        for c in range(s.shape[0]):
            v, f = _interp_nan(s[c])
            ch.append(v)
            flag = f or flag
        n_nan += flag == "nan"
        n_all_nan += flag == "all_nan"
        clean.append(np.stack(ch))
    lengths = np.array([s.shape[-1] for s in clean])
    steps = []
    if n_nan or n_all_nan:
        steps.append(f"interp_linear_nan(series={n_nan},all_nan_to_zero={n_all_nan})")
    if lengths.min() != lengths.max():
        X = Resizer(resized_length="max").fit_transform(clean)
        steps.append(f"resize_max(len_min={lengths.min()},len_max={lengths.max()})")
    else:
        X = np.stack(clean)
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=2, keepdims=True)
    sd = X.std(axis=2, keepdims=True)
    sd[sd == 0] = 1.0
    X = ((X - mu) / sd).astype(np.float32)
    steps.append("znorm_per_series")
    return X, ";".join(steps)


def load_dataset(name: str, data_dir: Path) -> dict:
    """Carrega (com cache .npz), concatena treino+teste e pre-processa.

    O cache guarda n_train para permitir as metricas restritas ao split original.
    Gravacao atomica: varios processos podem carregar o mesmo conjunto ao mesmo tempo.
    """
    cache_dir = data_dir / "processed"
    cache = cache_dir / f"{name}.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=False)
        return {"X": d["X"], "y": d["y"], "n_train": int(d["n_train"]),
                "preprocessing": str(d["preprocessing"])}

    from aeon.datasets import load_classification

    raw = data_dir / "ucr"
    raw.mkdir(parents=True, exist_ok=True)
    Xtr, ytr = load_classification(name, split="train", extract_path=str(raw))
    Xte, yte = load_classification(name, split="test", extract_path=str(raw))
    series = [np.asarray(s, dtype=np.float64) for s in list(Xtr) + list(Xte)]
    y = np.concatenate([np.asarray(ytr), np.asarray(yte)]).astype(str)
    X, prep = preprocess(series)
    prep = "concat_train_test;" + prep
    cache_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".npz", dir=cache_dir)
    os.close(fd)
    np.savez(tmp, X=X, y=y, n_train=np.array(len(Xtr)), preprocessing=np.array(prep))
    os.replace(tmp, cache)
    return {"X": X, "y": y, "n_train": len(Xtr), "preprocessing": prep}


# ---------------------------------------------------------------------- modelo
def _callbacks():
    """Mesmo ReduceLROnPlateau do padrao do aeon + registro por epoca.

    Quando recebe callbacks, o aeon acrescenta o proprio ModelCheckpoint(monitor="loss",
    save_best_only), que recarrega o modelo de menor perda no fim, como no padrao.
    """
    import tensorflow as tf

    class EpochLog(tf.keras.callbacks.Callback):
        def on_train_begin(self, logs=None):
            self.rows = []

        def on_epoch_begin(self, epoch, logs=None):
            self._t = time.perf_counter()

        def on_epoch_end(self, epoch, logs=None):
            dt = time.perf_counter() - self._t
            try:
                lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
            except Exception:  # noqa: BLE001
                lr = float("nan")
            self.rows.append({"epoch": epoch + 1, "loss": float((logs or {}).get("loss", np.nan)),
                              "learning_rate": lr, "epoch_time_s": dt})

    log = EpochLog()
    rlr = tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=50,
                                               min_lr=0.0001)
    return [rlr, log], log


def _check_encoder(clst, X: np.ndarray, latent: np.ndarray) -> str:
    """Confere que model_.layers[1] e o codificador (mesma checagem da linha de base)."""
    Xk = X.transpose(0, 2, 1)
    layers = clst.model_.layers
    sub = Xk[: min(256, len(Xk))]
    full = clst.model_.predict(sub, verbose=0)
    comp = layers[2].predict(layers[1].predict(sub, verbose=0), verbose=0)
    recon_ok = bool(np.allclose(full, comp, atol=1e-5))
    pred_ok = bool(np.array_equal(clst.predict(X), clst.estimator_.predict(latent)))
    ok = len(layers) == 3 and recon_ok and pred_ok
    return (f"{'OK' if ok else 'FALHOU'} layers={[ly.name for ly in layers]} "
            f"dec(enc(x))==model(x):{recon_ok} predict_igual:{pred_ok}")


def train_and_extract(model_name: str, X: np.ndarray, n_epochs: int, seed: int):
    import aeon.clustering.deep_learning as dl
    import tensorflow as tf

    shm = "/dev/shm" if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK) else None
    tmp = tempfile.mkdtemp(prefix="aeon_ckpt_", dir=shm)
    try:
        cbs, log = _callbacks()
        clst = getattr(dl, model_name)(
            callbacks=cbs, n_epochs=n_epochs, random_state=seed, batch_size=32,
            save_best_model=False, save_last_model=False,
            file_path=tmp.rstrip("/\\") + "/", verbose=False)
        t0 = time.perf_counter()
        clst.fit(X)
        train_t = time.perf_counter() - t0

        hist = clst.history.history if hasattr(clst.history, "history") else clst.history
        losses = [float(v) for v in hist["loss"]]
        Xk = X.transpose(0, 2, 1)
        t0 = time.perf_counter()
        latent = clst.model_.layers[1].predict(Xk, verbose=0)
        latent = np.asarray(latent).reshape(len(X), -1).astype(np.float32)
        latent_t = time.perf_counter() - t0
        check = _check_encoder(clst, X, latent)
        recon_mse = float(np.mean((clst.model_.predict(Xk, verbose=0) - Xk) ** 2))

        mem_peak = ""
        if tf.config.list_physical_devices("GPU"):
            try:
                mem_peak = round(tf.config.experimental.get_memory_info("GPU:0")["peak"] / 2**20, 1)
            except Exception:  # noqa: BLE001
                pass
        times = [r["epoch_time_s"] for r in log.rows]
        info = {
            "n_params": int(clst.model_.count_params()),
            "latent_dim": int(latent.shape[1]),
            "train_time_s": round(train_t, 3),
            "latent_time_s": round(latent_t, 3),
            "epochs_run": len(losses),
            "best_epoch": int(np.argmin(losses)) + 1,
            "first_epoch_s": round(times[0], 3) if times else "",
            "epoch_median_s": round(float(np.median(times[1:])), 4) if len(times) > 1 else "",
            "recon_loss_min": min(losses),
            "recon_loss_last": losses[-1],
            "recon_mse": recon_mse,
            "lr_final": log.rows[-1]["learning_rate"] if log.rows else "",
            "gpu_mem_peak_mb": mem_peak,
            "encoder_check": check,
        }
        return latent, info, log.rows
    finally:
        tf.keras.backend.clear_session()
        gc.collect()
        shutil.rmtree(tmp, ignore_errors=True)


# -------------------------------------------------------------------- busca K
def _external(y, lab, M) -> tuple[float, float, float]:
    return (M.rand_score(y, lab), M.adjusted_rand_score(y, lab),
            M.normalized_mutual_info_score(y, lab))


def k_search(latent, y, n_train, ks, k_mode, seed, base, recon_loss, recon_mse):
    from aeon.clustering import TimeSeriesKMeans
    from sklearn import metrics as M

    rows, notes = [], []

    def emit(k, vals):
        for m in METRICS:
            v = vals.get(m, float("nan"))
            rows.append({**base, "k": k, "k_mode": k_mode, "metric": m,
                         "value": "" if v == "" else repr(float(v))})

    for k in ks:
        km = TimeSeriesKMeans(n_clusters=k, distance="euclidean", averaging_method="mean",
                              random_state=seed)
        try:
            lab = np.asarray(km.fit_predict(latent))
        except Exception as e:  # noqa: BLE001  (ex.: latente degenerado)
            notes.append(f"K={k}: k-means falhou ({type(e).__name__}: {e}); metricas=NaN")
            emit(k, {"recon_loss": recon_loss, "recon_mse": recon_mse, "elbo": ""})
            continue
        sizes = np.bincount(np.unique(lab, return_inverse=True)[1])
        n_found = len(sizes)
        vals = {"n_clusters_produced": n_found, "n_empty_components": k - n_found,
                "degenerate": float(n_found < 2),
                "cluster_size_min": sizes.min(), "cluster_size_max": sizes.max(),
                "kmeans_inertia": getattr(km, "inertia_", float("nan")),
                "kmeans_n_iter": getattr(km, "n_iter_", float("nan"))}
        if 2 <= n_found <= len(lab) - 1:
            vals["silhouette"] = M.silhouette_score(latent, lab, metric="euclidean")
            vals["davies_bouldin"] = M.davies_bouldin_score(latent, lab)
            vals["calinski_harabasz"] = M.calinski_harabasz_score(latent, lab)
        else:
            notes.append(f"K={k}: {n_found} grupo(s) distinto(s); silhueta/DB/CH=NaN")
        if n_found < k:
            notes.append(f"K={k}: k-means devolveu {n_found} grupos")
        vals["ri"], vals["ari"], vals["nmi"] = _external(y, lab, M)
        vals["ami"] = M.adjusted_mutual_info_score(y, lab)
        # Mesmo agrupamento (ajustado no conjunto unido), avaliado so nas series que vieram
        # do treino e so nas que vieram do teste no split original do UCR.
        if 0 < n_train < len(y):
            vals["ri_train"], vals["ari_train"], vals["nmi_train"] = _external(
                y[:n_train], lab[:n_train], M)
            vals["ri_test"], vals["ari_test"], vals["nmi_test"] = _external(
                y[n_train:], lab[n_train:], M)
        vals.update(recon_loss=recon_loss, recon_mse=recon_mse, elbo="")
        emit(k, vals)
    return rows, notes


def ks_for(k_mode, n_cases, n_classes, cfg):
    if k_mode == "grid":
        return [k for k in range(cfg["k_min"], cfg["k_max"] + 1, cfg["k_step"]) if k < n_cases]
    return [n_classes]


# ----------------------------------------------------------------------- main
def pair_dir(out: Path, dataset: str, model: str) -> Path:
    return out / "pares" / model / dataset


def _write_csv(path: Path, cols, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def _write_status(d: Path, status: dict):
    tmp = d / "status.json.tmp"
    tmp.write_text(json.dumps(status, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, d / "status.json")


def run_pair(cfg: dict, dataset: str, model: str) -> int:
    out = Path(cfg["out"])
    d = pair_dir(out, dataset, model)
    d.mkdir(parents=True, exist_ok=True)
    for f in ("status.json", "metricas.csv", "curva_perda.csv"):
        (d / f).unlink(missing_ok=True)

    import aeon
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:  # noqa: BLE001
            pass
    gpu_name = ""
    if gpus:
        try:
            gpu_name = tf.config.experimental.get_device_details(gpus[0]).get("device_name", "")
        except Exception:  # noqa: BLE001
            pass
    status = {
        "timestamp_inicio": datetime.now().isoformat(timespec="seconds"),
        "dataset": dataset, "model": model, "status": "em_andamento",
        "seed": cfg["seed"], "n_epochs": cfg["epochs"],
        "k_grid": f"{cfg['k_min']}:{cfg['k_max']}:{cfg['k_step']}",
        "host": socket.gethostname(), "device": "GPU" if gpus else "CPU",
        "gpu_name": gpu_name, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "python": platform.python_version(), "tensorflow": tf.__version__,
        "aeon": aeon.__version__,
    }
    try:
        t0 = time.perf_counter()
        data = load_dataset(dataset, Path(cfg["data_dir"]))
        X, y, n_train = data["X"], data["y"], data["n_train"]
        n_cases, _, n_tp = X.shape
        n_classes = len(np.unique(y))
        status.update(load_time_s=round(time.perf_counter() - t0, 3), n_cases=n_cases,
                      n_train=n_train, n_test=n_cases - n_train, n_timepoints=n_tp,
                      n_classes=n_classes, preprocessing=data["preprocessing"])
        print(f"{dataset}/{model}: n={n_cases} m={n_tp} classes={n_classes} "
              f"[{status['device']} {gpu_name}]", flush=True)

        latent, info, curve = train_and_extract(model, X, cfg["epochs"], cfg["seed"])
        status.update(info)
        np.save(d / "latente.npy", latent)
        _write_csv(d / "curva_perda.csv",
                   ["dataset", "model", "epoch", "loss", "learning_rate", "epoch_time_s"],
                   [{"dataset": dataset, "model": model, **r} for r in curve])
        print(f"  treino {info['train_time_s']:.1f}s, {info['epochs_run']} epocas, "
              f"perda min {info['recon_loss_min']:.4g} | {info['encoder_check']}", flush=True)

        base = {"dataset": dataset, "model": model, "seed": cfg["seed"],
                "n_epochs": cfg["epochs"], "n_cases": n_cases, "n_timepoints": n_tp,
                "n_classes": n_classes, "preprocessing": data["preprocessing"]}
        rows, notes, n_k = [], [], 0
        t0 = time.perf_counter()
        for km in cfg["k_modes"]:
            ks = ks_for(km, n_cases, n_classes, cfg)
            r, nt = k_search(latent, y, n_train, ks, km, cfg["seed"], base,
                             info["recon_loss_min"], info["recon_mse"])
            rows += r
            notes += [f"[{km}] {s}" for s in nt]
            n_k += len(ks)
        status["ksearch_time_s"] = round(time.perf_counter() - t0, 3)
        status["n_k"] = n_k
        _write_csv(d / "metricas.csv", LONG_COLS, rows)
        status.update(status="ok", message=" | ".join(notes)[:4000])
        print(f"  busca de K: {n_k} valores em {status['ksearch_time_s']:.1f}s", flush=True)
        rc = 0
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc(limit=4)
        status.update(status="erro",
                      message=" ".join(f"{type(e).__name__}: {e} || {tb}".split())[:4000])
        print(f"  ERRO {type(e).__name__}: {e}", flush=True)
        rc = 1
    status["timestamp_fim"] = datetime.now().isoformat(timespec="seconds")
    _write_status(d, status)
    return rc


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--model", required=True, choices=MODELS)
    a = p.parse_args(argv)
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    import keras

    keras.utils.disable_interactive_logging()
    return run_pair(cfg, a.dataset, a.model)


if __name__ == "__main__":
    raise SystemExit(main())
