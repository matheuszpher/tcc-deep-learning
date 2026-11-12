"""Linha de base: agrupadores profundos do aeon nos conjuntos univariados do UCR.

Protocolo (ver CLAUDE.md, secao 4):
- treino e teste concatenados; rotulos usados apenas nas metricas externas;
- NaN: interpolacao linear dentro de cada serie, extremos com o valor valido mais proximo;
- comprimentos diferentes: Resizer(resized_length="max") (reamostragem linear);
- normalizacao z por serie (desvio 0 -> 1);
- cada (conjunto, modelo) e treinado uma vez; o latente (model_.layers[1]) e salvo;
- para cada K: TimeSeriesKMeans euclidiano sobre o latente + metricas.

Uso (exemplo):
    python experimentos/aeon_dl_baseline.py --datasets Coffee Beef GunPoint --epochs 5 \
        --out results/aeon_dl_teste
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

MODELS = [
    "AEFCNClusterer",
    "AEResNetClusterer",
    "AEDCNNClusterer",
    "AEDRNNClusterer",
    "AEBiGRUClusterer",
    "AEAttentionBiGRUClusterer",
]

LONG_COLS = [
    "dataset", "model", "k", "k_mode", "metric", "value", "seed", "n_epochs",
    "n_cases", "n_timepoints", "n_classes", "preprocessing",
]
EXEC_COLS = [
    "timestamp", "dataset", "model", "status", "k_mode", "seed", "n_epochs", "device",
    "n_cases", "n_timepoints", "load_time_s", "train_time_s", "epochs_run",
    "first_epoch_s", "epoch_median_s", "latent_time_s", "ksearch_time_s", "n_k",
    "recon_loss_min", "recon_loss_last", "recon_mse", "encoder_check", "message",
]
METRICS = ["silhouette", "davies_bouldin", "ri", "ari", "nmi", "recon_loss", "recon_mse",
           "elbo"]
# recon_loss = min(history["loss"]) (protocolo; no AEDCNN inclui a penalidade L2 do decodificador)
# recon_mse  = MSE pura do modelo restaurado sobre todo o conjunto (comparavel entre modelos)


# --------------------------------------------------------------------------- CSV
def append_rows(path: Path, cols: list[str], rows: list[dict]) -> None:
    """Anexa linhas. Se o arquivo ja existe, segue o cabecalho dele (colunas novas
    que ainda nao estao no cabecalho sao descartadas), para nunca desalinhar o CSV."""
    new = not path.exists()
    if not new:
        with open(path, newline="", encoding="utf-8") as f:
            header = next(csv.reader(f), None)
        if header:
            cols = header
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def done_pairs(path: Path, seed: int, n_epochs: int, k_mode: str) -> set[tuple[str, str]]:
    """Pares (conjunto, modelo) que ja tem linhas no CSV longo para esta configuracao."""
    if not path.exists():
        return set()
    out = set()
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if (r["seed"] == str(seed) and r["n_epochs"] == str(n_epochs)
                    and r["k_mode"] == k_mode):
                out.add((r["dataset"], r["model"]))
    return out


# ------------------------------------------------------------------------- dados
def _interp_nan(v: np.ndarray) -> tuple[np.ndarray, str]:
    mask = np.isnan(v)
    if not mask.any():
        return v, ""
    if mask.all():
        return np.zeros_like(v), "all_nan"
    idx = np.arange(v.size)
    v = v.copy()
    # np.interp repete o primeiro/ultimo valor valido fora do intervalo (extremos).
    v[mask] = np.interp(idx[mask], idx[~mask], v[~mask])
    return v, "nan"


def load_dataset(name: str, data_dir: Path, cache_dir: Path) -> dict:
    """Carrega, concatena treino+teste, trata NaN/comprimento e normaliza (cache .npz)."""
    cache = cache_dir / f"{name}.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=False)
        return {"X": d["X"], "y": d["y"], "preprocessing": str(d["preprocessing"])}

    from aeon.datasets import load_classification

    data_dir.mkdir(parents=True, exist_ok=True)
    Xtr, ytr = load_classification(name, split="train", extract_path=str(data_dir))
    Xte, yte = load_classification(name, split="test", extract_path=str(data_dir))
    series = [np.asarray(s, dtype=np.float64) for s in list(Xtr) + list(Xte)]
    y = np.concatenate([np.asarray(ytr), np.asarray(yte)]).astype(str)
    X, prep = preprocess(series)
    prep = "concat_train_test;" + prep

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(cache, X=X, y=y, preprocessing=np.array(prep))
    return {"X": X, "y": y, "preprocessing": prep}


def preprocess(series: list[np.ndarray]) -> tuple[np.ndarray, str]:
    """NaN -> interpolacao linear; comprimentos diferentes -> Resizer("max"); z-norm.

    `series`: lista de arrays (n_canais=1, comprimento_i). Devolve X float32
    (n_casos, 1, m) e a descricao dos passos aplicados.
    """
    from aeon.transformations.collection.unequal_length import Resizer

    if series[0].ndim != 2 or series[0].shape[0] != 1:
        raise ValueError(f"formato inesperado: {series[0].shape}")

    n_nan = n_all_nan = 0
    clean = []
    for s in series:
        ch = []
        flag = ""
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


# ------------------------------------------------------------------------ modelo
def make_callbacks():
    """Callbacks equivalentes ao padrao do aeon, mais um cronometro por epoca.

    Com `callbacks=None`, o aeon usa ReduceLROnPlateau(loss, 0.5, 50, 1e-4) +
    ModelCheckpoint(loss, save_best_only). Quando recebe callbacks, ele acrescenta o
    mesmo ModelCheckpoint (monitor="loss"). Repetimos o ReduceLROnPlateau com os
    mesmos parametros, entao o treino fica identico ao padrao (validation_split=0).
    """
    import tensorflow as tf

    class EpochTimer(tf.keras.callbacks.Callback):
        def on_train_begin(self, logs=None):
            self.times = []

        def on_epoch_begin(self, epoch, logs=None):
            self._t = time.perf_counter()

        def on_epoch_end(self, epoch, logs=None):
            self.times.append(time.perf_counter() - self._t)

    timer = EpochTimer()
    rlr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="loss", factor=0.5, patience=50, min_lr=0.0001)
    return [rlr, timer], timer


def build_clusterer(model_name: str, n_epochs: int, seed: int, tmp_dir: str,
                    callbacks=None):
    import aeon.clustering.deep_learning as dl

    cls = getattr(dl, model_name)
    return cls(
        callbacks=callbacks,
        n_epochs=n_epochs,
        random_state=seed,
        batch_size=32,
        save_best_model=False,
        save_last_model=False,
        file_path=tmp_dir.rstrip("/\\") + "/",
        verbose=False,
    )


def check_encoder(clst, X: np.ndarray, latent: np.ndarray) -> str:
    """Confere que model_.layers[1] e o codificador.

    1) model_ tem 3 camadas (entrada, codificador, decodificador);
    2) decodificador(codificador(x)) reproduz model_(x);
    3) o k-means interno do aeon, aplicado ao nosso latente, da o mesmo resultado
       que clst.predict(X) (que passa pelo caminho interno do aeon).
    """
    Xk = X.transpose(0, 2, 1)
    layers = clst.model_.layers
    names = [ly.name for ly in layers]
    sub = Xk[: min(256, len(Xk))]
    full = clst.model_.predict(sub, verbose=0)
    comp = layers[2].predict(layers[1].predict(sub, verbose=0), verbose=0)
    recon_ok = bool(np.allclose(full, comp, atol=1e-5))
    pred_aeon = clst.predict(X)
    pred_ours = clst.estimator_.predict(latent)
    pred_ok = bool(np.array_equal(pred_aeon, pred_ours))
    ok = len(layers) == 3 and recon_ok and pred_ok
    return (f"{'OK' if ok else 'FALHOU'} layers={names} latent_shape={latent.shape} "
            f"dec(enc(x))==model(x):{recon_ok} predict_igual:{pred_ok}")


def train_and_extract(model_name: str, X: np.ndarray, n_epochs: int, seed: int):
    import tensorflow as tf

    # Checkpoint do aeon (melhor `loss`) em RAM quando houver /dev/shm (Linux/WSL): mesmo arquivo,
    # mesmo modelo recarregado; so evita gravar o modelo inteiro em disco a cada melhora.
    shm = "/dev/shm" if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK) else None
    tmp = tempfile.mkdtemp(prefix="aeon_ckpt_", dir=shm)
    try:
        cbs, timer = make_callbacks()
        clst = build_clusterer(model_name, n_epochs, seed, tmp, callbacks=cbs)
        t0 = time.perf_counter()
        clst.fit(X)
        train_t = time.perf_counter() - t0

        hist = clst.history.history if hasattr(clst.history, "history") else clst.history
        losses = [float(v) for v in hist["loss"]]

        t0 = time.perf_counter()
        latent = clst.model_.layers[1].predict(X.transpose(0, 2, 1), verbose=0)
        latent = np.asarray(latent).reshape(len(X), -1).astype(np.float32)
        latent_t = time.perf_counter() - t0

        check = check_encoder(clst, X, latent)
        Xk = X.transpose(0, 2, 1)
        recon_mse = float(np.mean((clst.model_.predict(Xk, verbose=0) - Xk) ** 2))
        return latent, {
            "train_time_s": train_t,
            "latent_time_s": latent_t,
            "recon_loss_min": min(losses),
            "recon_loss_last": losses[-1],
            "recon_mse": recon_mse,
            "epochs_run": len(losses),
            "first_epoch_s": timer.times[0] if timer.times else "",
            "epoch_median_s": (float(np.median(timer.times[1:])) if len(timer.times) > 1
                               else (timer.times[0] if timer.times else "")),
            "encoder_check": check,
        }
    finally:
        tf.keras.backend.clear_session()
        gc.collect()
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------- busca K
def k_search(latent, y, ks, k_mode, seed, base_row, recon_loss, recon_mse=None):
    const = {"recon_loss": recon_loss, "recon_mse": recon_mse}
    from aeon.clustering import TimeSeriesKMeans
    from sklearn import metrics as M

    rows, notes = [], []
    for k in ks:
        km = TimeSeriesKMeans(n_clusters=k, distance="euclidean",
                              averaging_method="mean", random_state=seed)
        try:
            lab = km.fit_predict(latent)
        except Exception as e:  # noqa: BLE001  (ex.: grupos vazios em latente degenerado)
            notes.append(f"K={k}: k-means falhou ({type(e).__name__}: {e}); metricas=NaN")
            for m in METRICS:
                v = (const[m] if m in const else ("" if m == "elbo" else float("nan")))
                v = "" if v is None else v
                rows.append({**base_row, "k": k, "k_mode": k_mode, "metric": m,
                             "value": v if v == "" else repr(float(v))})
            continue
        n_found = len(np.unique(lab))
        vals = {}
        if 2 <= n_found <= len(lab) - 1:
            vals["silhouette"] = M.silhouette_score(latent, lab, metric="euclidean")
            vals["davies_bouldin"] = M.davies_bouldin_score(latent, lab)
        else:
            vals["silhouette"] = vals["davies_bouldin"] = float("nan")
            notes.append(f"K={k}: {n_found} grupo(s) distinto(s), silhueta/DB=NaN")
        if n_found < k:
            notes.append(f"K={k}: k-means devolveu {n_found} grupos")
        vals["ri"] = M.rand_score(y, lab)
        vals["ari"] = M.adjusted_rand_score(y, lab)
        vals["nmi"] = M.normalized_mutual_info_score(y, lab)
        vals["recon_loss"] = recon_loss
        vals["recon_mse"] = "" if recon_mse is None else recon_mse
        vals["elbo"] = ""  # nao se aplica aos autoencoders do aeon
        for m in METRICS:
            v = vals[m]
            rows.append({**base_row, "k": k, "k_mode": k_mode, "metric": m,
                         "value": v if v == "" else repr(float(v))})
    return rows, notes


def ks_for(k_mode, n_cases, n_classes, k_min, k_max, k_step):
    if k_mode == "grid":
        return [k for k in range(k_min, k_max + 1, k_step) if k < n_cases]
    return [n_classes]


# ------------------------------------------------------------------ selecao K
def write_selection(long_csv: Path, out_csv: Path) -> None:
    import pandas as pd

    if not long_csv.exists():
        return
    df = pd.read_csv(long_csv)
    df = df[df["k_mode"] == "grid"]
    if df.empty:
        return
    keys = ["dataset", "model", "seed", "n_epochs"]
    wide = df.pivot_table(index=keys + ["k", "n_classes"], columns="metric",
                          values="value", aggfunc="first").reset_index()
    out = []
    for _, g in wide.groupby(keys):
        g = g.dropna(subset=["silhouette"])
        if g.empty:
            continue
        best = g.sort_values(["silhouette", "k"], ascending=[False, True]).iloc[0]
        out.append(best)
    if out:
        sel = pd.DataFrame(out).rename(columns={"k": "k_selected"})
        cols = keys + ["n_classes", "k_selected", "silhouette", "davies_bouldin",
                       "ri", "ari", "nmi", "recon_loss"]
        sel[[c for c in cols if c in sel.columns]].to_csv(out_csv, index=False)


# ------------------------------------------------------------------------ main
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="*", default=None,
                   help="nomes do UCR (padrao: os 128 de tsc_datasets.univariate)")
    p.add_argument("--models", nargs="*", default=MODELS, choices=MODELS)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--k-min", type=int, default=2)
    p.add_argument("--k-max", type=int, default=70,
                   help="padrao 70 (pedido do colaborador: 2:70:2)")
    p.add_argument("--k-step", type=int, default=2)
    p.add_argument("--k-mode", choices=["grid", "n_classes", "both"], default="grid")
    p.add_argument("--out", default=str(ROOT / "results" / "aeon_dl"))
    p.add_argument("--data-dir", default=str(ROOT / "data" / "ucr"))
    p.add_argument("--device", choices=["auto", "cpu"], default="auto",
                   help="cpu: esconde a GPU (usado para o AEDRNN, que nao cabe em 4 GB)")
    p.add_argument("--stop-after-hours", type=float, default=None,
                   help="nao inicia novos treinos depois deste tempo (rodada noturna)")
    p.add_argument("--max-size", type=int, default=None,
                   help="so conjuntos com n_casos*comprimento <= este valor")
    p.add_argument("--no-isolate", action="store_true",
                   help="treina tudo no mesmo processo (padrao: um subprocesso por par)")
    p.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    t_start = time.perf_counter()
    if not a.child and not a.no_isolate:
        return run_isolated(a, t_start)

    import tensorflow as tf
    import keras

    keras.utils.disable_interactive_logging()
    if a.device == "cpu":
        tf.config.set_visible_devices([], "GPU")
    from aeon.datasets.tsc_datasets import univariate

    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
    device = "GPU" if gpus else "CPU"
    print(f"TensorFlow {tf.__version__} | dispositivo: {device} {gpus}", flush=True)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    long_csv = out / "resultados_longo.csv"
    exec_csv = out / "execucoes.csv"
    sel_csv = out / "selecao_k_silhueta.csv"
    lat_root = out / "latents" / f"e{a.epochs}_s{a.seed}"
    data_dir = Path(a.data_dir)
    cache_dir = data_dir.parent / "processed"

    names = a.datasets or list(univariate)
    k_modes = ["grid", "n_classes"] if a.k_mode == "both" else [a.k_mode]

    def log_exec(**kw):
        append_rows(exec_csv, EXEC_COLS, [{
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "seed": a.seed, "n_epochs": a.epochs, "device": device, **kw}])

    # 1) carregar tudo (com cache) para ordenar por n_casos x comprimento
    sizes, load_t = {}, {}
    for name in names:
        t0 = time.perf_counter()
        try:
            d = load_dataset(name, data_dir, cache_dir)
            sizes[name] = d["X"].shape[0] * d["X"].shape[2]
            del d
            load_t[name] = time.perf_counter() - t0
        except Exception as e:  # noqa: BLE001
            log_exec(dataset=name, model="*", status="erro_carga",
                     load_time_s=round(time.perf_counter() - t0, 3),
                     message=f"{type(e).__name__}: {e}")
            print(f"[erro carga] {name}: {e}", flush=True)
    order = sorted(sizes, key=lambda n: (sizes[n], n))
    if a.max_size is not None:
        order = [n for n in order if sizes[n] <= a.max_size]
    print(f"{len(order)} conjuntos na fila: {order}", flush=True)

    # 2) treinar / buscar K
    stop = False
    for name in order:
        if stop:
            break
        d = load_dataset(name, data_dir, cache_dir)
        X, y = d["X"], d["y"]
        n_cases, _, n_tp = X.shape
        n_classes = len(np.unique(y))
        base = {"dataset": name, "seed": a.seed, "n_epochs": a.epochs,
                "n_cases": n_cases, "n_timepoints": n_tp, "n_classes": n_classes,
                "preprocessing": d["preprocessing"]}
        print(f"\n== {name}: n={n_cases} m={n_tp} classes={n_classes} "
              f"[{d['preprocessing']}]", flush=True)

        for model in a.models:
            if (a.stop_after_hours is not None
                    and time.perf_counter() - t_start > a.stop_after_hours * 3600):
                print("Prazo atingido; nao inicio novos treinos.", flush=True)
                stop = True
                break
            pending = [km for km in k_modes
                       if (name, model) not in done_pairs(long_csv, a.seed, a.epochs, km)]
            if not pending:
                print(f"  {model}: ja feito, pulando", flush=True)
                continue
            lat_path = lat_root / model / f"{name}.npy"
            meta_path = lat_path.with_suffix(".json")
            info = {"train_time_s": "", "latent_time_s": "", "encoder_check": ""}
            try:
                if lat_path.exists() and meta_path.exists():
                    latent = np.load(lat_path)
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    info.update(meta, train_time_s="", latent_time_s="",
                                first_epoch_s="", epoch_median_s="",
                                encoder_check="latente reaproveitado")
                    print(f"  {model}: latente salvo reaproveitado", flush=True)
                else:
                    latent, info = train_and_extract(model, X, a.epochs, a.seed)
                    lat_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(lat_path, latent)
                    meta_path.write_text(json.dumps({k: info[k] for k in (
                        "recon_loss_min", "recon_loss_last", "recon_mse", "epochs_run",
                        "train_time_s", "first_epoch_s", "epoch_median_s",
                        "encoder_check")}, indent=1), encoding="utf-8")
                    print(f"  {model}: treino {info['train_time_s']:.1f}s | "
                          f"{info['encoder_check']}", flush=True)

                for km in pending:
                    ks = ks_for(km, n_cases, n_classes, a.k_min, a.k_max, a.k_step)
                    t0 = time.perf_counter()
                    rows, notes = k_search(latent, y, ks, km, a.seed,
                                           {**base, "model": model},
                                           info["recon_loss_min"],
                                           info.get("recon_mse"))
                    ks_t = time.perf_counter() - t0
                    append_rows(long_csv, LONG_COLS, rows)
                    log_exec(dataset=name, model=model, status="ok", k_mode=km,
                             n_cases=n_cases, n_timepoints=n_tp,
                             load_time_s=round(load_t[name], 3),
                             train_time_s=_r(info["train_time_s"]),
                             epochs_run=info.get("epochs_run", ""),
                             first_epoch_s=_r(info.get("first_epoch_s", "")),
                             epoch_median_s=_r(info.get("epoch_median_s", "")),
                             latent_time_s=_r(info["latent_time_s"]),
                             ksearch_time_s=round(ks_t, 3), n_k=len(ks),
                             recon_loss_min=info["recon_loss_min"],
                             recon_loss_last=info["recon_loss_last"],
                             recon_mse=info.get("recon_mse", ""),
                             encoder_check=info["encoder_check"],
                             message=f"k_mode={km}; " + " | ".join(notes))
                    print(f"  {model} [{km}]: {len(ks)} K em {ks_t:.1f}s", flush=True)
            except Exception as e:  # noqa: BLE001
                tb = traceback.format_exc(limit=3).replace("\n", " ")
                log_exec(dataset=name, model=model, status="erro",
                         n_cases=n_cases, n_timepoints=n_tp,
                         load_time_s=round(load_t[name], 3),
                         message=" ".join(f"{type(e).__name__}: {e} || {tb}".split())[:2000])
                print(f"  {model}: ERRO {type(e).__name__}: {e}", flush=True)
                try:
                    tf.keras.backend.clear_session()
                except Exception:  # noqa: BLE001
                    pass
                gc.collect()

    if not a.child:
        write_selection(long_csv, sel_csv)
        print(f"\nSaidas em {out}", flush=True)


def _clean_shm_checkpoints() -> None:
    """Remove pastas de checkpoint orfas em /dev/shm (filho morto nao limpa a propria)."""
    import glob

    for d in glob.glob("/dev/shm/aeon_ckpt_*"):
        shutil.rmtree(d, ignore_errors=True)


def run_isolated(a, t_start) -> int:
    """Um subprocesso por (conjunto, modelo).

    Libera toda a memoria do TF ao fim de cada par e, se o filho for morto (ex.: OOM
    killer, codigo -9/137), registra a falha em execucoes.csv e segue para o proximo.
    """
    import subprocess

    from aeon.datasets.tsc_datasets import univariate

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    long_csv, exec_csv = out / "resultados_longo.csv", out / "execucoes.csv"
    data_dir = Path(a.data_dir)
    cache_dir = data_dir.parent / "processed"
    names = a.datasets or list(univariate)
    k_modes = ["grid", "n_classes"] if a.k_mode == "both" else [a.k_mode]

    def log_exec(**kw):
        append_rows(exec_csv, EXEC_COLS, [{
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "seed": a.seed, "n_epochs": a.epochs, "device": a.device, **kw}])

    sizes, shapes = {}, {}
    for name in names:
        t0 = time.perf_counter()
        try:
            d = load_dataset(name, data_dir, cache_dir)
            sizes[name] = d["X"].shape[0] * d["X"].shape[2]
            shapes[name] = (d["X"].shape[0], d["X"].shape[2])
            del d
        except Exception as e:  # noqa: BLE001
            log_exec(dataset=name, model="*", status="erro_carga",
                     load_time_s=round(time.perf_counter() - t0, 3),
                     message=" ".join(f"{type(e).__name__}: {e}".split())[:2000])
            print(f"[erro carga] {name}: {e}", flush=True)
    order = sorted(sizes, key=lambda n: (sizes[n], n))
    if a.max_size is not None:
        order = [n for n in order if sizes[n] <= a.max_size]
    print(f"{len(order)} conjuntos na fila: {order}", flush=True)

    _clean_shm_checkpoints()
    passthru = ["--epochs", str(a.epochs), "--seed", str(a.seed),
                "--k-min", str(a.k_min), "--k-max", str(a.k_max), "--k-step", str(a.k_step),
                "--k-mode", a.k_mode, "--out", str(a.out), "--data-dir", str(a.data_dir),
                "--device", a.device]
    for name in order:
        for model in a.models:
            if (a.stop_after_hours is not None
                    and time.perf_counter() - t_start > a.stop_after_hours * 3600):
                print("Prazo atingido; nao inicio novos treinos.", flush=True)
                write_selection(long_csv, out / "selecao_k_silhueta.csv")
                return 0
            if all((name, model) in done_pairs(long_csv, a.seed, a.epochs, km)
                   for km in k_modes):
                print(f"{name}/{model}: ja feito, pulando", flush=True)
                continue
            cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--child",
                   "--datasets", name, "--models", model, *passthru]
            t0 = time.perf_counter()
            rc = subprocess.run(cmd).returncode
            if rc != 0:
                _clean_shm_checkpoints()
                n, m = shapes[name]
                msg = (f"subprocesso terminou com codigo {rc} sem registrar resultado"
                       + (" (SIGKILL: provavel falta de memoria)" if rc in (-9, 137) else ""))
                log_exec(dataset=name, model=model, status="erro_processo",
                         n_cases=n, n_timepoints=m,
                         train_time_s=round(time.perf_counter() - t0, 3), message=msg)
                print(f"{name}/{model}: {msg}", flush=True)
    write_selection(long_csv, out / "selecao_k_silhueta.csv")
    print(f"\nSaidas em {out}", flush=True)
    return 0


def _r(v):
    return round(v, 3) if isinstance(v, (int, float)) else v


if __name__ == "__main__":
    sys.exit(main())
