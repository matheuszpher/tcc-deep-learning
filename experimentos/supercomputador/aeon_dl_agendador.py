"""Agenda os pares (conjunto, modelo) em subprocessos, um por GPU, e consolida os CSVs.

    python aeon_dl_agendador.py --config <out>/config.json            # roda + consolida
    python aeon_dl_agendador.py --config <out>/config.json --so-consolidar

- Retomavel: pula pares com status "ok" (e, se repetir_falhas=False, tambem os que falharam).
- Um subprocesso por par: se faltar memoria ou o processo travar, so aquele par falha,
  e a falha fica registrada (status "erro", "erro_processo" ou "timeout").
- Ordem: conjuntos menores primeiro (n_casos x comprimento).
- Paralelismo: um "slot" por GPU listada em config["gpus"] (x jobs_por_gpu).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import aeon_dl_par as P  # noqa: E402

EXEC_COLS = [
    "timestamp_inicio", "timestamp_fim", "dataset", "model", "status", "seed", "n_epochs",
    "k_grid", "host", "device", "gpu_name", "cuda_visible_devices", "python", "tensorflow",
    "aeon", "n_cases", "n_train", "n_test", "n_timepoints", "n_classes", "preprocessing",
    "n_params", "latent_dim", "load_time_s", "train_time_s", "epochs_run", "best_epoch",
    "first_epoch_s", "epoch_median_s", "latent_time_s", "ksearch_time_s", "n_k",
    "recon_loss_min", "recon_loss_last", "recon_mse", "lr_final", "gpu_mem_peak_mb",
    "wall_time_s", "encoder_check", "message",
]


def log(msg: str, out: Path | None = None):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    if out is not None:
        with open(out / "agendador.log", "a", encoding="utf-8") as f:
            f.write(line + "\n")


def read_status(out: Path, dataset: str, model: str) -> dict | None:
    f = P.pair_dir(out, dataset, model) / "status.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def same_config(st: dict, cfg: dict) -> bool:
    return (str(st.get("seed")) == str(cfg["seed"])
            and str(st.get("n_epochs")) == str(cfg["epochs"])
            and st.get("k_grid") == f"{cfg['k_min']}:{cfg['k_max']}:{cfg['k_step']}")


def dataset_sizes(cfg: dict, out: Path) -> dict[str, tuple[int, int]]:
    """Carrega (e guarda em cache) todos os conjuntos para ordenar por tamanho."""
    shapes = {}
    for i, name in enumerate(cfg["datasets"], 1):
        try:
            d = P.load_dataset(name, Path(cfg["data_dir"]))
            shapes[name] = (int(d["X"].shape[0]), int(d["X"].shape[2]))
        except Exception as e:  # noqa: BLE001
            log(f"erro ao carregar {name}: {type(e).__name__}: {e}", out)
            for model in cfg["models"]:
                d = P.pair_dir(out, name, model)
                d.mkdir(parents=True, exist_ok=True)
                P._write_status(d, {"dataset": name, "model": model, "status": "erro_carga",
                                    "seed": cfg["seed"], "n_epochs": cfg["epochs"],
                                    "k_grid": f"{cfg['k_min']}:{cfg['k_max']}:{cfg['k_step']}",
                                    "timestamp_fim": datetime.now().isoformat(timespec="seconds"),
                                    "message": f"{type(e).__name__}: {e}"[:2000]})
        if i % 16 == 0:
            log(f"dados: {i}/{len(cfg['datasets'])} conjuntos prontos", out)
    return shapes


def run(cfg: dict) -> None:
    out = Path(cfg["out"])
    (out / "logs").mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    log(f"inicio | {len(cfg['datasets'])} conjuntos x {len(cfg['models'])} modelos | "
        f"epocas={cfg['epochs']} seed={cfg['seed']} K={cfg['k_min']}:{cfg['k_max']}:{cfg['k_step']} "
        f"k_modes={cfg['k_modes']} gpus={cfg['gpus']}", out)
    shapes = dataset_sizes(cfg, out)
    order = sorted(shapes, key=lambda n: (shapes[n][0] * shapes[n][1], n))
    if cfg.get("max_size"):
        order = [n for n in order if shapes[n][0] * shapes[n][1] <= cfg["max_size"]]

    queue = []
    for name in order:
        for model in cfg["models"]:
            st = read_status(out, name, model)
            if st and same_config(st, cfg):
                if st.get("status") == "ok":
                    continue
                # interrompido / em_andamento: o agendador parou no meio, entao roda de novo.
                if (not cfg.get("repetir_falhas", False)
                        and st.get("status") not in ("em_andamento", "interrompido")):
                    continue
            queue.append((name, model))
    log(f"fila: {len(queue)} pares pendentes", out)

    slots = [str(g) for g in cfg["gpus"]] * int(cfg.get("jobs_por_gpu", 1)) or ["cpu"]
    running: dict[str, tuple] = {}  # slot -> (proc, name, model, t0, logfile)
    timeout = float(cfg.get("timeout_par_horas") or 0) * 3600
    deadline = float(cfg.get("parar_apos_horas") or 0) * 3600
    last_consol = time.time()

    def launch(slot, name, model):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "" if (slot == "cpu" or model in cfg.get("modelos_cpu", [])) else slot
        env["TF_CPP_MIN_LOG_LEVEL"] = "2"
        lf = open(out / "logs" / f"{model}__{name}.log", "w", encoding="utf-8")
        cmd = [sys.executable, "-u", str(HERE / "aeon_dl_par.py"), "--config",
               str(out / "config.json"), "--dataset", name, "--model", model]
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
        running[slot] = (proc, name, model, time.time(), lf)
        log(f"[slot {slot}] inicia {name}/{model}", out)

    def finish(slot, rc, reason=""):
        proc, name, model, t0, lf = running.pop(slot)
        lf.close()
        wall = round(time.time() - t0, 1)
        st = read_status(out, name, model)
        d = P.pair_dir(out, name, model)
        if st is None or st.get("status") == "em_andamento" or reason:
            # O filho morreu sem gravar status (ex.: SIGKILL por falta de memoria) ou estourou o tempo.
            msg = reason or (f"subprocesso terminou com codigo {rc} sem registrar resultado"
                             + (" (SIGKILL: provavel falta de memoria)" if rc in (-9, 137) else ""))
            tail = ""
            try:
                tail = " ".join((out / "logs" / f"{model}__{name}.log").read_text(
                    encoding="utf-8", errors="replace").split()[-60:])
            except Exception:  # noqa: BLE001
                pass
            st = {"dataset": name, "model": model,
                  "status": ("timeout" if reason.startswith("timeout") else
                             "interrompido" if reason else "erro_processo"),
                  "seed": cfg["seed"], "n_epochs": cfg["epochs"],
                  "k_grid": f"{cfg['k_min']}:{cfg['k_max']}:{cfg['k_step']}",
                  "timestamp_fim": datetime.now().isoformat(timespec="seconds"),
                  "message": f"{msg} || log: {tail}"[:4000], **(shapes_info(name))}
        st["wall_time_s"] = wall
        d.mkdir(parents=True, exist_ok=True)
        P._write_status(d, st)
        log(f"[slot {slot}] fim {name}/{model}: {st['status']} ({wall:.0f}s)", out)
        for tmp in Path("/dev/shm").glob("aeon_ckpt_*") if Path("/dev/shm").is_dir() else []:
            if time.time() - tmp.stat().st_mtime > 60 and not running:
                shutil.rmtree(tmp, ignore_errors=True)

    def shapes_info(name):
        n, m = shapes.get(name, ("", ""))
        return {"n_cases": n, "n_timepoints": m}

    try:
        while queue or running:
            for slot in list(running):
                proc, name, model, t0, _ = running[slot]
                rc = proc.poll()
                if rc is not None:
                    finish(slot, rc)
                elif timeout and time.time() - t0 > timeout:
                    proc.kill()
                    proc.wait()
                    finish(slot, -9, f"timeout: passou de {cfg['timeout_par_horas']} h")
            stop = deadline and time.time() - t_start > deadline
            for slot in slots:
                if slot not in running and queue and not stop:
                    launch(slot, *queue.pop(0))
            if stop and queue:
                log(f"prazo de {cfg['parar_apos_horas']} h atingido; {len(queue)} pares ficam "
                    "pendentes (rode de novo para retomar)", out)
                queue.clear()
            if time.time() - last_consol > 600:
                consolidate(cfg)
                last_consol = time.time()
            time.sleep(2)
    finally:
        for slot in list(running):
            running[slot][0].kill()
            finish(slot, -9, "agendador interrompido")
        consolidate(cfg)
        log(f"fim | {time.time() - t_start:.0f}s", out)


# ---------------------------------------------------------------- consolidacao
def consolidate(cfg: dict) -> dict:
    """Junta as pastas dos pares nos CSVs finais (reescritos do zero a cada chamada)."""
    import pandas as pd

    out = Path(cfg["out"])
    base = out / "pares"
    status, longs, curves = [], [], []
    if base.exists():
        for f in sorted(base.glob("*/*/status.json")):
            try:
                st = json.loads(f.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            status.append(st)
            if st.get("status") == "ok":
                if (f.parent / "metricas.csv").exists():
                    longs.append(pd.read_csv(f.parent / "metricas.csv"))
                if (f.parent / "curva_perda.csv").exists():
                    curves.append(pd.read_csv(f.parent / "curva_perda.csv"))

    ex = pd.DataFrame(status)
    for c in EXEC_COLS:
        if c not in ex.columns:
            ex[c] = ""
    ex = ex[EXEC_COLS + [c for c in ex.columns if c not in EXEC_COLS]]
    if not ex.empty:
        ex = ex.sort_values(["dataset", "model"])
    ex.to_csv(out / "execucoes.csv", index=False)
    ex[ex["status"] != "ok"].to_csv(out / "falhas.csv", index=False)

    lg = pd.concat(longs, ignore_index=True) if longs else pd.DataFrame(columns=P.LONG_COLS)
    lg.to_csv(out / "resultados_longo.csv", index=False)
    (pd.concat(curves, ignore_index=True) if curves
     else pd.DataFrame(columns=["dataset", "model", "epoch", "loss", "learning_rate",
                                "epoch_time_s"])).to_csv(out / "curvas_perda.csv", index=False)

    keys = ["dataset", "model", "seed", "n_epochs", "n_cases", "n_timepoints", "n_classes",
            "preprocessing", "k_mode", "k"]
    if lg.empty:
        wide = pd.DataFrame(columns=keys + P.METRICS)
    else:
        wide = (lg.pivot_table(index=keys, columns="metric", values="value", aggfunc="first",
                               dropna=False).reset_index())
        wide.columns.name = None
        wide = wide[keys + [m for m in P.METRICS if m in wide.columns]]
        wide = wide.dropna(subset=["n_cases"]).sort_values(["dataset", "model", "k_mode", "k"])
    wide.to_csv(out / "resultados_largo.csv", index=False)

    # K escolhido pela silhueta (maior; empate -> menor K), entre solucoes nao degeneradas.
    sel = []
    g = wide[wide["k_mode"] == "grid"] if not wide.empty else wide
    for (ds, mdl), grp in (g.groupby(["dataset", "model"]) if not g.empty else []):
        ok = grp.dropna(subset=["silhouette"])
        if ok.empty:
            continue
        b = ok.sort_values(["silhouette", "k"], ascending=[False, True]).iloc[0]
        row = b.rename({"k": "k_selected"}).drop(labels=["k_mode"]).to_dict()
        orc = grp.dropna(subset=["ari"]).sort_values(["ari", "k"], ascending=[False, True])
        if not orc.empty:
            row["k_oracle_ari"], row["ari_oracle"] = orc.iloc[0]["k"], orc.iloc[0]["ari"]
        nc = wide[(wide.dataset == ds) & (wide.model == mdl) & (wide.k_mode == "n_classes")]
        for m in ("ri", "ari", "nmi", "ami"):
            row[f"{m}_k_n_classes"] = nc.iloc[0][m] if not nc.empty else ""
        sel.append(row)
    pd.DataFrame(sel).to_csv(out / "selecao_k_silhueta.csv", index=False)

    # Situacao por conjunto
    ds_rows = []
    if not ex.empty:
        for ds, grp in ex.groupby("dataset"):
            r = {"dataset": ds}
            for c in ("n_cases", "n_timepoints", "n_classes", "preprocessing"):
                v = grp[c].replace("", pd.NA).dropna()
                r[c] = v.iloc[0] if not v.empty else ""
            r["modelos_ok"] = int((grp.status == "ok").sum())
            r["modelos_falha"] = int((grp.status != "ok").sum())
            r["modelos_pendentes"] = len(cfg["models"]) - len(grp)
            r["falhas"] = ";".join(f"{m}:{s}" for m, s in zip(grp.model, grp.status) if s != "ok")
            ds_rows.append(r)
    pd.DataFrame(ds_rows).to_csv(out / "status_conjuntos.csv", index=False)

    resumo = {"pares_ok": int((ex.status == "ok").sum()) if not ex.empty else 0,
              "pares_falha": int((ex.status != "ok").sum()) if not ex.empty else 0,
              "pares_total": len(cfg["datasets"]) * len(cfg["models"]),
              "conjuntos_com_resultado": int(lg["dataset"].nunique()) if not lg.empty else 0,
              "linhas_resultados_longo": len(lg)}
    (out / "progresso.json").write_text(json.dumps(resumo, indent=1), encoding="utf-8")
    return resumo


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--so-consolidar", action="store_true")
    a = p.parse_args(argv)
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    if a.so_consolidar:
        print(json.dumps(consolidate(cfg), indent=1))
        return 0
    run(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
