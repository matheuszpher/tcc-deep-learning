"""Gera aeon_dl_supercomputador.ipynb embutindo aeon_dl_par.py e aeon_dl_agendador.py.

Uso: python experimentos/supercomputador/gerar_notebook.py
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": s.strip("\n").splitlines(True)}


par = (HERE / "aeon_dl_par.py").read_text(encoding="utf-8")
ag = (HERE / "aeon_dl_agendador.py").read_text(encoding="utf-8")

cells = [
    md("""
# Linha de base: agrupadores profundos do aeon no UCR (versão para supercomputador)

Treina os 6 agrupadores profundos do `aeon` 1.6.0 (AEFCN, AEResNet, AEDCNN, AEBiGRU,
AEAttentionBiGRU, AEDRNN) nos 128 conjuntos univariados do UCR e busca o número de grupos
**K de 2 a 70, de 2 em 2** (mais K = número de classes, para comparação).

**Protocolo** (Paparrizos e Bogireddy, 2025): treino e teste unidos, rótulos usados só nas
métricas; valores ausentes por interpolação linear; comprimentos diferentes reamostrados para o
maior (`Resizer("max")`); normalização z por série; 200 épocas; semente 0. Cada par
(conjunto, modelo) é treinado **uma vez**; o espaço latente (`model_.layers[1]`, o mesmo que o
aeon usa) é salvo, e o `TimeSeriesKMeans` euclidiano roda para cada K sobre ele.

**Como usar**
1. Rode a célula 1 (instalação) uma vez, se o ambiente ainda não tiver os pacotes.
2. Ajuste a célula 2 (configuração). Para conferir o ambiente, rode antes com `TESTE_RAPIDO = True`
   (3 conjuntos pequenos, 2 épocas, poucos minutos).
3. Rode o restante. Com `EM_SEGUNDO_PLANO = True`, o agendador continua rodando mesmo se o
   navegador ou o kernel forem fechados; use a célula de acompanhamento para ver o progresso.
4. **É retomável:** rodar de novo pula os pares já concluídos.

Cada par roda num subprocesso próprio. Se faltar memória ou travar, só aquele par falha e a
falha fica registrada em `execucoes.csv` / `falhas.csv`. Com várias GPUs, roda um par por GPU
em paralelo.
"""),
    md("## 1. Instalação (uma vez)"),
    code("""
# Descomente se o ambiente não tiver os pacotes. Versões usadas na validação:
# aeon 1.6.0, tensorflow 2.21.0, keras 3.15.1, scikit-learn 1.9.1, numpy 2.3.5, pandas 3.0.6.
# Em Linux com GPU NVIDIA, "tensorflow[and-cuda]" traz CUDA/cuDNN pelo pip.
# %pip install -q "aeon==1.6.0" "tensorflow[and-cuda]==2.21.0" scikit-learn pandas
"""),
    md("## 2. Configuração"),
    code(r'''
import os, json, subprocess, sys
from pathlib import Path

TESTE_RAPIDO = os.environ.get("TESTE_RAPIDO", "0") == "1"   # True: 3 conjuntos pequenos, 2 épocas

PASTA_TRABALHO = Path(os.environ.get("AEON_HPC_DIR", "aeon_dl_hpc")).resolve()
OUT      = PASTA_TRABALHO / ("resultados_teste" if TESTE_RAPIDO else "resultados")
DATA_DIR = PASTA_TRABALHO / "dados"   # baixa o UCR aqui; se o nó não tiver internet, copie para dados/ucr

EPOCHS  = 2 if TESTE_RAPIDO else 200
SEED    = 0
K_MIN, K_MAX, K_STEP = 2, 70, 2       # pedido do colaborador: 2:70:2
K_MODES = ["grid", "n_classes"]       # grade 2..70 e K = número de classes

MODELS = ["AEFCNClusterer", "AEResNetClusterer", "AEDCNNClusterer",
          "AEBiGRUClusterer", "AEAttentionBiGRUClusterer", "AEDRNNClusterer"]
MODELOS_CPU = [s for s in os.environ.get("MODELOS_CPU", "").split(",") if s]  # forçar CPU (ex.: GPU pequena)

DATASETS = None                       # None = os 128 univariados do UCR
if TESTE_RAPIDO:
    DATASETS = ["Coffee", "Beef", "GunPoint"]

def _gpus():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=30)
        return [l.strip() for l in r.stdout.splitlines() if l.strip()]
    except Exception:
        return []

GPUS = _gpus()                        # um par por GPU em paralelo; [] = CPU
JOBS_POR_GPU = 1                      # suba para 2 em GPUs grandes (>= 24 GB) se quiser
TIMEOUT_PAR_HORAS = 12                # par que passar disso é encerrado e marcado "timeout"
PARAR_APOS_HORAS = None               # ex.: 11 -> não inicia novos pares depois de 11 h
REPETIR_FALHAS = False                # True: tenta de novo os pares que falharam
EM_SEGUNDO_PLANO = not TESTE_RAPIDO   # True: agendador destacado do kernel (nohup)

print("GPUs:", GPUS or "nenhuma (CPU)")
print("saída:", OUT)
'''),
    md("## 3. Código (gravado em arquivos `.py` ao lado do notebook)"),
    code("%%writefile aeon_dl_par.py\n" + par),
    code("%%writefile aeon_dl_agendador.py\n" + ag),
    md("## 4. Conferência do ambiente"),
    code(r'''
import importlib, platform
print("python", platform.python_version())
for m in ["aeon", "tensorflow", "keras", "sklearn", "numpy", "pandas"]:
    try:
        print(f"{m:12s}", importlib.import_module(m).__version__)
    except Exception as e:
        print(f"{m:12s} FALTANDO: {e}")
r = subprocess.run([sys.executable, "-c",
    "import os; os.environ['TF_CPP_MIN_LOG_LEVEL']='2'; import tensorflow as tf; "
    "print('GPUs vistas pelo TensorFlow:', tf.config.list_physical_devices('GPU'))"],
    capture_output=True, text=True)
print(r.stdout.strip() or r.stderr[-2000:])
'''),
    md("## 5. Grava a configuração"),
    code(r'''
from aeon.datasets.tsc_datasets import univariate

cfg = dict(out=str(OUT), data_dir=str(DATA_DIR), epochs=EPOCHS, seed=SEED,
           k_min=K_MIN, k_max=K_MAX, k_step=K_STEP, k_modes=K_MODES,
           models=MODELS, modelos_cpu=MODELOS_CPU,
           datasets=DATASETS or sorted(univariate),
           gpus=GPUS, jobs_por_gpu=JOBS_POR_GPU, timeout_par_horas=TIMEOUT_PAR_HORAS,
           parar_apos_horas=PARAR_APOS_HORAS, repetir_falhas=REPETIR_FALHAS)
OUT.mkdir(parents=True, exist_ok=True)
CFG = OUT / "config.json"
CFG.write_text(json.dumps(cfg, indent=1))
print(f"{len(cfg['datasets'])} conjuntos x {len(MODELS)} modelos -> {CFG}")
'''),
    md("## 6. Executa"),
    code(r'''
cmd = [sys.executable, "-u", str(Path("aeon_dl_agendador.py").resolve()), "--config", str(CFG)]
if EM_SEGUNDO_PLANO:
    # Destacado do kernel: continua rodando se o notebook for fechado.
    lf = open(OUT / "agendador_stdout.log", "a")
    p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
    (OUT / "agendador.pid").write_text(str(p.pid))
    print("agendador rodando em segundo plano, PID", p.pid, "| log:", OUT / "agendador.log")
else:
    rc = subprocess.run(cmd).returncode
    print("código de saída:", rc)
'''),
    md("## 7. Acompanhamento (rode quantas vezes quiser)"),
    code(r'''
import subprocess, json
subprocess.run([sys.executable, "aeon_dl_agendador.py", "--config", str(CFG), "--so-consolidar"],
               capture_output=True)
print(json.loads((OUT / "progresso.json").read_text()))
log = OUT / "agendador.log"
if log.exists():
    print("".join(log.read_text().splitlines(True)[-15:]))
'''),
    code(r'''
# Para parar o agendador em segundo plano (os pares concluídos ficam salvos):
# import os, signal; os.killpg(int((OUT / "agendador.pid").read_text()), signal.SIGTERM)
'''),
    md("## 8. Resultados"),
    code(r'''
import pandas as pd
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 40)
for f in ["resultados_longo.csv", "resultados_largo.csv", "selecao_k_silhueta.csv",
          "execucoes.csv", "curvas_perda.csv", "status_conjuntos.csv", "falhas.csv"]:
    p = OUT / f
    df = pd.read_csv(p) if p.exists() and p.stat().st_size else pd.DataFrame()
    print(f"\n### {f}: {len(df)} linhas")
    display(df.head(5))
'''),
    md("""
## 9. Arquivos de saída e dicionário de colunas

Tudo em `OUT/`. Os CSVs da raiz são reconstruídos a partir de `pares/` a cada consolidação.

| Arquivo | Conteúdo |
|---|---|
| `resultados_longo.csv` | **Principal.** Uma linha por (conjunto, modelo, K, métrica): `dataset, model, k, k_mode, metric, value, seed, n_epochs, n_cases, n_timepoints, n_classes, preprocessing` |
| `resultados_largo.csv` | O mesmo, uma linha por (conjunto, modelo, k_mode, K), métricas em colunas |
| `selecao_k_silhueta.csv` | Por (conjunto, modelo): K escolhido pela maior silhueta (empate: menor K) e todas as métricas nesse K; também `ri/ari/nmi/ami_k_n_classes` e o melhor K pelo ARI (`k_oracle_ari`, `ari_oracle`, usa rótulos: só limite otimista) |
| `execucoes.csv` | Uma linha por par: status, máquina, GPU, versões, tamanhos, parâmetros da rede, tempos, épocas, perdas, memória de GPU, mensagem de erro |
| `curvas_perda.csv` | Perda, taxa de aprendizado e tempo de cada época de cada treino |
| `status_conjuntos.csv` | Por conjunto: quantos modelos ok, com falha e pendentes |
| `falhas.csv` | Só os pares que falharam, com a mensagem e o fim do log |
| `pares/<modelo>/<conjunto>/` | `metricas.csv`, `curva_perda.csv`, `latente.npy`, `status.json` de cada par |
| `logs/` | saída de cada subprocesso; `agendador.log` |

**Métricas por K** (`metric` no CSV longo):

| Métrica | Significado |
|---|---|
| `silhouette`, `davies_bouldin`, `calinski_harabasz` | Índices internos no espaço latente (sem rótulos). Silhueta e CH: maior é melhor; DB: menor é melhor |
| `ri`, `ari`, `nmi`, `ami` | Externos, contra as classes, no conjunto inteiro (treino+teste) |
| `ri_train`, `ari_train`, `nmi_train` / `ri_test`, `ari_test`, `nmi_test` | O **mesmo** agrupamento (ajustado no conjunto unido) avaliado só nas séries que vinham do treino / do teste no split original do UCR |
| `n_clusters_produced` | Grupos que o k-means de fato produziu |
| `n_empty_components` | K − grupos produzidos |
| `degenerate` | 1 se produziu menos de 2 grupos (silhueta/DB/CH ficam NaN) |
| `cluster_size_min`, `cluster_size_max` | Tamanho do menor e do maior grupo |
| `kmeans_inertia`, `kmeans_n_iter` | Inércia final e iterações do k-means |
| `recon_loss` | Menor perda de treino (o aeon recarrega esse modelo). No AEDCNN inclui a penalidade L2 |
| `recon_mse` | MSE pura de reconstrução do modelo final em todo o conjunto (comparável entre modelos) |
| `elbo` | Vazio: os autoencoders do aeon não são variacionais |

**`execucoes.csv`, colunas de treino:** `n_params` (parâmetros da rede), `latent_dim`,
`train_time_s`, `epochs_run`, `best_epoch` (época de menor perda), `first_epoch_s` (inclui
compilação), `epoch_median_s`, `latent_time_s`, `ksearch_time_s`, `n_k`, `recon_loss_min`,
`recon_loss_last`, `recon_mse`, `lr_final`, `gpu_mem_peak_mb`, `wall_time_s` (tempo total do
subprocesso), `encoder_check` (confere que `layers[1]` é o codificador), `status`
(`ok`, `erro`, `erro_processo`, `timeout`, `erro_carga`) e `message`.
"""),
]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                   "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
for i, c in enumerate(nb["cells"]):
    c["id"] = f"c{i:02d}"
dest = HERE / "aeon_dl_supercomputador.ipynb"
dest.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(dest)
