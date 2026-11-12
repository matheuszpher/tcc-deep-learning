"""Gera docs/apresentacao/benchmark_aeon_dl.xlsx a partir dos CSVs (fotografia do momento).

Abas: Leia-me | Resultados | Resumo por modelo | ARI por conjunto | Diagrama CD | Execucoes |
      TF x PyTorch | Compatibilidade
Resumos sao formulas sobre a aba Resultados; dados brutos sao valores.
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.drawing.image import Image
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.cell.rich_text import CellRichText, TextBlock
from openpyxl.cell.text import InlineFont
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "aeon_dl"
BEN = ROOT / "results" / "bench_frameworks"
import sys
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs" / "apresentacao" / "benchmark_aeon_dl.xlsx"

F = "Arial"
H_FONT = Font(name=F, bold=True, color="FFFFFF", size=10)
H_FILL = PatternFill("solid", fgColor="1F3864")
BODY = Font(name=F, size=10)
BOLD = Font(name=F, size=10, bold=True)
TITLE = Font(name=F, size=14, bold=True)
NOTE = Font(name=F, size=9, italic=True, color="595959")
THIN = Side(style="thin", color="BFBFBF")
BOX = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
SHORT = {"AEFCNClusterer": "AEFCN", "AEResNetClusterer": "AEResNet", "AEDCNNClusterer": "AEDCNN",
         "AEBiGRUClusterer": "AEBiGRU", "AEAttentionBiGRUClusterer": "AEAttentionBiGRU",
         "AEDRNNClusterer": "AEDRNN"}
MODELS = list(SHORT.values())
CEN = {"silhueta": "K pela silhueta", "k_igual_n_classes": "K = nº de classes",
       "melhor_k_por_ari (usa rotulos; limite otimista)": "Melhor K pelo ARI (usa rótulos)"}


_SIL = "coesão e separação dos grupos no espaço latente, de -1 a 1 (maior = melhor); serve para escolher K dentro de um modelo, não para comparar modelos"
_RI = "Rand Index: fração dos pares de séries em que o agrupamento concorda com as classes reais (0 a 1); tende a subir com K"
_ARI = "Rand ajustado ao acaso: 0 = agrupamento aleatório, 1 = idêntico às classes; pode ser negativo; é a métrica principal de comparação"
_NMI = "informação mútua normalizada (média aritmética, padrão do sklearn), de 0 a 1; tende a subir com K"
_MSE = "erro quadrático médio da reconstrução do modelo final em todo o conjunto; os dados são normalizados, então prever zero dá cerca de 1 (vazio nos 3 primeiros conjuntos, anteriores à métrica)"
DICIONARIO = {
    "Resultados": [
        ("Conjunto", "conjunto do arquivo UCR (treino e teste concatenados)"),
        ("Modelo", "agrupador profundo do aeon: autoencoder + k-means no espaço latente"),
        ("Cenário", "como o K foi escolhido. 'K pela silhueta' = maior silhueta na grade 2 a 70 (não supervisionado); "
                    "'K = nº de classes' = protocolo de Paparrizos e Bogireddy (2025); "
                    "'Melhor K pelo ARI' = usa os rótulos, é só um limite otimista"),
        ("K", "número de grupos usado no k-means"),
        ("Nº classes", "número real de classes do conjunto (usado só na avaliação)"),
        ("Silhueta", _SIL), ("RI", _RI), ("ARI", _ARI), ("NMI", _NMI), ("recon_mse", _MSE),
        ("Nº séries", "total de séries (treino + teste)"),
        ("Comprimento", "pontos por série depois da reamostragem"),
    ],
    "Resumo por modelo": [
        ("Modelo", "agrupador do aeon"),
        ("Nº conjuntos", "quantos conjuntos o modelo já tem nesse cenário (fórmula CONT.SES sobre a aba Resultados)"),
        ("RI / ARI / NMI (média)", "média simples entre os conjuntos daquele modelo e cenário (fórmula MÉDIASES)"),
        ("K médio", "média do K usado; no cenário 'K pela silhueta' mostra o viés para K pequeno"),
    ],
    "ARI por conjunto": [
        ("Conjunto", "conjunto do UCR, do menor para o maior"),
        ("AEFCN ... AEDRNN", "ARI de cada modelo com K = nº de classes; vazio = par ainda não executado"),
        ("Melhor modelo", "modelo com maior ARI na linha (em empate, o mais à esquerda)"),
        ("Linha 'Média'", "ARI médio do modelo nos conjuntos em que ele rodou"),
        ("Linha 'Vitórias'", "em quantos conjuntos o modelo teve o maior ARI"),
    ],
    "Diagrama CD": [
        ("Eixo", "posto médio do ARI entre os conjuntos (1 = melhor); o número ao lado de cada modelo é o posto médio"),
        ("Barra horizontal", "liga modelos sem diferença estatisticamente significativa (Wilcoxon pareado com correção de Holm, alfa = 0,1)"),
        ("Conjuntos usados", "só os conjuntos em que os 5 modelos de GPU têm resultado; o AEDRNN fica de fora por ter poucos conjuntos"),
    ],
    "Execucoes": [
        ("Conjunto / Modelo", "par treinado"),
        ("Dispositivo", "GPU (WSL2) ou CPU (Windows; só o AEDRNN, que não cabe na GPU de 4 GB)"),
        ("Status", "ok, erro (exceção registrada) ou erro_processo (subprocesso morto, por exemplo por falta de memória)"),
        ("Nº séries / Comprimento", "tamanho do conjunto"),
        ("Treino (s)", "tempo total do ajuste: compilação, 200 épocas e gravação do checkpoint do melhor modelo"),
        ("1ª época (s)", "inclui a compilação do grafo pelo TensorFlow (XLA); é o custo fixo de cada par"),
        ("Mediana por época (s)", "tempo típico de uma época depois da primeira"),
        ("Busca de K (s)", "k-means + métricas para todos os K da grade, sobre o latente salvo"),
        ("recon_loss (treino)", "menor perda de treino (protocolo); no AEDCNN inclui a penalidade L2 do decodificador, por isso não é comparável"),
        ("recon_mse", _MSE),
    ],
    "TF x PyTorch": [
        ("Modelo", "AEFCN do aeon (convolucional); BiGRU com tanh (usa cuDNN); BiGRU com ReLU (como nos recorrentes do aeon)"),
        ("Comprimento", "pontos por série sintética (512 séries, lote 32, 20 épocas)"),
        ("Parâmetros TF / PyTorch", "parâmetros treináveis; iguais nos dois = é o mesmo modelo"),
        ("Status", "ok; 'travou (30 min)' = processo parado sem usar CPU nem GPU, encerrado no limite; 'não suportado' = a GRU do PyTorch só aceita tanh"),
        ("1ª época (s)", "inclui compilação (XLA no TF) e aquecimento da GPU"),
        ("Época (s)", "mediana das épocas 2 a 20"),
        ("Razão TF / PyTorch", "época do TF dividida pela do PyTorch: perto de 1 = empate; acima de 1 = PyTorch mais rápido; n/d = um dos lados não rodou"),
        ("Pico GPU (MB)", "memória máxima de GPU alocada pelo framework durante o treino"),
    ],
    "Compatibilidade": [
        ("Teste", "A = resolver dependências; B1/B2 = mesmo ambiente nos dois modos do Keras"),
        ("Configuração", "pacotes e modo testados"),
        ("Redes do aeon", "se foi possível construir a rede AEFCN do aeon"),
        ("tfp.layers (VMM)", "se foi possível construir a camada MultivariateNormalTriL, usada pelo VMM oficial"),
        ("Conclusão", "resultado ou mensagem de erro resumida"),
    ],
}


def header(ws, row, cols, widths=None):
    for j, c in enumerate(cols, 1):
        cell = ws.cell(row=row, column=j, value=c)
        cell.font, cell.fill, cell.border = H_FONT, H_FILL, BOX
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    if widths:
        for j, w in enumerate(widths, 1):
            ws.column_dimensions[get_column_letter(j)].width = w
    ws.row_dimensions[row].height = 30


def body(ws, r0, rows, fmts=None):
    for i, row in enumerate(rows):
        for j, v in enumerate(row, 1):
            if isinstance(v, float) and np.isnan(v):
                v = None
            c = ws.cell(row=r0 + i, column=j, value=v)
            c.font, c.border = BODY, BOX
            if fmts and fmts[j - 1]:
                c.number_format = fmts[j - 1]


def main():
    long = pd.read_csv(RES / "resultados_longo.csv")
    exe = pd.read_csv(RES / "execucoes.csv")
    pc = pd.read_csv(RES / "resumo" / "por_conjunto.csv")
    sizes = long.groupby("dataset")[["n_cases", "n_timepoints"]].first()
    pc = pc.join(sizes, on="dataset")
    pc["model"] = pc.model.map(SHORT)
    pc["cenario"] = pc.cenario.map(CEN)
    pc = pc.sort_values(["n_cases", "dataset", "model", "cenario"])
    snap = pd.to_datetime(exe.timestamp).max()
    ok = exe[exe.status.eq("ok") & exe.k_mode.eq("grid")]

    wb = Workbook()
    wb.calculation.fullCalcOnLoad = True

    # ---------------------------------------------------------------- Leia-me
    ws = wb.active
    ws.title = "Leia-me"
    ws.column_dimensions["A"].width = 120
    lines = [
        ("Linha de base: agrupadores profundos do aeon no UCR (resultados preliminares)", TITLE),
        (f"Fotografia dos resultados em {snap:%d/%m/%Y %H:%M} (a execução continua; números parciais).", BOLD),
        (f"Cobertura: {ok.dataset.nunique()} dos 128 conjuntos com algum resultado; "
         f"{len(ok)} pares (conjunto, modelo) concluídos, 0 falhas.", BODY),
        ("", BODY),
        ("Protocolo", BOLD),
        ("• 6 agrupadores do aeon 1.6.0 (TensorFlow): AEFCN, AEResNet, AEDCNN, AEBiGRU, AEAttentionBiGRU, AEDRNN; hiperparâmetros padrão.", BODY),
        ("• 200 épocas, lote 32, semente 0. Treino e teste do UCR concatenados (Paparrizos e Bogireddy, 2025); rótulos só nas métricas.", BODY),
        ("• Pré-processamento: NaN por interpolação linear; comprimentos diferentes reamostrados para o maior (Resizer); normalização z por série.", BODY),
        ("• Cada (conjunto, modelo) é treinado uma vez; k-means euclidiano sobre o latente salvo, para cada K de 2 a 70 (passo 2) e para K = nº de classes.", BODY),
        ("• K escolhido pela maior silhueta (empate: menor K). Métricas: RI, ARI, NMI, silhueta, Davies-Bouldin, perda de reconstrução.", BODY),
        ("", BODY),
        ("Abas", BOLD),
        ("• Resultados: uma linha por conjunto, modelo e cenário (K pela silhueta, K = nº de classes, melhor K pelo ARI).", BODY),
        ("• Resumo por modelo: médias calculadas por fórmula sobre a aba Resultados.", BODY),
        ("• ARI por conjunto: matriz conjunto × modelo com K = nº de classes (protocolo de Paparrizos e Bogireddy).", BODY),
        ("• Diagrama CD: posto médio do ARI, Wilcoxon com correção de Holm (aeon.visualisation).", BODY),
        ("• Execucoes: dispositivo, tempos (1ª época = compilação; mediana por época) e perdas de cada par.", BODY),
        ("• TF x PyTorch e Compatibilidade: benchmark de frameworks para a etapa 5 (06/11/2026).", BODY),
        ("", BODY),
        ("Cuidados de leitura", BOLD),
        ("• Resultados preliminares: os conjuntos prontos são os menores do UCR; não generalizar para o arquivo inteiro.", BODY),
        ("• Melhor K pelo ARI usa os rótulos: é um limite otimista, não um resultado não supervisionado.", BODY),
        ("• AEDRNN roda em CPU (não cabe na GPU de 4 GB) e tem só 4 conjuntos; fica fora do diagrama CD.", BODY),
        ("• AEDCNN: a perda de treino inclui penalidade L2 e a weight normalization do aeon ignora viés e ativação; ver docs/notas/achados-linha-de-base.md.", BODY),
        ("• recon_mse (MSE pura do modelo final) existe a partir do conjunto ECG200; nos 3 primeiros fica vazio.", BODY),
        ("", BODY),
        ("Fontes: results/aeon_dl/*.csv e results/bench_frameworks/* (gerado por experimentos/entrega/gerar_planilha.py).", NOTE),
    ]
    for i, (t, f) in enumerate(lines, 1):
        c = ws.cell(row=i, column=1, value=t)
        c.font = f
        c.alignment = Alignment(wrap_text=True, vertical="top")

    # ------------------------------------------------------------- Resultados
    ws = wb.create_sheet("Resultados")
    cols = ["Conjunto", "Modelo", "Cenário", "K", "Nº classes", "Silhueta", "RI", "ARI", "NMI",
            "recon_mse", "Nº séries", "Comprimento"]
    header(ws, 1, cols, [26, 18, 30, 6, 10, 10, 9, 9, 9, 11, 10, 12])
    rows = pc[["dataset", "model", "cenario", "k", "n_classes", "silhouette", "ri", "ari", "nmi",
               "recon_mse", "n_cases", "n_timepoints"]].values.tolist()
    f3 = "0.000"
    body(ws, 2, rows, [None, None, None, "0", "0", f3, f3, f3, f3, "0.0000", "#,##0", "#,##0"])
    n_res = len(rows) + 1
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = f"A1:L{n_res}"

    # --------------------------------------------------- Resumo por modelo
    ws = wb.create_sheet("Resumo por modelo")
    ws["A1"] = "Médias por modelo e cenário (fórmulas sobre a aba Resultados)"
    ws["A1"].font = TITLE
    r = 3
    for cen in CEN.values():
        ws.cell(row=r, column=1, value=cen).font = BOLD
        r += 1
        header(ws, r, ["Modelo", "Nº conjuntos", "RI (média)", "ARI (média)", "NMI (média)",
                       "K médio"], [30, 13, 12, 12, 12, 10])
        for m in MODELS:
            r += 1
            crit = f'Resultados!$B$2:$B${n_res},$A{r},Resultados!$C$2:$C${n_res},"{cen}"'
            vals = [m, f"=COUNTIFS({crit})"]
            for col in ("G", "H", "I", "D"):
                vals.append(f"=IF($B{r}=0,\"\",AVERAGEIFS(Resultados!${col}$2:${col}${n_res},{crit}))")
            body(ws, r, [vals], [None, "0", f3, f3, f3, "0.0"])
        r += 2
    c = ws.cell(row=r, column=1, value="Médias simples entre conjuntos; para comparar modelos, "
                "prefira os postos (aba Diagrama CD). AEDRNN tem menos conjuntos que os demais.")
    c.font = NOTE

    # --------------------------------------------------- ARI por conjunto
    ws = wb.create_sheet("ARI por conjunto")
    ws["A1"] = "ARI com K = nº de classes (fórmulas sobre a aba Resultados; vazio = ainda não executado)"
    ws["A1"].font = TITLE
    cen = CEN["k_igual_n_classes"]
    header(ws, 3, ["Conjunto"] + MODELS + ["Melhor modelo"], [26] + [16] * 6 + [20])
    dsets = list(dict.fromkeys(pc.dataset))
    for i, d in enumerate(dsets):
        rr = 4 + i
        row = [d]
        for j, m in enumerate(MODELS):
            col = get_column_letter(2 + j)
            crit = (f'Resultados!$A$2:$A${n_res},$A{rr},Resultados!$B$2:$B${n_res},{col}$3,'
                    f'Resultados!$C$2:$C${n_res},"{cen}"')
            row.append(f"=IF(COUNTIFS({crit})=0,\"\",SUMIFS(Resultados!$H$2:$H${n_res},{crit}))")
        row.append(f"=IF(COUNT(B{rr}:G{rr})=0,\"\",INDEX($B$3:$G$3,MATCH(MAX(B{rr}:G{rr}),B{rr}:G{rr},0)))")
        body(ws, rr, [row], [None] + [f3] * 6 + [None])
    last = 3 + len(dsets)
    rr = last + 1
    row = ["Média"] + [f"=IF(COUNT({get_column_letter(2+j)}4:{get_column_letter(2+j)}{last})=0,\"\","
                       f"AVERAGE({get_column_letter(2+j)}4:{get_column_letter(2+j)}{last}))" for j in range(6)] + [""]
    body(ws, rr, [row], [None] + [f3] * 6 + [None])
    for j in range(1, 9):
        ws.cell(row=rr, column=j).font = BOLD
    rr += 1
    row = ["Vitórias (melhor ARI)"] + [f'=COUNTIF($H$4:$H${last},{get_column_letter(2+j)}$3)' for j in range(6)] + [""]
    body(ws, rr, [row], [None] + ["0"] * 6 + [None])
    for j in range(1, 9):
        ws.cell(row=rr, column=j).font = BOLD
    ws.freeze_panes = "B4"

    # --------------------------------------------------------- Diagrama CD
    ws = wb.create_sheet("Diagrama CD")
    ws["A1"] = "Diagrama de diferença crítica do ARI (5 modelos de GPU; Wilcoxon + Holm, alfa = 0,1)"
    ws["A1"].font = TITLE
    ws["A2"] = ("Eixo = posto médio (1 = melhor). Modelos ligados pela barra não diferem "
                "significativamente. AEDRNN fica de fora (só 4 conjuntos).")
    ws["A2"].font = NOTE
    for k, (png, title) in enumerate([("cd_n_classes_ari.png", "K = nº de classes"),
                                      ("cd_silhueta_ari.png", "K pela silhueta")]):
        row0 = 4 + k * 22
        ws.cell(row=row0, column=1, value=title).font = BOLD
        img = Image(str(RES / "resumo" / png))
        img.width, img.height = img.width * 0.55, img.height * 0.55
        ws.add_image(img, f"A{row0 + 1}")

    # ---------------------------------------------------------- Execucoes
    ws = wb.create_sheet("Execucoes")
    e = ok.copy()
    e["model"] = e.model.map(SHORT)
    e = e.sort_values(["n_cases", "dataset", "model"])
    cols = ["Conjunto", "Modelo", "Dispositivo", "Status", "Nº séries", "Comprimento",
            "Treino (s)", "1ª época (s)", "Mediana por época (s)", "Busca de K (s)",
            "recon_loss (treino)", "recon_mse"]
    header(ws, 1, cols, [26, 18, 11, 8, 10, 12, 11, 11, 13, 12, 14, 11])
    rows = e[["dataset", "model", "device", "status", "n_cases", "n_timepoints", "train_time_s",
              "first_epoch_s", "epoch_median_s", "ksearch_time_s", "recon_loss_min",
              "recon_mse"]].values.tolist()
    body(ws, 2, rows, [None, None, None, None, "#,##0", "#,##0", "#,##0.0", "#,##0.0", "0.000",
                       "0.0", "0.0000", "0.0000"])
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = f"A1:L{len(rows) + 1}"

    # -------------------------------------------------------- TF x PyTorch
    ws = wb.create_sheet("TF x PyTorch")
    ws["A1"] = "Benchmark TensorFlow x PyTorch (mesmo modelo, mesmos dados, mesma GPU)"
    ws["A1"].font = TITLE
    ws["A2"] = ("RTX 3050 Laptop 4 GB (WSL2); 512 séries sintéticas, lote 32, Adam 1e-3, MSE, 20 épocas; "
                "TF 2.21 / Keras 3.15 x PyTorch 2.14 (cuDNN 9.10). 06/11/2026.")
    ws["A2"].font = NOTE
    b = pd.DataFrame([json.loads(x) for x in (BEN / "bench.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()])
    desc = {"fcn_ae": "AEFCN do aeon (convolucional)", "bigru_tanh": "BiGRU, tanh (cuDNN)",
            "bigru_relu": "BiGRU, ReLU (como no aeon)"}
    cols = ["Modelo", "Comprimento", "Parâmetros TF", "Parâmetros PyTorch", "Status TF",
            "Status PyTorch", "1ª época TF (s)", "1ª época PyTorch (s)", "Época TF (s)",
            "Época PyTorch (s)", "Razão TF / PyTorch", "Pico GPU TF (MB)", "Pico GPU PyTorch (MB)"]
    header(ws, 4, cols, [30, 12, 14, 16, 14, 16, 13, 15, 12, 14, 14, 14, 16])
    r = 5
    for L in (286, 1024):
        for mdl in ("fcn_ae", "bigru_tanh", "bigru_relu"):
            t = b[(b.framework == "tensorflow") & (b.model == mdl) & (b.length == L)].iloc[0]
            p = b[(b.framework == "pytorch") & (b.model == mdl) & (b.length == L)].iloc[0]
            st = {"ok": "ok", "erro_processo": "travou (30 min)", "nao_suportado": "não suportado"}
            row = [desc[mdl], L, t.get("params_trainable"), p.get("params_trainable"),
                   st.get(t.status, t.status), st.get(p.status, p.status),
                   t.get("first_epoch_s"), p.get("first_epoch_s"), t.get("epoch_median_s"),
                   p.get("epoch_median_s"), f"=IF(AND(ISNUMBER(I{r}),ISNUMBER(J{r})),I{r}/J{r},\"n/d\")",
                   t.get("peak_gpu_mb"), p.get("peak_gpu_mb")]
            row = [None if (isinstance(v, float) and np.isnan(v)) else v for v in row]
            body(ws, r, [row], [None, "#,##0", "#,##0", "#,##0", None, None, "0.0", "0.0", "0.000",
                                "0.000", '0.00"x"', "#,##0", "#,##0"])
            r += 1
    notes = [
        "Leitura: convolucional empata por época (razão perto de 1); recorrente com tanh é ~15x mais lento no TF; "
        "ReLU (padrão do aeon) não usa cuDNN no TF e não existe no nn.GRU do PyTorch.",
        "Parâmetros idênticos nos dois frameworks confirmam que o modelo comparado é o mesmo.",
        "TF com tanh e 1.024 pontos travou (sem uso de CPU/GPU) e foi encerrado no limite de 30 min; causa não investigada.",
        "Uma execução por caso: diferenças pequenas (razão 0,84 a 0,99) devem ser lidas como empate.",
    ]
    for i, t in enumerate(notes):
        ws.cell(row=r + 1 + i, column=1, value=t).font = NOTE

    # ------------------------------------------------------ Compatibilidade
    ws = wb.create_sheet("Compatibilidade")
    ws["A1"] = "O VMM oficial (TF 2.15 + tensorflow-probability) convive com o aeon 1.6.0 (Keras 3)?"
    ws["A1"].font = TITLE
    header(ws, 3, ["Teste", "Configuração", "Redes do aeon", "tfp.layers (VMM)", "Conclusão"],
           [8, 60, 16, 18, 50])
    rows = [
        ["A", "aeon==1.6.0 + tensorflow==2.15.* + tensorflow-probability==0.23.0 (pilha do VMM)",
         "-", "-", "Dependências impossíveis de satisfazer (uv pip compile)"],
        ["B1", "aeon 1.6.0 + TF 2.21 + tfp 0.25 + tf-keras; modo padrão (Keras 3)", "ok", "falha",
         "KerasTensor não pode entrar em função do TF"],
        ["B2", "o mesmo, com TF_USE_LEGACY_KERAS=1 (Keras 2)", "falha", "ok",
         "nome de escopo '__act_encoder_block0/' inválido"],
    ]
    body(ws, 4, rows)
    ws.cell(row=8, column=1, value="Conclusão: não rodam juntos em nenhum modo; o VMM precisa ser portado "
            "(para Keras 3 ou para PyTorch) em qualquer caminho.").font = BOLD
    for row in ws.iter_rows(min_row=4, max_row=6):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")

    # ------------------------------------------- dicionario de colunas (fim de cada aba)
    for nome, itens in DICIONARIO.items():
        ws = wb[nome]
        r = (52 if nome == "Diagrama CD" else ws.max_row) + 3  # imagens nao contam em max_row
        ws.cell(row=r, column=1, value="O que significa cada coluna").font = Font(name=F, size=12, bold=True)
        for i, (col, sig) in enumerate(itens, 1):
            c = ws.cell(row=r + i, column=1)
            c.value = CellRichText(TextBlock(InlineFont(rFont=F, sz=10, b=True), f"{col}: "),
                                   TextBlock(InlineFont(rFont=F, sz=10), sig))
            c.alignment = Alignment(wrap_text=False, vertical="top")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    wb.save(OUT)
    print(OUT, "| linhas em Resultados:", n_res - 1, "| conjuntos:", len(dsets))


if __name__ == "__main__":
    main()
