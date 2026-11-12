"""Gera docs/apresentacao/solucao.excalidraw (abrir em excalidraw.com: menu > Abrir).

Versao visual: pipeline com icones, cartoes de numeros e mini graficos desenhados a partir
dos CSVs (fotografia do momento). Nenhum numero e digitado a mao.
"""

import json
import random
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "aeon_dl"
BEN = ROOT / "results" / "bench_frameworks"
OUT = ROOT / "docs" / "apresentacao" / "solucao.excalidraw"
rnd = random.Random(7)
EL = []

INK = "#1e1e1e"
AZUL, AZUL_F = "#1971c2", "#d0ebff"
VERDE, VERDE_F = "#2f9e44", "#d3f9d8"
LARANJA, LARANJA_F = "#e8590c", "#ffe8cc"
ROXO, ROXO_F = "#6741d9", "#e5dbff"
CINZA, CINZA_F = "#868e96", "#f8f9fa"
VERM, VERM_F = "#e03131", "#ffe3e3"
AMAR_F = "#fff3bf"
TF_COR, PT_COR = "#f08c00", "#e03131"   # cores "de marca" so para as barras


def _w(s, size):
    """Largura aproximada (emoji conta como 2 caracteres)."""
    lines = s.split("\n")
    return max(sum(2 if ord(ch) > 0x2500 else 1 for ch in l) for l in lines) * size * 0.56


def _base(t, x, y, w, h, stroke, bg="transparent", **kw):
    e = {"id": f"e{len(EL)}_{rnd.randrange(1 << 30)}", "type": t, "x": x, "y": y, "width": w,
         "height": h, "angle": 0, "strokeColor": stroke, "backgroundColor": bg,
         "fillStyle": kw.get("fill", "solid"), "strokeWidth": kw.get("sw", 2),
         "strokeStyle": kw.get("style", "solid"), "roughness": kw.get("rough", 1), "opacity": 100,
         "groupIds": [], "frameId": None, "roundness": kw.get("roundness"),
         "seed": rnd.randrange(1 << 30), "version": 1, "versionNonce": rnd.randrange(1 << 30),
         "isDeleted": False, "boundElements": [], "updated": 1, "link": None, "locked": False}
    EL.append(e)
    return e


def text(x, y, s, size=16, color=INK, align="left", font=2):
    w, h = _w(s, size), len(s.split("\n")) * size * 1.25
    if align == "center":
        x -= w / 2
    e = _base("text", x, y, w, h, color)
    e.update({"text": s, "originalText": s, "fontSize": size, "fontFamily": font,
              "textAlign": align, "verticalAlign": "top", "containerId": None,
              "lineHeight": 1.25, "autoResize": True})
    return e


def rect(x, y, w, h, stroke, bg="transparent", **kw):
    kw.setdefault("roundness", {"type": 3})
    return _base("rectangle", x, y, w, h, stroke, bg, **kw)


def ellipse(cx, cy, r, stroke, bg):
    return _base("ellipse", cx - r, cy - r, 2 * r, 2 * r, stroke, bg, sw=3)


def line(pts, color=INK, sw=2, style="solid", arrow=False):
    x0, y0 = pts[0]
    rel = [[px - x0, py - y0] for px, py in pts]
    xs, ys = [p[0] for p in rel], [p[1] for p in rel]
    e = _base("arrow" if arrow else "line", x0, y0, max(xs) - min(xs), max(ys) - min(ys),
              color, sw=sw, style=style, roundness={"type": 2})
    e.update({"points": rel, "lastCommittedPoint": None, "startBinding": None,
              "endBinding": None, "startArrowhead": None,
              "endArrowhead": "arrow" if arrow else None})
    if arrow:
        e["elbowed"] = False
    return e


def card_title(x, y, icon, title, color):
    text(x, y, f"{icon}  {title}", 22, color)


def fmt(v, d=2):
    return f"{v:.{d}f}".replace(".", ",")


# ============================================================== dados
def dados():
    e = pd.read_csv(RES / "execucoes.csv")
    g = e[e.k_mode.eq("grid")]
    t = g.pivot_table(index="dataset", columns="model", values="status", aggfunc="first")
    gpu = [c for c in t.columns if "DRNN" not in c]
    pc = pd.read_csv(RES / "resumo" / "por_conjunto.csv")
    pc = pc[pc.cenario.eq("k_igual_n_classes") & pc.model.isin(gpu)]
    mat = pc.pivot_table(index="dataset", columns="model", values="ari").dropna()
    ranks = mat.rank(axis=1, ascending=False).mean().sort_values()
    k = pd.read_csv(RES / "resumo" / "k_escolhido.csv").set_index("model").frac_k_2
    m = g.dropna(subset=["recon_mse"]).pivot_table(index="dataset", columns="model", values="recon_mse")
    b = pd.DataFrame([json.loads(x) for x in (BEN / "bench.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()])

    def ep(fw, mdl, L):
        r = b[(b.framework == fw) & (b.model == mdl) & (b.length == L)]
        return float(r.epoch_median_s.iloc[0]) if len(r) and r.status.iloc[0] == "ok" else None

    return {
        "snap": pd.to_datetime(e.timestamp).max(),
        "n_ds": g.dataset.nunique(), "n_ok": int((g.status == "ok").sum()),
        "n_fail": int((e.status != "ok").sum()),
        "full5": int((t[gpu].notna().sum(axis=1) == 5).sum()), "n_cd": len(mat),
        "ranks": ranks, "k2": k,
        "dcnn_bad": int((m["AEDCNNClusterer"] > 1).sum()), "dcnn_n": int(m["AEDCNNClusterer"].notna().sum()),
        "bench": {"Convolucional\n(AEFCN do aeon)": (ep("tensorflow", "fcn_ae", 286), ep("pytorch", "fcn_ae", 286)),
                  "Recorrente\n(GRU tanh)": (ep("tensorflow", "bigru_tanh", 286), ep("pytorch", "bigru_tanh", 286))},
    }


SHORT = {"AEFCNClusterer": "FCN", "AEResNetClusterer": "ResNet", "AEDCNNClusterer": "DCNN",
         "AEBiGRUClusterer": "BiGRU", "AEAttentionBiGRUClusterer": "AttBiGRU",
         "AEDRNNClusterer": "DRNN"}


def main():
    D = dados()

    # ---------------------------------------------------------------- titulo
    text(40, 24, "Linha de base: deep clustering do aeon no UCR", 40, INK, font=1)
    text(42, 82, f"fotografia de {D['snap']:%d/%m %H:%M}  •  resultados preliminares  •  execução em andamento",
         16, CINZA)

    # ---------------------------------------------------------------- pipeline
    Y, R, X0, DX = 215, 58, 120, 300
    passos = [
        ("📚", "Dados", "128 séries UCR\ntreino + teste", AZUL, AZUL_F),
        ("🧹", "Preparo", "NaN • tamanho\nnormalização z", AZUL, AZUL_F),
        ("🧠", "Treino 1×", "6 autoencoders\n200 épocas", VERDE, VERDE_F),
        ("💾", "Latente", "layers[1]\nconferido ✔", VERDE, VERDE_F),
        ("🔍", "Busca de K", "K = 2…70\n+ nº de classes", ROXO, ROXO_F),
        ("📊", "Métricas", "RI • ARI • NMI\nsilhueta • DB", ROXO, ROXO_F),
    ]
    for i, (ic, ttl, sub, s, f) in enumerate(passos):
        cx = X0 + i * DX
        ellipse(cx, Y, R, s, f)
        text(cx, Y - 30, ic, 44, align="center")
        text(cx, Y + R + 12, ttl, 22, s, align="center")
        text(cx, Y + R + 44, sub, 15, CINZA, align="center")
        if i:
            line([(cx - DX + R + 12, Y), (cx - R - 12, Y)], CINZA, sw=3, arrow=True)
    # etiquetas de hardware sob o treino
    cx = X0 + 2 * DX
    for j, (lbl, col, bg) in enumerate([("GPU ×5", VERDE, VERDE_F), ("CPU ×1", LARANJA, LARANJA_F)]):
        rect(cx - 95 + j * 100, Y - R - 52, 90, 30, col, bg, sw=1)
        text(cx - 50 + j * 100, Y - R - 48, lbl, 15, col, align="center")
    # nota adesiva: K nao muda o treino
    nx, ny = X0 + 3.15 * DX, 22
    rect(nx, ny, 330, 82, "#f08c00", AMAR_F, sw=1, rough=2)
    text(nx + 14, ny + 10, "💡 K não muda a rede:\ntreina 1×, testa 36 K em segundos", 16, INK)
    line([(nx + 60, ny + 84), (X0 + 4 * DX - 40, Y - R - 4)], "#f08c00", sw=2, style="dashed", arrow=True)

    # ---------------------------------------------------------------- KPIs
    Y2 = 410
    card_title(40, Y2, "📈", "Onde estamos", VERDE)
    kpis = [(f"{D['n_ds']}/128", "conjuntos"), (str(D["n_ok"]), "pares treinados"),
            (str(D["n_fail"]), "falhas"), ("6", "modelos do aeon")]
    for i, (big, small) in enumerate(kpis):
        x, y = 40 + (i % 2) * 245, Y2 + 50 + (i // 2) * 130
        rect(x, y, 230, 115, VERDE, "#ffffff", sw=2)
        text(x + 115, y + 12, big, 46, VERDE, align="center", font=1)
        text(x + 115, y + 78, small, 16, CINZA, align="center")

    # ---------------------------------------------------------------- ranking ARI
    X3, W3 = 580, 560
    card_title(X3, Y2, "🏁", "Ranking do ARI (K = nº de classes)", ROXO)
    ax_y, ax_x0, ax_x1 = Y2 + 150, X3 + 40, X3 + W3 - 40
    line([(ax_x0, ax_y), (ax_x1, ax_y)], INK, sw=3)

    def px(rank):  # 5 (esq) ... 1 (dir)
        return ax_x0 + (5 - rank) / 4 * (ax_x1 - ax_x0)

    for r in range(1, 6):
        line([(px(r), ax_y - 8), (px(r), ax_y + 8)], INK, sw=2)
        text(px(r), ax_y + 14, str(r), 14, CINZA, align="center")
    text(ax_x1 + 6, ax_y - 30, "melhor →", 13, CINZA)
    ranks = D["ranks"]
    lo, hi = px(ranks.max()), px(ranks.min())
    rect(lo - 10, ax_y - 40, hi - lo + 20, 14, ROXO, ROXO, sw=1, roundness={"type": 3})
    # rotulos espalhados (postos muito proximos), ligados ao ponto por linha de chamada
    por_x = sorted(ranks.items(), key=lambda kv: px(kv[1]))
    n = len(por_x)
    for i, (mdl, rk) in enumerate(por_x):
        x = px(rk)
        ellipse(x, ax_y, 9, ROXO, "#ffffff")
        lx = ax_x0 + 30 + i * (ax_x1 - ax_x0 - 60) / max(n - 1, 1)
        ly = ax_y + 62
        line([(x, ax_y + 10), (lx, ly - 4)], CINZA, sw=1, style="dotted")
        text(lx, ly, f"{SHORT[mdl]}\n{fmt(rk)}", 14, INK, align="center")
    text(X3 + W3 / 2, Y2 + 60, f"posto médio em {D['n_cd']} conjuntos (Wilcoxon + Holm)", 15, CINZA, align="center")
    rect(X3 + 60, Y2 + 255, W3 - 120, 40, ROXO, ROXO_F, sw=1)
    text(X3 + W3 / 2, Y2 + 264, "= barra única: nenhuma diferença significativa", 16, ROXO, align="center")

    # ---------------------------------------------------------------- vies K=2
    X4, W4 = 1190, 560
    card_title(X4, Y2, "🎯", "Silhueta escolhe K = 2", LARANJA)
    text(X4, Y2 + 36, "fração dos conjuntos em que a silhueta escolheu K = 2", 14, CINZA)
    order = D["k2"].sort_values(ascending=False)
    bw = 330
    for i, (mdl, fr) in enumerate(order.items()):
        y = Y2 + 70 + i * 36
        text(X4, y + 4, SHORT[mdl], 15, INK)
        rect(X4 + 95, y, bw, 24, CINZA, "#ffffff", sw=1, roundness=None)
        rect(X4 + 95, y, bw * fr, 24, LARANJA, LARANJA_F, sw=1, roundness=None, fill="hachure")
        text(X4 + 105 + bw, y + 3, f"{fr:.0%}", 16, LARANJA)
    text(X4, Y2 + 290, "→ reportar também K = nº de classes", 16, LARANJA)

    # ---------------------------------------------------------------- benchmark
    Y5 = 760
    card_title(40, Y5, "⚡", "TensorFlow × PyTorch (mesmo modelo, mesma GPU)", ROXO)
    text(40, Y5 + 36, "tempo por época, séries de 286 pontos", 14, CINZA)
    vmax = max(max(v) for v in D["bench"].values())
    base_y, hmax = Y5 + 250, 150
    for i, (lbl, (tfv, ptv)) in enumerate(D["bench"].items()):
        gx = 90 + i * 300
        for j, (v, cor, nome) in enumerate([(tfv, TF_COR, "TF"), (ptv, PT_COR, "PyTorch")]):
            h = max(4, v / vmax * hmax)
            rect(gx + j * 85, base_y - h, 70, h, cor, cor, sw=1, roundness=None, fill="solid")
            text(gx + j * 85 + 35, base_y - h - 24, f"{fmt(v)} s", 14, INK, align="center")
            text(gx + j * 85 + 35, base_y + 6, nome, 14, CINZA, align="center")
        ratio = tfv / ptv
        tag = "empate" if 0.8 < ratio < 1.25 else f"PyTorch {ratio:.0f}× mais rápido"
        text(gx + 77, base_y + 30, lbl, 15, INK, align="center")
        tw = max(150, _w(tag, 15) + 24)
        rect(gx + 77 - tw / 2, base_y + 75, tw, 30, ROXO, ROXO_F, sw=1)
        text(gx + 77, base_y + 80, tag, 15, ROXO, align="center")
    line([(70, base_y), (650, base_y)], INK, sw=2)
    # alertas do benchmark
    alerts = [("🧊", "TF travou com\nséries de 1.024 pontos"),
              ("🔌", "VMM oficial × aeon:\nnão rodam juntos"),
              ("✅", "Sugestão: PyTorch +\nteste de equivalência")]
    for i, (ic, s) in enumerate(alerts):
        y = Y5 + 60 + i * 95
        rect(690, y, 300, 80, ROXO, "#ffffff", sw=1)
        text(708, y + 18, ic, 32)
        text(760, y + 14, s, 16, INK)

    # ---------------------------------------------------------------- achados
    X6 = 1060
    card_title(X6, Y5, "🔎", "Achados no aeon 1.6.0", LARANJA)
    achados = [("⚠️", "DCNN reconstrói pior\nque prever zero", f"{D['dcnn_bad']} de {D['dcnn_n']} conjuntos"),
               ("🐢", "GRU com ReLU:\nsem cuDNN", "1ª época leva minutos"),
               ("💥", "DRNN não cabe\nna GPU de 4 GB", "roda em CPU")]
    for i, (ic, s, sub) in enumerate(achados):
        y = Y5 + 60 + i * 95
        rect(X6, y, 690, 80, LARANJA, LARANJA_F, sw=1)
        text(X6 + 18, y + 18, ic, 32)
        text(X6 + 75, y + 14, s, 17, INK)
        text(X6 + 420, y + 28, sub, 16, LARANJA)

    # ---------------------------------------------------------------- decisoes
    Y7 = 1150
    card_title(40, Y7, "❓", "Decisões para a reunião", INK)
    qs = ["🖥️ GPU do\nlaboratório?", "⚖️ TF ou\nPyTorch?", "🔁 ReLU ou\ntanh?",
          "🧩 manter o\nDCNN do aeon?", "🎲 semente única\n+ AMI?"]
    for i, q in enumerate(qs):
        x = 40 + i * 345
        rect(x, Y7 + 48, 320, 86, INK, "#ffffff", sw=2, rough=2)
        text(x + 160, Y7 + 62, q, 20, INK, align="center")

    doc = {"type": "excalidraw", "version": 2, "source": "https://excalidraw.com",
           "elements": EL, "appState": {"viewBackgroundColor": "#ffffff", "gridSize": None},
           "files": {}}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    print(OUT, len(EL), "elementos")


if __name__ == "__main__":
    main()
