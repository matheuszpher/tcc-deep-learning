"""Resumo dos resultados a partir do CSV longo (nao treina nada).

Gera em <out>/resumo/:
- por_conjunto.csv : RI/ARI/NMI por conjunto, modelo e cenario (com o K usado);
- cenarios.csv     : media/mediana de ARI, NMI e RI por modelo em tres cenarios
                     (K pela silhueta | K = n_classes | melhor K pelo ARI = limite otimista);
- k_escolhido.csv  : distribuicao do K escolhido pela silhueta (vies para K pequeno?);
- cobertura.csv    : quantos conjuntos cada modelo completou / falhou;
- cd_<cenario>_ari.png : diagrama de diferenca critica (Wilcoxon + Holm, padrao do aeon),
                     so com conjuntos em que todos os modelos tem resultado.

Uso: python experimentos/resumo.py --out results/aeon_dl
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXT = ["ari", "nmi", "ri"]


def wide(df: pd.DataFrame) -> pd.DataFrame:
    return df.pivot_table(index=["dataset", "model", "k", "k_mode", "n_classes"],
                          columns="metric", values="value", aggfunc="first").reset_index()


def scenarios(w: pd.DataFrame) -> pd.DataFrame:
    g = w[w.k_mode == "grid"]
    sil = (g.dropna(subset=["silhouette"])
             .sort_values(["silhouette", "k"], ascending=[False, True])
             .groupby(["dataset", "model"]).head(1).assign(cenario="silhueta"))
    ncl = w[w.k_mode == "n_classes"].assign(cenario="k_igual_n_classes")
    best = (g.dropna(subset=["ari"]).sort_values(["ari", "k"], ascending=[False, True])
              .groupby(["dataset", "model"]).head(1)
              .assign(cenario="melhor_k_por_ari (usa rotulos; limite otimista)"))
    return pd.concat([sil, ncl, best], ignore_index=True)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(ROOT / "results" / "aeon_dl"))
    p.add_argument("--min-datasets-cd", type=int, default=5)
    a = p.parse_args(argv)
    out = Path(a.out)
    dst = out / "resumo"
    dst.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(out / "resultados_longo.csv")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    w = wide(df)
    sc = scenarios(w)

    pc = sc[["dataset", "model", "cenario", "k", "n_classes", "silhouette", "ri", "ari", "nmi"]
            + [c for c in ("recon_mse",) if c in sc.columns]]
    pc.sort_values(["dataset", "model", "cenario"]).round(4).to_csv(dst / "por_conjunto.csv",
                                                                     index=False)

    tab = (sc.groupby(["cenario", "model"])[EXT]
             .agg(["mean", "median", "count"]).round(4))
    tab.to_csv(dst / "cenarios.csv")
    print("\n== ARI/NMI/RI por cenario e modelo (media, mediana, n conjuntos)")
    print(tab.to_string())

    sel = sc[sc.cenario == "silhueta"].merge(
        w[w.k_mode == "n_classes"][["dataset", "model", "k"]].rename(columns={"k": "k_real"}),
        on=["dataset", "model"], how="left")
    kd = (sel.assign(k_igual_2=sel.k == 2, k_menor_igual_real=sel.k <= sel.n_classes)
             .groupby("model").agg(n=("k", "size"), k_mediano=("k", "median"),
                                   frac_k_2=("k_igual_2", "mean"),
                                   frac_k_le_n_classes=("k_menor_igual_real", "mean"))
             .round(3))
    kd.to_csv(dst / "k_escolhido.csv")
    print("\n== K escolhido pela silhueta")
    print(kd.to_string())

    ex = out / "execucoes.csv"
    if ex.exists():
        e = pd.read_csv(ex)
        cov = e.groupby(["model", "status"])["dataset"].nunique().unstack(fill_value=0)
        cov.to_csv(dst / "cobertura.csv")
        print("\n== Cobertura (conjuntos distintos por status)")
        print(cov.to_string())

    try:
        import matplotlib
        matplotlib.use("Agg")
        from aeon.visualisation import plot_critical_difference
    except Exception as err:  # noqa: BLE001
        print(f"\n(sem diagrama CD: {err})")
        return
    for cen, tag in [("silhueta", "silhueta"), ("k_igual_n_classes", "n_classes")]:
        m = sc[sc.cenario == cen].pivot_table(index="dataset", columns="model", values="ari")
        cnt = m.notna().sum()
        # modelos com cobertura bem menor (< 75% do maximo) ficam de fora; os demais entram
        # e o diagrama usa so os conjuntos em que todos eles tem resultado
        fora = list(cnt[cnt < 0.75 * cnt.max()].index)
        if fora:
            print(f"\n(CD {tag}: fora por cobertura menor: {fora})")
        m = m.drop(columns=fora).dropna()
        if m.shape[0] < a.min_datasets_cd or m.shape[1] < 2:
            print(f"\n(CD {tag}: so {m.shape[0]} conjuntos completos; minimo {a.min_datasets_cd})")
            continue
        fig, _ = plot_critical_difference(m.to_numpy(), list(m.columns))
        fig.savefig(dst / f"cd_{tag}_ari.png", bbox_inches="tight", dpi=150)
        print(f"\nCD {tag}: {m.shape[0]} conjuntos x {m.shape[1]} modelos -> cd_{tag}_ari.png")


if __name__ == "__main__":
    main()
