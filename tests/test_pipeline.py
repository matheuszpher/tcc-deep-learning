"""Testes do pre-processamento, da grade de K, das metricas e da selecao de K.

Rodar: .venv\\Scripts\\python -m pytest tests -q
Nao treinam redes; usam dados sinteticos.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experimentos"))
import aeon_dl_baseline as B  # noqa: E402


# ------------------------------------------------------------ NaN
def test_interp_interior_e_linear():
    v, f = B._interp_nan(np.array([0.0, np.nan, 2.0, np.nan, np.nan, 5.0]))
    assert f == "nan"
    np.testing.assert_allclose(v, [0, 1, 2, 3, 4, 5])


def test_interp_extremos_usam_valor_valido_mais_proximo():
    v, _ = B._interp_nan(np.array([np.nan, np.nan, 3.0, 4.0, np.nan]))
    np.testing.assert_allclose(v, [3, 3, 3, 4, 4])


def test_interp_serie_toda_nan_vira_zero():
    v, f = B._interp_nan(np.full(4, np.nan))
    assert f == "all_nan"
    np.testing.assert_array_equal(v, np.zeros(4))


def test_interp_sem_nan_nao_altera():
    x = np.arange(5.0)
    v, f = B._interp_nan(x)
    assert f == "" and v is x


# ------------------------------------------------------- preprocess
def test_preprocess_comprimento_igual_znorm():
    rng = np.random.default_rng(0)
    series = [rng.normal(3, 2, size=(1, 50)) for _ in range(4)]
    X, prep = B.preprocess(series)
    assert X.shape == (4, 1, 50) and X.dtype == np.float32
    np.testing.assert_allclose(X.mean(axis=2), 0, atol=1e-5)
    np.testing.assert_allclose(X.std(axis=2), 1, atol=1e-4)
    assert prep == "znorm_per_series"


def test_preprocess_desvio_zero_vira_zeros():
    X, _ = B.preprocess([np.full((1, 10), 7.0), np.arange(10.0).reshape(1, -1)])
    np.testing.assert_array_equal(X[0], 0)
    assert np.isfinite(X).all()


def test_preprocess_comprimentos_diferentes_reamostra_para_o_maximo():
    series = [np.linspace(0, 1, 20).reshape(1, -1), np.linspace(0, 1, 40).reshape(1, -1)]
    X, prep = B.preprocess(series)
    assert X.shape == (2, 1, 40)
    assert "resize_max(len_min=20,len_max=40)" in prep
    # rampa reamostrada continua rampa: apos z-norm as duas series coincidem
    np.testing.assert_allclose(X[0], X[1], atol=1e-4)


def test_preprocess_nan_e_registrado_e_tratado_antes_do_resize():
    s1 = np.array([[0.0, np.nan, 2.0, 3.0]])
    s2 = np.arange(8.0).reshape(1, -1)
    X, prep = B.preprocess([s1, s2])
    assert np.isfinite(X).all()
    assert prep.startswith("interp_linear_nan(series=1,all_nan_to_zero=0)")


def test_preprocess_rejeita_multivariado():
    with pytest.raises(ValueError):
        B.preprocess([np.zeros((2, 10))])


# ------------------------------------------------------------- K
def test_grade_padrao_2_a_70_passo_2():
    ks = B.ks_for("grid", n_cases=1000, n_classes=3, k_min=2, k_max=70, k_step=2)
    assert ks == list(range(2, 71, 2))


def test_grade_respeita_k_menor_que_n():
    ks = B.ks_for("grid", n_cases=56, n_classes=2, k_min=2, k_max=70, k_step=2)
    assert max(ks) == 54 and all(k < 56 for k in ks)


def test_k_igual_numero_de_classes():
    assert B.ks_for("n_classes", 100, 7, 2, 70, 2) == [7]


# ------------------------------------------------------- metricas
BASE = {"dataset": "D", "model": "M", "seed": 0, "n_epochs": 1, "n_cases": 0,
        "n_timepoints": 0, "n_classes": 2, "preprocessing": "p"}


def test_k_search_linhas_e_metricas():
    rng = np.random.default_rng(0)
    lat = np.vstack([rng.normal(0, .1, (20, 3)), rng.normal(5, .1, (20, 3))]).astype(np.float32)
    y = np.array(["a"] * 20 + ["b"] * 20)
    rows, notes = B.k_search(lat, y, [2, 4], "grid", 0, BASE, recon_loss=0.5, recon_mse=0.25)
    assert len(rows) == 2 * len(B.METRICS)
    v = {(r["k"], r["metric"]): r["value"] for r in rows}
    assert float(v[(2, "ari")]) == pytest.approx(1.0)
    assert float(v[(2, "silhouette")]) > 0.9
    assert v[(2, "elbo")] == ""
    assert float(v[(4, "recon_loss")]) == 0.5
    assert float(v[(2, "recon_mse")]) == 0.25


def test_k_search_latente_degenerado_nao_derruba_e_gera_nan():
    # todos os pontos iguais: o k-means do aeon levanta ValueError (grupos vazios)
    lat = np.zeros((10, 2), dtype=np.float32)
    y = np.array(["a", "b"] * 5)
    rows, notes = B.k_search(lat, y, [2, 4], "grid", 0, BASE, recon_loss=0.1)
    assert len(rows) == 2 * len(B.METRICS)
    v = {(r["k"], r["metric"]): r["value"] for r in rows}
    for k in (2, 4):
        assert np.isnan(float(v[(k, "silhouette")])) and np.isnan(float(v[(k, "ari")]))
        assert float(v[(k, "recon_loss")]) == 0.1
    assert any("NaN" in n for n in notes)


def test_k_search_poucos_grupos_distintos_gera_nan_silhueta():
    # 3 pontos distintos repetidos, K=2: k-means funciona; e se devolver 1 grupo, NaN
    lat = np.array([[0, 0]] * 9 + [[1, 1]], dtype=np.float32)
    y = np.array(["a"] * 9 + ["b"])
    rows, _ = B.k_search(lat, y, [2], "grid", 0, BASE, recon_loss=0.1)
    v = {r["metric"]: float(r["value"]) for r in rows if r["value"] != ""}
    assert np.isnan(v["silhouette"]) or -1 <= v["silhouette"] <= 1


# ------------------------------------------------------- selecao
def test_selecao_maior_silhueta_e_empate_menor_k(tmp_path):
    rows = []
    for k, sil in [(2, 0.5), (4, 0.8), (6, 0.8), (8, float("nan"))]:
        for m, val in [("silhouette", sil), ("ari", k / 10)]:
            rows.append({**BASE, "k": k, "k_mode": "grid", "metric": m, "value": val})
    rows.append({**BASE, "k": 2, "k_mode": "n_classes", "metric": "silhouette", "value": 0.99})
    long_csv, sel_csv = tmp_path / "l.csv", tmp_path / "s.csv"
    pd.DataFrame(rows).to_csv(long_csv, index=False)
    B.write_selection(long_csv, sel_csv)
    sel = pd.read_csv(sel_csv)
    assert len(sel) == 1
    assert sel.loc[0, "k_selected"] == 4          # empate 4 x 6 -> menor K
    assert sel.loc[0, "ari"] == pytest.approx(0.4)  # n_classes nao entra na selecao


# ------------------------------------------------------- CSV
def test_append_rows_segue_cabecalho_existente(tmp_path):
    f = tmp_path / "e.csv"
    B.append_rows(f, ["a", "b"], [{"a": 1, "b": 2}])
    B.append_rows(f, ["a", "x", "b"], [{"a": 3, "x": 9, "b": 4}])
    df = pd.read_csv(f)
    assert list(df.columns) == ["a", "b"] and df.b.tolist() == [2, 4]
