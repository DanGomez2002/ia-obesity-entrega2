"""Entrega 2 (etapas 06-09): registro de pasos, archivos de resultados y split sin fugas.

Ninguna prueba entrena modelos: leen el código, los CSV de data/ y los splits.
"""
from pathlib import Path

import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]


def _etapas_e2():
    from etapas import e06_naive_bayes, e07_svm, e08_redes_neuronales, e09_comparacion_final
    return [m.etapa for m in (e06_naive_bayes, e07_svm, e08_redes_neuronales, e09_comparacion_final)]


def _linea(ubicacion):
    archivo, linea = ubicacion.rsplit(":", 1)
    return (RAIZ / archivo).read_text(encoding="utf-8").splitlines()[int(linea) - 1]


def test_numeracion_continua_y_pasos_unicos():
    etapas = _etapas_e2()
    assert [e.numero for e in etapas] == ["06", "07", "08", "09"]
    for e in etapas:
        ids = [p.id for p in e.pasos]
        assert len(ids) == len(set(ids)), f"etapa {e.numero}: pasos repetidos"
        assert ids[0] == "1"


def test_parametros_declarados_apuntan_a_su_definicion():
    from src.pasos import ubicar
    for etapa in _etapas_e2():
        modulo = etapa.pasos[0].fn.__globals__
        for nombre in list(etapa.globales) + [n for p in etapa.pasos for n in p.parametros]:
            ubicacion = ubicar(nombre, modulo)
            assert ubicacion, f"etapa {etapa.numero}: no se encuentra {nombre!r}"
            corto = nombre.rsplit(".", 1)[-1]
            linea = _linea(ubicacion)
            assert f"def {corto}(" in linea or linea.startswith(f"{corto} ="), (nombre, linea)


def test_run_all_y_main_conocen_las_nueve_etapas():
    import run_all
    assert [m[1:3] for m in run_all.MODULOS] == [f"{i:02d}" for i in range(1, 10)]
    import main
    assert list(main._cargar_etapas()) == [f"{i:02d}" for i in range(1, 10)]


# ----------------------------------------------------------------------------
# Archivos de resultados
# ----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def res():
    from src import resultados
    if not resultados.RESULTADOS_E2.exists():
        pytest.skip("falta data/resultados_entrega2.csv (se crea con run_all.py)")
    return resultados


def test_csv_con_el_esquema_comun(res):
    for ruta in (res.RESULTADOS_E1, res.RESULTADOS_E2):
        assert list(pd.read_csv(ruta).columns) == res.COLUMNAS, ruta.name


def test_entrega1_congelada_coincide_con_resultados_texto(res):
    actual = res.construir_entrega1(*res._huellas_split())
    pd.testing.assert_frame_equal(res.leer(res.RESULTADOS_E1), actual, check_dtype=False)


def test_mismo_test_de_418_dentro_de_cada_tarea(res):
    df = pd.concat([res.leer(res.RESULTADOS_E1), res.leer(res.RESULTADOS_E2)])
    sup = df[df["tarea"].isin(res.TAREAS_SUPERVISADAS)]
    for tarea, g in sup.groupby("tarea"):
        assert set(g["n_test"]) == {418}, tarea
        assert g["huella_test"].nunique() == 1, f"{tarea}: modelos evaluados en tests distintos"


def test_entrega2_una_fila_por_familia_y_tarea(res):
    e2 = res.leer(res.RESULTADOS_E2)
    esperado = {("clasificacion_A", "06"), ("clasificacion_B", "06"),
                ("clasificacion_A", "07"), ("clasificacion_B", "07"), ("regresion", "07"),
                ("clasificacion_A", "08"), ("clasificacion_B", "08"), ("regresion", "08")}
    vistos = {(t, f.split()[1]) for t, f in zip(e2["tarea"], e2["fuente"])}
    assert vistos == esperado and len(e2) == len(esperado)


def test_leer_tabla_md_deja_las_columnas_de_texto(tmp_path, monkeypatch, res):
    md = tmp_path / "t.md"
    md.write_text("### T\n\n|   | x | p |\n|:--|--:|:--|\n| a | 1.5 | {'C': 1} |\n| b | — | {'C': 2} |\n",
                  encoding="utf-8")
    monkeypatch.setattr(res, "ROOT", tmp_path)
    t = res.leer_tabla_md("t.md", "T")
    assert t["x"].dtype.kind == "f" and pd.isna(t.loc["b", "x"])
    assert t.loc["a", "p"] == "{'C': 1}"
    with pytest.raises(ValueError):
        res.leer_tabla_md("t.md", "no existe")


# ----------------------------------------------------------------------------
# Redes neuronales: la validación sale del train y el test no se toca
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("tarea", ["clasificacion_A", "clasificacion_B", "regresion"])
def test_validacion_de_la_red_sale_del_train(tarea):
    from etapas import e08_redes_neuronales as e08
    d = e08.datos(tarea)
    sub, val, te, tr = map(set, (d["idx_sub"], d["idx_val"], d["idx_test"], d["idx_train"]))
    assert (len(sub), len(val), len(te)) == (1418, 251, 418)
    assert sub | val == tr and not (sub & val)
    assert not (te & tr)
    assert len(d["columnas"]) == {"clasificacion_A": 20, "clasificacion_B": 18, "regresion": 19}[tarea]
    if tarea == "regresion":
        assert "NObeyesdad" not in d["columnas"] and "Weight" not in d["columnas"]
