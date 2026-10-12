"""Resultados de las dos entregas en un formato común, para compararlos en la etapa 09.

- data/resultados_entrega1.csv : CONGELADO. Se construye una sola vez LEYENDO las tablas ya
  ejecutadas de la Entrega 1 en resultados_texto/ (no se re-entrena ningún modelo viejo):
      python -m src.resultados --congelar-entrega1
  Si el archivo ya existe no se sobrescribe: se compara con lo que hay en resultados_texto/
  y se informa de cualquier diferencia (para sobrescribir: --forzar).
- data/resultados_entrega2.csv : lo escriben las etapas 06-08 al terminar.

Una fila por (tarea, modelo). Lo que no aplica o no se midió queda vacío, nunca se estima.
`huella_test` identifica las filas del test (hash de sus índices): si coincide en todas las
filas, todos los modelos se evaluaron sobre el MISMO test de 418 instancias.
"""
import hashlib
import sys

import numpy as np
import pandas as pd

from config import DATA_DIR, ROOT

RESULTADOS_E1 = DATA_DIR / "resultados_entrega1.csv"
RESULTADOS_E2 = DATA_DIR / "resultados_entrega2.csv"
_TEXTO = ROOT / "resultados_texto"

COLUMNAS = [
    "tarea", "modelo",
    "accuracy", "f1_macro",                      # clasificación (test)
    "MAE", "RMSE", "R2", "R2_adj",               # regresión (test)
    "cv_score", "cv_metrica",                    # validación cruzada del GridSearch
    "log_loss", "tiempo_entrenamiento_s", "n_parametros", "n_vectores_soporte", "epocas",
    "mejores_params", "n_test", "huella_test",
    "silueta", "ARI", "NMI", "pureza",           # sólo clustering_nota (no supervisado)
    "nota", "fuente",
]

TAREAS_SUPERVISADAS = ["clasificacion_A", "clasificacion_B", "regresion"]


def huella_test(indices) -> str:
    """Hash corto de los índices del test (independiente del orden)."""
    texto = ",".join(str(int(i)) for i in sorted(indices))
    return hashlib.sha256(texto.encode()).hexdigest()[:12]


def leer_tabla_md(ruta, titulo: str) -> pd.DataFrame:
    """Lee la tabla Markdown que va bajo '### <titulo>' en un .md de resultados_texto/.

    La primera columna es el índice (nombre del modelo). '—' se lee como NaN.
    Lanza ValueError si el título no está: así nunca se rellena un número a ciegas."""
    lineas = (ROOT / ruta).read_text(encoding="utf-8").splitlines()
    try:
        i = lineas.index(f"### {titulo}")
    except ValueError:
        raise ValueError(f"No está la tabla '{titulo}' en {ruta}") from None
    filas = []
    for linea in lineas[i + 1:]:
        if linea.startswith("|"):
            filas.append([c.strip() for c in linea.strip().strip("|").split("|")])
        elif filas:
            break
    cabecera, cuerpo = filas[0], filas[2:]          # filas[1] es el separador |:---|
    df = pd.DataFrame(cuerpo, columns=cabecera).set_index(cabecera[0])
    df.index.name = None
    df = df.replace("—", np.nan)
    for col in df.columns:                  # numéricas a número; las de texto se dejan igual
        try:
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            pass
    return df


# Qué tabla de resultados_texto/ alimenta cada tarea de la Entrega 1
_FUENTES_E1 = {
    "clasificacion_A": ("resultados_texto/02_clasificacion/3.md",
                        "Variante A — macro-F1 en CV y métricas en test"),
    "clasificacion_B": ("resultados_texto/02_clasificacion/4.md",
                        "Variante B — macro-F1 en CV y métricas en test"),
    "regresion": ("resultados_texto/03_regresion/2.md",
                  "Resultados: RMSE de CV y métricas en test (MAE, RMSE, R², R²_adj)"),
    "clustering_nota": ("resultados_texto/05_comparacion/1.md",
                        "Clustering K-Means con K=7 comparado con las clases reales"),
}


def construir_entrega1(huella_clf: str, huella_reg: str, n_test: int = 418) -> pd.DataFrame:
    """Tabla de la Entrega 1 en el formato común, leída de resultados_texto/."""
    filas = []
    for tarea in ("clasificacion_A", "clasificacion_B"):
        ruta, titulo = _FUENTES_E1[tarea]
        t = leer_tabla_md(ruta, titulo)
        for modelo, r in t.iterrows():
            filas.append({"tarea": tarea, "modelo": modelo, "accuracy": r["test_accuracy"],
                          "f1_macro": r["test_f1_macro"], "cv_score": r["cv_f1_macro"],
                          "cv_metrica": "f1_macro", "n_test": n_test, "huella_test": huella_clf,
                          "fuente": f"{ruta} · {titulo}"})

    ruta, titulo = _FUENTES_E1["regresion"]
    for modelo, r in leer_tabla_md(ruta, titulo).iterrows():
        filas.append({"tarea": "regresion", "modelo": modelo, "MAE": r["test_MAE"],
                      "RMSE": r["test_RMSE"], "R2": r["test_R2"], "R2_adj": r["test_R2_adj"],
                      "cv_score": r["cv_RMSE"], "cv_metrica": "RMSE" if pd.notna(r["cv_RMSE"]) else np.nan,
                      "n_test": n_test, "huella_test": huella_reg, "fuente": f"{ruta} · {titulo}"})

    ruta, titulo = _FUENTES_E1["clustering_nota"]
    for config_km, r in leer_tabla_md(ruta, titulo).iterrows():
        con_clase = config_km.startswith("Con clase")
        filas.append({"tarea": "clustering_nota", "modelo": f"K-Means K=7 · {config_km}",
                      "silueta": r["silueta"], "ARI": r["ARI"], "NMI": r["NMI"], "pureza": r["pureza"],
                      "nota": "no supervisado: NO entra en la comparación"
                              + ("; usa la clase como feature (fuga)" if con_clase else ""),
                      "fuente": f"{ruta} · {titulo}"})
    return _formato(pd.DataFrame(filas))


def _formato(df: pd.DataFrame) -> pd.DataFrame:
    """Columnas en el orden común y contadores como enteros (vacío = no aplica)."""
    df = df.reindex(columns=COLUMNAS)
    for col in ("n_test", "n_parametros", "n_vectores_soporte", "epocas"):
        df[col] = pd.to_numeric(df[col]).round().astype("Int64")
    return df


def leer(ruta) -> pd.DataFrame:
    """Lee un CSV de resultados con las columnas en el orden común.

    float_precision="round_trip": relee cada número exactamente como se escribió; sin eso, al
    reescribir el archivo los valores de OTRAS etapas podían cambiar en el último dígito."""
    return _formato(pd.read_csv(ruta, float_precision="round_trip"))


def guardar_entrega2(filas: list, etapa: str) -> pd.DataFrame:
    """Escribe en data/resultados_entrega2.csv las filas de una etapa (06, 07, 08).

    Reemplaza TODAS las filas que esa etapa escribió antes (su `fuente` empieza por
    'etapa <nn>'), así un modelo que deja de ser el elegido no se queda colgado."""
    prefijo = f"etapa {etapa}"
    nuevo = _formato(pd.DataFrame(filas))
    assert nuevo["fuente"].str.startswith(prefijo).all(), "cada fila debe indicar su etapa en 'fuente'"
    if RESULTADOS_E2.exists():
        viejo = leer(RESULTADOS_E2)
        viejo = viejo[~viejo["fuente"].fillna("").str.startswith(prefijo)]
        nuevo = pd.concat([viejo, nuevo], ignore_index=True) if len(viejo) else nuevo
    nuevo = nuevo.sort_values(["fuente", "tarea", "modelo"]).reset_index(drop=True)
    nuevo.to_csv(RESULTADOS_E2, index=False)
    print(f"   [resultados] {RESULTADOS_E2.relative_to(ROOT)}: {len(filas)} filas de la etapa {etapa} "
          f"({len(nuevo)} en total)")
    return nuevo


def _huellas_split():
    """Huellas del test de clasificación y de regresión (sólo hace el split, no entrena)."""
    sys.path.insert(0, str(ROOT))
    from etapas import e02_clasificacion as e02
    from etapas import e03_regresion as e03
    te_clf, te_reg = e02.datos()["X_test"], e03.datos()["X_test"]
    assert len(te_clf) == len(te_reg) == 418, (len(te_clf), len(te_reg))
    return huella_test(te_clf.index), huella_test(te_reg.index)


def congelar_entrega1(forzar: bool = False) -> pd.DataFrame:
    nuevo = construir_entrega1(*_huellas_split())
    if RESULTADOS_E1.exists() and not forzar:
        actual = leer(RESULTADOS_E1)
        try:
            pd.testing.assert_frame_equal(actual, nuevo, check_dtype=False)
            print(f"{RESULTADOS_E1.name} ya existe y coincide con resultados_texto/: no se toca.")
        except AssertionError as e:
            print(f"[aviso] {RESULTADOS_E1.name} ya existe y DIFIERE de resultados_texto/. "
                  f"No se sobrescribe (usa --forzar si es intencionado).\n{e}")
        return actual
    nuevo.to_csv(RESULTADOS_E1, index=False)
    print(f"Escrito {RESULTADOS_E1.relative_to(ROOT)} ({len(nuevo)} filas).")
    return nuevo


if __name__ == "__main__":
    if "--congelar-entrega1" in sys.argv:
        df = congelar_entrega1(forzar="--forzar" in sys.argv)
        with pd.option_context("display.width", 200, "display.max_columns", 30):
            print(df.dropna(axis=1, how="all").drop(columns=["fuente"]).to_string())
    else:
        print(__doc__)
