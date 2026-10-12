"""Resultados de las dos entregas en un formato común, para compararlos en la etapa 09.

- data/resultados_entrega1.csv : CONGELADO. Se construyó una sola vez (T1) leyendo las tablas
  ejecutadas del proyecto de la Entrega 1; aquí sólo se lee, ningún código lo escribe. La
  etapa 09 comprueba con una huella SHA-256 que no ha cambiado.
- data/resultados_entrega2.csv : lo escriben las etapas 06-08 al terminar.

Una fila por (tarea, modelo). Lo que no aplica o no se midió queda vacío, nunca se estima.
`huella_test` identifica las filas del test (hash de sus índices): si coincide en todas las
filas, todos los modelos se evaluaron sobre el MISMO test de 418 instancias.
"""
import hashlib

import numpy as np
import pandas as pd

from config import DATA_DIR, ROOT

RESULTADOS_E1 = DATA_DIR / "resultados_entrega1.csv"
RESULTADOS_E2 = DATA_DIR / "resultados_entrega2.csv"

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
