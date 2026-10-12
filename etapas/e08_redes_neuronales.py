"""ETAPA 08 — Redes neuronales con Keras: clasificación y regresión
====================================================================

Perceptrón multicapa (red densa, "feed-forward"): cada neurona calcula una suma ponderada de
sus entradas más un sesgo y le aplica una función de activación; las capas ocultas usan ReLU,
max(0, z). Los pesos se ajustan por descenso de gradiente (Adam) minimizando una función de
pérdida sobre el train; el gradiente se calcula por retropropagación.

Diseño (las mismas tareas y variantes que las etapas 02, 03, 06 y 07):
- Entradas: una neurona por atributo codificado — 20 en clasificación A, 18 en B (sin Weight
  ni Height) y 19 en regresión (sin NObeyesdad ni el propio Weight).
- Clasificación: la clase se codifica 0-6 en el orden de CLASS_ORDER (de menor a mayor IMC);
  salida de 7 neuronas softmax (una probabilidad por clase, suman 1) y pérdida
  sparse_categorical_crossentropy (log-loss con etiquetas enteras).
- Regresión: 1 neurona de salida lineal (sin activación: el peso no está acotado) y pérdida
  mse. El target se estandariza con un StandardScaler y la predicción se des-estandariza para
  reportar en kg.
- Capas ocultas: se comparan tres arquitecturas — pequeña [32], media [64, 32] y grande
  [128, 64, 32] — y se elige por la pérdida de VALIDACIÓN (media de 5 semillas), apoyándose
  en las curvas de pérdida train vs validación por época.

Protocolo y garantías contra fugas:
- Splits de las etapas 02 (clasificación) y 03 (regresión): mismos 1669 train / 418 test.
- La validación para early stopping y para elegir arquitectura se saca SÓLO del train (15 %,
  estratificada, random_state=42): sub-train 1418 / validación 251. El test no se toca hasta
  la evaluación final.
- Los StandardScaler (de X y, en regresión, de y) se ajustan SÓLO con el sub-train.
- EarlyStopping(monitor="val_loss", patience=PACIENCIA, restore_best_weights=True): se para
  cuando la pérdida de validación deja de mejorar y se recuperan los pesos de su mejor época.
- Reproducibilidad: keras.utils.set_random_seed(semilla) antes de construir y entrenar cada
  red, y tf.config.experimental.enable_op_determinism(). Con la misma semilla, el mismo
  resultado (comprobado en el paso 2).
- Las redes no son deterministas entre semillas (inicialización de pesos y orden de los
  mini-lotes), así que se reporta el modelo con semilla 42 Y la media ± desviación de 5
  semillas (42-46).

Pasos:
  1    Datos: sub-train / validación / test y comprobaciones de no-fuga
  2    Arquitecturas: entradas, salidas, capas, parámetros y comprobación de reproducibilidad
  3    Clasificación A: comparación de arquitecturas y curvas de pérdida
  3.1  Clasificación A: modelo elegido en test (semilla 42 y 5 semillas)
  4    Clasificación B: comparación de arquitecturas y curvas de pérdida
  4.1  Clasificación B: modelo elegido en test
  5    Regresión: comparación de arquitecturas y curvas de pérdida
  5.1  Regresión: modelo elegido en test (kg), predicho vs real y residuos
  6    Guardar resultados para la comparación final (data/resultados_entrega2.csv)
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")   # sin los avisos de arranque de TensorFlow
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")   # sólo CPU (evita el aviso de CUDA ausente)
os.environ.setdefault("KERAS_BACKEND", "tensorflow")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.dummy import DummyRegressor
from sklearn.metrics import classification_report, confusion_matrix, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import keras
import tensorflow as tf

from config import RANDOM_STATE
from src import cache, resultados
from src.evaluation import (clf_metrics, metricas_por_clase, plot_confusion,
                            plot_residuos_vs_pred, reg_metrics)
from src.pasos import Etapa, cli
from src.preprocessing import CLASS_ORDER
from src.salida import OPCIONES, nota, subtitulo, tabla

from etapas import e02_clasificacion as e02
from etapas import e03_regresion as e03

tf.config.experimental.enable_op_determinism()

etapa = Etapa("08", "Redes neuronales con Keras (clasificación y regresión)", "redes_neuronales",
              "Perceptrón multicapa: 3 arquitecturas, early stopping y 5 semillas",
              globales=("RANDOM_STATE", "e02.datos", "e03.datos", "VAL_FRAC", "SEMILLAS"))

ARQUITECTURAS = {"pequeña": [32], "media": [64, 32], "grande": [128, 64, 32]}
SEMILLAS = [42, 43, 44, 45, 46]   # la primera es la del "modelo final"
VAL_FRAC = 0.15                  # parte del TRAIN que se reserva para validación
EPOCAS_MAX = 500
PACIENCIA = 30
BATCH = 32
LR = 1e-3
TAREAS = {"clasificacion_A": "Clasificación A (todas las features)",
          "clasificacion_B": "Clasificación B (sin Weight/Height)",
          "regresion": "Regresión de Weight"}

_estado = {}


# ----------------------------------------------------------------------------
# Datos: sub-train / validación / test, escalados con el sub-train
# ----------------------------------------------------------------------------
def datos(tarea: str) -> dict:
    """Arrays listos para Keras. La validación sale del TRAIN; el test es el de e02/e03."""
    if tarea not in _estado:
        if tarea == "regresion":
            d = e03.datos()
            X_tr, X_te, y_tr, y_te = d["X_train"], d["X_test"], d["y_train"], d["y_test"]
            estratos = pd.qcut(y_tr, q=10, labels=False)     # deciles de peso, como e03
        else:
            d = e02.datos()
            quitar = [] if tarea == "clasificacion_A" else e02.DROP_WH
            X_tr, X_te = d["X_train"].drop(columns=quitar), d["X_test"].drop(columns=quitar)
            codigo = {c: i for i, c in enumerate(CLASS_ORDER)}
            y_tr, y_te = d["y_train"].map(codigo), d["y_test"].map(codigo)
            estratos = y_tr
        X_sub, X_val, y_sub, y_val = train_test_split(
            X_tr, y_tr, test_size=VAL_FRAC, stratify=estratos, random_state=RANDOM_STATE)
        sx = StandardScaler().fit(X_sub)                     # sólo sub-train
        out = dict(X_sub=sx.transform(X_sub).astype("float32"),
                   X_val=sx.transform(X_val).astype("float32"),
                   X_test=sx.transform(X_te).astype("float32"),
                   idx_sub=X_sub.index, idx_val=X_val.index, idx_test=X_te.index,
                   idx_train=X_tr.index, columnas=list(X_tr.columns),
                   X_sub_df=X_sub, X_test_df=X_te)
        if tarea == "regresion":
            sy = StandardScaler().fit(y_sub.to_numpy().reshape(-1, 1))   # sólo sub-train
            out.update(y_sub=sy.transform(y_sub.to_numpy().reshape(-1, 1)).ravel().astype("float32"),
                       y_val=sy.transform(y_val.to_numpy().reshape(-1, 1)).ravel().astype("float32"),
                       y_test_kg=y_te.to_numpy(), y_sub_kg=y_sub.to_numpy(), escala_y=sy)
        else:
            out.update(y_sub=y_sub.to_numpy(), y_val=y_val.to_numpy(), y_test=y_te.to_numpy())
        _estado[tarea] = out
    return _estado[tarea]


# ----------------------------------------------------------------------------
# Red: construcción y entrenamiento
# ----------------------------------------------------------------------------
def construir(n_entradas: int, capas: list, tarea: str) -> keras.Model:
    """Perceptrón multicapa: Dense(ReLU) por capa oculta + salida softmax(7) o lineal(1)."""
    modelo = keras.Sequential([keras.Input(shape=(n_entradas,))]
                              + [keras.layers.Dense(n, activation="relu") for n in capas])
    if tarea == "regresion":
        modelo.add(keras.layers.Dense(1))                    # lineal
        modelo.compile(optimizer=keras.optimizers.Adam(LR), loss="mse")
    else:
        modelo.add(keras.layers.Dense(len(CLASS_ORDER), activation="softmax"))
        modelo.compile(optimizer=keras.optimizers.Adam(LR), loss="sparse_categorical_crossentropy",
                       metrics=["accuracy"])
    return modelo


def entrenar(tarea: str, capas: list, semilla: int, epocas: int = EPOCAS_MAX) -> dict:
    """Entrena una red con early stopping sobre la validación. Devuelve historial, pesos,
    mejor época, épocas entrenadas, nº de parámetros y tiempo (no mira el test)."""
    d = datos(tarea)
    keras.utils.set_random_seed(semilla)
    modelo = construir(d["X_sub"].shape[1], capas, tarea)
    parada = keras.callbacks.EarlyStopping(monitor="val_loss", patience=PACIENCIA,
                                           restore_best_weights=True)
    t0 = time.perf_counter()
    h = modelo.fit(d["X_sub"], d["y_sub"], validation_data=(d["X_val"], d["y_val"]),
                   epochs=epocas, batch_size=BATCH, callbacks=[parada], verbose=0)
    tiempo = time.perf_counter() - t0
    hist = {k: np.asarray(v, dtype=float) for k, v in h.history.items()}
    return dict(capas=list(capas), semilla=semilla, historial=hist, pesos=modelo.get_weights(),
                mejor_epoca=int(np.argmin(hist["val_loss"])) + 1, epocas=len(hist["val_loss"]),
                val_loss=float(np.min(hist["val_loss"])), n_parametros=int(modelo.count_params()),
                tiempo_s=tiempo)


def reconstruir(tarea: str, run: dict) -> keras.Model:
    """La red de una corrida con sus pesos finales (los de su mejor época)."""
    modelo = construir(len(datos(tarea)["columnas"]), run["capas"], tarea)
    modelo.set_weights(run["pesos"])
    return modelo


def _comparar(tarea: str) -> dict:
    """Las 3 arquitecturas × 5 semillas (sólo sub-train y validación)."""
    runs = {}
    for nombre, capas in ARQUITECTURAS.items():
        for s in SEMILLAS:
            runs[(nombre, s)] = entrenar(tarea, capas, s)
            r = runs[(nombre, s)]
            print(f"   [{tarea}] {nombre} {capas} semilla {s}: val_loss {r['val_loss']:.4f} "
                  f"(mejor época {r['mejor_epoca']} de {r['epocas']}, {r['tiempo_s']:.1f} s)")
    return runs


def comparacion(tarea: str) -> dict:
    return cache.obtener(f"nn_{tarea}", lambda: _comparar(tarea))


def resumen_arquitecturas(tarea: str) -> pd.DataFrame:
    runs = comparacion(tarea)
    filas = {}
    for nombre, capas in ARQUITECTURAS.items():
        rs = [runs[(nombre, s)] for s in SEMILLAS]
        vl = np.array([r["val_loss"] for r in rs])
        filas[f"{nombre} {capas}"] = {
            "n_parametros": rs[0]["n_parametros"],
            "val_loss media": vl.mean(), "val_loss desv": vl.std(ddof=1),
            "val_loss semilla 42": rs[0]["val_loss"],
            "mejor época (media)": np.mean([r["mejor_epoca"] for r in rs]),
            "épocas entrenadas (media)": np.mean([r["epocas"] for r in rs]),
            "tiempo medio (s)": np.mean([r["tiempo_s"] for r in rs]),
        }
    return pd.DataFrame.from_dict(filas, orient="index")


def elegida(tarea: str) -> str:
    """Arquitectura con menor val_loss media en las 5 semillas (el test no interviene)."""
    t = resumen_arquitecturas(tarea)
    return t["val_loss media"].idxmin().split(" ")[0]


# ----------------------------------------------------------------------------
# Evaluación final en test (sólo de la arquitectura elegida)
# ----------------------------------------------------------------------------
def _evaluar(tarea: str) -> dict:
    """Métricas en test de la arquitectura elegida para cada una de las 5 semillas."""
    d = datos(tarea)
    nombre = elegida(tarea)
    runs = comparacion(tarea)
    por_semilla, preds = {}, {}
    for s in SEMILLAS:
        modelo = reconstruir(tarea, runs[(nombre, s)])
        salida = modelo.predict(d["X_test"], verbose=0)
        if tarea == "regresion":
            y_pred = d["escala_y"].inverse_transform(salida).ravel()     # de vuelta a kg
            m = reg_metrics(d["y_test_kg"], y_pred, n_features=len(d["columnas"]))
        else:
            y_pred = salida.argmax(axis=1)
            m = {**clf_metrics(d["y_test"], y_pred),
                 "log_loss": log_loss(d["y_test"], salida, labels=list(range(len(CLASS_ORDER))))}
            preds[f"proba_{s}"] = salida
        por_semilla[s] = {**m, "mejor_epoca": runs[(nombre, s)]["mejor_epoca"],
                          "epocas": runs[(nombre, s)]["epocas"], "tiempo_s": runs[(nombre, s)]["tiempo_s"]}
        preds[s] = y_pred
    return dict(arquitectura=nombre, por_semilla=pd.DataFrame.from_dict(por_semilla, orient="index"),
                preds=preds)


def evaluacion(tarea: str) -> dict:
    clave = f"eval_{tarea}"
    if clave not in _estado:
        _estado[clave] = _evaluar(tarea)
    return _estado[clave]


def _media_std(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"semilla 42": df.loc[SEMILLAS[0]], "media 5 semillas": df.mean(),
                         "desv. 5 semillas": df.std(ddof=1)})


# ----------------------------------------------------------------------------
# Figuras y bloques comunes
# ----------------------------------------------------------------------------
def _curvas(tarea: str, nombre_fig: str):
    """Pérdida train vs validación por época, un panel por arquitectura (semilla 42 en trazo
    fuerte, las otras 4 semillas en trazo fino)."""
    runs = comparacion(tarea)
    fig, axes = plt.subplots(1, 3, figsize=(19, 5), sharey=True)
    for ax, (nombre, capas) in zip(axes, ARQUITECTURAS.items()):
        for s in SEMILLAS[1:]:
            h = runs[(nombre, s)]["historial"]
            ax.plot(np.arange(1, len(h["val_loss"]) + 1), h["val_loss"], color="#ff7f0e", lw=0.7, alpha=0.35)
        r = runs[(nombre, SEMILLAS[0])]
        h = r["historial"]
        ep = np.arange(1, len(h["loss"]) + 1)
        ax.plot(ep, h["loss"], color="#1f77b4", lw=2, label="train (sub-train), semilla 42")
        ax.plot(ep, h["val_loss"], color="#ff7f0e", lw=2, label="validación, semilla 42")
        ax.plot([], [], color="#ff7f0e", lw=0.7, alpha=0.6, label="validación, semillas 43-46")
        ax.axvline(r["mejor_epoca"], ls="--", color="gray", lw=1.2,
                   label=f"mejor época = {r['mejor_epoca']} (pesos restaurados)")
        ax.set_title(f"{nombre} {capas} — {r['n_parametros']:,} parámetros")
        ax.set_xlabel("época")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("MSE (target estandarizado)" if tarea == "regresion"
                       else "pérdida (entropía cruzada)")
    tope = np.percentile(np.concatenate([runs[(n, 42)]["historial"]["val_loss"] for n in ARQUITECTURAS]), 95)
    axes[0].set_ylim(0, tope * 1.3)
    fig.suptitle(f"{TAREAS[tarea]}: curvas de pérdida por época", y=1.02)
    etapa.figura(fig, nombre_fig)


def _paso_comparacion(tarea: str, nombre_fig: str):
    t = resumen_arquitecturas(tarea)
    tabla(t, f"{TAREAS[tarea]} — arquitecturas (validación, 5 semillas)", 4)
    _curvas(tarea, nombre_fig)
    nombre = elegida(tarea)
    fila = t.loc[[i for i in t.index if i.startswith(nombre)][0]]
    otras = t.drop(index=fila.name)
    print(f"Arquitectura elegida (menor val_loss media): {fila.name}")
    for idx, r in otras.iterrows():
        dif = r["val_loss media"] - fila["val_loss media"]
        print(f"   {idx}: val_loss media {dif:+.4f} respecto a la elegida "
              f"({dif / max(fila['val_loss desv'], 1e-12):.1f} desviaciones de la elegida)")


def _paso_final_clf(tarea: str, tag: str):
    ev = evaluacion(tarea)
    d = datos(tarea)
    run = comparacion(tarea)[(ev["arquitectura"], SEMILLAS[0])]
    print(f"Arquitectura: {ev['arquitectura']} {run['capas']} · {run['n_parametros']:,} parámetros · "
          f"mejor época {run['mejor_epoca']} de {run['epocas']} entrenadas (semilla 42)")
    tabla(_media_std(ev["por_semilla"][["accuracy", "f1_macro", "log_loss", "mejor_epoca", "epocas", "tiempo_s"]]),
          f"Red neuronal ({tag}) en test: semilla 42 y media ± desviación de 5 semillas", 4)
    tabla(ev["por_semilla"], f"Red neuronal ({tag}) en test, por semilla", 4, grafica=False)

    y_true = pd.Series(d["y_test"]).map(dict(enumerate(CLASS_ORDER)))
    y_pred = pd.Series(ev["preds"][SEMILLAS[0]]).map(dict(enumerate(CLASS_ORDER)))
    subtitulo(f"Red neuronal {ev['arquitectura']} ({tag}), semilla 42")
    print(classification_report(y_true, y_pred, labels=CLASS_ORDER, digits=3))
    rep = pd.DataFrame(classification_report(y_true, y_pred, labels=CLASS_ORDER, digits=3,
                                             output_dict=True)).T
    tabla(rep.loc[CLASS_ORDER, ["precision", "recall", "f1-score"]],
          f"classification_report por clase — red ({tag})", 3, texto=False)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    plot_confusion(y_true, y_pred, CLASS_ORDER, f"Red neuronal {ev['arquitectura']} ({tag}), semilla 42", ax=ax)
    etapa.figura(fig, f"{'3.1' if tag == 'A' else '4.1'}_matriz_confusion_{tag}")
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_ORDER)
    tabla(metricas_por_clase(cm, CLASS_ORDER), f"Métricas por clase — red ({tag})", 3)
    conf = ev["preds"][f"proba_{SEMILLAS[0]}"].max(axis=1).mean()
    acc = ev["por_semilla"].loc[SEMILLAS[0], "accuracy"]
    print(f"Confianza media (prob. de la clase predicha) {conf:.3f} vs accuracy {acc:.3f} "
          f"({conf - acc:+.3f})")


# ----------------------------------------------------------------------------
# 1. Datos
# ----------------------------------------------------------------------------
@etapa.paso("1", "Datos: sub-train / validación (del train) / test y comprobaciones de no-fuga",
            parametros=("VAL_FRAC", "datos"))
def paso_1_datos():
    filas = {}
    for tarea in TAREAS:
        d = datos(tarea)
        sub, val, te, tr = map(set, (d["idx_sub"], d["idx_val"], d["idx_test"], d["idx_train"]))
        assert len(tr) == 1669 and len(te) == 418, "el split base debe ser 1669/418"
        assert val <= tr and sub <= tr and not (val & sub), "validación y sub-train salen del train"
        assert not (val & te) and not (sub & te), "ninguna fila del test en sub-train ni en validación"
        filas[TAREAS[tarea]] = {"entradas": len(d["columnas"]), "sub-train": len(sub),
                                "validación": len(val), "test": len(te),
                                "huella del test": resultados.huella_test(d["idx_test"])}
    tabla(pd.DataFrame.from_dict(filas, orient="index"), "Tamaños de cada parte (sin solapes)", grafica=False)
    dr = datos("regresion")
    print(f"Target de regresión en el sub-train: media {dr['escala_y'].mean_[0]:.2f} kg, "
          f"desviación {dr['escala_y'].scale_[0]:.2f} kg (con esto se estandariza y se des-estandariza)")
    m = dr["X_sub"].mean(axis=0)
    print(f"X del sub-train tras escalar: medias en [{m.min():.1e}, {m.max():.1e}]; "
          f"X del test tras escalar: medias en [{dr['X_test'].mean(axis=0).min():.2f}, "
          f"{dr['X_test'].mean(axis=0).max():.2f}] (no son 0: el escalador no vio el test)")
    nota("""
    La validación que usa el early stopping y la elección de arquitectura sale del TRAIN
    (15 %, estratificada). Si saliera del test, el test decidiría cuándo parar y qué red elegir,
    y su métrica final ya no sería una estimación honesta. Por la misma razón los escaladores se
    ajustan sólo con el sub-train: las medias del test escalado no son exactamente 0.
    """)


# ----------------------------------------------------------------------------
# 2. Arquitecturas
# ----------------------------------------------------------------------------
@etapa.paso("2", "Arquitecturas: entradas, salidas, capas ocultas, parámetros y reproducibilidad",
            parametros=("ARQUITECTURAS", "construir", "entrenar", "EPOCAS_MAX", "PACIENCIA", "BATCH", "LR"))
def paso_2_arquitecturas():
    filas = {}
    for tarea in TAREAS:
        n = len(datos(tarea)["columnas"])
        for nombre, capas in ARQUITECTURAS.items():
            m = construir(n, capas, tarea)
            filas[(TAREAS[tarea], f"{nombre} {capas}")] = {
                "entradas": n, "capas ocultas": len(capas),
                "neuronas ocultas": " → ".join(map(str, capas)),
                "salida": "7 softmax" if tarea != "regresion" else "1 lineal",
                "pérdida": "sparse_categorical_crossentropy" if tarea != "regresion" else "mse",
                "parámetros": m.count_params()}
    tabla(pd.DataFrame.from_dict(filas, orient="index"), "Las 3 arquitecturas en cada tarea", grafica=False)
    subtitulo("Desglose de parámetros (clasificación A, arquitectura media)")
    construir(20, ARQUITECTURAS["media"], "clasificacion_A").summary(print_fn=print)
    print(f"Optimizador Adam (lr={LR}), lotes de {BATCH}, hasta {EPOCAS_MAX} épocas, "
          f"EarlyStopping(val_loss, patience={PACIENCIA}, restore_best_weights=True)")

    # Reproducibilidad: misma semilla dos veces -> mismos pesos y misma pérdida
    a = entrenar("clasificacion_A", ARQUITECTURAS["pequeña"], 42, epocas=5)
    b = entrenar("clasificacion_A", ARQUITECTURAS["pequeña"], 42, epocas=5)
    c = entrenar("clasificacion_A", ARQUITECTURAS["pequeña"], 43, epocas=5)
    iguales = all(np.array_equal(x, y) for x, y in zip(a["pesos"], b["pesos"]))
    print(f"\nReproducibilidad (red pequeña, 5 épocas): semilla 42 dos veces -> val_loss "
          f"{a['val_loss']:.6f} y {b['val_loss']:.6f}, pesos idénticos: {iguales}; "
          f"semilla 43 -> {c['val_loss']:.6f}")
    assert iguales, "con la misma semilla el entrenamiento debe ser idéntico"
    nota(f"""
    Entradas: una neurona por atributo codificado (las mismas columnas que usan los demás
    modelos; las binarias y one-hot ya son numéricas y las ordinales van de 0 a 3), escaladas
    con StandardScaler porque el descenso de gradiente converge mal si los atributos tienen
    escalas muy distintas (Weight en kg frente a binarias 0/1).

    Salidas: en clasificación, 7 neuronas softmax porque hay 7 clases excluyentes: la salida
    se interpreta como P(clase | x) y se entrena con la entropía cruzada (sparse_ porque las
    etiquetas van como enteros 0-6, no one-hot). En regresión, 1 neurona lineal porque se
    predice un único número real sin acotar; se entrena con el error cuadrático medio.

    Capas ocultas: con ~1400 filas de entrenamiento y ~20 atributos, el problema es pequeño. Se
    prueban 1, 2 y 3 capas con anchura decreciente (32; 64-32; 128-64-32): la primera capa algo
    más ancha que la entrada para poder combinar atributos, y las siguientes más estrechas para
    resumir. ReLU evita la saturación de sigmoide/tanh. No se usa dropout ni regularización L2:
    el freno al sobreajuste es el early stopping, y las curvas por época muestran si hace falta
    más. La arquitectura se elige por la pérdida media de validación en 5 semillas.
    """)


# ----------------------------------------------------------------------------
# 3-4. Clasificación A y B
# ----------------------------------------------------------------------------
@etapa.paso("3", "Clasificación A: comparación de las 3 arquitecturas (validación) y curvas de pérdida",
            figuras=1, parametros=("ARQUITECTURAS", "SEMILLAS", "_comparar"))
def paso_3_comparacion_A():
    _paso_comparacion("clasificacion_A", "3_curvas_perdida_A")


@etapa.paso("3.1", "Clasificación A: red elegida en test (semilla 42 y media ± desv. de 5 semillas)",
            figuras=1, parametros=("ARQUITECTURAS", "SEMILLAS"))
def paso_3_1_final_A():
    _paso_final_clf("clasificacion_A", "A")


@etapa.paso("4", "Clasificación B: comparación de las 3 arquitecturas (validación) y curvas de pérdida",
            figuras=1, parametros=("ARQUITECTURAS", "SEMILLAS", "_comparar", "e02.DROP_WH"))
def paso_4_comparacion_B():
    _paso_comparacion("clasificacion_B", "4_curvas_perdida_B")


@etapa.paso("4.1", "Clasificación B: red elegida en test (semilla 42 y media ± desv. de 5 semillas)",
            figuras=1, parametros=("ARQUITECTURAS", "SEMILLAS", "e02.DROP_WH"))
def paso_4_1_final_B():
    _paso_final_clf("clasificacion_B", "B")


# ----------------------------------------------------------------------------
# 5. Regresión
# ----------------------------------------------------------------------------
@etapa.paso("5", "Regresión: comparación de las 3 arquitecturas (validación) y curvas de pérdida",
            figuras=1, parametros=("ARQUITECTURAS", "SEMILLAS", "_comparar"))
def paso_5_comparacion_reg():
    _paso_comparacion("regresion", "5_curvas_perdida_reg")


@etapa.paso("5.1", "Regresión: red elegida en test (kg), baseline, predicho vs real y residuos",
            figuras=2, parametros=("ARQUITECTURAS", "SEMILLAS"))
def paso_5_1_final_reg():
    ev = evaluacion("regresion")
    d = datos("regresion")
    run = comparacion("regresion")[(ev["arquitectura"], SEMILLAS[0])]
    print(f"Arquitectura: {ev['arquitectura']} {run['capas']} · {run['n_parametros']:,} parámetros · "
          f"mejor época {run['mejor_epoca']} de {run['epocas']} entrenadas (semilla 42)")
    base = DummyRegressor(strategy="mean").fit(d["X_sub_df"], d["y_sub_kg"])
    m_base = reg_metrics(d["y_test_kg"], base.predict(d["X_test_df"]), n_features=len(d["columnas"]))
    t = _media_std(ev["por_semilla"][["MAE", "RMSE", "R2", "R2_adj", "mejor_epoca", "epocas", "tiempo_s"]])
    t["baseline (media del sub-train)"] = pd.Series(m_base)
    tabla(t, "Red neuronal (regresión) en test, en kg: semilla 42, media ± desv. de 5 semillas y baseline", 4)
    tabla(ev["por_semilla"], "Red neuronal (regresión) en test, por semilla", 4, grafica=False)

    y_te, y_pred = d["y_test_kg"], ev["preds"][SEMILLAS[0]]
    r = ev["por_semilla"].loc[SEMILLAS[0]]
    lo, hi = min(y_te.min(), y_pred.min()) - 2, max(y_te.max(), y_pred.max()) + 2
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_te, y_pred, s=12, alpha=0.5, color="steelblue")
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.2)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Real (kg)")
    ax.set_ylabel("Predicho (kg)")
    ax.set_title(f"Red {ev['arquitectura']} (semilla 42)\nRMSE={r['RMSE']:.2f} kg  R²={r['R2']:.3f}  "
                 f"R²adj={r['R2_adj']:.3f}")
    etapa.figura(fig, "5.1_pred_vs_real_red")
    etapa.figura(plot_residuos_vs_pred(y_te, {f"Red {ev['arquitectura']} (semilla 42)": y_pred}),
                 "5.1_residuos_vs_predicho_red")
    y_sub = d["y_sub_kg"]
    print(f"Predicciones fuera del rango del sub-train [{y_sub.min():.1f}, {y_sub.max():.1f}] kg: "
          f"{int(((y_pred < y_sub.min()) | (y_pred > y_sub.max())).sum())} "
          f"(mín {y_pred.min():.2f}, máx {y_pred.max():.2f})")


# ----------------------------------------------------------------------------
# 6. Guardar resultados
# ----------------------------------------------------------------------------
@etapa.paso("6", "Guardar la red elegida (semilla 42, con media ± desv. de 5 semillas) en data/resultados_entrega2.csv")
def paso_6_guardar():
    filas = []
    for tarea in TAREAS:
        ev = evaluacion(tarea)
        d = datos(tarea)
        run = comparacion(tarea)[(ev["arquitectura"], SEMILLAS[0])]
        ps = ev["por_semilla"]
        r = ps.loc[SEMILLAS[0]]
        fila = {"tarea": tarea, "modelo": f"Red neuronal {ev['arquitectura']} {run['capas']}",
                "cv_score": run["val_loss"],
                "cv_metrica": "val_loss (validación 15 % del train)",
                "tiempo_entrenamiento_s": run["tiempo_s"], "n_parametros": run["n_parametros"],
                "epocas": run["epocas"],
                "mejores_params": str({"capas": run["capas"], "mejor_epoca": run["mejor_epoca"],
                                       "epocas_entrenadas": run["epocas"], "lr": LR, "batch": BATCH,
                                       "paciencia": PACIENCIA, "semilla": SEMILLAS[0]}),
                "n_test": len(d["idx_test"]), "huella_test": resultados.huella_test(d["idx_test"]),
                "fuente": f"etapa 08 · paso {'5.1' if tarea == 'regresion' else ('3.1' if tarea.endswith('A') else '4.1')}"}
        if tarea == "regresion":
            fila.update(MAE=r["MAE"], RMSE=r["RMSE"], R2=r["R2"], R2_adj=r["R2_adj"],
                        nota=f"semilla 42; 5 semillas: RMSE {ps['RMSE'].mean():.4f} ± {ps['RMSE'].std(ddof=1):.4f}, "
                             f"R2 {ps['R2'].mean():.4f} ± {ps['R2'].std(ddof=1):.4f}")
        else:
            fila.update(accuracy=r["accuracy"], f1_macro=r["f1_macro"], log_loss=r["log_loss"],
                        nota=f"semilla 42; 5 semillas: macro-F1 {ps['f1_macro'].mean():.4f} ± "
                             f"{ps['f1_macro'].std(ddof=1):.4f}, accuracy {ps['accuracy'].mean():.4f} ± "
                             f"{ps['accuracy'].std(ddof=1):.4f}")
        filas.append(fila)
    df = pd.DataFrame(filas).set_index(["tarea", "modelo"])
    tabla(df[["accuracy", "f1_macro", "log_loss", "RMSE", "R2", "n_parametros", "epocas"]],
          "Filas para data/resultados_entrega2.csv", grafica=False)
    for f in filas:
        print(f"   {f['tarea']}: {f['nota']}")
    if OPCIONES["guardar"]:
        resultados.guardar_entrega2(filas, "08")
    else:
        print("   (no se escribe data/resultados_entrega2.csv: sólo se guarda con --guardar o run_all.py)")


if __name__ == "__main__":
    sys.exit(cli(etapa))
