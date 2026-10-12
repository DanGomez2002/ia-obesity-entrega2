"""ETAPA 09 — Comparación final de la Entrega 2: los 6 modelos supervisados juntos
==================================================================================

Reúne en una sola comparación los modelos de la Entrega 1 (árbol, random forest, KNN) y los
de la Entrega 2 (Naive Bayes, SVM, red neuronal) en las tres tareas supervisadas:
clasificación A (todas las features), clasificación B (sin Weight ni Height) y regresión de
Weight.

- Los resultados de la Entrega 1 NO se re-entrenan: se leen congelados de
  data/resultados_entrega1.csv (construido en T1 a partir de resultados_texto/).
- Los de la Entrega 2 se leen de data/resultados_entrega2.csv, que escriben las etapas
  06, 07 y 08 al terminar. Cada familia aporta el modelo que eligió por CV/validación.
- Antes de comparar se comprueba que, dentro de cada tarea, todas las filas se evaluaron sobre
  el MISMO test de 418 instancias (misma huella de índices).
- K-Means (Entrega 1) no entra: es no supervisado. Sólo se cita como nota.

Todas las cifras de esta etapa se leen de esos archivos; las conclusiones se escriben con
ellas (f-strings), no a mano. Si una etapa anterior cambia, esta cambia con ella.

Pasos:
  1    Carga y verificaciones (mismo test, recuentos, E1 congelada vs resultados_texto/)
  1.1  Discrepancias con los valores de referencia del encargo
  2    Tablas por tarea: clasificación A, clasificación B y regresión
  3    Ranking y gráfica comparativa
  4    Métricas adicionales: tiempo, parámetros, vectores de soporte, log-loss
  5    Conclusiones
"""
import ast
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src import resultados
from src.pasos import Etapa, cli
from etapas import e02_clasificacion as e02
from src.salida import nota, subtitulo, tabla

etapa = Etapa("09", "Comparación final de la Entrega 2", "comparacion_final",
              "Árbol, RF y KNN (E1, congelados) frente a Naive Bayes, SVM y red neuronal (E2)")

TAREAS = {"clasificacion_A": "Clasificación A (todas las features)",
          "clasificacion_B": "Clasificación B (sin Weight ni Height)",
          "regresion": "Regresión de Weight"}
FAMILIAS = ["Árbol", "Random Forest", "KNN", "Naive Bayes", "SVM", "Red neuronal"]
BASELINE = "Baseline (media)"
N_ESPERADO = {"clasificacion_A": 6, "clasificacion_B": 6, "regresion": 5}   # sin el baseline
# Colores validados (paleta categórica, dos tonos): identidad = entrega, no el rango
COLOR_ENTREGA = {"E1": "#2a78d6", "E2": "#eb6834"}

# Valores de referencia que daba el ENCARGO de la Entrega 2 para la Entrega 1 (3 decimales en
# clasificación y 2-3 en regresión). NO son resultados: sólo se contrastan con los ejecutados.
REFERENCIAS_ENCARGO = {
    ("clasificacion_A", "Árbol de decisión"): {"accuracy": 0.938, "f1_macro": 0.938},
    ("clasificacion_A", "Random Forest"): {"accuracy": 0.957, "f1_macro": 0.956},
    ("clasificacion_A", "KNN"): {"accuracy": 0.821, "f1_macro": 0.814},
    ("clasificacion_B", "Árbol de decisión"): {"accuracy": 0.746, "f1_macro": 0.740},
    ("clasificacion_B", "Random Forest"): {"accuracy": 0.857, "f1_macro": 0.853},
    ("clasificacion_B", "KNN"): {"accuracy": 0.768, "f1_macro": 0.764},
    ("regresion", "Baseline (media)"): {"MAE": 21.71, "RMSE": 25.91, "R2": 0.000, "R2_adj": -0.048},
    ("regresion", "Árbol de regresión"): {"MAE": 6.58, "RMSE": 10.47, "R2": 0.837, "R2_adj": 0.829},
    ("regresion", "Random Forest"): {"MAE": 4.84, "RMSE": 8.25, "R2": 0.899, "R2_adj": 0.894},
    ("regresion", "KNN"): {"MAE": 5.05, "RMSE": 9.91, "R2": 0.854, "R2_adj": 0.847},
}

_estado = {}


def familia(modelo: str) -> str:
    if modelo == BASELINE:
        return "Baseline"
    for f in FAMILIAS:
        if modelo.startswith(f):
            return f
    raise ValueError(f"modelo sin familia: {modelo}")


def _semillas(nota_red: str, metrica: str):
    """(media, desv) de 5 semillas que la etapa 08 deja en la columna `nota`."""
    m = re.search(rf"{re.escape(metrica)} ([\d.]+) ± ([\d.]+)", str(nota_red))
    return (float(m.group(1)), float(m.group(2))) if m else (np.nan, np.nan)


def datos() -> pd.DataFrame:
    """E1 (congelada) + E2, con columnas de entrega y familia, sólo tareas supervisadas."""
    if "df" not in _estado:
        e1 = resultados.leer(resultados.RESULTADOS_E1).assign(entrega="E1")
        e2 = resultados.leer(resultados.RESULTADOS_E2).assign(entrega="E2")
        df = pd.concat([e1, e2], ignore_index=True)
        _estado["clustering"] = df[df["tarea"] == "clustering_nota"]
        df = df[df["tarea"].isin(TAREAS)].copy()
        df["familia"] = df["modelo"].map(familia)
        # media ± desv. de 5 semillas de la red (sólo la red las tiene)
        for col, met in (("f1_macro", "macro-F1"), ("accuracy", "accuracy"), ("RMSE", "RMSE"), ("R2", "R2")):
            par = df["nota"].map(lambda n: _semillas(n, met))
            df[f"{col}_media5"] = par.map(lambda p: p[0])
            df[f"{col}_desv5"] = par.map(lambda p: p[1])
        df["errores_test"] = ((1 - df["accuracy"]) * df["n_test"]).round().astype("Int64")
        _estado["df"] = df
    return _estado["df"]


def tabla_tarea(tarea: str) -> pd.DataFrame:
    """Filas de una tarea ordenadas por la métrica principal (macro-F1 ↓ / RMSE ↑)."""
    df = datos()
    t = df[df["tarea"] == tarea].set_index("modelo")
    if tarea == "regresion":
        sup = t.drop(index=BASELINE).sort_values("RMSE")
        sup["puesto"] = range(1, len(sup) + 1)
        out = pd.concat([sup, t.loc[[BASELINE]]])
        out["puesto"] = out["puesto"].astype("Int64")
        return out
    t = t.sort_values("f1_macro", ascending=False)
    t["puesto"] = range(1, len(t) + 1)
    return t


# ----------------------------------------------------------------------------
# 1. Carga y verificaciones
# ----------------------------------------------------------------------------
@etapa.paso("1", "Carga de los dos archivos de resultados y verificaciones (mismo test de 418)",
            parametros=("N_ESPERADO",))
def paso_1_carga():
    df = datos()
    print(f"Entrega 1 (congelada): {resultados.RESULTADOS_E1.name} · "
          f"Entrega 2: {resultados.RESULTADOS_E2.name}")
    filas = {}
    for tarea in TAREAS:
        t = df[df["tarea"] == tarea]
        sup = t[t["modelo"] != BASELINE]
        filas[TAREAS[tarea]] = {
            "modelos E1": int((sup["entrega"] == "E1").sum()),
            "modelos E2": int((sup["entrega"] == "E2").sum()),
            "familias": ", ".join(f for f in FAMILIAS if f in set(sup["familia"])),
            "n_test": ", ".join(map(str, sorted(t["n_test"].unique()))),
            "huellas del test distintas": t["huella_test"].nunique(),
            "huella": t["huella_test"].iloc[0]}
        assert len(sup) == N_ESPERADO[tarea], f"{tarea}: {len(sup)} modelos, se esperaban {N_ESPERADO[tarea]}"
        assert sup["familia"].is_unique, f"{tarea}: una familia aparece dos veces"
        assert set(t["n_test"]) == {418}, f"{tarea}: el test no tiene 418 filas en todas las filas"
        assert t["huella_test"].nunique() == 1, f"{tarea}: NO todos los modelos usan el mismo test"
    tabla(pd.DataFrame.from_dict(filas, orient="index"), "Recuento y mismo test por tarea", grafica=False)

    # ¿La E1 congelada sigue coincidiendo con lo que hay en resultados_texto/?
    actual = resultados.construir_entrega1(*resultados._huellas_split())
    congelada = resultados.leer(resultados.RESULTADOS_E1)
    try:
        pd.testing.assert_frame_equal(congelada, actual, check_dtype=False)
        print("Entrega 1 congelada == tablas de resultados_texto/ de las etapas 02, 03 y 05: coincide.")
    except AssertionError as e:
        print(f"[DISCREPANCIA] la Entrega 1 congelada ya no coincide con resultados_texto/:\n{e}")
    nota("""
    Clasificación y regresión tienen tests DISTINTOS (huellas distintas): el de clasificación
    está estratificado por clase y el de regresión por deciles de Weight, como en la Entrega 1.
    Lo que garantiza una comparación justa es que dentro de cada tarea todos los modelos usen
    el mismo test, y eso es lo que se comprueba (una sola huella por tarea).
    """)


@etapa.paso("1.1", "Discrepancias entre los valores de referencia del encargo y los ejecutados",
            parametros=("REFERENCIAS_ENCARGO",))
def paso_1_1_discrepancias():
    df = datos().set_index(["tarea", "modelo"])
    filas = []
    for (tarea, modelo), refs in REFERENCIAS_ENCARGO.items():
        for metrica, ref in refs.items():
            real = float(df.loc[(tarea, modelo), metrica])
            dec = 3 if tarea != "regresion" or metrica.startswith("R2") else 2
            coincide = round(real, dec) == round(ref, dec) or abs(real - ref) <= 0.5 * 10 ** -dec
            filas.append({"tarea": tarea, "modelo": modelo, "métrica": metrica,
                          "referencia del encargo": ref, "ejecutado": real,
                          "coincide": "sí" if coincide else "NO"})
    t = pd.DataFrame(filas).set_index(["tarea", "modelo", "métrica"])
    tabla(t, "Valores de referencia del encargo frente a la ejecución", 4, grafica=False)
    malas = t[t["coincide"] == "NO"]
    print(f"{len(t) - len(malas)} de {len(t)} valores coinciden (al número de decimales de la referencia).")
    for (tarea, modelo, metrica), r in malas.iterrows():
        print(f"   DISCREPANCIA {tarea} · {modelo} · {metrica}: encargo {r['referencia del encargo']}, "
              f"ejecutado {r['ejecutado']:.4f}")
    if len(malas):
        nota("""
        Las cifras correctas son las ejecutadas (resultados_texto/ de la Entrega 1, reproducidas
        en T1 al volver a correr las etapas 02 y 03). Las discrepancias son de transcripción en
        el encargo, no de los modelos: el macro-F1 del árbol A coincide con su ACCURACY (0.938),
        y el del KNN B con su macro-F1 de VALIDACIÓN CRUZADA (0.764), no con el de test.
        """)


# ----------------------------------------------------------------------------
# 2. Tablas por tarea
# ----------------------------------------------------------------------------
def _cols_clf(t):
    out = t[["puesto", "entrega", "familia", "accuracy", "f1_macro", "errores_test", "cv_score",
             "f1_macro_media5", "f1_macro_desv5"]].copy()
    return out.rename(columns={"f1_macro_media5": "f1 media 5 semillas", "f1_macro_desv5": "f1 desv 5 semillas",
                               "errores_test": "errores en test (de 418)",
                               "cv_score": "cv (macro-F1; red: val_loss)"})


@etapa.paso("2", "Tablas por tarea: clasificación A, clasificación B y regresión (mismo test)")
def paso_2_tablas():
    for tarea in ("clasificacion_A", "clasificacion_B"):
        tabla(_cols_clf(tabla_tarea(tarea)), f"{TAREAS[tarea]} — test (418), ordenado por macro-F1", 4)
    t = tabla_tarea("regresion")
    out = t[["puesto", "entrega", "familia", "MAE", "RMSE", "R2", "R2_adj", "cv_score",
             "RMSE_media5", "RMSE_desv5"]].rename(columns={
        "cv_score": "cv (RMSE; red: val_loss)", "RMSE_media5": "RMSE media 5 semillas",
        "RMSE_desv5": "RMSE desv 5 semillas"})
    tabla(out, f"{TAREAS['regresion']} — test (418), kg, ordenado por RMSE", 4)
    print("La red neuronal aparece con su modelo de semilla 42 (columnas principales) y con la "
          "media ± desviación de 5 semillas (columnas de la derecha).")


# ----------------------------------------------------------------------------
# 3. Ranking y figura
# ----------------------------------------------------------------------------
@etapa.paso("3", "Ranking por macro-F1 (clasificación) y por RMSE (regresión) y gráfica comparativa",
            figuras=1, parametros=("COLOR_ENTREGA",))
def paso_3_ranking():
    rk = {}
    for tarea in TAREAS:
        t = tabla_tarea(tarea)
        t = t[t["familia"] != "Baseline"]
        rk[TAREAS[tarea]] = pd.Series(t["puesto"].to_numpy(), index=t["familia"])
    ranking = pd.DataFrame(rk).reindex(FAMILIAS)
    ranking["puesto medio"] = ranking.mean(axis=1)
    tabla(ranking.sort_values("puesto medio"), "Puesto de cada familia en cada tarea (1 = mejor)", 2)

    fig, axes = plt.subplots(1, 3, figsize=(20, 5.6))
    paneles = [("clasificacion_A", "f1_macro", "macro-F1 en test", (0, 1.08)),
               ("clasificacion_B", "f1_macro", "macro-F1 en test", (0, 1.08)),
               ("regresion", "RMSE", "RMSE en test (kg) — menor es mejor", None)]
    for ax, (tarea, col, xlabel, lim) in zip(axes, paneles):
        t = tabla_tarea(tarea)
        base = t.loc[BASELINE, "RMSE"] if tarea == "regresion" else None
        t = t[t["familia"] != "Baseline"].iloc[::-1]          # el mejor arriba
        y = np.arange(len(t))
        ax.barh(y, t[col], height=0.62, color=[COLOR_ENTREGA[e] for e in t["entrega"]],
                edgecolor="white", linewidth=2)
        media, desv = t[f"{col}_media5"], t[f"{col}_desv5"]
        tiene = media.notna()
        ax.errorbar(media[tiene], y[tiene.to_numpy()], xerr=desv[tiene], fmt="o", mfc="white",
                    mec="#1b1b1b", ecolor="#1b1b1b", ms=7, capsize=4, lw=1.2, zorder=3,
                    label="red: media ± desv. 5 semillas")
        # la etiqueta va a la derecha de la barra y, en la red, también de su barra de error
        fin = np.fmax(t[col].to_numpy(dtype=float), (media + desv).to_numpy(dtype=float))
        for yi, x, v in zip(y, fin, t[col]):
            ax.text(x, yi, f"  {v:.4f}" if col != "RMSE" else f"  {v:.2f}", va="center", fontsize=9,
                    color="#1b1b1b")
        ax.set_yticks(y, [re.sub(r"^Red neuronal \w+ ", "Red neuronal ", m) for m in t.index], fontsize=9)
        if base is not None:
            ax.axvline(base, ls="--", color="#6b6b6b", lw=1.2)
            ax.text(base, -0.55, f"baseline (media) {base:.2f} kg  ", ha="right", va="center",
                    fontsize=8.5, color="#4a4a4a")
            ax.set_ylim(-0.9, len(t) - 0.4)
            ax.set_xlim(0, base * 1.12)
        else:
            ax.set_xlim(*lim)
        ax.set_xlabel(xlabel)
        ax.set_title(TAREAS[tarea])
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", alpha=0.3)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLOR_ENTREGA.values()]
    handles.append(plt.Line2D([], [], marker="o", ls="", mfc="white", mec="#1b1b1b", ms=7))
    fig.legend(handles, ["Entrega 1 (árbol, RF, KNN; congelados)",
                         "Entrega 2 (Naive Bayes, SVM, red neuronal)",
                         "red neuronal: media ± desv. de 5 semillas (la barra es la semilla 42)"],
               loc="lower center", ncol=3, fontsize=9.5, bbox_to_anchor=(0.5, -0.06), frameon=False)
    fig.suptitle("Comparación final sobre el mismo test de 418 instancias por tarea", y=1.02)
    etapa.figura(fig, "3_comparacion_final")


# ----------------------------------------------------------------------------
# 4. Métricas adicionales
# ----------------------------------------------------------------------------
@etapa.paso("4", "Métricas adicionales de la Entrega 2: tiempo, parámetros, vectores de soporte, log-loss")
def paso_4_adicionales():
    df = datos()
    e2 = df[df["entrega"] == "E2"].copy()
    e2["mejor época"] = e2["mejores_params"].map(
        lambda s: ast.literal_eval(s).get("mejor_epoca") if isinstance(s, str) and "mejor_epoca" in s else np.nan)
    t = e2.set_index(["tarea", "modelo"])[["tiempo_entrenamiento_s", "n_parametros", "n_vectores_soporte",
                                           "mejor época", "epocas", "log_loss"]]
    t = t.rename(columns={"tiempo_entrenamiento_s": "tiempo de entrenamiento (s)",
                          "n_parametros": "nº parámetros (red)", "n_vectores_soporte": "nº vectores de soporte",
                          "epocas": "épocas entrenadas", "log_loss": "log-loss en test"})
    tabla(t, "Modelos de la Entrega 2: coste y tamaño", 4, grafica=False)
    nota("""
    Qué significa cada hueco: el nº de parámetros y las épocas sólo existen en la red; los
    vectores de soporte sólo en la SVM; el log-loss sólo en los modelos que dan probabilidades
    calibradas por construcción (Naive Bayes y la red con softmax). La SVC de sklearn no da
    probabilidades salvo con probability=True, que añade un ajuste de Platt con validación
    cruzada interna y no se usó. En la Entrega 1 estas métricas no se midieron y, para no
    re-entrenar, quedan vacías (no se estiman).

    Tiempos: los de Naive Bayes y SVM son el reajuste del mejor modelo de su GridSearch sobre el
    train completo (refit_time_); el de la red es el entrenamiento con early stopping de la
    semilla 42. Son tiempos de UN ajuste en la CPU de esta máquina, sin contar la búsqueda de
    hiperparámetros (que en la SVM y en la red cuesta minutos), y sirven para comparar órdenes
    de magnitud, no décimas.
    """)


# ----------------------------------------------------------------------------
# 5. Conclusiones
# ----------------------------------------------------------------------------
@etapa.paso("5", "Conclusiones (escritas con las cifras de los pasos anteriores)")
def paso_5_conclusiones():
    A, B, R = (tabla_tarea(t) for t in TAREAS)
    R_sup = R.drop(index=BASELINE)
    base = R.loc[BASELINE]

    def fila(t, fam):
        return t[t["familia"] == fam].iloc[0]

    def ganador_cv(t, mayor=True):
        """Mejor por CV entre los modelos con CV de 5 folds (la red usa val_loss: fuera)."""
        c = t[~t["familia"].isin(["Red neuronal", "Baseline"])]
        return c["cv_score"].idxmax() if mayor else c["cv_score"].idxmin()

    gA, sA, gB, sB, gR, sR = A.iloc[0], A.iloc[1], B.iloc[0], B.iloc[1], R_sup.iloc[0], R_sup.iloc[1]
    cvA, cvB, cvR = ganador_cv(A), ganador_cv(B), ganador_cv(R_sup, mayor=False)
    rf = {k: fila(t, "Random Forest") for k, t in (("A", A), ("B", B), ("R", R_sup))}
    nn = {k: fila(t, "Red neuronal") for k, t in (("A", A), ("B", B), ("R", R_sup))}
    nb = {k: fila(t, "Naive Bayes") for k, t in (("A", A), ("B", B))}
    svA = fila(A, "SVM")
    try:
        sol = resultados.leer_tabla_md("resultados_texto/07_svm/3.2.md",
                                       "SVC con SÓLO Weight y Height (macro-F1 de CV, 5 folds)")
        rbf_wh = float(sol.loc["rbf sobre (Weight, Height)", "cv_f1_macro"])
    except (ValueError, KeyError, FileNotFoundError):
        rbf_wh = np.nan
    ep = {k: ast.literal_eval(nn[k]["mejores_params"])["mejor_epoca"] for k in ("A", "B", "R")}
    X = e02.datos()["X"]
    pct_sint = (1 - e02.filas_reales(X).mean()) * 100      # filas con respuestas no enteras
    clu = _estado["clustering"].set_index("modelo")
    clu_sin = clu[~clu["nota"].str.contains("fuga")]
    e2 = datos()[datos()["entrega"] == "E2"]
    t_nb = e2[e2["familia"] == "Naive Bayes"]["tiempo_entrenamiento_s"]
    t_sv = e2[e2["familia"] == "SVM"]["tiempo_entrenamiento_s"]
    t_nn = e2[e2["familia"] == "Red neuronal"]["tiempo_entrenamiento_s"]

    tabla(pd.DataFrame({
        TAREAS["clasificacion_A"]: [gA.name, f"macro-F1 {gA['f1_macro']:.4f}", f"{sA.name} ({sA['f1_macro']:.4f})", cvA],
        TAREAS["clasificacion_B"]: [gB.name, f"macro-F1 {gB['f1_macro']:.4f}", f"{sB.name} ({sB['f1_macro']:.4f})", cvB],
        TAREAS["regresion"]: [gR.name, f"RMSE {gR['RMSE']:.2f} kg", f"{sR.name} ({sR['RMSE']:.2f} kg)", cvR],
    }, index=["mejor en test", "métrica", "segundo", "mejor por CV (sin la red)"]).T,
        "Mejor modelo por tarea", grafica=False)

    # --- (1) Clasificación A
    dif_A = int(sA["errores_test"] - gA["errores_test"])
    txt_rf_A = ("" if sA["familia"] == "Random Forest" else
                f" El Random Forest de la Entrega 1 queda en {rf['A']['f1_macro']:.4f}.")
    igual_wh = ("prácticamente lo mismo que" if abs(rbf_wh - svA["cv_score"]) < 0.005 else "frente a")
    # --- (4) Red frente a RF
    nn_pierde_seed = all(nn[k]["puesto"] > rf[k]["puesto"] for k in ("A", "B", "R"))
    nn_pierde_media = (nn["A"]["f1_macro_media5"] < rf["A"]["f1_macro"]
                       and nn["B"]["f1_macro_media5"] < rf["B"]["f1_macro"]
                       and nn["R"]["RMSE_media5"] > rf["R"]["RMSE"])
    if nn_pierde_seed and nn_pierde_media:
        cabecera_nn = ("(4) LA RED NEURONAL NO SUPERA AL RANDOM FOREST EN NINGUNA TAREA, ni con la "
                       "semilla 42 ni con la media de 5 semillas")
    else:
        cabecera_nn = "(4) RED NEURONAL FRENTE AL RANDOM FOREST: el resultado depende de la tarea"
    arq = {k: re.search(r"Red neuronal (\w+)", nn[k].name).group(1) for k in ("A", "B", "R")}
    txt_ep = "; ".join(f"en {n} se eligió la {arq[k]} y su mejor época con la semilla 42 es la {ep[k]}"
                       for k, n in (("A", "A"), ("B", "B"), ("R", "regresión")))
    peor = [n for k, n in (("A", "A"), ("B", "B")) if nn[k]["f1_macro"] < nn[k]["f1_macro_media5"]]
    peor += ["regresión"] if nn["R"]["RMSE"] > nn["R"]["RMSE_media5"] else []
    txt_semilla = (f"El modelo de la semilla 42 queda por debajo de la media de sus 5 semillas en "
                   f"{' y '.join([', '.join(peor[:-1]), peor[-1]] if len(peor) > 1 else peor)}: comparar sólo esa fila penaliza a la red por azar, y por eso "
                   f"se dan las dos cifras." if peor else "")
    # --- (5) Naive Bayes
    ultimo_nb = all(nb[k]["puesto"] == len(t) for k, t in (("A", A), ("B", B)))

    nota(f"""
    (1) CLASIFICACIÓN A (con Weight y Height). Gana {gA.name} con macro-F1 {gA['f1_macro']:.4f}
    ({gA['errores_test']} errores de 418), seguido de {sA.name} con {sA['f1_macro']:.4f}
    ({sA['errores_test']} errores): {dif_A} instancias de diferencia.{txt_rf_A} Esta variante tiene
    fuga: la clase se define por el IMC = Weight/Height², y un SVC rbf con SÓLO esas dos columnas
    logra macro-F1 de CV {rbf_wh:.4f}, {igual_wh} el SVM con las 20 ({svA['cv_score']:.4f}; etapa 07,
    paso 3.2). Aquí gana el modelo que mejor traza una frontera suave en el plano (Weight,
    Height) y por eso hasta el kernel lineal compite: la variante A mide cómo se reconstruye una
    fórmula, no cuánto predicen los hábitos.

    (2) CLASIFICACIÓN B (sin Weight ni Height), la tarea realista. Gana {gB.name} con macro-F1
    {gB['f1_macro']:.4f}, frente a {sB.name} con {sB['f1_macro']:.4f}: {int(sB['errores_test'] - gB['errores_test'])}
    instancias de 418 de diferencia. Por CV el mejor es {cvB}{' (coincide)' if cvB == gB.name else ''}.
    Con hábitos de tipos mezclados (binarios, ordinales y continuos con ruido) un conjunto de
    árboles capta interacciones y umbrales sin necesitar escalado ni suponer una forma, y al
    promediar muchos árboles reduce la varianza.

    (3) REGRESIÓN DE WEIGHT. Gana {gR.name} con RMSE {gR['RMSE']:.2f} kg (R² {gR['R2']:.3f}), frente a
    {sR.name} con {sR['RMSE']:.2f} kg. Todos superan con holgura al baseline que predice la media
    (RMSE {base['RMSE']:.2f} kg, R² {base['R2']:.4f}).

    {cabecera_nn}. Puestos de la red: {nn['A']['puesto']}, {nn['B']['puesto']} y {nn['R']['puesto']} de
    {len(A)}, {len(B)} y {len(R_sup)} en A, B y regresión. Media ± desviación de 5 semillas frente al RF:
    A {nn['A']['f1_macro_media5']:.4f} ± {nn['A']['f1_macro_desv5']:.4f} frente a {rf['A']['f1_macro']:.4f};
    B {nn['B']['f1_macro_media5']:.4f} ± {nn['B']['f1_macro_desv5']:.4f} frente a {rf['B']['f1_macro']:.4f};
    regresión {nn['R']['RMSE_media5']:.2f} ± {nn['R']['RMSE_desv5']:.2f} kg frente a {rf['R']['RMSE']:.2f} kg.
    No indica un fallo de implementación: con ~{len(X)} filas tabulares (la red entrena con 1418,
    porque 251 del train se reservan para validación) y atributos heterogéneos, es habitual que
    un conjunto de árboles iguale o supere a una red densa, que necesita más datos para aprender
    interacciones que un árbol obtiene con un simple umbral. Las curvas de la etapa 08 muestran
    que las redes media y grande sobreajustan pronto: {txt_ep}. No se añadió regularización
    (decisión registrada en la bitácora). {txt_semilla}

    (5) NAIVE BAYES {'queda último en A y en B' if ultimo_nb else 'en A y en B'}
    ({nb['A']['f1_macro']:.4f} y {nb['B']['f1_macro']:.4f}). Su supuesto de independencia
    condicional se incumple de forma extrema (Weight y Height ligados por el IMC dentro de cada
    clase, etapa 06) y sus probabilidades salen sobreconfiadas. A cambio es el más barato de la
    Entrega 2: {t_nb.max():.3f} s por ajuste frente a {t_sv.min():.2f}-{t_sv.max():.2f} s de la SVM y
    {t_nn.min():.1f}-{t_nn.max():.1f} s de la red.

    (6) CAUTELAS. Las diferencias se miden en UN test de 418 filas: en clasificación, 0.01 de
    accuracy son {0.01 * 418:.1f} instancias, y la desviación entre semillas de la propia red es de
    {nn['A']['f1_macro_desv5']:.4f} (A) y {nn['B']['f1_macro_desv5']:.4f} (B) de macro-F1. Los ganadores por CV
    {'coinciden con los de test en las tres tareas' if (cvA, cvB, cvR) == (gA.name, gB.name, gR.name)
     else 'no coinciden con los de test en todas las tareas'} (sin contar la red, cuya validación
    es otra). Y el {pct_sint:.1f} % de las filas del dataset tiene respuestas de encuesta no enteras,
    la marca de las filas generadas con SMOTE (etapa 02, paso 8), lo que hace que todas las
    métricas sean optimistas respecto a datos reales.

    (7) K-MEANS (Entrega 1) no entra en la comparación porque no es supervisado. Como nota: sin
    la clase entre sus atributos, su mejor configuración
    ({clu_sin['ARI'].idxmax().split(' · ')[-1]}) recupera poco las clases (ARI {clu_sin['ARI'].max():.4f}).
    """)


if __name__ == "__main__":
    sys.exit(cli(etapa))
