"""ETAPA 03 — Regresión de `Weight`
===================================

Modelos: árbol de regresión, random forest y KNN.

Protocolo:
- Dataset preparado en la etapa 01 (data/reg.csv): mismas features codificadas, SIN
  NObeyesdad (la clase codifica directamente el rango de peso -> fuga de datos). Height sí se
  mantiene: es una medida física independiente, no se deriva del target.
- Split 80/20 (random_state=42), estratificado por deciles de Weight para que sea comparable
  al de clasificación.
- Hiperparámetros con GridSearchCV (5-fold, criterio RMSE negativo) sobre el train.
- KNN dentro de Pipeline con StandardScaler.
- Métricas en test: MAE, RMSE, R² y R² ajustado. Baseline que predice la media.
- Tres árboles con prepoda y curva del mejor K de KNN.

Variantes (las mismas preguntas que la etapa 02 se hace en clasificación):
- Sin Height: la otra medida corporal. ¿Cuánto del peso se explica sólo con hábitos, edad y sexo?
- Sólo filas reales: sin las filas que SMOTE generó, con y sin Height.
- Ruido (jittering) en los hábitos: cuánto empeora cada modelo, con y sin Height.

Pasos:
  1    Datos y split
  2    Modelos, grillas y ajuste (tabla)   4    Predicho vs real y residuos
  2.1  Curva del mejor K de KNN            5    Importancia de features (RF)
  3    Tres árboles de regresión           6    Comparación y R² ajustado
  8    Variante sin Height                 9    Sólo filas reales (sin SMOTE)
  10   Robustez ante ruido (jittering)
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (GridSearchCV, KFold, RepeatedKFold, cross_val_score,
                                     cross_validate, train_test_split)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor, plot_tree

from config import RANDOM_STATE
from src import cache
from etapas.e02_clasificacion import HABIT_NUM_COLS, RANGOS_HABITOS, _con_ruido, filas_reales
from src.data import TARGET_REG
from src.evaluation import (plot_k_curve, plot_n_estimators_curve, plot_residuos_vs_pred,
                            reg_metrics, results_table, tree_summary)
from src.pasos import Etapa, cli
from src.preprocessing import cargar_reg
from src.salida import subtitulo, tabla

etapa = Etapa("03", "Regresión de Weight", "regresion",
              "Árbol, Random Forest y KNN para predecir el peso sin usar NObeyesdad",
              globales=("RANDOM_STATE", "datos"))

_estado = {}


def datos():
    if "datos" not in _estado:
        df = cargar_reg()
        X = df.drop(columns=[TARGET_REG])
        y = df[TARGET_REG]
        strata = pd.qcut(y, q=10, labels=False)   # deciles de peso para estratificar
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=strata, random_state=RANDOM_STATE)
        _estado["datos"] = dict(X=X, y=y, X_train=X_train, X_test=X_test, y_train=y_train,
                                y_test=y_test, n_features=X.shape[1])
    return _estado["datos"]


def make_models():
    return {
        "Árbol de regresión": (
            DecisionTreeRegressor(random_state=RANDOM_STATE),
            {"max_depth": [4, 6, 8, 12, None], "min_samples_leaf": [1, 3, 5, 10]},
        ),
        "Random Forest": (
            RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            {"n_estimators": [200, 400], "max_depth": [None, 12], "min_samples_leaf": [1, 2]},
        ),
        "KNN": (
            Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsRegressor())]),
            {"knn__n_neighbors": list(range(1, 32, 2)), "knn__weights": ["uniform", "distance"]},
        ),
    }


DROP_H = ["Height"]   # columna que se quita en la variante sin Height (paso 8)


def _ajustar(sin=()):
    """GridSearch de los 3 modelos + baseline, sin las columnas `sin`. Devuelve (fitted,
    results, tabla)."""
    d = datos()
    X_tr, X_te = d["X_train"].drop(columns=list(sin)), d["X_test"].drop(columns=list(sin))
    n_features = X_tr.shape[1]
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fitted, results = {}, {}
    dummy = DummyRegressor(strategy="mean").fit(X_tr, d["y_train"])
    results["Baseline (media)"] = {
        "cv_RMSE": np.nan,
        **{f"test_{k}": v for k, v in
           reg_metrics(d["y_test"], dummy.predict(X_te), n_features=n_features).items()},
    }
    for name, (model, grid) in make_models().items():
        gs = GridSearchCV(model, grid, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
        gs.fit(X_tr, d["y_train"])
        fitted[name] = gs
        y_pred = gs.predict(X_te)
        results[name] = {
            "cv_RMSE": -gs.best_score_,
            **{f"test_{k}": v for k, v in
               reg_metrics(d["y_test"], y_pred, n_features=n_features).items()},
        }
        print(f"   {name}: mejores params = {gs.best_params_}")
    return fitted, results, results_table(results)


def modelos():
    """(fitted, results, tabla) — cacheado en disco."""
    return cache.obtener("reg_modelos", _ajustar)


def modelos_sin_height():
    """Lo mismo que modelos() pero sin Height (paso 8)."""
    return cache.obtener("reg_modelos_sin_height", lambda: _ajustar(sin=DROP_H))


def arboles_prepoda():
    if "arboles" not in _estado:
        d = datos()
        arboles = {
            "Completo": DecisionTreeRegressor(random_state=RANDOM_STATE),
            "Podado": DecisionTreeRegressor(max_depth=8, min_samples_leaf=5, random_state=RANDOM_STATE),
            "Muy podado": DecisionTreeRegressor(max_depth=3, min_samples_leaf=20, random_state=RANDOM_STATE),
        }
        for a in arboles.values():
            a.fit(d["X_train"], d["y_train"])
        _estado["arboles"] = arboles
    return _estado["arboles"]


def _pred_vs_real(items, titulos, nombre_fig):
    """Panel 1x3 predicho vs real con la MISMA escala en los tres ejes."""
    d = datos()
    y_test, X_test = d["y_test"], d["X_test"]
    preds = {n: m.predict(X_test) for n, m in items.items()}
    lo = min(y_test.min(), min(p.min() for p in preds.values())) - 2
    hi = max(y_test.max(), max(p.max() for p in preds.values())) + 2
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (nombre, y_pred) in zip(axes, preds.items()):
        ax.scatter(y_test, y_pred, s=12, alpha=0.5, color="steelblue")
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.2)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("Real (kg)")
        ax.set_ylabel("Predicho (kg)")
        ax.set_title(titulos[nombre])
    etapa.figura(fig, nombre_fig)


# ----------------------------------------------------------------------------
@etapa.paso("1", "Datos y split (estratificado por deciles de Weight)")
def paso_1_datos():
    d = datos()
    print("train:", d["X_train"].shape, "| test:", d["X_test"].shape, "| n_features:", d["n_features"])
    print("features:", d["X"].columns.tolist())
    print("¿NObeyesdad entre las features?:", "NObeyesdad" in d["X"].columns, "(debe ser False)")
    tabla(d["y_train"].describe().round(2), "Weight en train")


@etapa.paso("2", "Modelos, grillas y ajuste con GridSearchCV (tabla con R² ajustado)",
            parametros=("make_models", "_ajustar"))
def paso_2_modelos():
    for name, (model, grid) in make_models().items():
        subtitulo(name)
        for k, v in grid.items():
            print(f"   {k}: {v}")
    _, _, t = modelos()
    tabla(t, "Resultados: RMSE de CV y métricas en test (MAE, RMSE, R², R²_adj)")


@etapa.paso("2.1", "Curva del mejor K de KNN (RMSE de CV vs K)", figuras=1,
            parametros=("make_models",))
def paso_2_1_curva_k():
    fitted, _, _ = modelos()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax, mejor_k = plot_k_curve(fitted["KNN"].cv_results_, ylabel="RMSE (CV)", negate=True, ax=ax)
    etapa.figura(fig, "2.1_curva_k_knn")
    print("Mejor K (regresión):", mejor_k, "| params:", fitted["KNN"].best_params_)


# ----------------------------------------------------------------------------
# 2.2 ¿Cuántos árboles necesita el Random Forest?
# ----------------------------------------------------------------------------
N_ARBOLES = [10, 25, 50, 100, 200, 400, 800,1000, 2000, 4000, 8000]


def _barrido_n_arboles(n_arboles=N_ARBOLES):
    """Para cada nº de árboles: RMSE de CV (5 folds), métricas en test y tiempo de fit."""
    d = datos()
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    filas = []
    for n in n_arboles:
        rf = RandomForestRegressor(n_estimators=n, random_state=RANDOM_STATE, n_jobs=-1)
        scores = -cross_val_score(rf, d["X_train"], d["y_train"], cv=cv,
                                  scoring="neg_root_mean_squared_error", n_jobs=-1)
        t0 = time.perf_counter()
        rf.fit(d["X_train"], d["y_train"])
        t_fit = time.perf_counter() - t0
        m = reg_metrics(d["y_test"], rf.predict(d["X_test"]), n_features=d["n_features"])
        filas.append({"n_estimators": n, "cv_RMSE": scores.mean(), "cv_std": scores.std(),
                      "test_RMSE": m["RMSE"], "test_R2": m["R2"], "tiempo_fit_s": t_fit})
    return pd.DataFrame(filas).set_index("n_estimators")


def barrido_n_arboles():
    return cache.obtener("reg_barrido_n_arboles", _barrido_n_arboles)


@etapa.paso("2.2", "¿Cuántos árboles? Random Forest con "
            + ", ".join(map(str, N_ARBOLES[:-1])) + f" y {N_ARBOLES[-1]}", figuras=1,
            parametros=("N_ARBOLES", "_barrido_n_arboles"))
def paso_2_2_n_arboles():
    df = barrido_n_arboles()
    tabla(df, "Random Forest (regresión): error y coste según el número de árboles", 4)

    fig, n_suf = plot_n_estimators_curve(
        df, "cv_RMSE", "test_RMSE", "RMSE (kg)",
        "Random Forest: RMSE según el número de árboles",
        col_std="cv_std", n_proyecto=200, mejor="min")
    etapa.figura(fig, "2.2_n_estimators_rf")

    cv_r, t = df["cv_RMSE"], df["tiempo_fit_s"]
    n_min, n_max, n_mejor = int(df.index.min()), int(df.index.max()), int(cv_r.idxmin())
    print(f"\nRMSE de CV: {cv_r.loc[n_min]:.4f} kg con {n_min} árboles  ->  {cv_r.loc[n_max]:.4f} kg "
          f"con {n_max} (mejora total {cv_r.loc[n_min] - cv_r.loc[n_max]:.4f} kg; el mejor valor "
          f"de la tabla se da con {n_mejor} árboles)")
    print(f"Con {n_suf} árboles ya se consigue el 90% de toda la mejora del barrido.")
    print(f"Pasar de {n_suf} a {n_max} árboles multiplica el tiempo de entrenamiento por "
          f"{t.loc[n_max] / max(float(t.loc[n_suf]), 1e-9):.1f} "
          f"({t.loc[n_suf]:.2f} s -> {t.loc[n_max]:.2f} s) a cambio de "
          f"{cv_r.loc[n_suf] - cv_r.loc[n_max]:.4f} kg de RMSE.")


@etapa.paso("3", "Tres árboles de regresión: completo, podado y muy podado", figuras=4,
            parametros=("arboles_prepoda",))
def paso_3_tres_arboles():
    d = datos()
    X = d["X"]
    resumen = []
    for nombre, modelo in arboles_prepoda().items():
        s = tree_summary(modelo, X.columns, nombre)
        m = reg_metrics(d["y_test"], modelo.predict(d["X_test"]), n_features=d["n_features"])
        resumen.append({**s, **m})
        print(s)
        max_d = None if nombre == "Muy podado" else 3
        fig, ax = plt.subplots(figsize=(20, 8))
        plot_tree(modelo, max_depth=max_d, feature_names=X.columns, filled=True, rounded=True,
                  ax=ax, fontsize=9 if nombre == "Muy podado" else 8)
        ax.set_title(f"Árbol {nombre} — profundidad {s['profundidad']}, {s['n_hojas']} hojas, "
                     f"raíz: {s['raiz_atributo']} ≤ {s['raiz_umbral']}"
                     + (" (truncado a profundidad 3 para visualización)" if max_d else ""))
        etapa.figura(fig, f"3_arbol_{nombre.lower().replace(' ', '_')}")

    cols = ["modelo", "profundidad", "n_hojas", "raiz_atributo", "MAE", "RMSE", "R2", "R2_adj"]
    tabla(pd.DataFrame(resumen)[cols].set_index("modelo"), "Resumen de los tres árboles")

    titulos = {}
    for nombre, modelo in arboles_prepoda().items():
        m = reg_metrics(d["y_test"], modelo.predict(d["X_test"]), n_features=d["n_features"])
        titulos[nombre] = (f"Árbol {nombre}\nRMSE={m['RMSE']:.2f} kg  R²={m['R2']:.3f}  "
                           f"R²adj={m['R2_adj']:.3f}")
    _pred_vs_real(arboles_prepoda(), titulos, "3_pred_vs_real_tres_arboles")


@etapa.paso("4", "Predicho vs real y residuos — modelos principales (test)", figuras=3,
            parametros=("make_models",))
def paso_4_pred_vs_real():
    fitted, results, _ = modelos()
    titulos = {name: (f"{name}\nRMSE={results[name]['test_RMSE']:.2f} kg  "
                      f"R²={results[name]['test_R2']:.3f}  R²adj={results[name]['test_R2_adj']:.3f}")
               for name in fitted}
    _pred_vs_real(fitted, titulos, "4_pred_vs_real_modelos")

    d = datos()
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    for ax, (name, gs) in zip(axes, fitted.items()):
        resid = d["y_test"] - gs.predict(d["X_test"])
        sns.histplot(resid, bins=40, kde=True, ax=ax, color="steelblue")
        ax.axvline(0, c="r", ls="--", lw=1.2)
        ax.set_title(f"Residuos — {name}")
        ax.set_xlabel("Real − Predicho (kg)")
        ax.set_ylabel("Frecuencia")
    etapa.figura(fig, "4_residuos_modelos")

    preds = {name: gs.predict(d["X_test"]) for name, gs in fitted.items()}
    etapa.figura(plot_residuos_vs_pred(d["y_test"], preds), "4_residuos_vs_predicho")


@etapa.paso("5", "Importancia de features (Random Forest)", figuras=1,
            parametros=("make_models",))
def paso_5_importancias():
    fitted, _, _ = modelos()
    d = datos()
    rf = fitted["Random Forest"].best_estimator_
    imp = pd.Series(rf.feature_importances_, index=d["X_train"].columns).sort_values(ascending=False)
    tabla(imp.rename("importancia"), "feature_importances_")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=imp.values, y=imp.index, ax=ax, color="steelblue")
    ax.set_title("Importancia de features — Random Forest (regresión de Weight)")
    ax.set_xlabel("Importancia (reducción de impureza)")
    etapa.figura(fig, "5_importancias_rf")


@etapa.paso("6", "Comparación de modelos y explicación del R² ajustado", figuras=1,
            parametros=("make_models",))
def paso_6_comparacion():
    _, _, t = modelos()
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    t["test_RMSE"].plot.bar(ax=axes[0], color="steelblue", title="RMSE en test (kg)")
    t["test_R2"].plot.bar(ax=axes[1], color="indianred", title="R² en test")
    t["test_R2_adj"].plot.bar(ax=axes[2], color="forestgreen", title="R² ajustado en test")
    for ax in axes:
        ax.tick_params(axis="x", rotation=20)
    etapa.figura(fig, "6_comparacion_metricas")
    tabla(t, "Tabla de resultados")


# ----------------------------------------------------------------------------
# 8. Variante sin Height
# ----------------------------------------------------------------------------
@etapa.paso("8", "Variante sin Height: GridSearch, comparación con/sin Height e importancias",
            figuras=2, parametros=("make_models", "DROP_H"))
def paso_8_sin_height():
    _, _, t_con = modelos()
    fitted, _, t_sin = modelos_sin_height()
    for name, gs in fitted.items():
        print(f"   {name} (sin Height): mejores params = {gs.best_params_}")
    tabla(t_sin, "Sin Height — RMSE de CV y métricas en test")
    comp = pd.concat({"Con Height": t_con, "Sin Height": t_sin}, axis=0)
    tabla(comp, "Comparación con Height vs sin Height")
    etapa.figura(figura_con_sin_height(comp), "8_comparacion_con_sin_height")

    X_tr = datos()["X_train"].drop(columns=DROP_H)
    rf = fitted["Random Forest"].best_estimator_
    imp = pd.Series(rf.feature_importances_, index=X_tr.columns).sort_values(ascending=False)
    tabla(imp.rename("importancia"), "Importancias RF (sin Height)")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=imp.values, y=imp.index, ax=ax, color="indianred")
    ax.set_title("Importancia de features — Random Forest (sin Height)")
    ax.set_xlabel("Importancia (reducción de impureza)")
    etapa.figura(fig, "8_importancias_rf_sin_height")


def figura_con_sin_height(comp):
    """RMSE y R² en test por modelo, con y sin Height (comp: tablas concatenadas por variante)."""
    datos_plot = comp.reset_index().rename(columns={"level_0": "variante", "level_1": "modelo"})
    fig, axes = plt.subplots(1, 2, figsize=(16, 4.5))
    for ax, col, titulo in [(axes[0], "test_RMSE", "RMSE en test (kg) — más bajo es mejor"),
                            (axes[1], "test_R2", "R² en test — más alto es mejor")]:
        sns.barplot(data=datos_plot, x="modelo", y=col, hue="variante", ax=ax)
        ax.set_title(titulo)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=15)
    return fig


# ----------------------------------------------------------------------------
# 9. Sólo filas reales (sin SMOTE)
# ----------------------------------------------------------------------------
# Mismas filas reales que el paso 8 de la etapa 02 (filas_reales: respuestas de encuesta
# todas enteras), 491 de 2087.
BASELINE_REALES = "Línea base (media)"
METRICAS_REALES = {"MAE": "neg_mean_absolute_error", "RMSE": "neg_root_mean_squared_error",
                   "R2": "r2"}


def _modelos_reales():
    """Configuraciones fijas (sin GridSearch: con 491 filas haría falta CV anidada). El árbol
    es el 'Podado' del paso 3 y KNN usa el K que eligió el GridSearch del paso 2."""
    return {
        BASELINE_REALES: DummyRegressor(strategy="mean"),
        "Árbol (depth 8, leaf 5)": DecisionTreeRegressor(max_depth=8, min_samples_leaf=5,
                                                         random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE,
                                               n_jobs=-1),
        "KNN (k=3, distance)": Pipeline([("scaler", StandardScaler()),
                                         ("knn", KNeighborsRegressor(n_neighbors=3,
                                                                     weights="distance"))]),
    }


def cv_reales(X, y, repeticiones=10, seed=RANDOM_STATE):
    """MAE, RMSE y R² medios (y desviación del RMSE) con 3 folds repetidos."""
    cv = RepeatedKFold(n_splits=3, n_repeats=repeticiones, random_state=seed)
    filas = {}
    for nombre, modelo in _modelos_reales().items():
        s = cross_validate(modelo, X, y, cv=cv, scoring=METRICAS_REALES, n_jobs=-1)
        filas[nombre] = {"MAE": -s["test_MAE"].mean(), "RMSE": -s["test_RMSE"].mean(),
                         "R2": s["test_R2"].mean(), "RMSE_std": s["test_RMSE"].std()}
    return pd.DataFrame(filas).T


def _evaluacion_reales():
    d = datos()
    X, y = d["X"], d["y"]
    r = filas_reales(X)

    # (a) Los modelos del proyecto (pasos 2 y 8, entrenados con todo) evaluados por separado
    # en las filas reales y en las sintéticas del test
    m_te = filas_reales(d["X_test"])
    y_te = d["y_test"]
    real_vs_sint = {}
    for tag, (fitted, _, _), sin in (("Con Height", modelos(), []),
                                     ("Sin Height", modelos_sin_height(), DROP_H)):
        X_te = d["X_test"].drop(columns=sin)
        for nombre in ("Árbol de regresión", "Random Forest", "KNN"):
            p = fitted[nombre].predict(X_te)
            real, sint = reg_metrics(y_te[m_te], p[m_te]), reg_metrics(y_te[~m_te], p[~m_te])
            real_vs_sint[f"{tag} · {nombre}"] = {
                "MAE reales": real["MAE"], "MAE sintéticas": sint["MAE"],
                "RMSE reales": real["RMSE"], "RMSE sintéticas": sint["RMSE"]}

    X_r, y_r = X[r], y[r]
    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=RANDOM_STATE,
                               n_jobs=-1).fit(X_r, y_r)
    return dict(
        n_reales=int(r.sum()), n_total=len(X), n_test_reales=int(m_te.sum()),
        peso=pd.DataFrame({"filas reales": y[r].describe(), "filas sintéticas": y[~r].describe()}),
        real_vs_sint=pd.DataFrame(real_vs_sint).T,
        con=cv_reales(X_r, y_r),
        sin=cv_reales(X_r.drop(columns=DROP_H), y_r),
        importancias=pd.Series(rf.feature_importances_, index=X_r.columns).sort_values(
            ascending=False),
    )


def evaluacion_reales():
    return cache.obtener("reg_filas_reales", _evaluacion_reales)


def figura_reales(con, sin, subtitulo_cv):
    """Barras de RMSE (± desviación entre folds) y R² por modelo, con y sin Height."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    x, ancho = np.arange(len(con)), 0.38
    etiquetas = [n.replace(" (media)", "") for n in con.index]
    for desp, t, nombre, color in [(-ancho / 2, con, "Con Height", "#4c78a8"),
                                   (ancho / 2, sin, "Sin Height", "#f58518")]:
        axes[0].bar(x + desp, t["RMSE"], ancho, yerr=t["RMSE_std"], capsize=3, label=nombre,
                    color=color)
        axes[1].bar(x + desp, t["R2"], ancho, label=nombre, color=color)
    for ax, titulo in [(axes[0], "RMSE (kg) — más bajo es mejor"),
                       (axes[1], "R² — más alto es mejor")]:
        ax.set_xticks(x, etiquetas, rotation=15, ha="right")
        ax.set_title(f"Sólo filas reales · {titulo}")
        ax.set_xlabel(subtitulo_cv)
        ax.grid(axis="x", visible=False)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=9)
    axes[1].axhline(0, c="k", lw=0.8)
    return fig


@etapa.paso("9", "Sólo filas reales (sin SMOTE): el peso sin datos sintéticos, con y sin Height",
            figuras=1, parametros=("_modelos_reales", "cv_reales", "_evaluacion_reales",
                                   "filas_reales"))
def paso_9_filas_reales():
    ev = evaluacion_reales()
    print(f"Filas reales: {ev['n_reales']} de {ev['n_total']} "
          f"({ev['n_reales'] / ev['n_total']:.1%}); en el test: {ev['n_test_reales']}")
    tabla(ev["peso"].round(1), "Weight en las filas reales vs sintéticas", 1)
    tabla(ev["real_vs_sint"], "Modelos del proyecto: error en las filas reales vs sintéticas del test")
    tabla(ev["con"], "Sólo filas reales · con Height (CV 3×10)")
    tabla(ev["sin"], "Sólo filas reales · sin Height (CV 3×10)")
    tabla(ev["importancias"].head(8).rename("importancia"),
          "Importancia de atributos (Random Forest, filas reales, con Height)")
    etapa.figura(figura_reales(ev["con"], ev["sin"], "CV 3 folds × 10 repeticiones"),
                 "9_filas_reales")


# ----------------------------------------------------------------------------
# 10. Robustez ante ruido sintético (jittering)
# ----------------------------------------------------------------------------
# Mismo ruido que el paso 7 de la etapa 02: N(0, (nivel·σ)²) en los hábitos continuos
# (HABIT_NUM_COLS), recortado a RANGOS_HABITOS, en train y test. Height no se perturba.
NIVELES_RUIDO = [0.0, 0.05, 0.10, 0.15, 0.20]


def _modelos_ruido():
    return {
        "Árbol de regresión": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE,
                                               n_jobs=-1),
        "KNN (k=1)": Pipeline([("scaler", StandardScaler()),
                               ("knn", KNeighborsRegressor(n_neighbors=1))]),
        "KNN (k=3, distance)": Pipeline([("scaler", StandardScaler()),
                                         ("knn", KNeighborsRegressor(n_neighbors=3,
                                                                     weights="distance"))]),
    }


def barrido_ruido(X_tr, X_te, y_tr, y_te, niveles=NIVELES_RUIDO):
    """RMSE y R² en test de cada modelo para cada nivel de ruido (formato largo)."""
    filas = []
    for nl in niveles:
        Xa, Xb = _con_ruido(X_tr, X_te, nl) if nl > 0 else (X_tr, X_te)
        for nombre, modelo in _modelos_ruido().items():
            m = reg_metrics(y_te, modelo.fit(Xa, y_tr).predict(Xb))
            filas.append({"nivel": int(round(nl * 100)), "modelo": nombre,
                          "RMSE": m["RMSE"], "R2": m["R2"]})
    return pd.DataFrame(filas)


def figura_ruido(barridos):
    """Un panel por variante: RMSE en test vs nivel de ruido, una línea por modelo."""
    fig, axes = plt.subplots(1, len(barridos), figsize=(8 * len(barridos), 4.5), sharey=True)
    for ax, (variante, df) in zip(np.atleast_1d(axes), barridos.items()):
        ancho = df.pivot(index="nivel", columns="modelo", values="RMSE")
        for marcador, modelo in zip(["o", "s", "^", "D"], ancho.columns):
            ax.plot(ancho.index, ancho[modelo], marker=marcador, lw=2.2, label=modelo)
        ax.set_title(f"RMSE vs nivel de ruido en hábitos ({variante})")
        ax.set_xlabel("Nivel de perturbación gaussiana (% de desviación estándar)")
        ax.set_ylabel("RMSE en test (kg)")
        ax.set_xticks(list(ancho.index))
        ax.grid(True, linestyle="--", alpha=0.7)
    np.atleast_1d(axes)[-1].legend(title="Modelo", loc="center left", bbox_to_anchor=(1.01, 0.5))
    return fig


@etapa.paso("10", "Robustez ante ruido sintético (jittering) en los hábitos: barrido 0–20%, "
            "con y sin Height", figuras=1,
            parametros=("_modelos_ruido", "NIVELES_RUIDO", "_con_ruido", "HABIT_NUM_COLS",
                        "RANGOS_HABITOS"))
def paso_10_ruido():
    d = datos()
    barridos = {}
    for variante, sin in (("con Height", []), ("sin Height", DROP_H)):
        X_tr, X_te = d["X_train"].drop(columns=sin), d["X_test"].drop(columns=sin)
        df = barrido_ruido(X_tr, X_te, d["y_train"], d["y_test"])
        barridos[variante] = df
        rmse = df.pivot(index="nivel", columns="modelo", values="RMSE")
        tabla(rmse.rename_axis("Nivel de ruido (%)"), f"RMSE en test por nivel de ruido ({variante})")
        tabla((rmse - rmse.loc[0]).rename_axis("Nivel de ruido (%)"),
              f"Aumento del RMSE respecto a sin ruido, en kg ({variante})")
    etapa.figura(figura_ruido(barridos), "10_barrido_ruido")


if __name__ == "__main__":
    sys.exit(cli(etapa))
