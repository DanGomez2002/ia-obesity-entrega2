"""ETAPA 07 — Máquinas de vectores de soporte: clasificación (SVC) y regresión (SVR)
=====================================================================================

Una SVM busca la frontera de MÁXIMO MARGEN entre clases. Sólo los puntos que quedan sobre el
margen o lo violan (los vectores de soporte) definen el modelo. C controla cuánto se castiga
violar el margen (C grande = margen estrecho, ajusta más el train). Con un KERNEL la SVM
trabaja implícitamente en un espacio transformado:
- linear : K(x, z) = x·z                    frontera lineal (hiperplano)
- poly   : K(x, z) = (gamma·x·z + coef0)^d  fronteras polinómicas de grado d
- rbf    : K(x, z) = exp(-gamma·||x - z||²) fronteras flexibles; gamma = alcance de cada punto
SVR usa la misma idea para regresión: un "tubo" de ancho epsilon alrededor de la predicción
dentro del cual el error no se penaliza; los vectores de soporte son los puntos fuera del tubo.

Protocolo (el de las etapas 02 y 03, reutilizando sus splits):
- Clasificación: split de `e02.datos()` (1669/418, estratificado por clase), variantes A y B.
- Regresión: split de `e03.datos()` (1669/418, estratificado por deciles de Weight), sin
  NObeyesdad.
- SIEMPRE Pipeline(StandardScaler, SVC/SVR): la SVM depende de distancias y productos
  escalares, así que sin escalar dominarían los atributos de mayor rango (Weight en kg).
- Un GridSearchCV por kernel (5-fold; macro-F1 en clf, RMSE en reg) para poder comparar los
  kernels entre sí; el kernel que pasa a la comparación final se elige por CV, no por test.
- SVR: además del escalado de X, el TARGET se estandariza con TransformedTargetRegressor
  (ajustado sólo con el train de cada fold) para que epsilon y C tengan una escala razonable;
  las predicciones se devuelven en kg y las métricas se reportan en kg.

Pasos:
  1    Datos y split (clasificación y regresión)
  2    Modelos y grillas por kernel
  3    Clasificación A: tabla por kernel          4    Clasificación B: tabla por kernel
  3.1  A: reportes y matrices de los 3 kernels    4.1  B: reportes y matrices
  3.2  ¿Por qué el kernel lineal compite en A?    5    Comparación A vs B
  3.3  Frontera de decisión en 2D (PCA)           6    Regresión (SVR): tabla por kernel
  3.4  Vectores de soporte de los mejores modelos 6.1  Regresión: predicho vs real y residuos
  7    Guardar resultados para la comparación final (data/resultados_entrega2.csv)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.compose import TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC, SVR

from config import RANDOM_STATE
from src import cache, resultados
from src.evaluation import (clf_metrics, metricas_por_clase, plot_confusion,
                            plot_residuos_vs_pred, reg_metrics, results_table)
from src.pasos import Etapa, cli
from src.preprocessing import CLASS_ORDER
from src.salida import OPCIONES, nota, subtitulo, tabla

from etapas import e02_clasificacion as e02
from etapas import e03_regresion as e03

etapa = Etapa("07", "SVM: clasificación (SVC) y regresión (SVR)", "svm",
              "Kernels linear, poly y rbf en las variantes A y B y en la regresión de Weight",
              globales=("RANDOM_STATE", "e02.datos", "e03.datos", "e02.cv_estratificado"))

KERNELS = ["linear", "poly", "rbf"]
COLORES_KERNEL = {"linear": "#4c78a8", "poly": "#f58518", "rbf": "#54a24b"}
CACHE_MB = 1000   # caché de la matriz de kernel de libsvm (acelera las grillas)

_estado = {}


def _params(gs) -> dict:
    """best_params_ sin el prefijo del Pipeline y con números de Python, para imprimir."""
    return {k.split("__")[-1]: (v.item() if isinstance(v, np.generic) else v)
            for k, v in gs.best_params_.items()}


# ----------------------------------------------------------------------------
# Clasificación (SVC)
# ----------------------------------------------------------------------------
def make_models():
    """Un Pipeline(StandardScaler, SVC) por kernel, cada uno con su grilla."""
    def svc(kernel):
        return Pipeline([("scaler", StandardScaler()),
                         ("svc", SVC(kernel=kernel, cache_size=CACHE_MB))])
    # Grillas ampliadas en dos pasadas, hasta que ningún óptimo quedara en el borde:
    # 1ª: linear C=100 (B), poly C=100 (A) y degree=3/gamma=0.1 (B), rbf C=1000/gamma=0.001 (A).
    # 2ª: linear C=1000 (A), poly degree=4 (B), rbf C=1e5/gamma=1e-4 (A). Más allá de esos
    # bordes la CV ya empeoraba (probado aparte), así que el óptimo es un máximo real.
    # Excepción: linear se queda en C ≤ 1000. Con C=3000 y C=10000 la CV de A baja (0.9593 y
    # 0.9547, frente a 0.9599 con C=1000; medido aparte) y en B, que no es separable, un solo
    # ajuste con C=10000 pasa de 10 minutos (libsvm converge más despacio cuanto mayor es C).
    return {
        "linear": (svc("linear"), {"svc__C": [0.01, 0.1, 1, 10, 100, 1000]}),
        "poly": (svc("poly"), {"svc__C": [0.1, 1, 10, 100, 1000], "svc__degree": [2, 3, 4, 5],
                               "svc__gamma": ["scale", 0.01, 0.1, 0.5], "svc__coef0": [0, 1]}),
        "rbf": (svc("rbf"), {"svc__C": [0.1, 1, 10, 100, 1000, 10000, 100000, 1000000],
                             "svc__gamma": ["scale", 0.00001, 0.0001, 0.001, 0.01, 0.1, 1]}),
    }


def fit_evaluate(X_tr, y_tr, X_te, y_te, tag):
    """GridSearchCV de cada kernel. Devuelve (fitted, tabla_resultados)."""
    fitted, results = {}, {}
    for kernel, (model, grid) in make_models().items():
        gs = GridSearchCV(model, grid, cv=e02.cv_estratificado(), scoring="f1_macro", n_jobs=-1)
        gs.fit(X_tr, y_tr)
        fitted[kernel] = gs
        svc = gs.best_estimator_.named_steps["svc"]
        results[kernel] = {
            "cv_f1_macro": gs.best_score_,
            **{f"test_{k}": v for k, v in clf_metrics(y_te, gs.predict(X_te)).items()},
            "n_vectores_soporte": int(svc.n_support_.sum()),
            "tiempo_fit_s": gs.refit_time_,
        }
        print(f"   [{tag}] {kernel}: mejores params = {_params(gs)}")
    return fitted, results_table(results)


def variante(tag: str):
    """'A' = todas las features, 'B' = sin Weight/Height. Devuelve (fitted, tabla, X_tr, X_te)."""
    d = e02.datos()
    if tag == "A":
        X_tr, X_te = d["X_train"], d["X_test"]
    else:
        X_tr, X_te = d["X_train"].drop(columns=e02.DROP_WH), d["X_test"].drop(columns=e02.DROP_WH)
    fitted, tabla_res = cache.obtener(
        f"svm_clf_{tag}", lambda: fit_evaluate(X_tr, d["y_train"], X_te, d["y_test"], tag))
    return fitted, tabla_res, X_tr, X_te


def mejor_kernel(t: pd.DataFrame, col="cv_f1_macro", mayor=True) -> str:
    """Kernel con mejor score de CV (el test no interviene)."""
    return t[col].idxmax() if mayor else t[col].idxmin()


def _tabla_kernels(fitted, t):
    """La tabla de resultados con la columna de los mejores parámetros de cada kernel."""
    return t.assign(mejores_params=[str(_params(fitted[k])) for k in t.index])


def _reportes(tag: str):
    fitted, _, _, X_te = variante(tag)
    y_test = e02.datos()["y_test"]
    sufijo = "A" if tag == "A" else "B, sin Weight/Height"
    for kernel, gs in fitted.items():
        subtitulo(f"SVC {kernel} ({sufijo})")
        y_pred = gs.predict(X_te)
        print(classification_report(y_test, y_pred, labels=CLASS_ORDER, digits=3))
        rep = pd.DataFrame(classification_report(y_test, y_pred, labels=CLASS_ORDER,
                                                 digits=3, output_dict=True)).T
        tabla(rep.loc[CLASS_ORDER, ["precision", "recall", "f1-score"]],
              f"classification_report por clase — SVC {kernel} ({tag})", 3, texto=False)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for ax, (kernel, gs) in zip(axes, fitted.items()):
        plot_confusion(y_test, gs.predict(X_te), CLASS_ORDER, f"SVC {kernel} ({sufijo})", ax=ax)
    etapa.figura(fig, f"{'3.1' if tag == 'A' else '4.1'}_matrices_confusion_{tag}")
    for kernel, gs in fitted.items():
        cm = confusion_matrix(y_test, gs.predict(X_te), labels=CLASS_ORDER)
        tabla(metricas_por_clase(cm, CLASS_ORDER), f"Métricas por clase — SVC {kernel} ({tag})", 3)


# ----------------------------------------------------------------------------
# Regresión (SVR)
# ----------------------------------------------------------------------------
def make_models_reg():
    """Un TransformedTargetRegressor(Pipeline(StandardScaler, SVR)) por kernel.

    El target se estandariza (StandardScaler sobre y, ajustado con el train de cada fold):
    así epsilon se mide en desviaciones típicas del peso (0.1 ≈ 2.6 kg) y C no necesita
    valores enormes. La predicción se des-estandariza sola: sale en kg."""
    def svr(kernel):
        return TransformedTargetRegressor(
            regressor=Pipeline([("scaler", StandardScaler()),
                                ("svr", SVR(kernel=kernel, cache_size=CACHE_MB))]),
            transformer=StandardScaler())
    # Primera pasada: epsilon=0.2 (borde) en linear y poly, y C=0.1 (borde) en poly -> se
    # amplían. Se recortan dos zonas lentas y lejos del óptimo, medidas con un ajuste suelto:
    # poly con C=100 (~150 s por ajuste) y linear con C=1000 (~150 s). 2ª pasada: epsilon=0.5
    # (borde) en linear -> se añade 1.0 (con 0.75 y 1.0 la CV ya empeoraba).
    p = "regressor__svr__"
    eps = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
    return {
        "linear": (svr("linear"), {f"{p}C": [0.1, 1, 10, 100], f"{p}epsilon": eps}),
        "poly": (svr("poly"), {f"{p}C": [0.01, 0.1, 1, 10], f"{p}epsilon": eps,
                               f"{p}degree": [2, 3], f"{p}gamma": ["scale", 0.01, 0.1]}),
        "rbf": (svr("rbf"), {f"{p}C": [0.1, 1, 10, 100, 1000], f"{p}epsilon": eps,
                             f"{p}gamma": ["scale", 0.001, 0.01, 0.1, 1]}),
    }


def _ajustar_reg():
    """GridSearch de los 3 kernels + baseline. Devuelve (fitted, tabla)."""
    d = e03.datos()
    X_tr, X_te, y_tr, y_te = d["X_train"], d["X_test"], d["y_train"], d["y_test"]
    n_features = X_tr.shape[1]
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fitted, results = {}, {}
    dummy = DummyRegressor(strategy="mean").fit(X_tr, y_tr)
    results["Baseline (media)"] = {
        "cv_RMSE": np.nan,
        **{f"test_{k}": v for k, v in reg_metrics(y_te, dummy.predict(X_te), n_features).items()},
        "n_vectores_soporte": np.nan, "tiempo_fit_s": np.nan}
    for kernel, (model, grid) in make_models_reg().items():
        gs = GridSearchCV(model, grid, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
        gs.fit(X_tr, y_tr)
        fitted[kernel] = gs
        svr = gs.best_estimator_.regressor_.named_steps["svr"]
        results[kernel] = {
            "cv_RMSE": -gs.best_score_,
            **{f"test_{k}": v for k, v in reg_metrics(y_te, gs.predict(X_te), n_features).items()},
            "n_vectores_soporte": len(svr.support_),
            "tiempo_fit_s": gs.refit_time_,
        }
        print(f"   [reg] {kernel}: mejores params = {_params(gs)}")
    return fitted, results_table(results)


def modelos_reg():
    return cache.obtener("svm_reg", _ajustar_reg)


# ----------------------------------------------------------------------------
# 1. Datos y split
# ----------------------------------------------------------------------------
@etapa.paso("1", "Datos y split: el de la etapa 02 (clasificación) y el de la 03 (regresión)")
def paso_1_datos():
    dc, dr = e02.datos(), e03.datos()
    assert len(dc["X_train"]) == len(dr["X_train"]) == 1669 and len(dc["X_test"]) == len(dr["X_test"]) == 418
    assert "NObeyesdad" not in dr["X"].columns, "NObeyesdad no debe estar en la regresión"
    print("clasificación  train:", dc["X_train"].shape, "| test:", dc["X_test"].shape,
          "| huella del test:", resultados.huella_test(dc["X_test"].index))
    print("regresión      train:", dr["X_train"].shape, "| test:", dr["X_test"].shape,
          "| huella del test:", resultados.huella_test(dr["X_test"].index))
    print("features de regresión (sin NObeyesdad):", dr["X"].columns.tolist())
    rango = dc["X_train"].max() - dc["X_train"].min()
    tabla(rango.sort_values(ascending=False).head(5).rename("rango en train (sin escalar)"),
          "Por qué hay que escalar: rango de los atributos más amplios", 2)


# ----------------------------------------------------------------------------
# 2. Modelos y grillas
# ----------------------------------------------------------------------------
@etapa.paso("2", "Modelos y grillas por kernel (SVC y SVR)",
            parametros=("make_models", "make_models_reg", "KERNELS"))
def paso_2_modelos():
    for titulo, modelos in (("SVC", make_models()), ("SVR", make_models_reg())):
        for kernel, (model, grid) in modelos.items():
            subtitulo(f"{titulo} — kernel {kernel}")
            for k, v in grid.items():
                print(f"   {k.split('__')[-1]}: {v}")
            n = int(np.prod([len(v) for v in grid.values()]))
            print(f"   ({n} combinaciones × 5 folds = {n * 5} ajustes)")


# ----------------------------------------------------------------------------
# 3. Clasificación A
# ----------------------------------------------------------------------------
@etapa.paso("3", "Clasificación A (todas las features): un GridSearch por kernel",
            parametros=("make_models", "fit_evaluate"))
def paso_3_variante_A():
    fitted, t, _, _ = variante("A")
    tabla(_tabla_kernels(fitted, t), "Variante A — SVC por kernel: CV, test, nº de vectores de soporte y tiempo")
    print(f"Kernel elegido por macro-F1 de CV: {mejor_kernel(t)}")


@etapa.paso("3.1", "Clasificación A: reportes por clase y matrices de confusión de los 3 kernels",
            figuras=1, parametros=("make_models",))
def paso_3_1_reportes_A():
    _reportes("A")


def _por_que_lineal():
    """macro-F1 de CV de un SVC con SÓLO Weight y Height, en kg/m y en logaritmos."""
    d = e02.datos()
    X = d["X_train"][["Weight", "Height"]]
    log = FunctionTransformer(np.log, feature_names_out="one-to-one")
    casos = {
        "linear sobre (Weight, Height)": Pipeline([("scaler", StandardScaler()),
                                                   ("svc", SVC(kernel="linear"))]),
        "linear sobre (log Weight, log Height)": Pipeline([("log", log), ("scaler", StandardScaler()),
                                                           ("svc", SVC(kernel="linear"))]),
        "rbf sobre (Weight, Height)": Pipeline([("scaler", StandardScaler()),
                                                ("svc", SVC(kernel="rbf"))]),
    }
    filas = {}
    for nombre, modelo in casos.items():
        grid = {"svc__C": [0.1, 1, 10, 100, 1000]}
        if "rbf" in nombre:
            grid["svc__gamma"] = ["scale", 0.1, 1, 10]
        gs = GridSearchCV(modelo, grid, cv=e02.cv_estratificado(), scoring="f1_macro", n_jobs=-1)
        gs.fit(X, d["y_train"])
        filas[nombre] = {"cv_f1_macro": gs.best_score_, "mejores_params": str(_params(gs))}
    return pd.DataFrame.from_dict(filas, orient="index")


@etapa.paso("3.2", "¿Por qué el kernel lineal compite en A? (la fuga hace la clase casi lineal)",
            parametros=("make_models", "_por_que_lineal"))
def paso_3_2_por_que_lineal():
    fA, tA, _, _ = variante("A")
    _, tB, _, _ = variante("B")
    hueco = pd.DataFrame({
        "linear (cv)": [tA.loc["linear", "cv_f1_macro"], tB.loc["linear", "cv_f1_macro"]],
        "mejor no lineal (cv)": [tA.loc[["poly", "rbf"], "cv_f1_macro"].max(),
                                 tB.loc[["poly", "rbf"], "cv_f1_macro"].max()],
    }, index=["A (con Weight/Height)", "B (sin Weight/Height)"])
    hueco["no lineal − linear"] = hueco["mejor no lineal (cv)"] - hueco["linear (cv)"]
    tabla(hueco, "macro-F1 de CV: kernel lineal vs mejor kernel no lineal")
    sol = cache.obtener("svm_por_que_lineal", _por_que_lineal)
    tabla(sol, "SVC con SÓLO Weight y Height (macro-F1 de CV, 5 folds)")
    nota(f"""
    La clase NObeyesdad se define por cortes del IMC = Weight / Height². Tomando logaritmos,
    log IMC = log Weight − 2·log Height: cada corte "IMC = c" es una RECTA en el plano
    (log Weight, log Height), así que las clases quedan separadas por rectas paralelas; en
    (Weight, Height) son las curvas Weight = c·Height², casi rectas en el rango de estaturas
    observado (figura del paso 1.1 de la etapa 06). Por eso en la variante A un clasificador
    lineal ya tiene la información decisiva casi en forma lineal: el hueco entre el mejor kernel
    no lineal y el lineal es de {hueco.iloc[0, 2]:+.4f} en A frente a {hueco.iloc[1, 2]:+.4f} en
    B, donde, sin Weight/Height, la relación de los hábitos con la clase no tiene esa forma.
    Con sólo dos atributos, el SVC lineal sobre (log Weight, log Height) logra macro-F1 de CV
    {sol.iloc[1, 0]:.4f}, frente a {sol.iloc[0, 0]:.4f} en kg/m y {sol.iloc[2, 0]:.4f} con rbf.
    Y un rbf con sólo esos dos atributos llega a {sol.iloc[2, 0]:.4f}, frente a
    {tA.loc["rbf", "cv_f1_macro"]:.4f} del SVC rbf con las 20 features: en la variante A casi todo
    el rendimiento sale de Weight y Height. Las clases no son cortes perfectos del IMC (sus
    rangos se solapan, etapa 06 paso 1.1), lo que pone un techo a cualquier frontera basada sólo
    en Weight y Height.

    El rbf elegido en A también apunta a lo mismo: {_params(fA["rbf"])}. Con gamma muy pequeño
    cada punto "alcanza" a casi todos los demás y la frontera se vuelve muy suave; compensado
    con un C enorme, el rbf se comporta prácticamente como un clasificador lineal. Por eso su
    óptimo se desplaza hacia la esquina C grande / gamma pequeño de la grilla.
    """)


@etapa.paso("3.3", "Frontera de decisión de los 3 kernels sobre 2 componentes PCA (sólo visualización)",
            figuras=1, parametros=("make_models",))
def paso_3_3_frontera_pca():
    fitted, _, X_tr, X_te = variante("A")
    d = e02.datos()
    codigo = {c: i for i, c in enumerate(CLASS_ORDER)}
    y_tr = d["y_train"].map(codigo).to_numpy()
    y_te = d["y_test"].map(codigo).to_numpy()
    proy = Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_components=2))]).fit(X_tr)
    Z_tr, Z_te = proy.transform(X_tr), proy.transform(X_te)
    var = proy.named_steps["pca"].explained_variance_ratio_
    print(f"Varianza explicada por PC1 y PC2: {var[0]:.3f} + {var[1]:.3f} = {var.sum():.3f}")

    margen = 0.5
    xx, yy = np.meshgrid(np.linspace(Z_tr[:, 0].min() - margen, Z_tr[:, 0].max() + margen, 300),
                         np.linspace(Z_tr[:, 1].min() - margen, Z_tr[:, 1].max() + margen, 300))
    cmap = ListedColormap(sns.color_palette("viridis", len(CLASS_ORDER)).as_hex())
    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5), sharex=True, sharey=True)
    filas = {}
    for ax, kernel in zip(axes, KERNELS):
        params = {k.split("__")[-1]: v for k, v in fitted[kernel].best_params_.items()}
        svc = SVC(kernel=kernel, cache_size=CACHE_MB, **params).fit(Z_tr, y_tr)
        zona = svc.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax.contourf(xx, yy, zona, levels=np.arange(-0.5, len(CLASS_ORDER)), cmap=cmap, alpha=0.35)
        ax.scatter(Z_te[:, 0], Z_te[:, 1], c=y_te, cmap=cmap, vmin=0, vmax=len(CLASS_ORDER) - 1,
                   s=14, edgecolors="k", linewidths=0.3)
        acc2d = accuracy_score(y_te, svc.predict(Z_te))
        filas[kernel] = {"params (de la grilla de A)": str(params), "accuracy test en 2D": acc2d,
                         "n_vectores_soporte en 2D": int(svc.n_support_.sum())}
        ax.set_title(f"SVC {kernel} en 2D — acc test 2D {acc2d:.3f}")
        ax.set_xlabel(f"PC1 ({var[0] * 100:.1f} % varianza)")
    axes[0].set_ylabel(f"PC2 ({var[1] * 100:.1f} % varianza)")
    handles = [plt.Line2D([], [], marker="o", ls="", color=cmap(i), label=c)
               for i, c in enumerate(CLASS_ORDER)]
    fig.legend(handles=handles, loc="lower center", ncol=7, fontsize=9, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Fronteras de los 3 kernels en el plano PCA de la variante A (puntos = test)", y=1.01)
    etapa.figura(fig, "3.3_frontera_pca")
    tabla(pd.DataFrame.from_dict(filas, orient="index"),
          "SVC reentrenado sobre 2 componentes PCA (sólo para visualizar)",
          grafica=False)
    nota(f"""
    Esta figura es SÓLO una visualización: cada SVC se reentrena con las 2 primeras componentes
    principales (que retienen el {var.sum() * 100:.1f} % de la varianza de las 20 features
    escaladas), con los hiperparámetros elegidos para su kernel en la variante A. Su accuracy en
    2D no es la del modelo real (paso 3), que usa las 20 features. Lo que muestra es la FORMA
    de la frontera: rectas en linear, curvas polinómicas en poly y regiones flexibles en rbf.
    """)


def _tabla_sv(gs, X_tr, y_tr, titulo):
    svc = gs.best_estimator_.named_steps["svc"]
    n_clase = pd.Series(svc.n_support_, index=svc.classes_).reindex(CLASS_ORDER)
    total = y_tr.value_counts().reindex(CLASS_ORDER)
    t = pd.DataFrame({"vectores de soporte": n_clase, "filas en train": total,
                      "% de la clase": n_clase / total * 100})
    t.loc["TOTAL"] = [n_clase.sum(), total.sum(), n_clase.sum() / total.sum() * 100]
    tabla(t, titulo, 1)


@etapa.paso("3.4", "Vectores de soporte de los mejores modelos (A, B y regresión)",
            parametros=("make_models", "make_models_reg"))
def paso_3_4_vectores_soporte():
    y_tr = e02.datos()["y_train"]
    for tag in ("A", "B"):
        fitted, t, X_tr, _ = variante(tag)
        k = mejor_kernel(t)
        _tabla_sv(fitted[k], X_tr, y_tr, f"Vectores de soporte — SVC {k} (variante {tag}, elegido por CV)")
    fitted, t = modelos_reg()
    k = mejor_kernel(t.drop(index="Baseline (media)"), "cv_RMSE", mayor=False)
    n_sv = len(fitted[k].best_estimator_.regressor_.named_steps["svr"].support_)
    print(f"SVR {k} (regresión, elegido por CV): {n_sv} vectores de soporte de 1669 filas de train "
          f"({n_sv / 1669 * 100:.1f} %) — los puntos fuera del tubo de ancho epsilon")
    nota("""
    Pocos vectores de soporte = frontera definida por pocos puntos (modelo "simple" y rápido de
    predecir). Una fracción alta indica clases solapadas o un margen muy ancho (C pequeño): la
    mayoría de los puntos tocan o violan el margen. En SVR, cuanto más estrecho es el tubo
    (epsilon pequeño), más puntos quedan fuera y más vectores de soporte hay.
    """)


# ----------------------------------------------------------------------------
# 4. Clasificación B
# ----------------------------------------------------------------------------
@etapa.paso("4", "Clasificación B (sin Weight/Height): un GridSearch por kernel",
            parametros=("make_models", "e02.DROP_WH"))
def paso_4_variante_B():
    fitted, t, _, _ = variante("B")
    tabla(_tabla_kernels(fitted, t), "Variante B — SVC por kernel: CV, test, nº de vectores de soporte y tiempo")
    print(f"Kernel elegido por macro-F1 de CV: {mejor_kernel(t)}")


@etapa.paso("4.1", "Clasificación B: reportes por clase y matrices de confusión de los 3 kernels",
            figuras=1, parametros=("make_models", "e02.DROP_WH"))
def paso_4_1_reportes_B():
    _reportes("B")


# ----------------------------------------------------------------------------
# 5. Comparación A vs B
# ----------------------------------------------------------------------------
@etapa.paso("5", "Comparación A vs B por kernel", figuras=1, parametros=("make_models",))
def paso_5_comparacion():
    _, tA, _, _ = variante("A")
    _, tB, _, _ = variante("B")
    comp = pd.concat({"A (todas)": tA, "B (sin Weight/Height)": tB}, axis=0)
    tabla(comp, "Comparación A vs B")
    fig, ax = plt.subplots(figsize=(8, 4))
    datos_plot = comp.reset_index().rename(columns={"level_0": "variante", "level_1": "kernel"})
    sns.barplot(data=datos_plot, x="kernel", y="test_f1_macro", hue="variante", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("SVC: macro-F1 en test por kernel y variante")
    etapa.figura(fig, "5_comparacion_A_vs_B")


# ----------------------------------------------------------------------------
# 6. Regresión (SVR)
# ----------------------------------------------------------------------------
@etapa.paso("6", "Regresión de Weight (SVR): un GridSearch por kernel, con baseline y R² ajustado",
            parametros=("make_models_reg", "_ajustar_reg"))
def paso_6_regresion():
    fitted, t = modelos_reg()
    params = {"Baseline (media)": "—", **{k: str(_params(g)) for k, g in fitted.items()}}
    tabla(t.assign(mejores_params=pd.Series(params)),
          "SVR por kernel: RMSE de CV y métricas en test (kg), nº de vectores de soporte y tiempo")
    print("Kernel elegido por RMSE de CV:",
          mejor_kernel(t.drop(index="Baseline (media)"), "cv_RMSE", mayor=False))


@etapa.paso("6.1", "Regresión: predicho vs real, residuos y extrapolación de los 3 kernels", figuras=2,
            parametros=("make_models_reg",))
def paso_6_1_pred_vs_real():
    fitted, t = modelos_reg()
    d = e03.datos()
    y_te = d["y_test"]
    preds = {f"SVR {k}": g.predict(d["X_test"]) for k, g in fitted.items()}
    lo = min(y_te.min(), min(p.min() for p in preds.values())) - 2
    hi = max(y_te.max(), max(p.max() for p in preds.values())) + 2
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (nombre, y_pred) in zip(axes, preds.items()):
        k = nombre.split()[-1]
        ax.scatter(y_te, y_pred, s=12, alpha=0.5, color="steelblue")
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.2)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("Real (kg)")
        ax.set_ylabel("Predicho (kg)")
        ax.set_title(f"{nombre}  RMSE={t.loc[k, 'test_RMSE']:.2f} kg  R²={t.loc[k, 'test_R2']:.3f}"
                     f"  R²adj={t.loc[k, 'test_R2_adj']:.3f}")
    etapa.figura(fig, "6.1_pred_vs_real_svr")
    etapa.figura(plot_residuos_vs_pred(y_te, preds), "6.1_residuos_vs_predicho_svr")

    # Predicciones fuera del rango de pesos visto en el train (extrapolación)
    y_tr = d["y_train"]
    fuera = pd.DataFrame.from_dict({nombre: {
        "pred. mínima (kg)": p.min(), "pred. máxima (kg)": p.max(),
        "fuera de [mín, máx] del train": int(((p < y_tr.min()) | (p > y_tr.max())).sum()),
        "residuo máx. |kg|": np.abs(y_te.to_numpy() - p).max(),
    } for nombre, p in preds.items()}, orient="index")
    tabla(fuera, f"Extrapolación: predicciones fuera del rango del train "
                 f"[{y_tr.min():.1f}, {y_tr.max():.1f}] kg", 2)
    nota(f"""
    El kernel poly tiene mejor RMSE de CV que el lineal ({t.loc["poly", "cv_RMSE"]:.2f} frente a
    {t.loc["linear", "cv_RMSE"]:.2f} kg) pero peor en test ({t.loc["poly", "test_RMSE"]:.2f} frente a
    {t.loc["linear", "test_RMSE"]:.2f} kg). La tabla muestra por qué: un polinomio crece sin límite
    fuera de la zona donde hay datos de entrenamiento, y unas pocas filas del test producen
    predicciones absurdas (un peso negativo o muy por encima del máximo observado). El RMSE eleva
    al cuadrado esos errores, así que unas pocas predicciones disparatadas bastan para empeorarlo.
    El rbf no tiene ese problema: lejos de los datos su predicción tiende a la media.
    """)


# ----------------------------------------------------------------------------
# 7. Guardar resultados
# ----------------------------------------------------------------------------
@etapa.paso("7", "Guardar el mejor kernel (por CV) de A, B y regresión en data/resultados_entrega2.csv")
def paso_7_guardar():
    dc, dr = e02.datos(), e03.datos()
    filas = []
    for tag in ("A", "B"):
        fitted, t, _, _ = variante(tag)
        k = mejor_kernel(t)
        r = t.loc[k]
        filas.append({
            "tarea": f"clasificacion_{tag}", "modelo": f"SVM (SVC {k})",
            "accuracy": r["test_accuracy"], "f1_macro": r["test_f1_macro"],
            "cv_score": r["cv_f1_macro"], "cv_metrica": "f1_macro",
            "tiempo_entrenamiento_s": r["tiempo_fit_s"], "n_vectores_soporte": r["n_vectores_soporte"],
            "mejores_params": str(_params(fitted[k])),
            "n_test": len(dc["X_test"]), "huella_test": resultados.huella_test(dc["X_test"].index),
            "nota": "kernel elegido por macro-F1 de CV entre linear, poly y rbf",
            "fuente": f"etapa 07 · paso {'3' if tag == 'A' else '4'} (variante {tag})",
        })
    fitted, t = modelos_reg()
    k = mejor_kernel(t.drop(index="Baseline (media)"), "cv_RMSE", mayor=False)
    r = t.loc[k]
    filas.append({
        "tarea": "regresion", "modelo": f"SVM (SVR {k})",
        "MAE": r["test_MAE"], "RMSE": r["test_RMSE"], "R2": r["test_R2"], "R2_adj": r["test_R2_adj"],
        "cv_score": r["cv_RMSE"], "cv_metrica": "RMSE",
        "tiempo_entrenamiento_s": r["tiempo_fit_s"], "n_vectores_soporte": r["n_vectores_soporte"],
        "mejores_params": str(_params(fitted[k])),
        "n_test": len(dr["X_test"]), "huella_test": resultados.huella_test(dr["X_test"].index),
        "nota": "kernel elegido por RMSE de CV entre linear, poly y rbf",
        "fuente": "etapa 07 · paso 6 (regresión)",
    })
    df = pd.DataFrame(filas).set_index(["tarea", "modelo"])
    tabla(df[["accuracy", "f1_macro", "RMSE", "R2", "cv_score", "n_vectores_soporte",
              "tiempo_entrenamiento_s"]], "Filas para data/resultados_entrega2.csv", grafica=False)
    if OPCIONES["guardar"]:
        resultados.guardar_entrega2(filas, "07")
    else:
        print("   (no se escribe data/resultados_entrega2.csv: sólo se guarda con --guardar o run_all.py)")


if __name__ == "__main__":
    sys.exit(cli(etapa))
