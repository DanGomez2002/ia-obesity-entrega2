"""ETAPA 06 — Naive Bayes (sólo clasificación de `NObeyesdad`)
==============================================================

Naive Bayes no tiene versión de regresión: es un clasificador que aplica el teorema de Bayes
P(clase | x) ∝ P(clase) · Π_j P(x_j | clase) suponiendo que los atributos son INDEPENDIENTES
dada la clase. Se usan las implementaciones de scikit-learn.

Protocolo (el mismo de la etapa 02, reutilizando su split):
- Mismo split estratificado 80/20 de la etapa 02 (`e02.datos()`, random_state=42): 1669/418.
- Hiperparámetros con GridSearchCV (5-fold estratificado de la etapa 02, criterio macro-F1).
- El NB que pasa a la comparación final se elige por macro-F1 de CV, NUNCA por el test.
- Métricas en test: accuracy, macro-F1, classification_report, matriz de confusión y, por ser
  un modelo probabilístico, log-loss.

Dos Naive Bayes:
- GaussianNB: cada atributo, dentro de cada clase, se modela como una normal (media y varianza
  por clase). Natural para las 8 continuas; las binarias se tratan igual (aproximación).
  Hiperparámetro: var_smoothing (se suma a todas las varianzas var_smoothing · máx. varianza).
- CategoricalNB: cada atributo es una variable categórica con su tabla de frecuencias por
  clase. Las 8 continuas se discretizan en n_bins tramos por cuantiles (KBinsDiscretizer
  dentro del Pipeline, ajustado sólo con el train de cada fold); binarias, ordinales y one-hot
  ya son categorías. Hiperparámetros: n_bins y alpha (suavizado de Laplace).
- BernoulliNB NO se usa: modela atributos binarios (0/1) y aquí 8 de 20 son continuos y 2
  ordinales de 4 niveles; habría que binarizarlos con un único umbral y se perdería casi toda
  su información.

Sin escalado: Naive Bayes estima cada atributo por separado (media/varianza o frecuencias por
clase), así que reescalar un atributo no cambia sus predicciones (ver paso 1).

Dos variantes, como en la etapa 02: A (todas las features) y B (sin Weight ni Height).

Pasos:
  1    Datos y split (y por qué no hace falta escalar)
  1.1  El supuesto de independencia y por qué no se cumple aquí
  2    Modelos y grillas
  3    Variante A: tabla de resultados      4    Variante B: tabla de resultados
  3.1  Variante A: reportes y matrices      4.1  Variante B: reportes y matrices
  3.2  Efecto de var_smoothing y alpha      5    Comparación A vs B y Naive Bayes elegido
  6    Guardar resultados para la comparación final (data/resultados_entrega2.csv)
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, log_loss
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import CategoricalNB, GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler

from config import RANDOM_STATE
from src import cache, resultados
from src.evaluation import clf_metrics, metricas_por_clase, plot_confusion, results_table
from src.pasos import Etapa, cli
from src.preprocessing import CLASS_ORDER, NUMERIC_COLS
from src.salida import OPCIONES, nota, subtitulo, tabla

from etapas import e02_clasificacion as e02

etapa = Etapa("06", "Naive Bayes (clasificación)", "naive_bayes",
              "GaussianNB vs CategoricalNB en las variantes A y B de la etapa 02",
              globales=("RANDOM_STATE", "e02.datos", "e02.cv_estratificado"))

# KBinsDiscretizer avisa cuando un tramo por cuantiles queda vacío y lo elimina (pasa con
# atributos casi enteros como FCVC o NCP). Es lo esperado: se queda con menos tramos.
warnings.filterwarnings("ignore", message="Bins whose width are too small")

NOMBRES = {"GaussianNB": "Naive Bayes (GaussianNB)", "CategoricalNB": "Naive Bayes (CategoricalNB)"}
# CAEC y CALC tienen 4 niveles (0-3) y CALC=3 ("Always") aparece UNA vez en el train: un fold
# sin esa fila no conocería la categoría y CategoricalNB fallaría al verla. min_categories=4
# reserva 4 categorías por atributo (las que no aparecen sólo reciben el suavizado alpha).
MIN_CATEGORIAS = 4
# Tramos por atributo continuo. Con [3, 5, 7, 10] y luego hasta 20 el óptimo caía en el borde;
# la grilla llega a 60 para que la curva (paso 3.2) muestre la meseta y la caída posterior.
N_BINS = [3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 60]

_estado = {}


def make_models(columnas):
    """Los 2 Naive Bayes con su grilla. `columnas`: las del X de la variante (A o B)."""
    continuas = [c for c in NUMERIC_COLS if c in columnas]
    discretizar = ColumnTransformer(
        [("bins", KBinsDiscretizer(encode="ordinal", strategy="quantile",
                                   quantile_method="averaged_inverted_cdf"), continuas)],
        remainder="passthrough")
    return {
        "GaussianNB": (
            GaussianNB(),
            {"var_smoothing": np.logspace(-12, 0, 13)},
        ),
        "CategoricalNB": (
            Pipeline([("discretizar", discretizar),
                      ("nb", CategoricalNB(min_categories=MIN_CATEGORIAS))]),
            {"discretizar__bins__n_bins": N_BINS, "nb__alpha": [0.01, 0.1, 0.5, 1.0]},
        ),
    }


def fit_evaluate(X_tr, y_tr, X_te, y_te, tag):
    """Ajusta los 2 NB con GridSearchCV y devuelve (fitted, tabla_resultados)."""
    fitted, results = {}, {}
    for name, (model, grid) in make_models(X_tr.columns).items():
        gs = GridSearchCV(model, grid, cv=e02.cv_estratificado(), scoring="f1_macro", n_jobs=-1)
        gs.fit(X_tr, y_tr)
        y_pred = gs.predict(X_te)
        fitted[name] = gs
        results[name] = {
            "cv_f1_macro": gs.best_score_,
            **{f"test_{k}": v for k, v in clf_metrics(y_te, y_pred).items()},
            "test_log_loss": log_loss(y_te, gs.predict_proba(X_te), labels=gs.classes_),
            # probabilidad media que el modelo da a la clase que predice; si supera a la
            # accuracy, el modelo está más seguro de lo que acierta (sobreconfianza)
            "test_confianza_media": gs.predict_proba(X_te).max(axis=1).mean(),
            "tiempo_fit_s": gs.refit_time_,
        }
        print(f"   [{tag}] {name}: mejores params = {_params(gs)}")
    return fitted, results_table(results)


def _params(gs) -> dict:
    """best_params_ con números de Python (np.float64(0.001) -> 0.001) para imprimir."""
    return {k: v.item() if isinstance(v, np.generic) else v for k, v in gs.best_params_.items()}


def variante(tag: str):
    """'A' = todas las features, 'B' = sin Weight/Height. Devuelve (fitted, tabla, X_tr, X_te)."""
    d = e02.datos()
    if tag == "A":
        X_tr, X_te = d["X_train"], d["X_test"]
    else:
        X_tr, X_te = d["X_train"].drop(columns=e02.DROP_WH), d["X_test"].drop(columns=e02.DROP_WH)
    fitted, tabla_res = cache.obtener(
        f"nb_variante_{tag}", lambda: fit_evaluate(X_tr, d["y_train"], X_te, d["y_test"], tag))
    return fitted, tabla_res, X_tr, X_te


def elegido(tag: str) -> str:
    """El NB de la variante con mejor macro-F1 de CV (el test no interviene)."""
    _, t, _, _ = variante(tag)
    return t["cv_f1_macro"].idxmax()


def _reportes(tag: str):
    """classification_report, matrices de confusión y métricas por clase de los 2 NB."""
    fitted, _, _, X_te = variante(tag)
    y_test = e02.datos()["y_test"]
    sufijo = "A" if tag == "A" else "B, sin Weight/Height"
    for name, gs in fitted.items():
        subtitulo(f"{name} ({sufijo})")
        y_pred = gs.predict(X_te)
        print(classification_report(y_test, y_pred, labels=CLASS_ORDER, digits=3))
        rep = pd.DataFrame(classification_report(y_test, y_pred, labels=CLASS_ORDER,
                                                 digits=3, output_dict=True)).T
        tabla(rep.loc[CLASS_ORDER, ["precision", "recall", "f1-score"]],
              f"classification_report por clase — {name} ({tag})", 3, texto=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (name, gs) in zip(axes, fitted.items()):
        plot_confusion(y_test, gs.predict(X_te), CLASS_ORDER, f"{name} ({sufijo})", ax=ax)
    etapa.figura(fig, f"{'3.1' if tag == 'A' else '4.1'}_matrices_confusion_{tag}")
    for name, gs in fitted.items():
        cm = confusion_matrix(y_test, gs.predict(X_te), labels=CLASS_ORDER)
        tabla(metricas_por_clase(cm, CLASS_ORDER), f"Métricas por clase — {name} ({tag})", 3)


# ----------------------------------------------------------------------------
# 1. Datos y split
# ----------------------------------------------------------------------------
def _escalado():
    """GaussianNB con y sin StandardScaler: macro-F1 de CV (mismo GridSearch de var_smoothing)."""
    d = e02.datos()
    grid = {"var_smoothing": np.logspace(-12, 0, 13)}
    filas = {}
    for tag in ("A", "B"):
        X = d["X_train"] if tag == "A" else d["X_train"].drop(columns=e02.DROP_WH)
        for nombre, modelo, g in (
                ("sin escalar", GaussianNB(), grid),
                ("con StandardScaler", Pipeline([("scaler", StandardScaler()), ("nb", GaussianNB())]),
                 {f"nb__{k}": v for k, v in grid.items()})):
            gs = GridSearchCV(modelo, g, cv=e02.cv_estratificado(), scoring="f1_macro", n_jobs=-1)
            gs.fit(X, d["y_train"])
            filas[(tag, nombre)] = {"cv_f1_macro": gs.best_score_,
                                    "mejor var_smoothing": float(list(gs.best_params_.values())[0])}
    return pd.DataFrame(filas).T


@etapa.paso("1", "Datos y split estratificado 80/20 (el de la etapa 02) y por qué no se escala")
def paso_1_datos():
    d = e02.datos()
    assert len(d["X_train"]) == 1669 and len(d["X_test"]) == 418, "el split debe ser 1669/418"
    print("train:", d["X_train"].shape, "| test:", d["X_test"].shape)
    print("features:", d["X"].columns.tolist())
    print("huella del test:", resultados.huella_test(d["X_test"].index))

    X_tr, X_te, y_tr = d["X_train"], d["X_test"], d["y_train"]
    sc = StandardScaler().fit(X_tr)
    var_clase = X_tr.groupby(y_tr.values).var(ddof=0)
    n_cero = int((var_clase < 1e-12).sum().sum())
    var_max = X_tr.var(ddof=0)
    # (a) Mismo var_smoothing (el de sklearn por defecto) con y sin escalar
    sin = GaussianNB().fit(X_tr, y_tr)
    con = GaussianNB().fit(sc.transform(X_tr), y_tr)
    iguales = np.mean(sin.predict(X_te) == con.predict(sc.transform(X_te)))
    # (b) Misma varianza suavizada, sólo cambiada de unidades: la invariancia es exacta
    con.var_ = sin.var_ / sc.scale_ ** 2
    iguales_exacto = np.mean(sin.predict(X_te) == con.predict(sc.transform(X_te)))
    print(f"Pares (clase, atributo) con varianza 0 en el train: {n_cero} "
          f"(atributos: {', '.join(var_clase.columns[(var_clase < 1e-12).any()])})")
    print(f"Mayor varianza sin escalar: {var_max.max():.1f} ({var_max.idxmax()}); escalada: 1")
    print(f"GaussianNB(var_smoothing=1e-9) sin escalar vs escalado: {iguales * 100:.1f} % de "
          "predicciones iguales en el test")
    print(f"... con la MISMA varianza suavizada expresada en las nuevas unidades: "
          f"{iguales_exacto * 100:.1f} % iguales")
    esc = cache.obtener("nb_escalado", _escalado)
    tabla(esc, "GaussianNB: macro-F1 de CV sin escalar vs con StandardScaler", 4)
    nota(f"""
    Naive Bayes no necesita escalado porque no mide distancias ni optimiza por gradiente: estima
    cada atributo por separado dentro de cada clase. En GaussianNB, si un atributo se multiplica
    por una constante, su media y su desviación por clase se multiplican por lo mismo y la
    densidad de cada punto cambia por el mismo factor en TODAS las clases: la clase más probable
    no cambia. Se comprueba arriba: con la misma varianza suavizada expresada en las nuevas
    unidades, el {iguales_exacto * 100:.1f} % de las predicciones coincide.

    El matiz es var_smoothing: sklearn suma a todas las varianzas var_smoothing · (mayor varianza
    entre atributos), que sin escalar es la de {var_max.idxmax()} ({var_max.max():.0f}) y escalado es 1.
    Esto pesa mucho aquí porque hay {n_cero} pares (clase, atributo) con varianza 0 en el train
    (binarias u ordinales constantes dentro de una clase): su varianza ES el suavizado, así que
    al escalar cambia su densidad y con el mismo var_smoothing sólo coincide el
    {iguales * 100:.1f} % de las predicciones. No es que haga falta escalar: var_smoothing es un
    hiperparámetro y el GridSearch lo ajusta en la escala en que se trabaje. Escalar no aporta y
    aquí empeora la CV (tabla). CategoricalNB trabaja con tramos y categorías, que no tienen
    escala. Por eso no hay StandardScaler en los Pipelines de Naive Bayes.
    """)


@etapa.paso("1.1", "El supuesto de independencia condicional y por qué no se cumple aquí", figuras=1)
def paso_1_1_independencia():
    d = e02.datos()
    tr = d["X_train"].assign(clase=d["y_train"].values)
    imc = tr["Weight"] / tr["Height"] ** 2

    filas = {}
    for clase in CLASS_ORDER:
        g = tr[tr["clase"] == clase]
        filas[clase] = {"n": len(g), "corr(Weight, Height | clase)": g["Weight"].corr(g["Height"]),
                        "IMC mín": imc[g.index].min(), "IMC máx": imc[g.index].max()}
    tabla(pd.DataFrame(filas).T, "Dentro de cada clase Weight y Height siguen correlacionados (train)", 3)

    # Pares de atributos más correlacionados DENTRO de las clases (media de la |corr| por clase)
    num = [c for c in NUMERIC_COLS]
    corr_media = sum(tr[tr["clase"] == c][num].corr().abs() for c in CLASS_ORDER) / len(CLASS_ORDER)
    pares = (corr_media.where(np.triu(np.ones(corr_media.shape, dtype=bool), k=1)).stack()
             .sort_values(ascending=False).head(6).rename("|corr| media dentro de las clases"))
    pares.index = [f"{a} – {b}" for a, b in pares.index]
    tabla(pares, "Pares de atributos numéricos más dependientes dada la clase (train)", 3)

    fig, ax = plt.subplots(figsize=(9, 6))
    paleta = sns.color_palette("viridis", len(CLASS_ORDER))
    for color, clase in zip(paleta, CLASS_ORDER):
        g = tr[tr["clase"] == clase]
        ax.scatter(g["Height"], g["Weight"], s=10, alpha=0.6, color=color, label=clase)
    h = np.linspace(tr["Height"].min(), tr["Height"].max(), 100)
    for corte in (18.5, 25, 27.5, 30, 35, 40):   # cortes de IMC (los de e02, paso 8)
        ax.plot(h, corte * h ** 2, "k--", lw=0.8, alpha=0.6)
    ax.set_xlabel("Height (m)")
    ax.set_ylabel("Weight (kg)")
    ax.set_title("La clase la fija IMC = Weight/Height²: dada la clase, Weight y Height NO son independientes")
    ax.legend(fontsize=8, markerscale=2, loc="upper left")
    etapa.figura(fig, "1.1_independencia_weight_height")
    nota("""
    Supuesto de Naive Bayes: dada la clase, los atributos son independientes, es decir
    P(x1, ..., xn | clase) = P(x1 | clase) · ... · P(xn | clase). Aquí NO se cumple, y de la
    forma más fuerte posible para los dos atributos más informativos: la clase se define por el
    IMC = Weight / Height² (líneas discontinuas de la figura). Dentro de una clase el IMC está
    confinado a un intervalo estrecho, de modo que si se conoce Height el rango posible de Weight
    queda casi fijado: Weight y Height están ligados dada la clase (tabla 1; su correlación dentro
    de cada clase es claramente positiva). Naive Bayes multiplica P(Weight | clase) y
    P(Height | clase) como si fueran dos evidencias independientes: no puede representar que lo
    que decide la clase es su COCIENTE, sólo mira cada uno por separado. Además hay otras
    dependencias entre hábitos (tabla 2). Consecuencias esperadas: Naive Bayes rinde por debajo
    de los modelos que sí combinan atributos (árbol, RF) y, al contar dos veces la misma
    evidencia, sus probabilidades tienden a ser demasiado extremas (el paso 5 lo comprueba
    comparando la confianza media con la accuracy). Aun así puede clasificar bien si ese error en
    las probabilidades no cambia cuál es la clase más probable.
    """)


# ----------------------------------------------------------------------------
# 2. Modelos y grillas
# ----------------------------------------------------------------------------
@etapa.paso("2", "Definición de modelos y grillas de hiperparámetros",
            parametros=("make_models", "MIN_CATEGORIAS", "N_BINS"))
def paso_2_modelos():
    for name, (model, grid) in make_models(e02.datos()["X"].columns).items():
        subtitulo(name)
        print("   modelo:", model)
        for k, v in grid.items():
            print(f"   {k}: {[float(x) for x in v] if isinstance(v, np.ndarray) else v}")


# ----------------------------------------------------------------------------
# 3. Variante A
# ----------------------------------------------------------------------------
@etapa.paso("3", "Variante A (todas las features): GridSearch y tabla de resultados",
            parametros=("make_models", "fit_evaluate"))
def paso_3_variante_A():
    _, tabla_A, _, _ = variante("A")
    tabla(tabla_A, "Variante A — macro-F1 en CV, métricas en test, log-loss y tiempo")


@etapa.paso("3.1", "Variante A: reportes por clase y matrices de confusión (test)", figuras=1,
            parametros=("make_models",))
def paso_3_1_reportes_A():
    _reportes("A")


@etapa.paso("3.2", "Efecto de var_smoothing (GaussianNB) y de n_bins/alpha (CategoricalNB) en la CV",
            figuras=1, parametros=("make_models",))
def paso_3_2_hiperparametros():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.8))
    for tag, estilo in (("A", "-"), ("B", "--")):
        fitted, _, _, _ = variante(tag)
        g = fitted["GaussianNB"].cv_results_
        ax1.plot(np.asarray(g["param_var_smoothing"], dtype=float), g["mean_test_score"],
                 marker="o", ls=estilo, label=f"variante {tag}")
        c = pd.DataFrame({"n_bins": np.asarray(fitted["CategoricalNB"].cv_results_
                                               ["param_discretizar__bins__n_bins"], dtype=int),
                          "alpha": np.asarray(fitted["CategoricalNB"].cv_results_["param_nb__alpha"],
                                              dtype=float),
                          "f1": fitted["CategoricalNB"].cv_results_["mean_test_score"]})
        curva = c.groupby("n_bins")["f1"].max()
        ax2.plot(curva.index, curva.values, marker="s", ls=estilo, label=f"variante {tag}")
        print(f"[{tag}] GaussianNB:", _params(fitted["GaussianNB"]),
              "| CategoricalNB:", _params(fitted["CategoricalNB"]))
    ax1.set_xscale("log")
    ax1.set_xlabel("var_smoothing (escala log)")
    ax1.set_ylabel("macro-F1 (CV)")
    ax1.set_title("GaussianNB: macro-F1 de CV según var_smoothing")
    ax2.set_xlabel("n_bins (tramos por atributo continuo)")
    ax2.set_ylabel("macro-F1 (CV, mejor alpha)")
    ax2.set_title("CategoricalNB: macro-F1 de CV según n_bins")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.legend()
    etapa.figura(fig, "3.2_hiperparametros_nb")


# ----------------------------------------------------------------------------
# 4. Variante B
# ----------------------------------------------------------------------------
@etapa.paso("4", "Variante B (sin Weight/Height): GridSearch y tabla de resultados",
            parametros=("make_models", "e02.DROP_WH"))
def paso_4_variante_B():
    _, tabla_B, _, _ = variante("B")
    tabla(tabla_B, "Variante B — macro-F1 en CV, métricas en test, log-loss y tiempo")


@etapa.paso("4.1", "Variante B: reportes por clase y matrices de confusión (test)", figuras=1,
            parametros=("make_models", "e02.DROP_WH"))
def paso_4_1_reportes_B():
    _reportes("B")


# ----------------------------------------------------------------------------
# 5. Comparación A vs B
# ----------------------------------------------------------------------------
@etapa.paso("5", "Comparación A vs B y Naive Bayes elegido (por macro-F1 de CV)", figuras=1,
            parametros=("make_models",))
def paso_5_comparacion():
    _, tabla_A, _, _ = variante("A")
    _, tabla_B, _, _ = variante("B")
    comp = pd.concat({"A (todas)": tabla_A, "B (sin Weight/Height)": tabla_B}, axis=0)
    tabla(comp, "Comparación A vs B")
    elegidos = pd.DataFrame({tag: {"NB elegido (mejor CV)": elegido(tag),
                                   "cv_f1_macro": t.loc[elegido(tag), "cv_f1_macro"],
                                   "test_f1_macro": t.loc[elegido(tag), "test_f1_macro"]}
                             for tag, t in (("A", tabla_A), ("B", tabla_B))}).T
    tabla(elegidos, "Naive Bayes que pasa a la comparación final", grafica=False)

    subtitulo("¿Probabilidades sobreconfiadas? (confianza media vs accuracy en test)")
    for (var, modelo), r in comp.iterrows():
        dif = r["test_confianza_media"] - r["test_accuracy"]
        juicio = ("sobreconfiado" if dif > 0.02 else "infraconfiado" if dif < -0.02
                  else "bien calibrado en media")
        print(f"   {var:<22} {modelo:<14} confianza media {r['test_confianza_media']:.3f} vs "
              f"accuracy {r['test_accuracy']:.3f} ({dif:+.3f}): {juicio}")

    fig, ax = plt.subplots(figsize=(8, 4))
    datos_plot = comp.reset_index().rename(columns={"level_0": "variante", "level_1": "modelo"})
    sns.barplot(data=datos_plot, x="modelo", y="test_f1_macro", hue="variante", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("Naive Bayes: macro-F1 en test por modelo y variante")
    etapa.figura(fig, "5_comparacion_A_vs_B")


# ----------------------------------------------------------------------------
# 6. Guardar resultados
# ----------------------------------------------------------------------------
@etapa.paso("6", "Guardar el Naive Bayes elegido de A y B en data/resultados_entrega2.csv")
def paso_6_guardar():
    d = e02.datos()
    filas = []
    for tag in ("A", "B"):
        fitted, t, _, _ = variante(tag)
        m = elegido(tag)
        r = t.loc[m]
        filas.append({
            "tarea": f"clasificacion_{tag}", "modelo": NOMBRES[m],
            "accuracy": r["test_accuracy"], "f1_macro": r["test_f1_macro"],
            "cv_score": r["cv_f1_macro"], "cv_metrica": "f1_macro",
            "log_loss": r["test_log_loss"], "tiempo_entrenamiento_s": r["tiempo_fit_s"],
            "mejores_params": str(_params(fitted[m])),
            "n_test": len(d["X_test"]), "huella_test": resultados.huella_test(d["X_test"].index),
            "nota": "elegido por macro-F1 de CV entre GaussianNB y CategoricalNB",
            "fuente": f"etapa 06 · pasos 3/4 (variante {tag})",
        })
    df = pd.DataFrame(filas).set_index(["tarea", "modelo"])
    tabla(df[["accuracy", "f1_macro", "cv_score", "log_loss", "tiempo_entrenamiento_s"]],
          "Filas para data/resultados_entrega2.csv", grafica=False)
    if OPCIONES["guardar"]:
        resultados.guardar_entrega2(filas, "06")
    else:
        print("   (no se escribe data/resultados_entrega2.csv: sólo se guarda con --guardar o run_all.py)")


if __name__ == "__main__":
    sys.exit(cli(etapa))
