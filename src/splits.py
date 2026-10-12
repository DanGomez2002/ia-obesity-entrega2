"""Splits train/test y validación cruzada compartidos por todas las etapas.

Es el MISMO protocolo de la Entrega 1 (mismo código, misma semilla), así que los tests son
exactamente los mismos y los resultados congelados de la Entrega 1
(data/resultados_entrega1.csv) se pueden comparar fila a fila con los de la Entrega 2:

- Clasificación: split 80/20 estratificado por clase (random_state=42) -> 1669 / 418.
- Regresión de Weight (sin NObeyesdad): split 80/20 estratificado por deciles de Weight
  (random_state=42) -> 1669 / 418. Es OTRA partición distinta de la de clasificación.
- Validación cruzada de clasificación: StratifiedKFold de 5 folds con shuffle (42).

La huella de cada test (src/resultados.huella_test) lo comprueba: clasificación
8d9af7b40ee9 y regresión 021799f95367, las mismas de la Entrega 1.
"""
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from config import RANDOM_STATE
from src.data import TARGET_CLF, TARGET_REG
from src.preprocessing import cargar_clf, cargar_reg

DROP_WH = ["Weight", "Height"]   # columnas que se quitan en la variante B de clasificación

# Respuestas de encuesta que SMOTE deja con decimales. Una fila con TODAS ellas enteras no fue
# interpolada: es una respuesta original (491 de 2087 filas, el ~23 % real que declara UCI).
# Es una heurística, no una etiqueta del dataset.
COLS_ENCUESTA = ["Age", "FCVC", "NCP", "CH2O", "FAF", "TUE"]

_estado = {}


def datos_clf() -> dict:
    """Split estratificado 80/20. Devuelve dict con X, y, X_train, X_test, y_train, y_test."""
    if "clf" not in _estado:
        df = cargar_clf()
        X = df.drop(columns=[TARGET_CLF])
        y = df[TARGET_CLF].astype(str)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
        _estado["clf"] = dict(X=X, y=y, X_train=X_train, X_test=X_test,
                              y_train=y_train, y_test=y_test)
    return _estado["clf"]


def datos_reg() -> dict:
    """Split 80/20 estratificado por deciles de Weight. Mismas claves + n_features."""
    if "reg" not in _estado:
        df = cargar_reg()
        X = df.drop(columns=[TARGET_REG])
        y = df[TARGET_REG]
        strata = pd.qcut(y, q=10, labels=False)   # deciles de peso para estratificar
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=strata, random_state=RANDOM_STATE)
        _estado["reg"] = dict(X=X, y=y, X_train=X_train, X_test=X_test, y_train=y_train,
                              y_test=y_test, n_features=X.shape[1])
    return _estado["reg"]


def cv_estratificado():
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def filas_reales(X):
    """Máscara booleana: True en las filas cuyas respuestas de encuesta son todas enteras."""
    return (X[COLS_ENCUESTA] % 1 == 0).all(axis=1)
