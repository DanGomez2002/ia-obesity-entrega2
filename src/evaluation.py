"""Helpers de evaluación y gráficos compartidos por las etapas de modelado (06-09)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def clf_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
    }


def adjusted_r2(r2: float, n: int, p: int) -> float:
    """R² ajustado. n = nº de observaciones, p = nº de predictores."""
    if n - p - 1 <= 0:
        return float("nan")
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


def reg_metrics(y_true, y_pred, n_features: int | None = None) -> dict:
    """MAE, RMSE, R2 y R2_adj. Mantener el orden de claves.
    Si n_features es None, R2_adj = nan (retrocompatibilidad)."""
    r2 = r2_score(y_true, y_pred)
    r2_adj = adjusted_r2(r2, len(y_true), n_features) if n_features is not None else float("nan")
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2,
        "R2_adj": r2_adj,
    }


def plot_confusion(y_true, y_pred, labels, title, ax=None, normalize=False, vmin=None, vmax=None):
    """Matriz de confusión. vmin/vmax fijan la escala de color (útil para comparar
    varios paneles con la misma escala)."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if normalize:
        cm = cm / cm.sum(axis=1, keepdims=True)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        cbar=False,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel("Clase predicha")
    ax.set_ylabel("Clase real")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    return ax


def metricas_por_clase(cm, labels) -> pd.DataFrame:
    """TP, FN, FP, TN, Precision, Recall y F1 de cada clase, leídos de la matriz de confusión.

    cm: filas = clase real, columnas = clase predicha (convención de sklearn).
    Para la clase i:  TP = cm[i, i]          (diagonal)
                      FN = fila i sin TP     (era i y se predijo otra)
                      FP = columna i sin TP  (se predijo i y era otra)
                      TN = todo lo demás
    Precision = TP / (TP + FP)   Recall = TP / (TP + FN)   F1 = 2·P·R / (P + R)
    Si un denominador es 0 (p. ej. nunca se predijo esa clase) la métrica queda en NaN; si
    Precision y Recall valen 0, F1 = 0 (igual que scikit-learn).
    """
    cm = np.asarray(cm)
    tp = np.diag(cm)
    fn = cm.sum(axis=1) - tp
    fp = cm.sum(axis=0) - tp
    tn = cm.sum() - tp - fn - fp
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), np.nan)
        recall = np.where(tp + fn > 0, tp / (tp + fn), np.nan)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    f1 = np.where(np.isnan(precision) | np.isnan(recall), np.nan, f1)
    df = pd.DataFrame({"TP": tp, "FN": fn, "FP": fp, "TN": tn,
                       "Precision": precision, "Recall": recall, "F1": f1},
                      index=pd.Index(labels, name="Clase"))
    return df


def results_table(results: dict) -> pd.DataFrame:
    """results: {nombre_modelo: {metrica: valor}} -> DataFrame ordenado."""
    return pd.DataFrame(results).T.round(4)


def plot_residuos_vs_pred(y_test, preds):
    """Residuos (real − predicho) frente al valor predicho, un panel por modelo."""
    y_test = np.asarray(y_test)
    lim = max(np.abs(y_test - np.asarray(p)).max() for p in preds.values()) * 1.05
    fig, axes = plt.subplots(1, len(preds), figsize=(6 * len(preds), 4.5), sharey=True)
    for ax, (nombre, y_pred) in zip(np.atleast_1d(axes), preds.items()):
        y_pred = np.asarray(y_pred)
        ax.scatter(y_pred, y_test - y_pred, s=12, alpha=0.5, color="steelblue")
        ax.axhline(0, c="r", ls="--", lw=1.2)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel("Predicho (kg)")
        ax.set_ylabel("Residuo: real − predicho (kg)")
        ax.set_title(f"Residuos vs predicho — {nombre}")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig
