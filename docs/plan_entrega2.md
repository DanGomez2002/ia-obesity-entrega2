# Plan de la Entrega 2 — SVM, Naive Bayes y Redes neuronales (Keras)

Dataset: UCI 544 (Estimation of Obesity Levels), 2111 filas en crudo, 2087 sin duplicados.
Este plan amplía el proyecto de la Entrega 1 **con su misma estructura**: la versión Python plana
de `etapas/` + pasos numerados (`src/pasos.py`), con salidas en `figures/` y `resultados_texto/`
y ejecución completa con `run_all.py`.

> **Nota (T8):** al terminar se quitó del proyecto todo lo de la Entrega 1 (etapas 01-05, sus
> salidas, la app de Streamlit, `estudio_etapas.html` y `study/`), que sigue en el proyecto
> anterior. De la Entrega 1 sólo queda `data/resultados_entrega1.csv`. Las secciones T0-T7
> describen el proyecto tal como era en cada momento.

Estado: **T9 terminada. Espera tu verificación.**

---

## 0. Lo que encontré al inspeccionar el repo (T0)

### 0.1 La estructura real no es la de notebooks

El encargo original hablaba de `notebooks/`, `docs/html/`, `run_all.sh`, `docs/plan_entrega.md`,
`docs/decisiones.md` y la guía `guia_estudio_sustentacion_v2.md`. **Nada de eso existe en este
repo.** Lo que hay es la versión plana (commit `c5c72c9`), que conserva la numeración de los
notebooks originales:

| Encargo original (notebooks) | Equivalente real en este repo |
|---|---|
| `notebooks/0X_*.ipynb` con secciones numeradas | `etapas/e0X_*.py` con `@etapa.paso("3.4", ...)` |
| celdas markdown | docstring del módulo + `salida.nota()` |
| salidas de las celdas / HTML exportado | `resultados_texto/<etapa>/<paso>.md` + `figures/<etapa>/*.png` |
| `run_all.sh` | `run_all.py` (y `main.py` como menú) |
| `fit_evaluate`, `make_models`, `variante(tag)` | en `etapas/e02_clasificacion.py` / `etapas/e03_regresion.py` |
| `reg_metrics` (con R² ajustado), `plot_confusion`, `metricas_por_clase` | `src/evaluation.py` |

**Decisión (confirmada por ti):** replicar la estructura plana. Las cuatro etapas nuevas siguen
la numeración: **06, 07, 08, 09**. La etapa 05 ya es la comparación de la Entrega 1.
Respecto a `decisiones.md` y la guía de estudio, dijiste que no son necesarias para el proyecto:
**no se crean** (ver T6). Las decisiones quedan documentadas como en la Entrega 1: en el
docstring de cada etapa y en las `nota()` de los pasos.

### 0.2 Discrepancias con los valores de referencia del encargo

Lo comprobé contra la salida ejecutada guardada en `resultados_texto/` (Entrega 1). Se usará
**el valor ejecutado**. En T1 se confirma además re-ejecutando las etapas 02 y 03 en este
entorno, solo para verificar; esas salidas no se guardan.

| Valor | Referencia del encargo | Ejecutado (`resultados_texto/`) | Comentario |
|---|---|---|---|
| Clasificación A · Árbol · F1 | 0.938 | **0.9362** | 0.938 es su *accuracy* (0.9378) |
| Clasificación A · KNN (k=1) · F1 | 0.814 | **0.8075** | accuracy 0.8206 ✓ |
| Clasificación B · KNN (k=3) · F1 | 0.764 | **0.7567** | 0.764 es su macro-F1 de **CV** (0.7640); accuracy 0.7679 ✓ |
| Resto de clasificación A/B | — | coincide | Árbol B 0.7464/0.7401, RF A 0.9569/0.9558, RF B 0.8565/0.8528 |
| Regresión (baseline, Árbol, RF, KNN) | — | coincide | Baseline R² = −0.0001 (se muestra 0.000), R²adj −0.0479 |

### 0.3 Acoplamientos que hay que respetar (para no romper la app ni los tests)

- `tests/test_navegacion.py` exige que `estudio_etapas.html` y el registro de la app
  (`webapp/navegacion.py`) tengan **exactamente** las mismas etapas y pasos. Por eso
  **`estudio_etapas.html` no se toca** y las etapas 06-09 no se registran en la app.
- `main.py` y `run_all.py` son de la CLI, no de Streamlit: se amplían con las etapas 06-09.
- `src/cache.py` calcula la huella de cada resultado a partir del código del que depende.
  Si cambia `src/evaluation.py`, los GridSearch de 02/03 se recalculan una vez (unos 40 s) con
  los mismos números. Para no tocarlo, los helpers nuevos van en un módulo nuevo
  (`src/resultados.py`) o dentro de la etapa que los usa. **No se modifica el comportamiento
  de nada existente.**
- `requirements.txt` lo comparte la app: añadir `tensorflow`/`keras` hace más pesada su
  instalación, pero no la rompe. Te aviso si veo cualquier otro riesgo.

---

## 1. Protocolo común (idéntico al de la Entrega 1)

- **Mismo split, sin copiarlo.** Las etapas nuevas importan `e02.datos()`, `e02.cv_estratificado()`,
  `e02.DROP_WH` y `e03.datos()`, como ya hace `e03` con `e02`. Resultado: 80/20,
  `random_state=42`, estratificado por clase (clf) y por deciles de Weight (reg), **1669 train / 418 test**.
  Cada paso 1 hace `assert` de esos tamaños y de que `NObeyesdad` no esté en regresión.
- **Huella del test.** Se calcula un hash de los índices del test y se guarda junto a cada fila de
  resultados. La etapa 09 comprueba que todos los modelos (E1 y E2) se evaluaron sobre las
  mismas 418 filas.
- **Validación.** Hiperparámetros con 5-fold CV (`StratifiedKFold` / `KFold`, `shuffle=True`, 42),
  `GridSearchCV` con `f1_macro` (clf) y `neg_root_mean_squared_error` (reg). La selección
  (kernel, variante de NB, arquitectura) se hace **siempre por CV/validación, nunca por test**.
  El test se usa una vez, al final.
- **Escalado.** Siempre dentro de un `Pipeline` (o, en Keras, un `StandardScaler` ajustado solo
  con la parte de entrenamiento).
- **Variantes.** Clasificación A (20 atributos) y B (18, sin Weight ni Height). Regresión de
  Weight con 19 atributos (sin NObeyesdad, con su `assert`).
- **Métricas.**
  - Clasificación: accuracy, macro-F1, `classification_report`, matriz de confusión y la tabla
    TP/FN/FP/TN por clase.
  - Regresión: MAE, RMSE, R², R² ajustado + baseline `DummyRegressor(mean)`, predicho vs real
    y residuos.
  - Adicionales de la Entrega 2: tiempo de entrenamiento (`refit_time_` del GridSearch / tiempo
    del `fit` final en Keras), nº de vectores de soporte (SVM), nº de parámetros y épocas
    efectivas (red), log-loss (Naive Bayes y red).
- **Caché.** Los GridSearch y entrenamientos van a `cache/` con `cache.obtener(...)`, igual que en 02/03.
- **Salidas.** `python etapas/e0X_*.py --todo --guardar --sin-ventanas` escribe
  `resultados_texto/0X_*/` y `figures/0X_*/` (el equivalente a "notebook ejecutado con salidas").
- **Resultados para la comparación.** El último paso de cada etapa 06-08 escribe sus filas en
  `data/resultados_entrega2.csv` (upsert por `tarea` + `modelo`). La E1 queda congelada en
  `data/resultados_entrega1.csv`.

Esquema común de ambos CSV:

```
tarea, modelo, accuracy, f1_macro, MAE, RMSE, R2, R2_adj, cv_score, cv_metrica,
log_loss, tiempo_entrenamiento_s, n_parametros, n_vectores_soporte, epocas,
mejores_params, n_test, huella_test, fuente
```

`tarea` ∈ {`clasificacion_A`, `clasificacion_B`, `regresion`} (+ `clustering_nota` solo en E1).
Lo que no aplica o no se midió queda vacío. **Nunca se estima.**

---

## 2. Tareas

| Tarea | Modelo recomendado | Motivo |
|---|---|---|
| T0. Inspeccionar el repo y escribir este plan | Opus (modo plan) | Define todo lo demás |
| T1. Congelar resultados de la Entrega 1 (`data/resultados_entrega1.csv`) | Sonnet | Lectura y copia, verificada |
| T2. Etapa 06 — Naive Bayes | Sonnet | Modelo simple, estructura existente |
| T3. Etapa 07 — SVM (SVC y SVR) | Sonnet | Código sklearn similar a KNN |
| T4. Etapa 08 — Redes neuronales (Keras) | Opus | Fuga test→validación, escalado del target, semillas |
| T5. Etapa 09 — Comparación final y conclusiones | Opus | Conclusiones honestas, discrepancias con E1 |
| ~~T6. Guía de estudio y `decisiones.md`~~ | — | **Eliminada:** dijiste que no son necesarias |
| T7. Verificación final: `run_all.py` completo, números del plan vs salidas, tests | Opus | Auditoría de detalle |

Cada tarea termina con: (1) la etapa ejecutada con `--todo --guardar`, (2) los números reales
en la BITÁCORA, (3) commit + push a `claude/laughing-cori-nprv9n` y (4) parada para tu
verificación, indicando el modelo de la siguiente tarea.

### T1. Congelar resultados de la Entrega 1 — *Sonnet*

- Construir `data/resultados_entrega1.csv` con el esquema común, **leyendo** de
  `resultados_texto/02_clasificacion/3.md`, `4.md`, `resultados_texto/03_regresion/2.md` y
  `resultados_texto/05_comparacion/1.md`. Sin figuras.
  - Filas: clasificacion_A × {Árbol, RF, KNN}, clasificacion_B × {Árbol, RF, KNN},
    regresion × {Baseline (media), Árbol, RF, KNN}.
  - `cv_score` es el de la E1. Tiempo, log-loss, params, SV y épocas van vacíos: no se
    re-entrena.
  - `clustering_nota`: K-Means K=7 (silueta, ARI, NMI, pureza), marcado como **no comparable**.
- Script reproducible: `src/resultados.py` con `congelar_entrega1()`, que parsea esas tablas.
  Si una tabla no aparece, falla en lugar de inventar.
- Verificación de reproducibilidad, sin guardar nada: instalar `requirements.txt` y ejecutar
  02 pasos 3/4 y 03 paso 2 en este entorno. Los números deben coincidir con `resultados_texto/`.
  Si no coinciden, me detengo y te aviso.
- **Criterio de aceptación**: CSV con 10 filas supervisadas + clustering; cada número coincide
  con `resultados_texto/`; `huella_test` calculada de `e02.datos()` / `e03.datos()` (solo el
  split); BITÁCORA con la tabla congelada y el resultado de la re-ejecución.

### T2. Etapa 06 — Naive Bayes (`etapas/e06_naive_bayes.py`) — *Sonnet*

Solo clasificación; variantes A y B. Pasos con la numeración de la etapa 02:

| Paso | Contenido |
|---|---|
| 1 | Datos y split (reutiliza `e02.datos()`; asserts 1669/418). Nota: por qué NB **no necesita escalado** (estima una media y una varianza por clase y atributo; reescalar un atributo no cambia las predicciones de GaussianNB, salvo el efecto de `var_smoothing`, que suma un ε proporcional a la mayor varianza; CategoricalNB trabaja con categorías) |
| 1.1 | **Supuesto de independencia condicional**: nota + evidencia real. Correlación Weight–Height **dentro de cada clase** y peso de la relación IMC↔clase; por qué se incumple aquí |
| 2 | `make_models()`: **GaussianNB** (`var_smoothing` en `logspace(-12, 0, 13)`) vs **CategoricalNB** sobre atributos discretizados (`KBinsDiscretizer` por cuantiles ajustado solo con train para las 8 continuas; binarias, ordinales y one-hot tal cual; `n_bins` ∈ {3, 5, 7, 10}, `alpha` ∈ {0.01, 0.1, 0.5, 1}; `min_categories` fijado para que un fold sin una categoría rara no falle). BernoulliNB se descarta y se explica por qué: la mayoría de atributos no son binarios y binarizarlos tira información |
| 3 | Variante A: `fit_evaluate` → tabla cv_f1_macro, test_accuracy, test_f1_macro, test_log_loss, tiempo |
| 3.1 | Variante A: classification_report + matrices de confusión + tabla por clase |
| 4 | Variante B: lo mismo |
| 5 | Comparación A vs B y **elección del NB por macro-F1 de CV** (no por test) |
| 6 | Escribir filas en `data/resultados_entrega2.csv` |

**Criterio de aceptación**: `--todo --guardar` sin errores; tablas y matrices en
`resultados_texto/06_naive_bayes/`; el NB elegido sale del CV; log-loss reportado; BITÁCORA
con números.

### T3. Etapa 07 — SVM (`etapas/e07_svm.py`) — *Sonnet*

**Clasificación (SVC)**, variantes A y B, con `Pipeline(StandardScaler, SVC)`. Un GridSearch por
kernel para poder compararlos:

- `linear`: C ∈ {0.01, 0.1, 1, 10, 100}
- `poly`: C ∈ {0.1, 1, 10, 100}, degree ∈ {2, 3}, gamma ∈ {scale, 0.01, 0.1}, coef0 ∈ {0, 1}
- `rbf`: C ∈ {0.1, 1, 10, 100, 1000}, gamma ∈ {scale, 0.001, 0.01, 0.1, 1}

Se elige el mejor kernel por macro-F1 de CV. Las grillas pueden ajustarse si un óptimo cae en el
borde; cualquier cambio queda anotado en la bitácora.

| Paso | Contenido |
|---|---|
| 1 | Datos y split (clf y reg), asserts |
| 2 | Modelos y grillas por kernel |
| 3 | Variante A: tabla por kernel (cv, test acc/F1, tiempo, nº SV) y mejor por CV |
| 3.1 | Variante A: reportes, matrices, tabla por clase del mejor |
| 3.2 | **¿Por qué el kernel lineal compite en A?** Nota sobre la fuga: la clase se define por cortes de IMC = W/H²; en (log W, log H) las fronteras son rectas y en el rango observado de (W, H) son casi rectas. Evidencia real: hueco lineal−rbf en A vs en B, y un SVC lineal solo con Weight+Height |
| 3.3 | Frontera de decisión de los 3 kernels sobre 2 componentes PCA (PCA ajustado con train escalado, mejores hiperparámetros de cada kernel reentrenados en 2D). **Solo visualización**: se avisa de que su accuracy no es la del modelo |
| 3.4 | Vectores de soporte del mejor modelo (total y por clase, % del train) |
| 4 | Variante B: tabla por kernel + reportes/matrices del mejor |
| 5 | Comparación A vs B |

**Regresión (SVR)**: `TransformedTargetRegressor(Pipeline(StandardScaler, SVR), transformer=StandardScaler())`.
El target se estandariza, ajustado solo con el train de cada fold, para que `epsilon` y `C`
tengan una escala razonable; las métricas se reportan en kg. Kernels:

- `linear`: C ∈ {0.1, 1, 10, 100}, epsilon ∈ {0.01, 0.05, 0.1, 0.2}
- `poly`: lo mismo + degree ∈ {2, 3}, gamma ∈ {scale, 0.01, 0.1}
- `rbf`: C ∈ {0.1, 1, 10, 100, 1000}, gamma ∈ {scale, 0.001, 0.01, 0.1, 1}, epsilon ∈ {0.01, 0.05, 0.1, 0.2}

| Paso | Contenido |
|---|---|
| 6 | Tabla baseline + 3 kernels (cv_RMSE, MAE, RMSE, R², R²adj, tiempo, nº SV) y mejor por CV |
| 6.1 | Predicho vs real y residuos (`plot_residuos_vs_pred`) |
| 7 | Escribir filas en `data/resultados_entrega2.csv` |

**Criterio de aceptación**: ejecución completa sin errores; los 3 kernels comparados en A, B y
regresión; frontera PCA de los 3 kernels; nº de SV reportado; si el lineal gana o empata en A,
se reporta tal cual con su explicación; BITÁCORA.

### T4. Etapa 08 — Redes neuronales con Keras (`etapas/e08_redes_neuronales.py`) — *Opus*

- **Dependencias**: `tensorflow` + `keras` con versiones **fijas** en `requirements.txt`
  (las exactas se deciden al instalar; Python 3.13 necesita TF ≥ 2.20).
- **Reproducibilidad**: `keras.utils.set_random_seed(42)` al inicio de cada entrenamiento y
  `tf.config.experimental.enable_op_determinism()`. Se comprueba que dos ejecuciones con la
  misma semilla dan el mismo número.
- **Validación para early stopping, solo del train**: el train (1669) se parte en sub-train
  85 % / validación 15 % (estratificado, 42). El test no se toca hasta el final. El
  `StandardScaler` de X (y de y en regresión) se ajusta **solo con el sub-train**.
- **Entradas / salidas**: 20 (A) / 18 (B) / 19 (reg) neuronas de entrada, una por atributo
  codificado. Clasificación: clase 0-6 codificada con `CLASS_ORDER`, salida 7 softmax,
  `sparse_categorical_crossentropy`. Regresión: 1 neurona lineal, `mse`, target escalado y
  predicción invertida a kg.
- **Arquitecturas**: pequeña `[32]`, media `[64, 32]`, grande `[128, 64, 32]`; ReLU, Adam
  (lr 1e-3), batch 32, hasta 500 épocas, `EarlyStopping(monitor="val_loss", patience=30,
  restore_best_weights=True)`. Se elige por **val_loss media sobre 5 semillas** (no por test),
  apoyado en las curvas loss train vs val por época.
- **Reporte final**: modelo con semilla 42 en test **y** media ± desviación sobre 5 semillas
  (42-46). Nº de parámetros (`count_params()`), épocas entrenadas y mejor época, log-loss
  (clf) y tiempo de entrenamiento.
- **Caché**: se guardan historiales, predicciones y métricas (no el objeto Keras), con
  `cache.obtener`.

| Paso | Contenido |
|---|---|
| 1 | Datos, split train/val/test, asserts de tamaños y de que val ⊂ train |
| 2 | Arquitecturas: entradas, salidas, capas, neuronas y nº de parámetros (justificación) |
| 3 | Clasificación A: comparación de las 3 arquitecturas (val_loss 5 semillas) + curvas por época |
| 3.1 | Clasificación A: modelo elegido en test (semilla 42 + media ± std 5 semillas), reportes, matriz |
| 4 / 4.1 | Clasificación B: lo mismo |
| 5 | Regresión: comparación de arquitecturas + curvas |
| 5.1 | Regresión: modelo elegido en test (kg), baseline, predicho vs real, residuos, media ± std |
| 6 | Escribir filas en `data/resultados_entrega2.csv` |

**Criterio de aceptación**: ninguna fila del test se usa en validación (assert); target de
regresión invertido a kg; curvas por época de las 3 arquitecturas en las 3 tareas;
semilla fija + media ± std; params y épocas efectivas reportados; BITÁCORA.

### T5. Etapa 09 — Comparación final (`etapas/e09_comparacion_final.py`) — *Opus*

| Paso | Contenido |
|---|---|
| 1 | Carga `resultados_entrega1.csv` (congelado) + `resultados_entrega2.csv`; comprueba que `huella_test` y `n_test=418` coinciden en todas las filas |
| 2 | Tablas: clasificación A (6 modelos), clasificación B (6), regresión (5 + baseline) |
| 3 | Ranking por macro-F1 (clf) y por RMSE (reg) + gráficas de barras comparativas (nuevas) |
| 4 | Métricas adicionales: tiempo, nº parámetros, nº SV, log-loss (vacías donde no aplican, explicado) |
| 5 | Conclusiones generadas desde los datos (f-strings, nada escrito a mano): mejor modelo por tarea y por qué; advertencia de que con ~2087 filas tabulares un RF puede igualar o superar a la red; K-Means solo como nota |

**Criterio de aceptación**: ninguna cifra escrita a mano; el ganador sale de los datos, sin
forzarlo; las discrepancias con la E1 quedan señaladas; BITÁCORA.

### T7. Verificación final — *Opus*

- `main.py` (menú 6-9) y `run_all.py` actualizados. `run_all.py` sin argumentos ejecuta las 9
  etapas; `--etapas 06 07 08 09` permite correr solo las nuevas.
- Ejecutar `python run_all.py --limpiar-cache` completo sin errores.
- Comprobar que `resultados_texto/` de las etapas 01-05 **no cambia**: así se reproduce la E1 en
  este entorno. Si solo cambian bytes de PNG (versión de matplotlib), se revierten.
- `pytest` en verde. Se añade `tests/test_entrega2.py`: parámetros de los pasos resolubles,
  esquema de los CSV y tamaños del split. **No se ejecuta ni se modifica la app.**
- Contrastar cada número de este plan y de la BITÁCORA con `resultados_texto/0[6-9]_*`.
- Resumen final con hallazgos y discrepancias.

---

## 3. Entregables (versión adaptada a la estructura real)

- `etapas/e06_naive_bayes.py`, `e07_svm.py`, `e08_redes_neuronales.py`, `e09_comparacion_final.py`,
  ejecutadas, con sus salidas en `resultados_texto/0[6-9]_*/` y `figures/0[6-9]_*/`.
- `data/resultados_entrega1.csv` (congelado) y `data/resultados_entrega2.csv`.
- `src/resultados.py` (lectura/escritura de los CSV de resultados).
- `main.py` y `run_all.py` ampliados; `requirements.txt` con TF/Keras fijados; `tests/test_entrega2.py`.
- Este plan con su BITÁCORA.

**Fuera de alcance**: la app de Streamlit (`app.py`, `webapp/`, `estudio_etapas.html`), guía de
estudio y `decisiones.md`.

---

## 4. BITÁCORA (números reales, se rellena al cerrar cada tarea)

| Tarea | Fecha | Modelo recomendado | Modelo usado | Estado | Commit |
|---|---|---|---|---|---|
| T0 | 2026-10-12 | Opus (modo plan) | Opus | Terminada (OK recibido) | `500aab5` |
| T1 | 2026-10-12 | Sonnet | Opus (no se cambió con /model) | Terminada (OK recibido) | `2267801` |
| T2 | 2026-10-12 | Sonnet | Opus (no se cambió con /model) | Terminada (OK recibido) | `c56fe7d` |
| T3 | 2026-10-12 | Sonnet | Opus (no se cambió con /model) | Terminada (OK recibido) | `f7a96a0` |
| T4 | 2026-10-12 | Opus | Opus | Terminada (OK recibido) | `d9c6c32` |
| T5 | 2026-10-12 | Opus | Opus | Terminada (OK recibido) | `60baf41` |
| T7 | 2026-10-12 | Opus | Opus | Terminada (OK recibido) | `4814738` |
| T8. Quitar la Entrega 1 del proyecto (conservar sólo sus resultados) | 2026-10-12 | Opus | Opus | Terminada (OK recibido) | `d1c4a0d` |
| T9. HTML visual de la Entrega 2 (`estudio_entrega2.html`) | 2026-10-12 | Sonnet | Opus (no se cambió con /model) | Terminada, espera verificación | (este commit) |

### T0 — Inspección

- Estructura real: versión plana (`etapas/`, `src/pasos.py`, `resultados_texto/`, `run_all.py`).
  Sin notebooks ni `docs/`.
- Datos: 2111 filas crudas → 2087 sin duplicados.
- Discrepancias con las referencias: Árbol A F1 0.9362 (no 0.938), KNN A F1 0.8075 (no 0.814),
  KNN B F1 0.7567 (no 0.764). Detalle en §0.2.

### T1 — Resultados de la Entrega 1 congelados

**Entorno**: `.venv` con Python 3.13.16, scikit-learn 1.9.1, pandas 3.0.6, numpy 2.5.4,
matplotlib 3.11.2. Se instaló `requirements.txt` sin PyQt5, que solo hace falta para abrir
ventanas.

**Re-ejecución de verificación** (`e02 --paso 3 3.6 4 4.1` y `e03 --paso 2`, sin `--guardar`;
`git status` limpio después). Los 10 modelos reproducen **exactamente** las cifras de
`resultados_texto/`. Los mejores hiperparámetros salieron por consola; no están en
`resultados_texto/` y por eso se anotan aquí y no en el CSV:

| Tarea | Árbol | Random Forest | KNN |
|---|---|---|---|
| clasificación A | max_depth=None, min_samples_leaf=1 | n_estimators=400, max_depth=None, min_samples_leaf=1 | k=1, uniform |
| clasificación B | max_depth=None, min_samples_leaf=1 | n_estimators=200, max_depth=None, min_samples_leaf=1 | k=3, distance |
| regresión | max_depth=12, min_samples_leaf=10 | n_estimators=400, max_depth=None, min_samples_leaf=1 | k=3, distance |

**`data/resultados_entrega1.csv`**: 14 filas, generadas con `python -m src.resultados --congelar-entrega1`.
El script lee las tablas de `resultados_texto/`, no tiene ningún número escrito a mano y no
sobrescribe el CSV si ya existe (sin `--forzar`).

| tarea | modelo | accuracy | macro-F1 | cv | MAE | RMSE | R² | R²adj |
|---|---|---|---|---|---|---|---|---|
| clasificacion_A | Árbol de decisión | 0.9378 | 0.9362 | 0.9230 | | | | |
| clasificacion_A | Random Forest | 0.9569 | 0.9558 | 0.9437 | | | | |
| clasificacion_A | KNN | 0.8206 | 0.8075 | 0.8152 | | | | |
| clasificacion_B | Árbol de decisión | 0.7464 | 0.7401 | 0.7440 | | | | |
| clasificacion_B | Random Forest | 0.8565 | 0.8528 | 0.8506 | | | | |
| clasificacion_B | KNN | 0.7679 | 0.7567 | 0.7640 | | | | |
| regresion | Baseline (media) | | | — | 21.7064 | 25.9126 | −0.0001 | −0.0479 |
| regresion | Árbol de regresión | | | 12.4728 | 6.5810 | 10.4655 | 0.8369 | 0.8291 |
| regresion | Random Forest | | | 9.1156 | 4.8386 | 8.2468 | 0.8987 | 0.8939 |
| regresion | KNN | | | 10.3709 | 5.0484 | 9.9085 | 0.8538 | 0.8468 |

`clustering_nota` (silueta / ARI / NMI / pureza), marcado como **no comparable**:

| Configuración | silueta | ARI | NMI | pureza |
|---|---|---|---|---|
| Sin clase (20 features) | 0.1315 | 0.0883 | 0.1575 | 0.3057 |
| Solo numéricas (8) | 0.1647 | 0.1564 | 0.2618 | 0.3824 |
| Con clase ordinal (fuga) | 0.1555 | 0.2520 | 0.3584 | 0.4255 |
| Con clase one-hot (fuga) | 0.2279 | 0.9989 | 0.9984 | 0.9995 |

**Cambios al esquema del plan**: se añadieron las columnas `silueta, ARI, NMI, pureza, nota`,
que solo usan las filas de clustering. Tiempo, log-loss, nº de parámetros, nº de vectores de
soporte y épocas quedan vacíos en la Entrega 1 porque no se re-entrena.

**Hallazgo — dos tests distintos de 418 filas.** `huella_test` vale `8d9af7b40ee9` en
clasificación y `021799f95367` en regresión. Es lo esperado: la Entrega 1 estratifica por
clase en un caso y por deciles de Weight en el otro, así que son particiones distintas. Lo que
garantiza la comparación es que **dentro de cada tarea** todos los modelos (E1 y E2) usen el
mismo test. La etapa 09 lo comprueba por tarea.

### T2 — Etapa 06, Naive Bayes

Ejecución: `python etapas/e06_naive_bayes.py --todo --guardar --sin-ventanas`. Sin errores,
10 pasos, salidas en `resultados_texto/06_naive_bayes/` y `figures/06_naive_bayes/`. El split
es el de la etapa 02 (1669/418, huella `8d9af7b40ee9`, la misma que en la E1).

**Resultados (test = 418)**

| Variante | Modelo | mejores params (CV) | cv macro-F1 | test acc | test macro-F1 | log-loss | confianza media |
|---|---|---|---|---|---|---|---|
| A | GaussianNB | var_smoothing=1e-4 | 0.6424 | 0.6699 | 0.6567 | 0.8145 | 0.7813 |
| A | **CategoricalNB** ← elegido | n_bins=25, alpha=0.01 | **0.7810** | 0.8086 | **0.8030** | 0.7642 | 0.8864 |
| B | GaussianNB | var_smoothing=1e-3 | 0.4939 | 0.5813 | 0.5333 | 1.5729 | 0.7779 |
| B | **CategoricalNB** ← elegido | n_bins=20, alpha=0.1 | **0.6523** | 0.6914 | **0.6624** | 1.1014 | 0.7733 |

- Tiempo de `refit` del modelo elegido: unos 0.02 s, con ruido de milisegundos entre
  ejecuciones; GaussianNB unos 0.005 s. El tiempo exacto de cada ejecución queda en
  `resultados_texto/06_naive_bayes/5.md`.
- En el CSV entra, por variante, **solo el NB elegido por CV**: CategoricalNB en A y en B.
- Referencia frente a la E1 (test macro-F1; la interpretación es de T5):
  - A: NB 0.8030 · Árbol 0.9362 · RF 0.9558 · KNN 0.8075.
  - B: NB 0.6624 · Árbol 0.7401 · RF 0.8528 · KNN 0.7567.
  - NB es el último en ambas variantes; en A queda a 0.0045 de KNN.

**Hallazgos que obligaron a corregir el código o las notas (reportados tal cual):**

1. **"NB no necesita escalado" era inexacto tal como lo había escrito.** Con el mismo
   `var_smoothing` (1e-9), GaussianNB con y sin `StandardScaler` coincide solo en el **89.5 %**
   de las predicciones del test.
   - Causa: hay **18 pares (clase, atributo) con varianza 0** en el train (Gender,
     family_history, FCVC, NCP, CAEC, SCC, MTRANS_Bike/Motorbike/Walking constantes dentro de
     alguna clase). En esos pares, la varianza que usa GaussianNB es solo el suavizado
     `var_smoothing × máx. varianza`. Esa máxima es 685.2 (Weight) sin escalar y 1 escalado.
   - Con la misma varianza suavizada expresada en las nuevas unidades coinciden el **100.0 %**.
   - Escalar empeora la CV: A 0.6424 → 0.5061; B 0.4939 → 0.4377.
   - Conclusión: no se escala. La nota del paso 1 lo explica con estos números calculados.
2. **El óptimo de `n_bins` caía en el borde de la grilla** con {3, 5, 7, 10} y otra vez con
   tope 20. La grilla final, `N_BINS = [3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 60]`, muestra una
   meseta entre 20 y 40 tramos y una caída a partir de 50 (figura 3.2).
3. **CALC = "Always" aparece una sola vez en el train.** Por eso `CategoricalNB(min_categories=4)`:
   sin eso, cualquier fold sin esa fila fallaría.
4. **Sobreconfianza comprobada, no supuesta.** En los 4 modelos la confianza media supera a la
   accuracy en test: +0.111 y +0.078 en A, +0.197 y +0.082 en B (GaussianNB y CategoricalNB).
5. **Independencia condicional** (paso 1.1): corr(Weight, Height | clase) entre 0.827
   (Normal_Weight) y 0.980 (Overweight_Level_I). |corr| media dentro de las clases:
   Height–Weight 0.930; los siguientes pares (Height–FAF 0.332, Weight–FAF 0.308,
   Age–TUE 0.303) están muy por debajo.
   - Los rangos de IMC por clase **se solapan** (p. ej. Overweight_Level_I llega a 28.26 y
     Overweight_Level_II empieza en 25.71). La clase no es un corte exacto de IMC en todas las
     filas; es coherente con lo que ya vio la etapa 02 en el paso 8.

**Código nuevo**
- `etapas/e06_naive_bayes.py`.
- `src/resultados.py`: nueva función `guardar_entrega2(filas, etapa)`. Reemplaza las filas
  previas de la misma etapa y solo escribe con `--guardar` o `run_all.py`, la misma convención
  que las tablas y figuras.

### T3 — Etapa 07, SVM (SVC y SVR)

Ejecución: `python etapas/e07_svm.py --todo --guardar --sin-ventanas`. Sin errores, 13 pasos,
unos 8 minutos sin caché. Salidas en `resultados_texto/07_svm/` y `figures/07_svm/`. Los
splits son los de E1: clasificación con huella `8d9af7b40ee9` y regresión con `021799f95367`.
Re-ejecutada desde cero tres veces con las grillas finales: los mismos hiperparámetros y las
mismas métricas en todas.

**Clasificación A (todas las features)**

| kernel | mejores params (CV) | cv macro-F1 | test acc | test macro-F1 | nº SV |
|---|---|---|---|---|---|
| linear | C=1000 | 0.9599 | 0.9689 | 0.9675 | 239 |
| poly | C=100, degree=3, gamma=0.01, coef0=1 | 0.9508 | 0.9569 | 0.9546 | 546 |
| **rbf** ← elegido | C=1e5, gamma=1e-4 | **0.9650** | 0.9689 | **0.9678** | 312 |

**Clasificación B (sin Weight/Height)**

| kernel | mejores params (CV) | cv macro-F1 | test acc | test macro-F1 | nº SV |
|---|---|---|---|---|---|
| linear | C=100 | 0.6235 | 0.6411 | 0.6091 | 1182 |
| poly | C=1, degree=5, gamma=scale, coef0=1 | 0.7821 | 0.7799 | 0.7705 | 934 |
| **rbf** ← elegido | C=10, gamma=0.1 | **0.7964** | 0.7871 | **0.7830** | 998 |

**Regresión de Weight (test, kg; R²adj con p = 19)**

| kernel | mejores params (CV) | cv RMSE | MAE | RMSE | R² | R²adj | nº SV |
|---|---|---|---|---|---|---|---|
| Baseline (media) | — | — | 21.7064 | 25.9126 | −0.0001 | −0.0479 | — |
| linear | C=10, epsilon=0.5 | 17.9842 | 14.1639 | 17.3358 | 0.5524 | 0.5310 | 752 |
| poly | C=1, degree=3, gamma=scale, epsilon=0.3 | 15.9654 | 11.7900 | 18.3846 | 0.4966 | 0.4725 | 874 |
| **rbf** ← elegido | C=1, gamma=0.1, epsilon=0.05 | **9.4351** | 6.2617 | **10.0951** | 0.8482 | 0.8410 | 1250 |

`epsilon` de SVR está en desviaciones típicas del peso, porque el target se estandariza. Los
tiempos de `refit` del modelo elegido son unos 0.1 s; el lineal tarda más (≈1 s en A, ≈4 s en B
y ≈1 s en regresión). Los tiempos varían entre ejecuciones; los exactos de la ejecución final
están en `resultados_texto/07_svm/`.

**Vectores de soporte (paso 3.4)**

| Modelo elegido | Vectores de soporte | % del train |
|---|---|---|
| SVC rbf A | 312 de 1669 | 18.7 % |
| SVC rbf B | 998 de 1669 | 59.8 % (Normal_Weight: 93.8 % de su clase) |
| SVR rbf | 1250 de 1669 | 74.9 % (tubo estrecho, epsilon = 0.05) |

**Hallazgos (reportados tal cual)**

1. **En A el kernel lineal compite**: CV 0.9599, frente a 0.9650 de rbf (hueco +0.0051). En B
   el hueco es +0.1729. En test, linear y rbf empatan en accuracy (0.9689) y casi en macro-F1
   (0.9675 frente a 0.9678).
   - Paso 3.2: un SVC con **solo Weight y Height** logra CV 0.9484 (linear en kg/m), 0.9505
     (linear en logaritmos) y **0.9650 (rbf), igual que el SVC rbf con las 20 features**. En A,
     casi todo el rendimiento sale de la fuga del IMC.
   - El rbf elegido en A (C=1e5, gamma=1e-4) está en el régimen "gamma pequeño / C grande",
     donde se comporta casi como un clasificador lineal.
2. **El SVM supera en A al mejor modelo de la E1**: SVC rbf 0.9678 frente a RF 0.9558 de
   macro-F1 en test. En B y en regresión **no**: B 0.7830 frente a RF 0.8528, y regresión
   RMSE 10.10 frente a RF 8.25 y KNN 9.91. La interpretación es de T5.
3. **SVR poly: mejor CV que linear pero peor test**, por extrapolación (tabla del paso 6.1):
   - SVR poly predice entre **−39.95 kg y 254.26 kg**; el rango del train es [39.4, 173.0] kg.
   - 18 predicciones caen fuera de ese rango y el residuo máximo es 142.26 kg.
   - Como comparación, linear tiene 12 fuera de rango (residuo máximo 47.14) y rbf solo 1
     (residuo máximo 49.45).
4. **Grillas: tres pasadas hasta que ningún óptimo quedara en el borde** (anotado en el código):
   - 1.ª pasada: bordes en linear-B C=100, poly-A C=100, poly-B degree=3 y gamma=0.1,
     rbf-A C=1000 y gamma=0.001, y en SVR (epsilon=0.2 en linear y poly, C=0.1 en poly).
   - 2.ª pasada: quedaban linear-A C=1000, rbf-A C=1e5 / gamma=1e-4, poly-B degree=4 y
     SVR-linear epsilon=0.5. Se probó **fuera de la grilla** y todo era peor:
     - linear-A C=3000: 0.9593; C=10000: 0.9547.
     - rbf-A con (C, gamma) = (1e6, 1e-5), (1e6, 1e-4), (1e5, 1e-5) y (1e7, 1e-5): ≤ 0.9583.
     - SVR-linear epsilon=0.75: 18.04; epsilon=1.0: 18.30.
   - Por esas pruebas se añadieron rbf C=1e6 y gamma=1e-5, poly degree=5 y SVR epsilon=1.0, y
     ya son interiores.
   - **Reproducible desde T7**: el paso 2.1 de la etapa 07 recalcula estas pruebas. Muestra la
     superficie C × gamma del rbf de A, la curva de epsilon del SVR lineal y los valores fuera
     de la grilla. Los resultados están en `resultados_texto/07_svm/2.1.md`.
   - **Excepción, linear C ≤ 1000**: C=10000 empeora A y en B un solo ajuste pasaba de 10
     minutos. Queda documentado que el óptimo C=1000 de linear-A está en el borde de la grilla
     pero es un máximo verificado.
   - **poly-B degree=5 sigue en el borde**: degree 6 y 7 dan 0.7818 y 0.7829 (meseta ±0.001) y
     rbf gana por 0.014, así que no cambia la elección.
   - **Recortes en SVR** por coste: poly C=100 (unos 150 s por ajuste) y linear C=1000 (unos
     150 s), ambos lejos del óptimo.
5. **Frontera PCA (paso 3.3)**: PC1 + PC2 explican solo el 25.6 % de la varianza. La accuracy
   en 2D es 0.404 (linear), 0.445 (poly) y 0.421 (rbf). Es solo visualización y así lo dice la
   nota.

**Código**: `etapas/e07_svm.py`, con `make_models` / `fit_evaluate` / `variante` para
clasificación y `make_models_reg` / `_ajustar_reg` / `modelos_reg` para regresión, con los
mismos nombres que `e02` y `e03`.

### T4 — Etapa 08, Redes neuronales (Keras)

**Dependencias**: `tensorflow==2.21.0` y `keras==3.15.1` (backend TensorFlow, solo CPU),
fijadas en `requirements.txt`. Hay ruedas `win_amd64` para Python 3.10-3.13.

Ejecución: `python etapas/e08_redes_neuronales.py --todo --guardar --sin-ventanas`. Sin
errores, 45 entrenamientos (3 tareas × 3 arquitecturas × 5 semillas), unos 12 minutos.

**Comprobaciones (paso 1 y paso 2)**
- Sub-train 1418 / validación 251, ambas sacadas del train. Test de 418, con las huellas de la
  E1: `8d9af7b40ee9` en clasificación y `021799f95367` en regresión. Hay `assert` de que
  validación ⊂ train, que sub-train y validación no se solapan y que ninguna de las dos toca
  el test.
- Escaladores ajustados solo con el sub-train: las medias del X del test escalado están en
  [−0.08, 0.11], no son 0. Target de regresión: media 86.98 kg, desviación 26.32 kg.
- Reproducibilidad: la semilla 42 dos veces da pesos idénticos (val_loss 1.235814 en ambas,
  5 épocas); la semilla 43 da 1.274095. Entre procesos distintos, la semilla 42 reprodujo
  val_loss 0.2380 (A, pequeña) y 0.1086 (regresión, grande).
- `EarlyStopping` de Keras 3.15 restaura los mejores pesos al terminar siempre, también si
  se agotan las épocas (revisado en el código fuente).

**Parámetros por arquitectura**

| Tarea | pequeña [32] | media [64, 32] | grande [128, 64, 32] |
|---|---|---|---|
| Clasificación A (20 entradas) | 903 | 3655 | 13255 |
| Clasificación B (18 entradas) | 839 | 3527 | 12999 |
| Regresión (19 entradas) | 673 | 3393 | 12929 |

**Elección por val_loss media de 5 semillas** (validación = 15 % del train; el test no
interviene):

| Tarea | pequeña | media | grande | Elegida |
|---|---|---|---|---|
| A (entropía cruzada) | **0.2370 ± 0.0395** | 0.2526 ± 0.0634 | 0.2750 ± 0.0518 | pequeña |
| B (entropía cruzada) | 0.9395 ± 0.0142 | 0.8697 ± 0.0239 | **0.8493 ± 0.0168** | grande |
| Regresión (MSE, y estandarizado) | 0.1493 ± 0.0176 | 0.1216 ± 0.0088 | **0.1142 ± 0.0064** | grande |

Mejor época media / épocas entrenadas medias:

| Tarea | pequeña | media | grande |
|---|---|---|---|
| A | 163 / 193 | 42 / 72 | 22 / 52 |
| B | 62.8 / 92.8 | 25.2 / 55.2 | 12.2 / 42.2 |
| Regresión | 126.8 / 156.8 | 30.4 / 60.4 | 19.8 / 49.8 |

**Resultados en test** (semilla 42 = fila del CSV; media ± desviación de las 5 semillas)

| Tarea | Red | Métrica principal, semilla 42 | Media ± desv. (5 semillas) | Otras (semilla 42) |
|---|---|---|---|---|
| A | pequeña [32], 903 parámetros, mejor época 157 de 187 | macro-F1 0.9227 | 0.9261 ± 0.0077 | acc 0.9258 (media 0.9292 ± 0.0071); log-loss 0.3097 |
| B | grande, 12999 parámetros, mejor época 9 de 39 | macro-F1 0.7327 | 0.7489 ± 0.0143 | acc 0.7464 (media 0.7589 ± 0.0095); log-loss 0.8034 |
| Regresión | grande, 12929 parámetros, mejor época 10 de 40 | RMSE 11.6764 kg | 10.4756 ± 0.6898 | MAE 7.3545 (media 6.7590 ± 0.3672); R² 0.7969 (media 0.8360 ± 0.0224); R²adj 0.7872 |

- Baseline de regresión, la media del sub-train: RMSE 25.9134 y R² −0.0002. Es casi igual al
  de E1 (25.9126 y −0.0001), que usaba la media de los 1669 del train completo.
- La red de regresión no hace ninguna predicción fuera del rango del sub-train: van de 43.35
  a 165.99 kg.
- Confianza media frente a accuracy, semilla 42: A 0.944 frente a 0.926 (+0.018, algo
  sobreconfiada); B 0.743 frente a 0.746 (−0.004, bien calibrada).

**Hallazgos (reportados tal cual, sin ajustar nada)**

1. **Sobreajuste rápido de las redes media y grande**, visible en las curvas de los pasos 3,
   4 y 5: la pérdida de train baja casi a 0 mientras la de validación toca su mínimo pronto
   y vuelve a subir.
   - B grande: mínimo en la época 9 con la semilla 42.
   - Regresión grande: mínimo en la época 10.
   - El early stopping recupera esos pesos, pero no hay regularización (dropout/L2), que el
     plan dejó fuera a propósito. **Decisión del usuario: no añadir regularización.** Se
     mantiene el protocolo aprobado; el sobreajuste se discute en las conclusiones de T5.
2. **La semilla 42 es la peor de las 5 en B y en regresión.**
   - B, macro-F1: 0.7327, frente a 0.7449-0.7715 de las otras semillas.
   - Regresión, RMSE: 11.68, frente a 9.97-10.41.
   - En A queda cuarta: 0.9227, frente a 0.9171-0.9380.
   - Si en T5 se compara solo la fila de la semilla 42, la red sale penalizada por azar; por
     eso la media ± desviación va en la columna `nota` del CSV.
3. **En A, la elección pequeña frente a media está dentro del ruido**: la media está a
   +0.0156, que son 0.4 desviaciones. La validación de 251 filas es ruidosa: con la semilla 42
   la media logra 0.1503, frente a 0.2380 de la pequeña.
4. **Referencia frente a la E1 y otros modelos de E2** (macro-F1 / RMSE de test; la
   interpretación es de T5). La red no supera al RF en ninguna tarea:
   - A: red 0.9227 (media 0.9261) frente a RF 0.9558 y SVM 0.9678.
   - B: red 0.7327 (media 0.7489) frente a RF 0.8528, SVM 0.7830, KNN 0.7567 y Árbol 0.7401.
   - Regresión: red 11.68 (media 10.48) frente a RF 8.25, KNN 9.91, SVR 10.10 y Árbol 10.47.

**Código**: `etapas/e08_redes_neuronales.py`, con `datos(tarea)`, `construir`, `entrenar`,
`_comparar` / `comparacion` (en caché), `resumen_arquitecturas`, `elegida`, `_evaluar` /
`evaluacion`. En caché se guardan historiales, pesos y métricas, no el objeto Keras; el
modelo se reconstruye con `set_weights`.

### T5 — Etapa 09, Comparación final

Ejecución: `python etapas/e09_comparacion_final.py --todo --guardar --sin-ventanas`. Sin
errores, 6 pasos, menos de 3 s, porque solo lee `data/resultados_entrega{1,2}.csv`. Salidas en
`resultados_texto/09_comparacion_final/` y `figures/09_comparacion_final/`.

**Verificaciones (paso 1)**
- Por tarea hay una sola huella de test y n_test = 418 en todas las filas: A y B
  `8d9af7b40ee9`, regresión `021799f95367`.
- Recuento: 6 / 6 / 5 modelos (más el baseline), una fila por familia.
- La E1 congelada coincide con las tablas actuales de `resultados_texto/` de las etapas 02,
  03 y 05.

**Discrepancias con el encargo (paso 1.1)**: coinciden 25 de 28 valores. Las 3 discrepancias
son las ya detectadas en T0:
- Árbol A, F1: 0.938 en el encargo, 0.9362 ejecutado. 0.938 es la accuracy.
- KNN A, F1: 0.814 frente a 0.8075.
- KNN B, F1: 0.764 frente a 0.7567. 0.764 es el F1 de CV.

**Tablas finales (test de 418; la red, con semilla 42 y media ± desviación de 5 semillas)**

Clasificación A, por macro-F1:

| Puesto | Modelo | Entrega | Accuracy | macro-F1 | Errores en test | CV |
|---|---|---|---|---|---|---|
| 1 | SVM (SVC rbf) | E2 | 0.9689 | **0.9678** | 13 | 0.9650 |
| 2 | Random Forest | E1 | 0.9569 | 0.9558 | 18 | 0.9437 |
| 3 | Árbol de decisión | E1 | 0.9378 | 0.9362 | 26 | 0.9230 |
| 4 | Red neuronal [32] | E2 | 0.9258 | 0.9227 (media 0.9261 ± 0.0077) | 31 | val_loss 0.2380 |
| 5 | KNN | E1 | 0.8206 | 0.8075 | 75 | 0.8152 |
| 6 | Naive Bayes (CategoricalNB) | E2 | 0.8086 | 0.8030 | 80 | 0.7810 |

Clasificación B, por macro-F1:

| Puesto | Modelo | Entrega | Accuracy | macro-F1 | Errores en test | CV |
|---|---|---|---|---|---|---|
| 1 | Random Forest | E1 | 0.8565 | **0.8528** | 60 | 0.8506 |
| 2 | SVM (SVC rbf) | E2 | 0.7871 | 0.7830 | 89 | 0.7964 |
| 3 | KNN | E1 | 0.7679 | 0.7567 | 97 | 0.7640 |
| 4 | Árbol de decisión | E1 | 0.7464 | 0.7401 | 106 | 0.7440 |
| 5 | Red neuronal [128, 64, 32] | E2 | 0.7464 | 0.7327 (media 0.7489 ± 0.0143) | 106 | val_loss 0.8455 |
| 6 | Naive Bayes (CategoricalNB) | E2 | 0.6914 | 0.6624 | 129 | 0.6523 |

Regresión, por RMSE (kg):

| Puesto | Modelo | Entrega | MAE | RMSE | R² | R²adj | CV |
|---|---|---|---|---|---|---|---|
| 1 | Random Forest | E1 | 4.8386 | **8.2468** | 0.8987 | 0.8939 | 9.1156 |
| 2 | KNN | E1 | 5.0484 | 9.9085 | 0.8538 | 0.8468 | 10.3709 |
| 3 | SVM (SVR rbf) | E2 | 6.2617 | 10.0951 | 0.8482 | 0.8410 | 9.4351 |
| 4 | Árbol de regresión | E1 | 6.5810 | 10.4655 | 0.8369 | 0.8291 | 12.4728 |
| 5 | Red neuronal [128, 64, 32] | E2 | 7.3545 | 11.6764 (media 10.4756 ± 0.6898) | 0.7969 | 0.7872 | val_loss 0.1086 |
| — | Baseline (media) | E1 | 21.7064 | 25.9126 | −0.0001 | −0.0479 | — |

**Puesto medio por familia (paso 3)**: Random Forest 1.33 · SVM 2.00 · KNN 3.33 · Árbol 3.67
· Red neuronal 4.67 · Naive Bayes 6.00 (solo clasificación).

**Conclusiones (paso 5, generadas con f-strings sobre los CSV)**
- A: gana el SVM, por 5 instancias sobre el RF. Hay fuga: un rbf solo con Weight y Height
  logra la misma CV, 0.9650.
- B: gana el RF por 29 instancias sobre el SVM.
- Regresión: gana el RF con 8.25 kg.
- La red no supera al RF en ninguna tarea, ni con la semilla 42 ni con la media de 5
  semillas.
- Naive Bayes queda último en A y en B, aunque es el más barato (0.018 s por ajuste).
- Los ganadores por CV coinciden con los de test en las tres tareas (sin contar la red).
- El 76.5 % de las filas tiene respuestas no enteras, la marca de SMOTE; está calculado con
  `e02.filas_reales`.
- K-Means: solo como nota (ARI 0.1564).

**Cambios en código compartido**: `src/resultados.leer_tabla_md` ahora deja como texto las
columnas no numéricas; antes fallaba con la columna `mejores_params`. Como las etapas 06-08
importan `src/resultados.py`, su caché se recalculará en la próxima ejecución, la de T7. Sus
números no deberían cambiar, y T7 lo comprobará.

### T7 — Verificación final

**Cambios**
- `main.py`: el menú carga las 9 etapas (opciones 1-9).
- `run_all.py`: ejecuta las 9 etapas, con la opción `--etapas 06 07 08 09` para correr solo
  algunas; valida los números de etapa.
- `tests/test_entrega2.py`: 11 pruebas que no entrenan nada.
  - Numeración y pasos únicos; parámetros declarados que apuntan a su definición.
  - `run_all` y `main` conocen las 9 etapas.
  - Esquema común de los CSV; la E1 congelada coincide con `resultados_texto/`.
  - Una sola huella de 418 por tarea y una fila por familia y tarea.
  - `leer_tabla_md` con columnas de texto.
  - La validación de la red sale del train (1418 + 251 = train; test disjunto).
- `etapas/e07_svm.py`: nuevo paso 2.1, "Bordes de la grilla". Hace reproducibles las pruebas
  fuera de la grilla que en T3 se habían corrido a mano.

**`python run_all.py --limpiar-cache` completo**: exit 0, las 9 etapas en 1574 s.

| Etapa | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 | 09 |
|---|---|---|---|---|---|---|---|---|---|
| Tiempo (s) | 4 | 81 | 252 | 27 | 5 | 14 | 507 | 682 | 2 |

Después, con el paso 2.1 añadido, `run_all.py --etapas 07 09`: exit 0, la 07 en 525 s.

**Reproducibilidad desde cero (diff numérico archivo a archivo contra lo versionado)**
- **Entrega 2 (06-09)**: idéntica salvo las columnas de **tiempo**, incluidos los 45
  entrenamientos de redes. Por ejemplo, el RMSE de la red de regresión es
  `11.676357233131657` en las dos ejecuciones. En `data/resultados_entrega2.csv` cambian
  exactamente las 8 celdas de tiempo. Se versionan las salidas de esta ejecución final.
- **Entrega 1 (01-05)**: las métricas congeladas (tablas de los pasos 3 y 4 de la etapa 02,
  paso 2 de la 03 y clustering de la 05) son **idénticas**. Hay diferencias pequeñas en
  análisis secundarios que **no entran en la comparación**:
  - Etapa 02, paso 8 (filas reales, CV 3×10): macro-F1 0.194203 → 0.194181 y
    0.395685 → 0.395671.
  - Etapa 03, paso 9 (filas reales): RMSE 17.4617 → 17.4598 y 12.8886 → 12.8864;
    R² −0.0631 → −0.0629.
  - Tiempos de los pasos 3.7 y 2.2.
  - `04_clustering/4.4.md` gana una sección de datos que la versión versionada no tenía.
  - Los CSV de puntos PCA solo difieren en la representación de los decimales.

  La causa probable son versiones de librería distintas a las del entorno donde se generó la
  E1. Aquí: scikit-learn 1.9.1, pandas 3.0.6, numpy 2.5.4, scipy 1.18.1. **Como la E1 no se
  modifica, sus salidas se restauraron a la versión versionada.**

**Tests**: `pytest tests/test_entrega2.py tests/test_cli.py tests/test_cache.py
tests/test_sin_postpoda.py` → **27 passed**.
- No se ejecutaron `test_app_smoke`, `test_modelos`, `test_clustering` ni `test_navegacion`:
  importan o ejecutan la app (`webapp/`), que está fuera de alcance.
- `estudio_etapas.html` y `webapp/` no se tocaron.

**Números de esta bitácora frente a las salidas**: un script compara cada número decimal de
las secciones T1-T7 con `resultados_texto/**/*.md` y con los logs de las ejecuciones
completas. El resultado está al final de esta sección.

**Fallo encontrado y corregido durante la verificación (commit `5acf985`)**:
`resultados.leer()` no releía los CSV de forma exacta. Al reescribir
`data/resultados_entrega2.csv`, los valores de otras etapas cambiaban en el último dígito
(p. ej. 11.676357233131657 → …656, ~2·10⁻¹⁵). Se corrigió con
`pd.read_csv(..., float_precision="round_trip")`: ahora la relectura es exacta, comprobado
escribiendo y releyendo el CSV completo.

**Ejecución final con el código definitivo**: `python run_all.py --etapas 06 07 08 09`, exit 0.

| Etapa | 06 | 07 | 08 | 09 |
|---|---|---|---|---|
| Tiempo (s) | 16 | 524 | 682 | 2 |

Las etapas 01-05 no dependen de `src/resultados.py` y sus salidas son las versionadas de la E1.
Comparando **celda a celda** todas las tablas CSV de `resultados_texto/06-09` y
`data/resultados_entrega2.csv` con la versión anterior, **solo cambian columnas de tiempo**:
`tiempo_fit_s`, `tiempo_entrenamiento_s`, `tiempo_s`, `tiempo medio (s)` y
`tiempo de entrenamiento (s)`. El RMSE de la red de regresión vuelve a ser exactamente
`11.676357233131657`. Tras la ejecución, `pytest` (los mismos 4 archivos) → **27 passed**.

**Resultado del contraste de números** (bitácora frente a `resultados_texto/` final y el log
de la ejecución final): todos los números decimales de T1-T7 aparecen en las salidas excepto
estos, que están justificados:
- `3.13` y `3.11` son versiones de Python y matplotlib, no resultados.
- `11.676357233131657` está en `data/resultados_entrega2.csv`, que el script no lee.
- `0.194181`, `0.395671`, `17.4598`, `12.8864` y `−0.0629` son de la regeneración de la E1
  que se descartó al restaurarla; están en el log de esa ejecución (`run_all --limpiar-cache`),
  no en las salidas versionadas.

Ningún número de la bitácora es una estimación. Los tiempos citados en las secciones T2-T4
son aproximados y varían entre ejecuciones; los exactos de la última ejecución están en
`resultados_texto/`.

### T8 — Quitar la Entrega 1 del proyecto

Decisión del usuario: este proyecto contiene solo la Entrega 2. Lo de la Entrega 1 queda en
su propio proyecto; aquí solo se conservan sus resultados congelados
(`data/resultados_entrega1.csv`).

**Eliminado (`git rm`; sigue en el historial)**
- `etapas/e01`-`e05`, más `figures/` y `resultados_texto/` de 01-05 y `tablas_sueltas`
  (K-Means).
- La app de Streamlit: `app.py`, `webapp/`, `estudio_etapas.html` y `study/`.
- Sus tests: `test_app_smoke`, `test_modelos`, `test_clustering`, `test_navegacion` y
  `test_sin_postpoda`.
- De `requirements.txt`: `streamlit`.
- Helpers que solo usaba la E1:
  - En `src/evaluation.py`: `tree_summary`, `plot_k_curve`, `plot_n_estimators_curve`,
    `plot_pred_vs_real` y los colores de las curvas.
  - En `src/salida.py`: `RECEPTOR` (gancho de la app).
  - En `src/resultados.py`: `construir_entrega1` / `congelar_entrega1`.
  - En `src/data.py`: los CSV intermedios `clf.csv` / `reg.csv`.

**Movido sin cambiar la lógica**
- Las etapas 06-08 usaban el split de `e02`/`e03`. Ahora está en **`src/splits.py`**
  (`datos_clf`, `datos_reg`, `cv_estratificado`, `DROP_WH`, `filas_reales`), copiado línea a
  línea.
- Comprobado antes de borrar: `X_train`, `X_test`, `y_train` e `y_test` son iguales
  (`DataFrame.equals`) a los de `e02.datos()` / `e03.datos()`; huellas `8d9af7b40ee9` y
  `021799f95367`; 491 filas reales en ambos casos.
- `cargar_clf` / `cargar_reg` construyen siempre desde `data/obesity_raw.csv`. Comprobado: dan
  valores idénticos a los del antiguo `clf.csv` / `reg.csv`.

**Sustituido**
- La comprobación "E1 congelada = `resultados_texto/` de la E1" ya no es posible. La etapa 09
  ahora verifica la **huella SHA-256 del CSV congelado** (`HUELLA_E1 = 09d093c7…`), que no ha
  cambiado desde T1 (commit `2267801`).

**Actualizado**
- `main.py` (menú 6-9), `run_all.py` (4 etapas), los ejemplos de `src/pasos.py`,
  `src/cache.py`, `config.py` y `.gitignore`.
- `tests/test_cli.py` y `tests/test_cache.py`: prueban lo mismo de antes sobre las etapas 06-09.
- `tests/test_entrega2.py`: nuevas pruebas de que `src/splits.py` da los splits de la E1 y de
  la huella del CSV congelado.

**Verificación**
- `pytest` (suite completa, ya sin nada que excluir) → **26 passed**.
- `python run_all.py --limpiar-cache` (sin caché y sin salidas previas de 06-09): exit 0,
  1208 s.

| Etapa | 06 | 07 | 08 | 09 |
|---|---|---|---|---|
| Tiempo (s) | 15 | 522 | 668 | 2 |

- Comparación celda a celda de las 39 tablas CSV de `resultados_texto/06-09` y de
  `data/resultados_entrega2.csv` con la versión anterior: **solo cambian las columnas de
  tiempo**. Ningún archivo nuevo ni desaparecido. Todas las métricas, hiperparámetros, nº de
  vectores de soporte, épocas y conclusiones son idénticos.

### T9 — HTML visual de la Entrega 2 (`estudio_entrega2.html`)

**Cómo se hace**: la página se **genera** con `python generar_estudio.py` (y al final de
`run_all.py`), igual que exportar un notebook ejecutado. Así ninguna cifra se escribe a mano.

| Contenido de la página | De dónde sale |
|---|---|
| Tablas y notas de cada paso | `resultados_texto/<etapa>/<paso>.md` |
| Salida completa de consola (desplegable) | `resultados_texto/<etapa>/<paso>_consola.txt` |
| Figuras (ampliables) y tablas como imagen | `figures/<etapa>/` |
| Indicadores de la portada | `data/resultados_entrega{1,2}.csv`, vía la etapa 09 |
| Idea de cada paso (`IDEA`) y presentación de cada etapa (`INTRO`) | Escritas a mano, **sin resultados** |

**Diseño**: el CSS y el JS de la antigua `estudio_etapas.html`, en `estudio/estilo.css` y
`estudio/interaccion.js`. Incluye barra lateral, progreso por sección guardado en el
navegador, pasos plegables, visor de figuras, filtro de tablas largas y tema oscuro. Se quitaron
la tira de clases y la calculadora de IMC porque traían cifras de la E1 escritas a mano.
"Ver código" enlaza a `etapas/…` en la rama `main` de GitHub, así que los enlaces funcionan
cuando la rama se fusione.

**Cambios en `src/` para que las notas y la consola queden guardadas** (solo con `--guardar`
o `run_all.py`, como el resto de salidas):
- `salida.nota()` añade la nota al `.md` del paso.
- `Etapa.ejecutar()` guarda la consola del paso en `<paso>_consola.txt`.

**Verificación**
- `python run_all.py --limpiar-cache`: exit 0.

| Etapa | 06 | 07 | 08 | 09 |
|---|---|---|---|---|
| Tiempo (s) | 15 | 524 | 674 | 2 |

- Comparación celda a celda con lo versionado: **solo cambian columnas de tiempo**. Archivos
  nuevos: las 39 consolas.
- Página: 39 pasos, 69 figuras.
  - En Chromium (Playwright): ninguna imagen rota, ningún error de JavaScript y sin
    desbordamiento horizontal a 390 px.
  - Se revisaron capturas de la portada, de los pasos 07·3.2 y 08·3 y de la vista móvil.
- Cifras: **las 2838 cifras decimales del texto visible aparecen en las salidas ejecutadas**.
  El texto escrito a mano solo contiene constantes de protocolo: 80/20, 418, 5 semillas, 42,
  7 clases, capas 32/64/128 y 15 %.
- `pytest` → **27 passed**. Nuevo test: el HTML tiene todos los pasos, cada paso tiene su
  `IDEA` y el texto escrito a mano no lleva cifras con decimales.
