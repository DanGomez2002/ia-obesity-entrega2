# Plan de la Entrega 2 — SVM, Naive Bayes y Redes neuronales (Keras)

Dataset: UCI 544 (Estimation of Obesity Levels), 2111 filas en crudo, 2087 sin duplicados.
Este plan amplía el proyecto de la Entrega 1 **con su misma estructura**: la versión Python plana
de `etapas/` + pasos numerados (`src/pasos.py`), con salidas en `figures/` y `resultados_texto/`
y ejecución completa con `run_all.py`.

Estado: **T1 terminada. Espera tu verificación para empezar T2.**

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
| T1 | 2026-10-12 | Sonnet | Opus (no se cambió con /model) | Terminada, espera verificación | (este commit) |
| T2 | | Sonnet | | Pendiente | |
| T3 | | Sonnet | | Pendiente | |
| T4 | | Opus | | Pendiente | |
| T5 | | Opus | | Pendiente | |
| T7 | | Opus | | Pendiente | |

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
