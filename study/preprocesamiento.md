# Preprocesamiento del dataset: qué usamos y qué no

Dataset: **UCI 544 – Estimation of Obesity Levels** (2111 filas × 17 columnas en crudo, sin valores faltantes, 24 filas duplicadas exactas).

Cómo leer las referencias de este documento:

| Etiqueta | Significa | Dónde se ve |
|---|---|---|
| **[Código]** | Archivo y función donde está implementado | `src/`, `etapas/`, `webapp/` |
| **[Gráfica]** | Imagen `.png` generada | `figures/<etapa>/` (y en `estudio_etapas.html`) |
| **[Tabla]** | Tabla impresa en consola y guardada como `.csv`/`.md`/`.png` | `figures/<etapa>/tablas/`, `resultados_texto/` |
| **[Datos]** | Archivo de datos resultante | `data/` |

Los pasos se nombran como `etapa·paso`, por ejemplo **E01·3** = etapa 01, paso 3. Puedes ejecutar uno solo con `python etapas/e01_eda.py --paso 3`, o verlo en la pestaña correspondiente de la app de Streamlit.

---

## 1. Resumen en una tabla

| Técnica | ¿Se usa? | Dónde |
|---|---|---|
| Revisión de valores faltantes | Sí (no hay ninguno) | E01·1 |
| Eliminación de filas duplicadas exactas | **Sí** (24) | `build_datasets` |
| Codificación binaria (0/1) | **Sí** | `encode_features` |
| Codificación ordinal (0–3) | **Sí** (CAEC, CALC) | `encode_features` |
| One-hot encoding | **Sí** (MTRANS, 5 columnas) | `encode_features` |
| Split train/test estratificado 80/20 | **Sí** | E02·1, E03·1 |
| Estandarización (`StandardScaler`) | **Sí**, solo para KNN y K-Means | Pipeline de KNN; E04·1 |
| Escalado en árboles y Random Forest | No (no lo necesitan) | — |
| Detección de fuga de datos (data leakage) | **Sí** (Weight/Height vs IMC) | E01·2.3 |
| Quitar Weight y Height (variante B) | **Sí** | E02·4 en adelante |
| Ruido gaussiano sintético (jittering) | **Sí**, como prueba de robustez | E02·7 |
| Identificar filas reales vs. sintéticas | **Sí** (heurística) | E02·8 |
| Agrupar las 7 clases en 3 | **Sí**, solo en el análisis de filas reales | E02·8 |
| PCA | **Sí**, solo para visualizar, no como preprocesamiento | E04·3.4 |
| Imputación de faltantes | No (no hay faltantes) | — |
| Tratamiento / eliminación de outliers | **No** | — |
| Normalización Min-Max | No | — |
| Transformaciones (log, Box-Cox) | No | — |
| Selección de variables previa al modelado | No (la importancia del RF solo se usa para interpretar) | E02·3.2, E03·5 |
| SMOTE / oversampling / undersampling propio | **No** (el dataset ya trae SMOTE de sus autores) | — |
| Ingeniería de variables (BMI como feature) | No (BMI solo se calcula para el análisis de fuga) | E01·2.3 |
| Pipeline que evita fuga entre train y test | **Sí** | `make_models` |

---

## 2. Técnicas que SÍ usamos

### 2.1 Revisión de calidad de datos (faltantes y duplicados)
- **Qué hace:** cuenta valores nulos y filas duplicadas exactas antes de tocar nada.
- **Resultado:** 0 faltantes, 24 duplicados.
- **[Código]** `etapas/e01_eda.py` → `paso_1_carga`
- **[Tabla]** "Primeras 5 filas" y "Estadísticos de las columnas numéricas": `figures/01_eda/tablas/1_tabla1_primeras_5_filas.png` y `1_tabla2_estadisticos_de_las_columnas_numericas.png`. Los conteos de faltantes y duplicados salen **solo por consola** (no son tabla guardada).

### 2.2 Eliminación de duplicados exactos
- **Qué hace:** `df_raw.drop_duplicates()` **antes** de cualquier split. Pasa de 2111 a **2087 filas**.
- **Por qué:** una fila duplicada puede caer en train y en test a la vez e inflar las métricas.
- **[Código]** `src/preprocessing.py` → `build_datasets`
- **[Datos]** `data/clf.csv` y `data/reg.csv` (2087 filas).
- **Límite:** solo quita duplicados *exactos*. Las filas sintéticas de SMOTE son casi iguales entre sí pero no idénticas, y esas **no** se quitan (ver sección 4).

### 2.3 Codificación de variables categóricas
Se hace en `encode_features` (`src/preprocessing.py`) según el tipo de variable:

| Tipo | Columnas | Técnica | Resultado |
|---|---|---|---|
| Binarias | `Gender`, `family_history_with_overweight`, `FAVC`, `SMOKE`, `SCC` | Mapeo directo | Female/no = 0, Male/yes = 1 |
| Ordinales | `CAEC`, `CALC` | Mapeo ordenado | no=0 < Sometimes=1 < Frequently=2 < Always=3 |
| Nominal | `MTRANS` | One-hot (`get_dummies`, **sin** `drop_first`) | 5 columnas: Automobile, Bike, Motorbike, Public_Transportation, Walking |
| Numéricas | `Age`, `Height`, `Weight`, `FCVC`, `NCP`, `CH2O`, `FAF`, `TUE` | Se dejan tal cual | — |
| Target clasificación | `NObeyesdad` | Categoría ordenada de 7 niveles (`CLASS_ORDER`) | de Insufficient_Weight a Obesity_Type_III |

- **Por qué `drop_first=False`:** para que los árboles vean cada categoría de transporte por separado.
- **Por qué ordinal y no one-hot en CAEC/CALC:** tienen un orden natural (nunca < a veces < frecuente < siempre).
- **Efecto en el tamaño:** `reg.csv` es 2087×20 y `clf.csv` es 2087×21 (con el target).
- **[Tabla]** "clf.csv — primeras filas" y "reg.csv — primeras filas": `figures/01_eda/tablas/3_tabla1_clf_csv_primeras_filas.png` y `3_tabla2_reg_csv_primeras_filas.png`.
- **[Código]** `etapas/e01_eda.py` → `paso_3_preprocesamiento` (ejecuta y guarda los CSV).

### 2.4 Dos datasets: clasificación y regresión
- **`clf.csv`:** todas las features codificadas + `NObeyesdad` como target.
- **`reg.csv`:** las mismas features **sin** `NObeyesdad` (sería hacer trampa) y con `Weight` como target, puesto al final.
- **[Código]** `build_datasets` en `src/preprocessing.py`; `cargar_clf()` y `cargar_reg()` los leen (o los regeneran si no existen).

### 2.5 División train/test estratificada
- **Clasificación:** 80/20, `stratify=y` (mantiene la proporción de las 7 clases), `random_state=42`.
- **Regresión:** 80/20, estratificado por **deciles de Weight** (`pd.qcut(y, q=10)`), porque el target es continuo.
- **Regla del proyecto:** el test se usa **una sola vez**, al final. Los hiperparámetros se eligen solo con CV sobre el train (5 folds; estratificado en clasificación, `KFold` normal en regresión).
- **[Código]** `etapas/e02_clasificacion.py` → `datos()`; `etapas/e03_regresion.py` → `datos()`.
- **[Tabla]** "Distribución de clases en train (%)": `figures/02_clasificacion/tablas/1_tabla1_distribucion_de_clases_en_train.png`.

### 2.6 Estandarización (`StandardScaler`) dentro de un Pipeline
- **Qué hace:** deja cada variable con media 0 y desviación 1.
- **Dónde:** solo **KNN** (clasificación y regresión) y **K-Means**, que usan distancias y se distorsionan si una variable tiene una escala mucho mayor (por ejemplo, `Age` frente a las binarias).
- **Cómo evita la fuga:** en KNN va **dentro del `Pipeline`**, así que se ajusta solo con el train de cada fold. Es una decisión explícita (está en el docstring de `src/preprocessing.py`).
- **No se aplica** a árboles ni Random Forest: sus cortes no cambian con la escala.
- **[Código]** `make_models` en `etapas/e02_clasificacion.py` y `etapas/e03_regresion.py` (`Pipeline([("scaler", StandardScaler()), ("knn", ...)])`).
- **Excepción a tener presente:** en clustering (`etapas/e04_clustering.py` → `datos()`) el scaler se ajusta con **todos** los datos. No es un problema porque no hay split ni evaluación supervisada, pero es distinto al caso de KNN.

### 2.7 Detección de fuga de datos (Weight/Height ↔ clase)
- **Hallazgo:** `NObeyesdad` se define a partir del IMC = Weight / Height². Si se dan Weight y Height al modelo, este "reconstruye" la etiqueta en lugar de aprender hábitos.
- **Cómo se comprobó:** boxplot del IMC por clase con los umbrales de la OMS (18.5 / 25 / 30 / 35 / 40), y un árbol de decisión con solo `BMI`, solo `Weight`, solo `Height` y `Weight`+`Height`, evaluado por CV.
- **[Gráfica]** `figures/01_eda/2.3_imc_por_clase.png`.
- **[Tabla]** "Correlación Weight / Height / BMI / Age": `figures/01_eda/tablas/2.3_tabla2_correlacion_weight_height_bmi_age.png`. La accuracy de CV de cada combinación sale por consola.
- **[Código]** `etapas/e01_eda.py` → `paso_2_3_fuga`.
- **Consecuencia:** dos variantes de clasificación.
  - **Variante A:** todas las features (como pide el enunciado).
  - **Variante B:** sin `Weight` ni `Height` (`DROP_WH = ["Weight", "Height"]`), que mide cuánto predicen los hábitos por sí solos (18 features).
- **[Gráfica]** `figures/02_clasificacion/5_comparacion_A_vs_B.png`; **[Tabla]** `5_tabla1_comparacion_a_vs_b.png`.

### 2.8 Exploración (EDA) que justifica las decisiones
| Qué | [Gráfica] | Paso |
|---|---|---|
| Balance de las 7 clases | `figures/01_eda/2.1_distribucion_clases.png` | E01·2.1 |
| Distribución de Weight (histograma + boxplot por clase) | `figures/01_eda/2.2_distribucion_weight.png` | E01·2.2 |
| Correlaciones de Pearson | `figures/01_eda/2.4_correlaciones.png` | E01·2.4 |

### 2.9 Ruido sintético (jittering) como prueba de robustez
- **Qué hace:** suma ruido gaussiano N(0, (nivel·σ)²) a los hábitos numéricos (`Age`, `FCVC`, `NCP`, `CH2O`, `FAF`, `TUE`) en train y test, y recorta a rangos válidos (por ejemplo `FCVC` entre 1 y 3). Niveles: 0, 5, 10, 15 y 20 %.
- **Objetivo:** ver qué modelo se degrada menos con datos imperfectos. **No** se usa para entrenar el modelo final; es un experimento.
- **[Código]** `etapas/e02_clasificacion.py` → `_con_ruido`, `paso_7_ruido`.
- **[Gráfica]** `figures/02_clasificacion/7_barrido_ruido.png`.
- **[Tabla]** `7_tabla1_robustez_ante_5_de_ruido_en_habitos_vari.png` y `7_tabla2_accuracy_en_test_por_nivel_de_ruido.png`.
- **Nota:** la caché `cache/cmp_ruido_20.joblib` guarda el caso con 20 % de ruido que usa la comparación final (E05).

### 2.10 Separar filas reales de sintéticas
- **Contexto:** según los autores del dataset (UCI), el **77 % de las filas son sintéticas**, generadas con SMOTE; solo el 23 % viene de encuestas reales.
- **Heurística usada:** una fila cuyas 6 respuestas de encuesta (`Age`, `FCVC`, `NCP`, `CH2O`, `FAF`, `TUE`) son todas **enteras** no fue interpolada. Da 491 de 2087 filas (23.5 %). El propio código aclara que es una heurística, no una etiqueta del dataset.
- **Qué se hace con ellas:** se entrena y evalúa **solo** con esas 491 filas (CV de 3 folds repetida 10 veces, sin GridSearch por el tamaño), con una línea base de clase mayoritaria, en 7 clases y en **3 clases agrupadas** (Bajo/Normal, Sobrepeso, Obesidad).
- **[Código]** `etapas/e02_clasificacion.py` → `filas_reales`, `COLS_ENCUESTA`, `GRUPOS_3`, `_cv_reales`, `paso_8_filas_reales`.
- **[Gráfica]** `figures/02_clasificacion/8_filas_reales_sin_WH.png`; en E05 también aparece con y sin SMOTE (`figures/05_comparacion/1_comparacion_final.png`).
- **[Tabla]** `8_tabla1_...` a `8_tabla6_...` en `figures/02_clasificacion/tablas/`.
- **Resultado clave:** con solo filas reales y sin Weight/Height, el macro-F1 baja a ~0.19 (7 clases, KNN k=1) y ~0.40 (3 clases, KNN k=1), frente a ~0.85 de la variante B con todo el dataset. Buena parte de ese 0.85 se explica porque las filas sintéticas se parecen a sus "vecinas" reales.

### 2.11 PCA (solo visualización)
- **Qué hace:** reduce a 2 componentes para dibujar los clusters de K-Means frente a las clases reales.
- **No es preprocesamiento:** no se entrenan modelos sobre las componentes.
- **[Código]** `etapas/e04_clustering.py` → `_pca_2d`.
- **[Gráfica]** `figures/04_clustering/3.4_pca_2d.png`.

---

## 3. Técnicas que NO usamos (y por qué)

| Técnica | Por qué no |
|---|---|
| **Imputación de faltantes** | El dataset tiene 0 faltantes (E01·1). |
| **Tratamiento de outliers** | No se detectan ni se eliminan. Los boxplots de E01·2.2 y E01·2.3 muestran valores extremos, pero se dejan: son personas del dataset (por ejemplo, pesos altos en Obesity_Type_III). Árboles y RF son robustos a ellos; KNN y K-Means sí pueden verse afectados. Es una decisión que conviene poder justificar. |
| **Min-Max / normalización a [0,1]** | Se eligió `StandardScaler`, que basta para KNN y K-Means. |
| **Transformaciones (log, Box-Cox, raíz)** | Los modelos usados no exigen normalidad de las variables ni del target. |
| **Selección de variables antes de modelar** | Se entrena con todas las features. La importancia del RF (`3.2_importancias_rf_A.png`, `5_importancias_rf.png`) se usa solo para **interpretar**. |
| **Ingeniería de variables (BMI como feature)** | El BMI se calcula solo para demostrar la fuga. Usarlo como feature haría trivial la clasificación. |
| **SMOTE / over o undersampling propio** | No se rebalancea. Las clases ya vienen casi balanceadas porque los autores del dataset aplicaron SMOTE. |
| **Reducción de dimensionalidad previa** | PCA solo se usa para dibujar. |
| **Eliminación de casi-duplicados** | Solo se quitan los 24 duplicados exactos. |
| **Split por grupos / por persona** | No hay un identificador de persona en el dataset. |
| **Análisis de multicolinealidad (VIF) o eliminación de variables correlacionadas** | Se miró la correlación (E01·2.4) pero no se eliminó nada; los modelos usados toleran la correlación. |

---

## 4. Limitaciones que conviene mencionar al defender el trabajo

1. **Fuga por filas sintéticas casi iguales (no por duplicados exactos).** El split train/test es aleatorio *después* de que SMOTE ya generó filas interpoladas. Una fila sintética en test puede tener a su "padre" o "hermana" casi idéntica en train. Eso favorece a KNN y a los árboles profundos, y explica por qué el resultado con solo filas reales (paso 8) es mucho peor que con todo el dataset.
2. **Weight/Height en la variante A es fuga por definición** (la clase sale del IMC). Por eso la variante A rinde ~0.96 de macro-F1 y la B ~0.85.
3. **La heurística de "fila real"** (todo entero) es una aproximación: puede marcar como real alguna fila sintética que cayó en enteros por casualidad.
4. **Escalado en clustering con todos los datos:** aceptable por no haber evaluación supervisada.
5. **Reproducibilidad:** los números dependen de `random_state=42` y de los modelos guardados en `cache/`. Si cambias hiperparámetros, borra la caché con `python run_all.py --limpiar-cache`.

---

## 5. Orden real del flujo de preprocesamiento

```
data/obesity_raw.csv (2111 × 17)
   │  E01·1  revisar faltantes (0) y duplicados (24)
   ▼
drop_duplicates()                              → 2087 filas
   │  encode_features: binarias 0/1, ordinales 0–3, one-hot MTRANS
   ▼
data/clf.csv (2087 × 21)    data/reg.csv (2087 × 20, target = Weight)
   │                              │
   │ split 80/20 estratificado    │ split 80/20 estratificado por deciles de Weight
   ▼                              ▼
Clasificación                  Regresión
 ├─ Variante A: 20 features     └─ 3 modelos, KNN con StandardScaler en Pipeline
 ├─ Variante B: sin Weight/Height (18 features)
 ├─ Robustez: ruido gaussiano 0–20 %
 └─ Solo filas reales (491) en 7 y 3 clases

Clustering (E04): features sin la clase → StandardScaler → K-Means; PCA solo para dibujar
```

## 6. Dónde ejecutar o ver cada cosa

| Quiero ver… | Cómo |
|---|---|
| Todo el pipeline y todas las figuras | `python run_all.py` |
| Solo el preprocesamiento | `python etapas/e01_eda.py --paso 3` |
| Un paso concreto con ventana | `python main.py 01 2.3` |
| Versión interactiva | `streamlit run app.py` (http://localhost:8501) |
| Explicación narrativa con todas las figuras | `estudio_etapas.html` |
