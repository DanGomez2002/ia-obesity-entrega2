# Cómo leer las 3 gráficas de regresión (paso 4, modelos principales)

Contexto: se predice **`Weight` (kg)** a partir de los hábitos y datos de la persona. Los tres modelos son **Árbol de regresión**, **Random Forest** y **KNN**, todos evaluados sobre el **conjunto de test** (datos que no vieron al entrenar).

Las tres gráficas muestran lo mismo desde ángulos distintos: **cuánto se equivoca cada modelo y cómo**.

| # | Archivo | Pregunta que responde |
|---|---|---|
| 1 | `4_pred_vs_real_modelos.png` | ¿Qué tan cerca está lo predicho de lo real? |
| 2 | `4_residuos_modelos.png` | ¿Cómo se distribuyen los errores? ¿Están centrados en 0? |
| 3 | `4_residuos_vs_predicho.png` | ¿El error depende del peso que se predice? ¿Hay sesgo? |

Archivos en `figures/03_regresion/`. Código: `etapas/e03_regresion.py` → `paso_4_pred_vs_real`; la gráfica 3 se dibuja con `plot_residuos_vs_pred` en `src/evaluation.py`.

---

## Vocabulario base

- **Residuo** = `real − predicho` (en kg).
  - Residuo **positivo**: el modelo **subestimó** (predijo menos de lo que la persona pesa).
  - Residuo **negativo**: el modelo **sobreestimó**.
  - Residuo **0**: acierto exacto.
- **Error absoluto** = `|real − predicho|`. Es el residuo sin signo.
- **Error relativo (%)** = `100 · |real − predicho| / real`. Sirve para comparar: 10 kg de error es mucho en una persona de 45 kg y poco en una de 130 kg.
- **RMSE** (kg): raíz del promedio de los errores al cuadrado. Penaliza mucho los errores grandes. Menor es mejor.
- **R²**: fracción de la variación del peso que explica el modelo (1 = perfecto, 0 = igual que predecir siempre la media). Mayor es mejor.
- **R² ajustado**: R² penalizado por el número de variables (p = 19–20). Aquí casi no cambia porque hay muchas más filas que variables.

Métricas que aparecen en los títulos de la gráfica 1 (test):

| Modelo | RMSE (kg) | R² | R² ajustado |
|---|---|---|---|
| Árbol de regresión | 10.47 | 0.837 | 0.829 |
| **Random Forest** | **8.25** | **0.899** | **0.894** |
| KNN | 9.91 | 0.854 | 0.847 |

---

## Gráfica 1 — Predicho vs real (`4_pred_vs_real_modelos.png`)

**Tipo:** diagrama de dispersión (scatter), un panel por modelo.

**Cómo está construida**
- **Eje X:** peso real (kg). **Eje Y:** peso predicho (kg).
- **Cada punto** es una persona del test.
- **Línea roja discontinua:** la diagonal `y = x`. Un punto sobre ella es una predicción perfecta.
- Los tres paneles usan **la misma escala** en ambos ejes, para poder compararlos a simple vista.

**Cómo leerla**
- **Cerca de la diagonal = buena predicción.** Cuanto más delgada la nube alrededor de la línea, mejor el modelo.
- **Punto por encima de la línea:** el modelo predijo **más** de lo real (sobreestimó).
- **Punto por debajo:** predijo **menos** (subestimó).
- **Distancia vertical a la diagonal** = el error de esa persona.
- **Patrón de escalones o filas horizontales de puntos** (se nota en el árbol): el árbol da el mismo valor a todas las personas que caen en la misma hoja, por eso sus predicciones se repiten.

**Qué se ve en tu imagen**
- **Random Forest:** la nube más pegada a la diagonal en casi todo el rango; coincide con su RMSE más bajo.
- **Árbol de regresión:** más dispersión y un par de errores muy grandes. El más evidente: una persona con peso real ≈ 55 kg que el árbol predice ≈ 134 kg (un punto muy por encima de la línea, arriba a la izquierda).
- **KNN:** intermedio. Muchos puntos pegados a la línea, pero con algunos puntos alejados (por ejemplo reales de 60–90 kg predichos en 110–125 kg).
- **En los tres, la dispersión es mayor entre 50 y 100 kg** que en los extremos, donde hay menos datos o el comportamiento es más regular.

**Para qué sirve:** da una impresión global rápida. No dice *dónde* falla con precisión; para eso están las gráficas 2 y 3.

---

## Gráfica 2 — Histograma de residuos (`4_residuos_modelos.png`)

**Tipo:** histograma con curva de densidad suavizada (KDE), un panel por modelo.

**Cómo está construida**
- **Eje X:** residuo `real − predicho` (kg).
- **Eje Y:** frecuencia, es decir cuántas personas del test caen en ese rango de error (barras de ~2 kg de ancho).
- **Barras azules:** conteo por rango. **Curva azul:** la misma distribución suavizada.
- **Línea roja punteada en 0:** error nulo.

**Cómo leerla**
- **Centrada en 0:** el modelo no tiende a sobre ni subestimar en promedio.
- **Pico alto y angosto:** muchos casos con error casi nulo. Mejor.
- **Colas largas (a izquierda o derecha):** casos con errores grandes. Importan mucho para el RMSE.
- **Asimetría:** una cola más larga a la izquierda significa más sobreestimaciones grandes; a la derecha, más subestimaciones grandes.

**Qué se ve en tu imagen**
- Los tres modelos tienen un **pico muy marcado en 0** (en el árbol y el Random Forest la barra central pasa de ~120 personas; en KNN llega a ~175). Eso significa que **para una gran parte de las personas el error es casi cero**.
- **Cuidado al interpretar ese pico:** que tantos casos tengan error prácticamente nulo es en parte consecuencia de que el dataset contiene filas sintéticas (SMOTE) muy parecidas entre sí. Un modelo como KNN o un árbol profundo puede "reconocer" a la vecina casi idéntica del entrenamiento. Ver `study/preprocesamiento.md`, sección 4. Es la razón por la que KNN tiene el pico más alto pero también colas largas.
- **Árbol:** cola izquierda que llega a ≈ −80 kg (el caso del punto atípico de la gráfica 1). Cola más ancha en general.
- **Random Forest:** colas más cortas (aprox. de −40 a +42 kg): sus peores errores son menores.
- **KNN:** pico más alto, pero cola derecha que llega a ≈ +55 kg, es decir, algunas subestimaciones grandes.

**Para qué sirve:** ver la *forma* del error y detectar sesgo global. Ojo: un histograma centrado en 0 **no** garantiza que el modelo sea bueno en todo el rango; para eso está la gráfica 3.

---

## Gráfica 3 — Residuos vs predicho (`4_residuos_vs_predicho.png`)

**Tipo:** dispersión (scatter), un panel por modelo. Es la gráfica de **diagnóstico** clásica de regresión.

**Cómo está construida**
- **Eje X:** peso **predicho** (kg). **Eje Y:** residuo `real − predicho` (kg).
- **Línea roja en y = 0:** error nulo.
- Los tres paneles comparten el eje Y, así que la altura de la nube es directamente comparable.

**Cómo leerla (lo que se busca)**
1. **Nube plana, sin forma, centrada en 0** → el modelo es sano: el error no depende del peso predicho.
2. **Forma de embudo o abanico** (la nube se ensancha hacia un lado) → **heterocedasticidad**: el error crece en cierto rango de peso.
3. **Curva (por ejemplo, positivos a la izquierda y negativos a la derecha)** → **sesgo sistemático**: el modelo subestima unos pesos y sobreestima otros.
4. **Puntos muy alejados** → casos atípicos que el modelo no logra explicar.

**Qué se ve en tu imagen**
- **Random Forest:** la nube más compacta alrededor de 0, con pocos puntos a más de ±20 kg. Algunos aislados cerca de ±40 kg.
- **Árbol:** más ancha; un punto extremo cerca de −80 kg (predicho ≈ 134).
- **KNN:** muchos puntos pegados a 0, pero con **errores positivos altos** (hasta ≈ +55 kg), concentrados cuando predice entre ≈ 50 y 90 kg.
- **A pesos predichos altos (> 120 kg) los residuos son pequeños** en los tres modelos: ahí los modelos aciertan más. Es consistente con la gráfica 1.
- **No hay una curva clara ni un embudo muy pronunciado**: no se observa sesgo sistemático fuerte; sí se ve más dispersión en el rango medio (≈ 60–100 kg).

**Diferencia con la gráfica 1:** es la misma información rotada. En la gráfica 1 el error es la distancia a la diagonal; aquí se "aplana" la diagonal para que el error se vea directo en el eje Y y los patrones sean más fáciles de detectar. Es la gráfica que normalmente se pide para validar un modelo de regresión.

---

## Cómo se complementan: "por qué RF es el mejor si KNN tiene el pico más alto en 0"

| Mirada | Quién destaca | Por qué |
|---|---|---|
| Pico del histograma en 0 (muchos errores casi nulos) | **KNN** | Acierta casi exacto a muchas personas, probablemente por tener vecinas muy parecidas en el entrenamiento |
| Errores grandes (RMSE, colas del histograma, residuos extremos) | **Random Forest** | Promedia muchos árboles y se equivoca menos en los casos difíciles |
| Rendimiento global (R², RMSE) | **Random Forest** | 0.899 y 8.25 kg |
| Peor desempeño general | **Árbol** | Un solo árbol, con saltos y algunos errores muy grandes |

**Resumen para la defensa:** el Random Forest es el modelo más fiable (menos errores grandes y mejor R²/RMSE); KNN es muy preciso en la mayoría de casos pero con fallos grandes ocasionales; el árbol único es el más inestable.

---

## Checklist rápido para leer cualquier gráfica de regresión

1. ¿Qué hay en cada eje y en qué unidades? (kg, % o frecuencia)
2. ¿Dónde está el "valor ideal"? (diagonal en la 1, línea en 0 en la 2 y 3)
3. ¿Hay patrones (embudo, curva, escalones) o es una nube sin forma?
4. ¿Hay puntos atípicos? ¿A qué persona o rango corresponden?
5. ¿Concuerda con las métricas (RMSE, R²)? Si no, busca la causa (colas largas, muestras sintéticas, etc.).
