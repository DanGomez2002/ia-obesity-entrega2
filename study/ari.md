# ARI y pureza: cómo se calculan, con el caso real de K-Means K=7

Este documento explica, con la matriz de confusión de **K=7 mapeada con el algoritmo húngaro** (`3.2_matriz_mapeada_k7`), qué miden el **ARI** (Adjusted Rand Index) y la **pureza**, cómo se calculan paso a paso y por qué salen valores tan bajos aquí.

**[Código]** `etapas/e04_clustering.py` → `analizar_clustering()` (y su copia en `webapp/clustering.py`).

```python
ct = pd.crosstab(y_true, labels)                 # tabla de contingencia clase real × cluster
r_ind, c_ind = linear_sum_assignment(-ct.values) # asignación húngara cluster -> clase
...
"ARI":      adjusted_rand_score(y_true, labels),
"acc_mapeo": np.trace(cm) / cm.sum(),
"pureza":   ct.max(axis=0).sum() / ct.values.sum(),
```

---

## 1. La matriz de partida

Filas = clase real (`NObeyesdad`). Columnas = clase que el húngaro le asignó al cluster. Son 2087 filas (2111 menos 24 duplicados).

| Clase real \ cluster→clase | Insuf. | Normal | Sobrep. I | Sobrep. II | Ob. I | Ob. II | Ob. III | **Total fila** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Insufficient_Weight | **141** | 6 | 0 | 0 | 45 | 30 | 45 | 267 |
| Normal_Weight | 125 | **31** | 4 | 6 | 31 | 32 | 53 | 282 |
| Overweight_Level_I | 60 | 9 | **2** | 1 | 65 | 91 | 48 | 276 |
| Overweight_Level_II | 12 | 6 | 0 | **1** | 94 | 106 | 71 | 290 |
| Obesity_Type_I | 4 | 2 | 0 | 3 | **110** | 122 | 110 | 351 |
| Obesity_Type_II | 1 | 1 | 1 | 0 | 95 | **148** | 51 | 297 |
| Obesity_Type_III | 0 | 0 | 0 | 0 | 1 | 125 | **198** | 324 |
| **Total columna (tamaño del cluster)** | 343 | 55 | 7 | 11 | 441 | 654 | 576 | **2087** |

En negrita está la diagonal, es decir, lo que el mapeo cuenta como "acierto".

Dos cosas que se ven de inmediato:

- **Los clusters 3 y 4 casi no existen** (7 y 11 filas). K-Means los dejó diminutos; el húngaro igual les tiene que asignar una clase distinta a cada uno (Sobrepeso I y II), porque la asignación es 1 a 1.
- **Los clusters grandes mezclan clases vecinas**: el cluster "Ob. II" tiene 654 filas repartidas entre 6 clases reales; el de "Ob. I" mezcla Sobrepeso I/II, Ob. I y Ob. II.

---

## 2. Qué hace el algoritmo húngaro (y qué NO hace)

K-Means entrega números de cluster (0…6) sin ningún significado. Para poder hablar de "aciertos" hay que darle a cada cluster el nombre de una clase.

El húngaro (`linear_sum_assignment(-ct.values)`) busca la **asignación 1 a 1 que maximiza la suma de la diagonal**. Aquí esa suma es **631**. Con 7 clusters y 7 clases es una permutación: ningún nombre de clase se repite.

Importante: es una asignación **óptima en el total**, no "el cluster se parece a esa clase". Por eso el cluster 3 (11 filas, 6 de ellas Normal) se llama "Sobrepeso II" solo porque era lo que dejaba mejor la suma global.

El húngaro **solo afecta a la matriz de confusión y a `acc_mapeo`**. El ARI y la pureza **no dependen del mapeo**: se calculan con los clusters tal cual (ver abajo). Por eso el ARI es una métrica "limpia" y la accuracy con mapeo es optimista.

---

## 3. Accuracy con mapeo (para comparar)

$$
\text{acc\_mapeo} = \frac{\text{traza}}{N} = \frac{141+31+2+1+110+148+198}{2087} = \frac{631}{2087} = 0.302
$$

Es la métrica más intuitiva: "30,2 % de las filas quedaron en el cluster cuyo nombre coincide con su clase". Con 7 clases y reparto parejo, el azar con mapeo óptimo ya daría ~14 % + algo de ventaja del húngaro, así que 30 % es poco.

---

## 4. Pureza

**Idea:** a cada cluster le damos la clase que **más abunda dentro de él** (sin exigir que sea 1 a 1) y contamos cuántas filas coinciden.

$$
\text{pureza} = \frac{1}{N}\sum_{\text{clusters}} \max_{\text{clase}} n_{\text{clase, cluster}}
$$

Código: `ct.max(axis=0).sum() / ct.values.sum()` — el máximo de **cada columna**, sumado, dividido entre el total.

Máximo de cada columna en nuestra matriz:

| Cluster (columna) | Valores | Máximo | Clase dominante |
|---|---|---:|---|
| →Insufficient | 141, 125, 60, 12, 4, 1, 0 | **141** | Insufficient |
| →Normal | 6, 31, 9, 6, 2, 1, 0 | **31** | Normal |
| →Sobrep. I | 0, 4, 2, 0, 0, 1, 0 | **4** | Normal (no Sobrep. I) |
| →Sobrep. II | 0, 6, 1, 1, 3, 0, 0 | **6** | Normal (no Sobrep. II) |
| →Ob. I | 45, 31, 65, 94, 110, 95, 1 | **110** | Ob. I |
| →Ob. II | 30, 32, 91, 106, 122, 148, 125 | **148** | Ob. II |
| →Ob. III | 45, 53, 48, 71, 110, 51, 198 | **198** | Ob. III |

$$
\text{pureza} = \frac{141+31+4+6+110+148+198}{2087} = \frac{638}{2087} = 0.306
$$

### Por qué pureza (638) ≠ acc_mapeo (631)

Los clusters 3 y 4 son los únicos donde el máximo de la columna **no está en la diagonal**: su clase dominante es Normal (4 y 6 filas), pero el húngaro ya usó "Normal" para otro cluster (el de 31) y los forzó a ser Sobrep. I y II (2 y 1 filas). La pureza sí permite repetir clase; la accuracy con mapeo no. Diferencia: (4−2) + (6−1) = 7 filas → 638 − 631.

Si la pureza fuera mucho mayor que `acc_mapeo`, sería señal de que varios clusters "compiten" por la misma clase (K demasiado grande para lo que hay en los datos).

### Limitación de la pureza

La pureza **sube sola al aumentar K** (con K = N sería 1.0) y no penaliza clusters diminutos. Los clusters de 7 y 11 filas casi no aportan, pero tampoco restan. Además, el piso de referencia no es 0: asignar todo a la clase más grande ya daría 351/2087 = 0,168.

---

## 5. ARI (Adjusted Rand Index)

**Idea:** no mira nombres de cluster ni de clase. Mira **pares de filas** y pregunta: "¿los dos clusterings (real y K-Means) coinciden en si este par va junto o separado?". Luego lo corrige por azar.

### 5.1 Conteo de pares

Para un número `n` de filas, los pares son `C(n,2) = n(n−1)/2`. Se calculan cuatro cantidades:

| Símbolo | Significado | Cómo se calcula | Valor aquí |
|---|---|---|---:|
| **S** | pares juntos **en la clase real Y en el mismo cluster** | suma de `C(n_ij, 2)` sobre las 49 celdas | **107 739** |
| **A** | pares que comparten **clase real** | suma de `C(total fila, 2)` | 312 694 |
| **B** | pares que comparten **cluster** | suma de `C(total columna, 2)` | 536 365 |
| **T** | todos los pares posibles | `C(2087, 2)` | 2 176 741 |

Ejemplos de cómo se arma S (celda por celda, `C(n,2)`):

- Celda de 198 (Ob. III real en cluster Ob. III): C(198,2) = 19 503 pares.
- Celda de 148: C(148,2) = 10 878. Celda de 141: C(141,2) = 9 870. Celda de 125 (dos veces: Normal→Insuf. y Ob. III→Ob. II): C(125,2) = 7 750 cada una.
- Las celdas de 0 o 1 aportan 0.

Sumando las 49 celdas sale S = 107 739.

### 5.2 Corrección por azar

Si los dos agrupamientos fueran independientes, se esperaría que S valiera:

$$
E[S] = \frac{A \cdot B}{T} = \frac{312\,694 \times 536\,365}{2\,176\,741} \approx 77\,050
$$

Y el máximo posible es el promedio `(A+B)/2 = 424 529,5` (el valor de S si ambos agrupamientos fueran idénticos).

### 5.3 Fórmula final

$$
\text{ARI} = \frac{S - E[S]}{\tfrac{A+B}{2} - E[S]}
= \frac{107\,739 - 77\,050}{424\,529{,}5 - 77\,050}
= \frac{30\,689}{347\,479}
\approx \mathbf{0{,}088}
$$

### 5.4 Cómo leerlo

| ARI | Interpretación |
|---|---|
| 1 | agrupamientos idénticos (salvo renombrar clusters) |
| ≈ 0 | tan buenos como asignar al azar |
| < 0 | peor que el azar |

Con **0,088**, K-Means K=7 está apenas por encima del azar. Dicho en pares: de los pares que "deberían" estar juntos por clase real, el clustering solo capturó una fracción pequeña más de lo que el azar ya capturaría (107 739 contra 77 050 esperados por azar).

---

## 6. Por qué dan tan bajos en este caso

1. **Las clases son un continuo, no grupos separados.** `NObeyesdad` se define por IMC en cortes (18,5 / 25 / 30 / 35 / 40). Las clases consecutivas se tocan, así que no hay fronteras naturales que K-Means pueda encontrar. Se ve en la matriz: cada fila reparte sus filas entre 3–4 columnas contiguas.
2. **K-Means agrupa por otras cosas.** Con todas las variables escaladas (género, transporte, hábitos…), los clusters se forman por similitud global, no por IMC. Por ejemplo, 125 filas de Obesity_Type_III y 122 de Obesity_Type_I caen en el cluster "Ob. II" aunque son clases muy distintas.
3. **Dos clusters degenerados** (7 y 11 filas) desperdician 2 de los 7 "huecos" y obligan al húngaro a inventar nombres.
4. **Las clases bajas se apiñan.** Normal (125 de 282) cae en el cluster "Insuficiente": el cluster de bajo peso es más un cluster de "peso bajo/normal" que de una sola clase.

El único bloque razonablemente nítido es el extremo: Obesity_Type_III (198 de 324 en su cluster y solo 1 en el de Ob. I) y Insufficient (141 de 267).

---

## 7. Resumen del caso

| Métrica | Valor | Depende del mapeo húngaro | Qué dice |
|---|---:|:---:|---|
| ARI | **0,088** | No | casi indistinguible del azar |
| Pureza | **0,306** | No | cada cluster tiene ~31 % de su clase dominante (ponderado) |
| acc_mapeo | **0,302** | Sí | 631 de 2087 en la diagonal |
| Referencia "todo a la clase mayoritaria" | 0,168 | — | piso de la pureza |

**Conclusión:** la pureza (0,306) parece "no tan mala" porque su piso es 0,17 y porque ignora el azar; el ARI (0,088) es el que dice la verdad estricta: **el clustering no recupera las clases de obesidad**. Eso es esperable en un problema de clasificación supervisada por umbrales de IMC que se intenta resolver sin etiquetas, y es justo la razón por la que en la etapa de comparación (E05) estas métricas se usan para mostrar que la estructura de clusters no coincide con `NObeyesdad`.
