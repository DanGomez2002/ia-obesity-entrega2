### Datos de la figura: 3_comparacion_final

#### Clasificación A (todas las features)

| categoria                   |    valor |
|:----------------------------|---------:|
| Naive Bayes (CategoricalNB) | 0.803    |
| KNN                         | 0.8075   |
| Red neuronal [32]           | 0.922715 |
| Árbol de decisión           | 0.9362   |
| Random Forest               | 0.9558   |
| SVM (SVC rbf)               | 0.9678   |


#### Clasificación B (sin Weight ni Height)

| categoria                   |    valor |
|:----------------------------|---------:|
| Naive Bayes (CategoricalNB) | 0.6624   |
| Red neuronal [128, 64, 32]  | 0.732701 |
| Árbol de decisión           | 0.7401   |
| KNN                         | 0.7567   |
| SVM (SVC rbf)               | 0.783    |
| Random Forest               | 0.8528   |


#### Regresión de Weight

| categoria                  |   valor |
|:---------------------------|--------:|
| Red neuronal [128, 64, 32] | 11.6764 |
| Árbol de regresión         | 10.4655 |
| SVM (SVR rbf)              | 10.0951 |
| KNN                        |  9.9085 |
| Random Forest              |  8.2468 |


