### Modelos de la Entrega 2: coste y tamaño

|                                                          |   tiempo de entrenamiento (s) | nº parámetros (red)   | nº vectores de soporte   | mejor época   | épocas entrenadas   | log-loss en test   |
|:---------------------------------------------------------|------------------------------:|:----------------------|:-------------------------|:--------------|:--------------------|:-------------------|
| ('clasificacion_A', 'Naive Bayes (CategoricalNB)')       |                        0.0174 | —                     | —                        | —             | —                   | 0.7642             |
| ('clasificacion_B', 'Naive Bayes (CategoricalNB)')       |                        0.0152 | —                     | —                        | —             | —                   | 1.1014             |
| ('clasificacion_A', 'SVM (SVC rbf)')                     |                        0.1027 | —                     | 312                      | —             | —                   | —                  |
| ('clasificacion_B', 'SVM (SVC rbf)')                     |                        0.0706 | —                     | 998                      | —             | —                   | —                  |
| ('regresion', 'SVM (SVR rbf)')                           |                        0.101  | —                     | 1,250                    | —             | —                   | —                  |
| ('clasificacion_A', 'Red neuronal pequeña [32]')         |                       30.4628 | 903                   | —                        | 157           | 187                 | 0.3097             |
| ('clasificacion_B', 'Red neuronal grande [128, 64, 32]') |                        8.4448 | 12,999                | —                        | 9             | 39                  | 0.8034             |
| ('regresion', 'Red neuronal grande [128, 64, 32]')       |                        7.3637 | 12,929                | —                        | 10            | 40                  | —                  |

