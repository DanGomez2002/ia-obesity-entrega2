### Regresión de Weight — test (418), kg, ordenado por RMSE

| modelo                            | puesto   | entrega   | familia       |     MAE |    RMSE |      R2 |   R2_adj | cv (RMSE; red: val_loss)   | RMSE media 5 semillas   | RMSE desv 5 semillas   |
|:----------------------------------|:---------|:----------|:--------------|--------:|--------:|--------:|---------:|:---------------------------|:------------------------|:-----------------------|
| Random Forest                     | 1        | E1        | Random Forest |  4.8386 |  8.2468 |  0.8987 |   0.8939 | 9.1156                     | —                       | —                      |
| KNN                               | 2        | E1        | KNN           |  5.0484 |  9.9085 |  0.8538 |   0.8468 | 10.3709                    | —                       | —                      |
| SVM (SVR rbf)                     | 3        | E2        | SVM           |  6.2617 | 10.0951 |  0.8482 |   0.841  | 9.4351                     | —                       | —                      |
| Árbol de regresión                | 4        | E1        | Árbol         |  6.581  | 10.4655 |  0.8369 |   0.8291 | 12.4728                    | —                       | —                      |
| Red neuronal grande [128, 64, 32] | 5        | E2        | Red neuronal  |  7.3545 | 11.6764 |  0.7969 |   0.7872 | 0.1086                     | 10.4756                 | 0.6898                 |
| Baseline (media)                  | —        | E1        | Baseline      | 21.7064 | 25.9126 | -0.0001 |  -0.0479 | —                          | —                       | —                      |

