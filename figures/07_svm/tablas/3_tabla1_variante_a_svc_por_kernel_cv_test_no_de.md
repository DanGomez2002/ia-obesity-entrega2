### Variante A — SVC por kernel: CV, test, nº de vectores de soporte y tiempo

|        |   cv_f1_macro |   test_accuracy |   test_f1_macro |   n_vectores_soporte |   tiempo_fit_s | mejores_params           |
|:-------|--------------:|----------------:|----------------:|---------------------:|---------------:|:-------------------------|
| linear |        0.9599 |          0.9689 |          0.9675 |                  239 |          1.045 | {'C': 1000}              |
| poly   |        0.9508 |          0.9569 |          0.9546 |                  546 |          0.056 | {'C': 100, 'coef0': 1, … |
| rbf    |        0.965  |          0.9689 |          0.9678 |                  312 |          0.122 | {'C': 100000, 'gamma': … |

