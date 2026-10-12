### Clasificación A (todas las features) — arquitecturas (validación, 5 semillas)

|                      |   n_parametros |   val_loss media |   val_loss desv |   val_loss semilla 42 |   mejor época (media) |   épocas entrenadas (media) |   tiempo medio (s) |
|:---------------------|---------------:|-----------------:|----------------:|----------------------:|----------------------:|----------------------------:|-------------------:|
| pequeña [32]         |            903 |           0.237  |          0.0395 |                0.238  |                   163 |                         193 |            30.8232 |
| media [64, 32]       |          3,655 |           0.2526 |          0.0634 |                0.1503 |                    42 |                          72 |            12.3769 |
| grande [128, 64, 32] |         13,255 |           0.275  |          0.0518 |                0.3164 |                    22 |                          52 |             9.4842 |

