"""Ejecuta las etapas completas, sin abrir ventanas, y guarda todas las figuras en figures/.

    python run_all.py                  # las 4 etapas; usa la caché de modelos si existe
    python run_all.py --limpiar-cache  # recalcula todo desde cero (~20 min)
    python run_all.py --etapas 07 09   # sólo esas etapas

Etapas de la Entrega 2 (la numeración continúa la de la Entrega 1, que vive en su propio
proyecto): 06 Naive Bayes, 07 SVM, 08 redes neuronales con Keras y 09 comparación final.
Es la forma de reproducir el proyecto entero de una vez.
"""
import argparse
import importlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import cache, salida

MODULOS = ["e06_naive_bayes", "e07_svm", "e08_redes_neuronales", "e09_comparacion_final"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Ejecuta las etapas del proyecto y guarda sus salidas")
    ap.add_argument("--limpiar-cache", action="store_true", help="recalcular todo desde cero")
    ap.add_argument("--etapas", nargs="+", metavar="NN", help="sólo estas etapas, p. ej. 06 07")
    args = ap.parse_args()

    elegidas = [m for m in MODULOS if not args.etapas or m[1:3] in {e.zfill(2) for e in args.etapas}]
    if args.etapas and len(elegidas) != len(args.etapas):
        ap.error(f"etapas válidas: {', '.join(m[1:3] for m in MODULOS)}")
    salida.configurar(mostrar=False, guardar=True)
    if args.limpiar_cache:
        cache.limpiar()
    t0 = time.time()
    for nombre in elegidas:
        t_etapa = time.time()
        importlib.import_module(f"etapas.{nombre}").etapa.ejecutar_todo()
        print(f"\n[run_all] {nombre} terminada en {time.time() - t_etapa:.0f} s")
    print(f"\n{len(elegidas)} etapas ejecutadas en {time.time() - t0:.0f} s. Figuras en figures/, "
          "tablas en resultados_texto/.")
