"""Configuración global del proyecto: rutas, semilla y estilo de gráficos.

Todos los scripts importan de aquí para que las rutas funcionen sin importar
desde qué carpeta se ejecuten (VS Code, terminal, etc.).
"""
from pathlib import Path

# Carpeta raíz del proyecto (donde está este archivo)
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"        # dataset crudo + resultados de las entregas 1 y 2
FIG_DIR = ROOT / "figures"      # PNG de cada gráfica, por etapa
CACHE_DIR = ROOT / "cache"      # modelos ya entrenados (GridSearch), para no repetir

# Semilla única para todo el proyecto: splits, validación cruzada y redes neuronales.
# Con 42 los números coinciden exactamente con los notebooks originales.
RANDOM_STATE = 42

# Estilo de matplotlib/seaborn compartido por todas las etapas
ESTILO_MPL = {
    "figure.dpi": 110,
    "savefig.dpi": 110,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.autolayout": True,
}
