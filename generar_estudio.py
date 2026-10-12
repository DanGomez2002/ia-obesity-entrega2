"""Genera estudio_entrega2.html: la Entrega 2 paso a paso, con sus figuras, tablas y notas.

    python run_all.py            # primero: deja las salidas en resultados_texto/ y figures/
    python generar_estudio.py    # después: escribe estudio_entrega2.html

La página es el equivalente a exportar a HTML un notebook ejecutado. Todo lo que tiene cifras
sale de las salidas ejecutadas, no de este archivo:
- las tablas y las notas de cada paso, de resultados_texto/<etapa>/<paso>.md;
- la salida completa de consola, de resultados_texto/<etapa>/<paso>_consola.txt;
- las figuras, de figures/<etapa>/;
- los indicadores de la portada, de data/resultados_entrega{1,2}.csv (vía la etapa 09).
Lo único escrito aquí a mano es la idea de cada paso (IDEA, sin resultados) y la
presentación de cada etapa (INTRO). El diseño (estudio/estilo.css y estudio/interaccion.js)
es el mismo de la página de estudio de la Entrega 1. Las imágenes se enlazan con rutas
relativas: abre el HTML desde la raíz del proyecto.
"""
import html
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))

import main  # noqa: E402  (registro de etapas y pasos)
from src.preprocessing import CLASS_ORDER  # noqa: E402

SALIDA = RAIZ / "estudio_entrega2.html"
RES = RAIZ / "resultados_texto"
FIG = RAIZ / "figures"
REPO_URL = "https://github.com/DanGomez2002/ia-obesity-entrega2/blob/main"
NAV = {"06": "Naive Bayes", "07": "SVM", "08": "Redes neuronales", "09": "Comparación final"}

# ----------------------------------------------------------------------------
# Texto escrito a mano: sólo conceptos y protocolo (los resultados salen de las salidas)
# ----------------------------------------------------------------------------
INTRO = {
    "06": """<p><strong>Naive Bayes</strong> aplica el teorema de Bayes suponiendo que los atributos son
independientes dada la clase: P(clase | x) ∝ P(clase) · Π P(x<sub>j</sub> | clase). Sólo existe para
clasificación. Se comparan <strong>GaussianNB</strong> (cada atributo es una normal por clase) y
<strong>CategoricalNB</strong> (cada atributo es una categoría; las continuas se discretizan en tramos),
en las variantes A (todas las features) y B (sin Weight ni Height).</p>""",
    "07": """<p>Una <strong>SVM</strong> busca la frontera de máximo margen; sólo los vectores de soporte la
definen. Con un <strong>kernel</strong> trabaja en un espacio transformado: <code>linear</code> (hiperplano),
<code>poly</code> (polinomio de grado d) y <code>rbf</code> (fronteras flexibles; gamma = alcance de cada
punto). Se usa para clasificación (<strong>SVC</strong>, variantes A y B) y regresión de Weight
(<strong>SVR</strong>, con un tubo de ancho epsilon), siempre con <code>StandardScaler</code> dentro del
Pipeline.</p>""",
    "08": """<p>Perceptrón multicapa con <strong>Keras</strong>: capas densas con ReLU, salida softmax de 7
neuronas en clasificación y 1 neurona lineal en regresión. Se comparan tres arquitecturas
(<code>[32]</code>, <code>[64, 32]</code>, <code>[128, 64, 32]</code>) por la pérdida de
<strong>validación</strong>, que sale del train (15 %), con early stopping. Como las redes no son
deterministas, se informa el modelo con semilla 42 y la media ± desviación de 5 semillas.</p>""",
    "09": """<p>Reúne los seis modelos supervisados sobre el <strong>mismo test de 418 instancias</strong> por
tarea: árbol, Random Forest y KNN de la Entrega 1 (congelados en
<code>data/resultados_entrega1.csv</code>, no se re-entrenan) frente a Naive Bayes, SVM y red neuronal
de la Entrega 2. Las conclusiones se escriben con las cifras de los archivos de resultados.</p>""",
}

IDEA = {
    ("06", "1"): "Mismo split 80/20 estratificado que la Entrega 1. Se comprueba por qué Naive Bayes no necesita escalado y qué papel juega <code>var_smoothing</code>.",
    ("06", "1.1"): "El supuesto de Naive Bayes es la independencia de los atributos dada la clase. Aquí se mide cuánto se incumple: la clase la fija el IMC = Weight / Height².",
    ("06", "2"): "Las dos variantes de Naive Bayes y sus grillas: <code>var_smoothing</code> en GaussianNB; número de tramos y suavizado <code>alpha</code> en CategoricalNB.",
    ("06", "3"): "GridSearchCV (5 folds estratificados, macro-F1) en la variante A. Además de accuracy y macro-F1 se informa el log-loss, porque Naive Bayes da probabilidades.",
    ("06", "3.1"): "Precisión, recall y F1 por clase y matrices de confusión en el test.",
    ("06", "3.2"): "Cómo cambia la macro-F1 de CV con cada hiperparámetro, para ver si el óptimo está dentro de la grilla.",
    ("06", "4"): "Lo mismo sin Weight ni Height: cuánto predicen los hábitos por sí solos.",
    ("06", "4.1"): "Métricas por clase y matrices de confusión de la variante B.",
    ("06", "5"): "Se elige el Naive Bayes que pasa a la comparación final por macro-F1 de <strong>CV</strong>, no de test, y se comprueba si sus probabilidades son sobreconfiadas.",
    ("06", "6"): "Fila de cada variante para <code>data/resultados_entrega2.csv</code>.",
    ("07", "1"): "Los mismos splits de la Entrega 1: por clase en clasificación y por deciles de Weight en regresión. Por qué la SVM sí necesita escalado.",
    ("07", "2"): "Un Pipeline <code>StandardScaler</code> + SVC/SVR por kernel, cada uno con su grilla (C, gamma, degree, coef0 y, en SVR, epsilon).",
    ("07", "2.1"): "Comprobación de que el óptimo de cada grilla no sigue mejorando fuera de ella.",
    ("07", "3"): "Un GridSearch por kernel en la variante A; el kernel que pasa a la comparación se elige por CV.",
    ("07", "3.1"): "Métricas por clase y matrices de confusión de los tres kernels en A.",
    ("07", "3.2"): "Por qué un kernel lineal compite cuando están Weight y Height: con logaritmos, los cortes del IMC son rectas.",
    ("07", "3.3"): "Forma de la frontera de cada kernel en el plano de las 2 primeras componentes principales. Es sólo visualización: su accuracy no es la del modelo.",
    ("07", "3.4"): "Cuántos puntos definen cada modelo elegido (vectores de soporte), por clase.",
    ("07", "4"): "Los tres kernels sin Weight ni Height.",
    ("07", "4.1"): "Métricas por clase y matrices de confusión de los tres kernels en B.",
    ("07", "5"): "Comparación de los kernels entre variantes.",
    ("07", "6"): "SVR con el target estandarizado (<code>TransformedTargetRegressor</code>), un GridSearch por kernel, baseline que predice la media y R² ajustado.",
    ("07", "6.1"): "Predicho frente a real, residuos y cuántas predicciones caen fuera del rango de pesos del train.",
    ("07", "7"): "Fila del mejor kernel de cada tarea para <code>data/resultados_entrega2.csv</code>.",
    ("08", "1"): "La validación para early stopping sale del train, nunca del test, y los escaladores se ajustan sólo con el sub-train. Se comprueba que no hay solapes.",
    ("08", "2"): "Entradas (una por atributo), salidas (7 softmax o 1 lineal), capas ocultas y número de parámetros de cada arquitectura. Se comprueba la reproducibilidad con la misma semilla.",
    ("08", "3"): "Las tres arquitecturas en la variante A, elegidas por la pérdida media de validación en 5 semillas. Curvas de pérdida train frente a validación por época.",
    ("08", "3.1"): "La red elegida en el test: modelo con semilla 42 y media ± desviación de las 5 semillas.",
    ("08", "4"): "Las tres arquitecturas sin Weight ni Height.",
    ("08", "4.1"): "La red elegida en B, en el test.",
    ("08", "5"): "Las tres arquitecturas en la regresión de Weight (pérdida MSE sobre el target estandarizado).",
    ("08", "5.1"): "La red elegida en el test, con las predicciones devueltas a kg, frente al baseline.",
    ("08", "6"): "Fila de cada tarea para <code>data/resultados_entrega2.csv</code>, con la media ± desviación de las 5 semillas.",
    ("09", "1"): "Antes de comparar se comprueba que dentro de cada tarea todos los modelos se evaluaron sobre el mismo test, y que los resultados congelados de la Entrega 1 no han cambiado.",
    ("09", "1.1"): "Contraste de los valores de referencia del encargo con los ejecutados.",
    ("09", "2"): "Las tres tablas finales, ordenadas por la métrica principal de cada tarea.",
    ("09", "3"): "Puesto de cada familia de modelos y gráfica comparativa.",
    ("09", "4"): "Coste y tamaño de los modelos de la Entrega 2: tiempo, parámetros, vectores de soporte y log-loss.",
    ("09", "5"): "Conclusiones generadas con las cifras de las tablas anteriores.",
}

# Líneas de la consola que no aportan al lector (rutas de archivos guardados, avisos de caché)
_RUIDO_CONSOLA = re.compile(r"^\s*(\[figura guardada\]|\[caché\]|\[calculando\]|\[recalculando\]|"
                            r"\(no se escribe)")


def e(texto) -> str:
    return html.escape(str(texto), quote=True)


def con_clases(texto_html: str) -> str:
    """Envuelve los nombres de clase en <code> (el JS les pone su punto de color)."""
    for c in sorted(CLASS_ORDER, key=len, reverse=True):
        texto_html = re.sub(rf"(?<![\w>]){c}(?![\w<])", f"<code>{c}</code>", texto_html)
    return texto_html


def slug(etapa, paso) -> str:
    return f"etapa{etapa}-p{paso.replace('.', '-')}"


# ----------------------------------------------------------------------------
# Lectura de las salidas
# ----------------------------------------------------------------------------
def bloques_md(ruta: Path) -> list:
    """Bloques del .md de un paso en orden: ('tabla', título, filas) o ('nota', texto).
    Se omiten las secciones 'Datos de la figura' (los datos de las gráficas)."""
    if not ruta.exists():
        return []
    bloques, modo, titulo, filas, nota = [], None, None, [], []

    def cerrar():
        nonlocal filas, nota
        if modo == "tabla" and filas:
            bloques.append(("tabla", titulo, filas))
        elif modo == "nota" and nota:
            bloques.append(("nota", "\n".join(nota).strip()))
        filas, nota = [], []

    for linea in ruta.read_text(encoding="utf-8").splitlines():
        if linea.startswith("# Etapa"):
            continue
        if linea.startswith("### "):
            cerrar()
            titulo = linea[4:].strip()
            modo = "omitir" if titulo.startswith("Datos de la figura") else "tabla"
        elif linea.strip() == "<!-- nota -->":
            cerrar()
            modo = "nota"
        elif modo == "tabla" and linea.startswith("|"):
            filas.append([c.strip() for c in linea.strip().strip("|").split("|")])
        elif modo == "nota":
            nota.append(linea)
    cerrar()
    return bloques


def tabla_html(titulo: str, filas: list) -> str:
    cab, cuerpo = filas[0], [f for f in filas[2:]]          # filas[1] es |:---|
    num = re.compile(r"^[−-]?[\d,]*\.?\d+(e[-+]?\d+)?$")

    def celda(v, tag):
        clase = ' class="num"' if tag == "td" and num.match(v.replace(" ", "")) else ""
        return f"<{tag}{clase}>{con_clases(e(v))}</{tag}>"

    h = "".join(celda(c, "th") for c in cab)
    b = "".join("<tr>" + "".join(celda(c, "td") for c in f) + "</tr>" for f in cuerpo)
    return (f'<p class="tab-title"><strong>{e(titulo)}</strong></p><div class="table-wrap">'
            f"<table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>")


def nota_html(texto: str) -> str:
    """Párrafos (separados por línea en blanco) y listas '- ' de una nota."""
    partes = []
    for parrafo in re.split(r"\n\s*\n", texto):
        lineas = [l.strip() for l in parrafo.splitlines() if l.strip()]
        if not lineas:
            continue
        if all(l.startswith("- ") for l in lineas):
            partes.append("<ul>" + "".join(f"<li>{con_clases(e(l[2:]))}</li>" for l in lineas) + "</ul>")
        else:
            partes.append(f"<p>{con_clases(e(' '.join(lineas)))}</p>")
    return '<div class="prose nota">' + "".join(partes) + "</div>"


def consola(ruta: Path) -> str:
    if not ruta.exists():
        return ""
    lineas = [l.rstrip() for l in ruta.read_text(encoding="utf-8").splitlines()
              if not _RUIDO_CONSOLA.match(l)]
    texto = re.sub(r"\n{3,}", "\n\n", "\n".join(lineas)).strip()
    return (f'<details class="texto-plano"><summary>Salida completa de consola del paso</summary>'
            f"<pre><code>{e(texto)}</code></pre></details>")


def figura(ruta: Path, grupo: str) -> str:
    rel = ruta.relative_to(RAIZ).as_posix()
    cap = ruta.stem.replace("_", " ")
    cap = re.sub(r"^(\d+(?:\.\d+)?) ", r"\1 · ", cap)
    return (f'<figure class="fig"><button type="button" class="fig-btn" data-src="{e(rel)}" '
            f'data-cap="{e(cap)}" data-group="{grupo}" aria-label="Ampliar {e(cap)}">'
            f'<img src="{e(rel)}" alt="{e(cap)}" loading="lazy"></button>'
            f"<figcaption>{e(cap)}</figcaption></figure>")


# ----------------------------------------------------------------------------
# Construcción de la página
# ----------------------------------------------------------------------------
def paso_html(etapa, p) -> str:
    carpeta = etapa.carpeta_figuras
    sid = slug(etapa.numero, p.id)
    figs = sorted((FIG / carpeta).glob(f"{p.id}_*.png"))
    tabs = sorted((FIG / carpeta / "tablas").glob(f"{p.id}_tabla*.png"))
    archivo, linea = (p.ubicacion.split(" ")[0].rsplit(":", 1) if p.ubicacion else ("", ""))
    badge = f'<span class="badge">{len(figs)} figura{"s" if len(figs) != 1 else ""}</span>' if figs else ""
    codigo = (f'<a class="codelink" href="{REPO_URL}/{archivo}#L{linea}" target="_blank" rel="noopener" '
              f'title="Abre {Path(archivo).name} en la línea {linea}. Ejecutar sólo este paso: '
              f'python {archivo} --paso {p.id}">Ver código<code>{Path(archivo).name}:{linea}</code></a>'
              if archivo else "")
    cuerpo = []
    idea = IDEA.get((etapa.numero, p.id))
    if idea:
        cuerpo.append(f'<div class="prose"><p>{idea}</p></div>')
    if figs:
        clase = "one" if len(figs) == 1 else ""
        cuerpo.append(f'<div class="figs {clase}">' + "".join(figura(f, sid) for f in figs) + "</div>")
    md = bloques_md(RES / carpeta / f"{p.id}.md")
    if md:
        partes = [tabla_html(b[1], b[2]) if b[0] == "tabla" else nota_html(b[1]) for b in md]
        cuerpo.append('<div class="prose">' + "".join(partes) + "</div>")
    cuerpo.append(consola(RES / carpeta / f"{p.id}_consola.txt"))
    if tabs:
        cuerpo.append(f'<details class="tabfigs"><summary>Tablas de este paso como imagen '
                      f'<span class="count">{len(tabs)}</span></summary><div class="figs mini">'
                      + "".join(figura(t, sid + "-t") for t in tabs) + "</div></details>")
    return (f'<article class="step" id="{sid}"><header class="step-head"><span class="step-id">{e(p.id)}</span>'
            f"<h3>{con_clases(e(p.titulo))}</h3>{badge}{codigo}"
            f'<button type="button" class="step-toggle" aria-expanded="true">Contraer</button></header>'
            f'<div class="step-body">{"".join(cuerpo)}</div></article>')


def seccion_html(etapa) -> str:
    sec = f"etapa{etapa.numero}"
    chips = "".join(f'<a class="chip" href="#{slug(etapa.numero, p.id)}"><b>{e(p.id)}</b> '
                    f"{e(p.titulo[:46])}</a>" for p in etapa.pasos)
    modulo = Path(etapa.pasos[0].ubicacion.split(":")[0]).as_posix()
    return (f'<section class="sec" id="{sec}" data-sec="{sec}"><header class="sec-head">'
            f'<p class="eyebrow">Etapa {etapa.numero}</p><h2>{e(etapa.nombre)}</h2></header>'
            f'<div class="prose intro">{INTRO[etapa.numero]}<p><strong>Archivo:</strong> <code>{modulo}</code>. '
            f"<strong>Figuras:</strong> <code>figures/{etapa.carpeta_figuras}/</code>. "
            f"<strong>Tablas y notas:</strong> <code>resultados_texto/{etapa.carpeta_figuras}/</code>.</p></div>"
            f'<nav class="chips" aria-label="Pasos de la etapa"><span class="chips-label">Ir al paso</span>{chips}</nav>'
            f'<div class="steps-tools"><button type="button" class="btn-ghost js-expand-all">Contraer todos los pasos</button></div>'
            + "".join(paso_html(etapa, p) for p in etapa.pasos)
            + f'<footer class="sec-foot"><button type="button" class="done-btn" data-sec="{sec}" aria-pressed="false">'
              f'<span class="tick" aria-hidden="true"></span><span class="lbl">Marcar sección como estudiada</span>'
              f"</button></footer></section>")


def kpis_html() -> str:
    """Indicadores de la portada, calculados con las tablas de la etapa 09."""
    from etapas import e09_comparacion_final as e09
    out = []
    for tarea, nombre in e09.TAREAS.items():
        t = e09.tabla_tarea(tarea)
        t = t[t["familia"] != "Baseline"]
        g = t.iloc[0]
        if tarea == "regresion":
            v, s = f"{g['RMSE']:.2f} kg", f"RMSE en test · R² {g['R2']:.3f}"
        else:
            v, s = f"{g['f1_macro']:.4f}", f"macro-F1 en test · accuracy {g['accuracy']:.4f}"
        out.append(f'<div class="kpi"><span class="k">{e(nombre)} · {e(g.name)}</span>'
                   f'<span class="v">{v}</span><span class="s">{e(s)}</span></div>')
    nn = [e09.tabla_tarea(t) for t in e09.TAREAS]
    puestos = [int(t[t["familia"] == "Red neuronal"]["puesto"].iloc[0]) for t in nn]
    totales = [len(t[t["familia"] != "Baseline"]) for t in nn]
    out.append('<div class="kpi"><span class="k">Red neuronal (Keras)</span>'
               f'<span class="v">{puestos[0]}.º · {puestos[1]}.º · {puestos[2]}.º</span>'
               f'<span class="s">puesto en A, B y regresión (de {totales[0]}, {totales[1]} y {totales[2]} modelos)</span></div>')
    return '<div class="kpis">' + "".join(out) + "</div>"


def pagina() -> str:
    etapas = list(main._cargar_etapas().values())
    css = (RAIZ / "estudio/estilo.css").read_text(encoding="utf-8")
    js = (RAIZ / "estudio/interaccion.js").read_text(encoding="utf-8")
    nav = "".join(f'<li><a href="#etapa{et.numero}" data-nav="etapa{et.numero}"><span class="dot" aria-hidden="true">'
                  f"</span>{et.numero} · {NAV[et.numero]}</a></li>" for et in etapas)
    css_extra = """
.tab-title{margin:18px 0 6px;font-size:14px}
td.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.prose.nota{border-left:3px solid var(--accent);padding:2px 0 2px 14px;margin:14px 0}
"""
    return f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Entrega 2 · Niveles de obesidad</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&family=JetBrains+Mono:wght@400;600&display=swap">
<style>
{css}{css_extra}</style>
</head>
<body>
<div class="topbar"><button type="button" class="btn-ghost" id="nav-open" aria-controls="side" aria-expanded="false">Secciones</button><b>Proyecto IA2 · Entrega 2</b><span class="count" id="top-prog">0/{len(etapas)}</span></div>
<div class="layout">
<aside class="side" id="side" aria-label="Secciones">
  <div class="brand"><b>Proyecto IA2 · Entrega 2</b><span>Estimación de niveles de obesidad · UCI 544</span></div>
  <div class="prog"><div class="bar"><span id="prog-bar"></span></div><span id="prog-txt">0 de {len(etapas)} secciones estudiadas</span></div>
  <nav><p class="nav-group">Etapas</p><ul>{nav}</ul></nav>
</aside>
<main>
<div class="wrap">
<header class="hero" id="top">
  <p class="eyebrow">Inteligencia Artificial 2 · Entrega 2 · UCI 544</p>
  <h1>Naive Bayes, SVM y redes neuronales, paso a paso</h1>
  <p class="lede">Cada paso de las etapas 06 a 09 con sus figuras, tablas y notas. Todas las cifras salen de ejecutar <code>python run_all.py</code>; esta página la escribe <code>python generar_estudio.py</code> a partir de esas salidas.</p>
  {kpis_html()}
  <ol class="route" aria-label="Ruta de estudio sugerida"><li>Los tres modelos nuevos: <a href="#etapa06">06</a>, <a href="#etapa07">07</a> y <a href="#etapa08">08</a>.</li><li>La comparación con la Entrega 1: <a href="#etapa09">09</a>.</li></ol>
</header>
{"".join(seccion_html(et) for et in etapas)}
</div>
</main>
</div>
<div class="lb" id="lb" hidden role="dialog" aria-modal="true" aria-label="Figura ampliada">
  <div class="lb-top"><span class="lb-cap" id="lb-cap"></span><button type="button" id="lb-zoom">Tamaño real</button><button type="button" id="lb-close">Cerrar</button></div>
  <div class="lb-stage" id="lb-stage"><img id="lb-img" alt=""></div>
  <div class="lb-bottom"><button type="button" class="lb-nav" id="lb-prev">← Anterior</button><span id="lb-pos"></span><button type="button" class="lb-nav" id="lb-next">Siguiente →</button></div>
</div>
<script>
{js}</script>
</body>
</html>
"""


if __name__ == "__main__":
    faltan = [f"{et.numero}/{p.id}" for et in main._cargar_etapas().values() for p in et.pasos
              if not (RES / et.carpeta_figuras / f"{p.id}_consola.txt").exists()]
    if faltan:
        sys.exit(f"Faltan salidas de {len(faltan)} pasos ({', '.join(faltan[:5])}…): "
                 "ejecuta antes python run_all.py")
    sin_idea = [k for k in ((et.numero, p.id) for et in main._cargar_etapas().values() for p in et.pasos)
                if k not in IDEA]
    if sin_idea:
        print(f"[aviso] pasos sin texto IDEA: {sin_idea}")
    SALIDA.write_text(pagina(), encoding="utf-8")
    print(f"Escrito {SALIDA.name} ({SALIDA.stat().st_size / 1024:.0f} KB).")
