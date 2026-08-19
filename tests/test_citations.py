"""Tests de citas, vigencia y evaluación (Fase 6). Sin Azure."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from langchain_core.documents import Document

from mexlex.evaluation import _evaluar
from mexlex.ingestion.structure import detectar_vigencia
from mexlex.retrieval.formatting import (
    cita_de,
    fuente_de,
    nombre_legible,
    titulo_de,
)


def _doc(**metadata) -> Document:
    base = {
        "ley": "LEY FEDERAL DE CINEMATOGRAFÍA",
        "ley_id": "LFC",
        "articulos": [19, 20],
        "page": 3,
        "ref": "LFC#12",
        "vigencia": "22-03-2021",
    }
    base.update(metadata)
    return Document(page_content="texto del artículo", metadata=base)


# --- nombres legibles ------------------------------------------------------


def test_normaliza_el_nombre_en_mayusculas():
    # Los PDFs oficiales traen los títulos gritados; citarlos así se lee mal.
    assert (
        nombre_legible("LEY FEDERAL DE CINEMATOGRAFÍA")
        == "Ley Federal de Cinematografía"
    )


def test_no_toca_un_nombre_ya_formateado():
    assert nombre_legible("Ley Federal del Trabajo") == "Ley Federal del Trabajo"


def test_la_primera_palabra_siempre_va_capitalizada():
    assert nombre_legible("DEL TRABAJO Y PREVISIÓN").startswith("Del ")


# --- vigencia --------------------------------------------------------------


def test_detecta_la_fecha_de_ultima_reforma():
    paginas = ["LEY FEDERAL DE CINEMATOGRAFÍA\nÚltima Reforma DOF 22-03-2021\n"]
    assert detectar_vigencia(paginas) == "22-03-2021"


def test_detecta_la_variante_publicada():
    paginas = ["Última reforma publicada DOF 14-11-2025\n"]
    assert detectar_vigencia(paginas) == "14-11-2025"


def test_sin_fecha_regresa_none():
    assert detectar_vigencia(["un documento sin encabezado de reforma"]) is None


# --- citas -----------------------------------------------------------------


def test_la_cita_del_llm_incluye_vigencia_y_ref():
    # El LLM necesita la vigencia para poder advertir, y la ref para
    # poder pedir contexto vecino.
    cita = cita_de(_doc())
    assert "vigente al 22-03-2021" in cita
    assert "ref LFC#12" in cita


def test_el_titulo_del_usuario_no_expone_la_ref_interna():
    titulo = titulo_de(_doc())
    assert "ref" not in titulo
    assert titulo == "Ley Federal de Cinematografía · arts. 19-20 · p. 3"


def test_articulo_unico_se_cita_en_singular():
    assert "art. 19" in titulo_de(_doc(articulos=[19]))


def test_seccion_sin_articulos_se_cita_por_su_nombre():
    assert "transitorios" in titulo_de(_doc(articulos=[], seccion="TRANSITORIOS"))


def test_la_fuente_marca_su_origen():
    # La UI usa esto para advertir cuando la respuesta salió a internet.
    assert fuente_de(_doc())["origen"] == "corpus"


# --- evaluadores -----------------------------------------------------------


def test_evaluador_detecta_tool_incorrecta():
    caso = {"tool_esperada": "obtener_articulo"}
    fallos = _evaluar(caso, "respuesta", ["buscar_en_leyes"], [])
    assert any("obtener_articulo" in f for f in fallos)


def test_evaluador_detecta_articulo_no_recuperado():
    caso = {"articulo_esperado": 19}
    fallos = _evaluar(caso, "respuesta", ["obtener_articulo"], [18, 20])
    assert any("19" in f for f in fallos)


def test_evaluador_acepta_caso_correcto():
    caso = {"tool_esperada": "obtener_articulo", "articulo_esperado": 19}
    assert _evaluar(caso, "el artículo 19 dice", ["obtener_articulo"], [19, 20]) == []


def test_evaluador_detecta_alucinacion():
    # Ante un artículo inexistente, inventar contenido es el peor fallo
    # posible en este dominio.
    caso = {"debe_admitir_ignorancia": True}
    inventado = "El artículo 999 establece que los exhibidores deberán..."
    assert _evaluar(caso, inventado, ["obtener_articulo"], []) != []

    honesto = "No encontré el artículo 999 en el corpus indexado."
    assert _evaluar(caso, honesto, ["obtener_articulo"], []) == []


def test_evaluador_acepta_parafraseo_con_alternativas():
    # "diez por ciento" y "10%" son la misma respuesta; el evaluador no
    # debe reportar un fallo por la forma de escribirlo.
    caso = {"debe_contener": [["diez por ciento", "10%"]]}
    assert _evaluar(caso, "reservarán el 10% del tiempo", [], []) == []
    assert _evaluar(caso, "reservarán el diez por ciento", [], []) == []
    assert _evaluar(caso, "reservarán una parte del tiempo", [], []) != []


def test_evaluador_exige_cero_tools_cuando_se_pide():
    caso = {"tools_esperadas": []}
    assert _evaluar(caso, "con gusto", [], []) == []
    assert _evaluar(caso, "con gusto", ["buscar_en_leyes"], []) != []
