"""Tests del agente que NO requieren Azure ni Tavily.

Lo que verificamos aquí es el *contrato* de las tools: nombre, schema y
descripción. Suena trivial, pero es literalmente lo único que el LLM ve
para decidir si una tool le sirve — si se rompe, el agente empieza a
elegir mal y es difícil de diagnosticar.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mexlex.config import settings
from mexlex.tools.article_tool import expandir_contexto, obtener_articulo
from mexlex.tools.legal_search_tool import buscar_en_leyes
from mexlex.tools.web_search_tool import get_web_search_tool


def test_tool_legal_expone_nombre_y_schema():
    assert buscar_en_leyes.name == "buscar_en_leyes"
    assert "consulta" in buscar_en_leyes.args


def test_tool_legal_documenta_cuando_usarla():
    # El docstring es el prompt que lee el modelo: debe decir para qué
    # sirve y exigir consultas autónomas.
    descripcion = buscar_en_leyes.description.lower()
    assert "ley" in descripcion
    assert "autónoma" in descripcion or "autonoma" in descripcion


@pytest.fixture
def sin_tavily(monkeypatch):
    """Simula un .env sin TAVILY_API_KEY.

    El cliente está cacheado con lru_cache, así que hay que limpiarlo o el
    resultado dependería de qué test corrió antes.
    """
    from mexlex.tools.web_search_tool import _cliente_tavily

    monkeypatch.setattr(settings, "tavily_api_key", None)
    _cliente_tavily.cache_clear()
    yield
    _cliente_tavily.cache_clear()


def test_web_search_deshabilitada_sin_api_key(sin_tavily):
    assert get_web_search_tool() is None


def test_get_tools_incluye_las_tres_del_corpus(sin_tavily):
    from mexlex.agent.graph import get_tools

    nombres = [t.name for t in get_tools()]
    assert nombres == ["buscar_en_leyes", "obtener_articulo", "expandir_contexto"]


async def test_web_search_avisa_en_vez_de_reventar_sin_key(sin_tavily):
    # Si el agente la llama sin key configurada, debe recibir un mensaje
    # accionable, no una excepción que tumbe el turno.
    from mexlex.tools.web_search_tool import buscar_en_web

    mensaje = await buscar_en_web.ainvoke(
        {
            "type": "tool_call",
            "id": "1",
            "name": "buscar_en_web",
            "args": {"consulta": "reforma cinematografía"},
        }
    )
    assert "no está disponible" in mensaje.content
    assert mensaje.artifact == []


# --- tools de acceso exacto (Fase 5) --------------------------------------


def test_tool_articulo_expone_numero_y_ley():
    assert obtener_articulo.name == "obtener_articulo"
    assert "numero" in obtener_articulo.args
    assert "ley" in obtener_articulo.args


def test_tool_articulo_dirige_al_agente_a_usarla_por_numero():
    # Sin esta instrucción el agente sigue yendo a buscar_en_leyes, que es
    # justo lo que no funciona para localizar un artículo.
    descripcion = obtener_articulo.description.lower()
    assert "número" in descripcion
    assert "siempre" in descripcion


def test_busqueda_tematica_desvia_los_lookups_por_numero():
    # La descripción de buscar_en_leyes debe mandar los casos por número
    # a obtener_articulo, o el agente elige mal.
    assert "obtener_articulo" in buscar_en_leyes.description


def test_tools_devuelven_fuentes_como_artifact():
    # El panel de fuentes de la UI depende de esto; si alguna tool
    # regresara solo texto, la UI se quedaría sin citas.
    for herramienta in (buscar_en_leyes, obtener_articulo, expandir_contexto):
        assert herramienta.response_format == "content_and_artifact"


async def test_expandir_contexto_rechaza_una_ref_invalida():
    # No debe reventar el turno del agente: regresa un mensaje utilizable.
    # Se invoca como tool_call (no como dict suelto) para recibir el
    # ToolMessage completo, con su artifact.
    mensaje = await expandir_contexto.ainvoke(
        {
            "type": "tool_call",
            "id": "1",
            "name": "expandir_contexto",
            "args": {"ref": "no-es-una-ref", "direccion": "siguiente", "cuantos": 1},
        }
    )
    assert "inválida" in mensaje.content.lower()
    assert mensaje.artifact == []
