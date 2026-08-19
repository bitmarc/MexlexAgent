"""Fase 4-6: búsqueda web (Tavily) como tool de respaldo.

Sirve para lo que el corpus indexado no puede cubrir: reformas
recientes, noticias jurídicas, o leyes que simplemente no has subido a
`data/raw_pdfs/`.

Es **opcional**: si no hay `TAVILY_API_KEY` en el .env, `get_web_search_tool()`
regresa None y el agente se arma solo con las tools del corpus. Así el
proyecto corre completo sin obligarte a dar de alta otra cuenta.

Para habilitarla: consigue una API key gratuita en https://tavily.com
y agrégala al .env como TAVILY_API_KEY.

Desde la Fase 6 no exponemos `TavilySearch` directamente, sino una tool
propia que la envuelve. El motivo son las citas: `TavilySearch` usa
`response_format="content"`, así que sus resultados llegan al modelo
como texto plano y **no producen `artifact`**. Sin eso, las fuentes web
nunca aparecerían en el panel de la UI ni podrían marcarse como "de
internet, no del corpus oficial".

Nota histórica: el plan original consideró la API de Bing Search de
Azure, pero fue retirada en agosto de 2025. Tavily es hoy el estándar
de facto en LangChain para esto.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from langchain_core.tools import tool
from langchain_tavily import TavilySearch

from mexlex.config import settings

logger = logging.getLogger(__name__)

# Priorizamos fuentes oficiales mexicanas: reduce el ruido de blogs y
# despachos que reinterpretan la ley.
DOMINIOS_OFICIALES = [
    "legislacion.edomex.gob.mx",
    "diputados.gob.mx",
    "dof.gob.mx",
    "scjn.gob.mx",
    "gob.mx",
]


@lru_cache(maxsize=1)
def _cliente_tavily() -> TavilySearch | None:
    """Cliente de Tavily, o None si no hay API key configurada."""
    if not settings.tavily_api_key:
        logger.info(
            "TAVILY_API_KEY no configurada: el agente correrá sin búsqueda web."
        )
        return None

    return TavilySearch(
        tavily_api_key=settings.tavily_api_key,
        max_results=settings.web_search_max_results,
        topic="general",
        include_domains=DOMINIOS_OFICIALES,
    )


@tool(response_format="content_and_artifact")
async def buscar_en_web(consulta: str) -> tuple[str, list[dict]]:
    """Busca información jurídica actual en internet (fuentes oficiales mexicanas).

    Úsala SOLO cuando:
    - las tools del corpus no encontraron la información, o
    - la pregunta es sobre reformas recientes, noticias o jurisprudencia
      que un PDF estático no contendría.

    NO la uses como primera opción para preguntas sobre el contenido de
    una ley: para eso están `obtener_articulo` y `buscar_en_leyes`, que
    consultan los documentos oficiales indexados y son más confiables.

    Cuando uses esta tool, **avisa siempre** al usuario que esa parte de
    la respuesta viene de internet y no del corpus oficial.

    Args:
        consulta: Qué buscar, redactado de forma autónoma y específica.

    Returns:
        Los resultados con su título y URL para poder citarlos.
    """
    cliente = _cliente_tavily()
    if cliente is None:
        return (
            "La búsqueda web no está disponible (falta TAVILY_API_KEY). "
            "Responde solo con el corpus indexado o admite que no lo tienes.",
            [],
        )

    try:
        datos = await cliente.ainvoke({"query": consulta})
    except Exception as exc:  # noqa: BLE001
        logger.exception("Falló la búsqueda web")
        return f"Error consultando la búsqueda web: {exc}", []

    resultados = datos.get("results", []) if isinstance(datos, dict) else []
    if not resultados:
        return "La búsqueda web no devolvió resultados para esa consulta.", []

    partes, fuentes = [], []
    for r in resultados:
        titulo = r.get("title") or "(sin título)"
        url = r.get("url", "")
        contenido = r.get("content", "")

        partes.append(f"[WEB · {titulo} · {url}]\n{contenido}")
        fuentes.append(
            {
                # La UI usa esto para advertir que no viene del corpus.
                "origen": "web",
                "titulo": f"🌐 {titulo}",
                "url": url,
                "texto": f"{contenido}\n\n{url}",
                "cita": f"{titulo} ({url})",
            }
        )

    return "\n\n---\n\n".join(partes), fuentes


def get_web_search_tool():
    """Regresa la tool de búsqueda web, o None si no hay API key.

    Devolver None (en vez de lanzar error) es deliberado: permite que el
    agente se arme con las tools que sí estén disponibles.
    """
    if _cliente_tavily() is None:
        return None
    return buscar_en_web
