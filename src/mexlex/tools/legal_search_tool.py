"""Fase 4-5: el retriever híbrido envuelto como `tool` del agente.

En las fases 1-3 el retrieval era un paso *fijo* del flujo: siempre se
ejecutaba, con la pregunta que le llegara. Como tool, pasa a ser una
capacidad que el LLM **decide** usar, y encima decide con qué consulta.

Lo más importante de este archivo no es el código, sino el **docstring**
de la función: ese texto es literalmente lo que el modelo lee para
decidir si esta tool le sirve. Un docstring vago = un agente que elige
mal.

Desde la Fase 5 esta tool es para preguntas TEMÁTICAS. Si el usuario
pide un artículo por número, `obtener_articulo` es exacta y esta no.
"""

from __future__ import annotations

from langchain_core.tools import tool

from mexlex.retrieval.formatting import format_result
from mexlex.retrieval.vectorstore import get_retriever


@tool(response_format="content_and_artifact")
async def buscar_en_leyes(consulta: str) -> tuple[str, list[dict]]:
    """Busca por TEMA en el corpus de leyes mexicanas indexadas.

    Úsala cuando la pregunta sea sobre un concepto, obligación, derecho,
    requisito o supuesto, y NO se sepa en qué artículo está. Por ejemplo:
    "qué dice la ley sobre precios por exhibición pública" o "qué
    obligaciones tiene el responsable del tratamiento de datos".

    ⚠️ Si el usuario menciona un artículo POR NÚMERO ("el artículo 19 de
    la Ley Federal de Cinematografía"), usa `obtener_articulo` en vez de
    esta: la búsqueda por similitud no distingue bien entre números de
    artículo y puede traerte el artículo equivocado.

    La consulta debe ser autónoma y descriptiva: usa las palabras del
    tema, no referencias al historial ("el anterior", "ese mismo").

    Args:
        consulta: El tema o pregunta legal a buscar, redactada de forma autónoma.

    Returns:
        Los fragmentos de ley más relevantes, cada uno precedido de su
        cita (ley, artículos, página y una `ref` para pedir más contexto).
    """
    docs = await get_retriever().ainvoke(consulta)
    return format_result(
        docs,
        vacio=(
            "No se encontraron fragmentos relevantes en el corpus indexado "
            "para esa consulta. Prueba con otros términos o, si conoces el "
            "número de artículo, usa obtener_articulo."
        ),
    )
