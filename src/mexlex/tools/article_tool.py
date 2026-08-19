"""Fase 5: tools de acceso exacto al articulado.

`buscar_en_leyes` resuelve preguntas temáticas, pero falla al pedirle un
artículo por número: para los embeddings "artículo 19" y "artículo 18"
son casi idénticos, y para BM25 la palabra "artículo" no discrimina nada
porque está en todos los chunks.

Estas dos tools no buscan: consultan por filtro exacto sobre la metadata
estructural del índice (ver `retrieval/lookup.py`). Son deterministas.
"""

from __future__ import annotations

import logging

from langchain_core.tools import tool

from mexlex.retrieval import lookup
from mexlex.retrieval.formatting import format_result

logger = logging.getLogger(__name__)


@tool(response_format="content_and_artifact")
async def obtener_articulo(numero: int, ley: str = "") -> tuple[str, list[dict]]:
    """Devuelve el texto EXACTO de un artículo por su número.

    Úsala SIEMPRE que la pregunta mencione un artículo por número, aunque
    también mencione un tema. Es exacta: no depende de la búsqueda por
    similitud, que suele traer el artículo equivocado en estos casos.

    Ejemplos de cuándo usarla:
    - "¿de qué habla el artículo 19 de la Ley Federal de Cinematografía?"
    - "¿y el siguiente artículo?" (resuelve tú el número: sería el 20)
    - "explícame el artículo 47"

    Si un artículo es largo y ocupa varios fragmentos, los devuelve todos
    en orden.

    Args:
        numero: El número del artículo (solo el dígito, ej. 19).
        ley: Identificador corto de la ley para desambiguar, ej. "LFC" o
            "LFPDPPP". Déjalo vacío si el usuario no especificó cuál; en
            ese caso se devuelve ese artículo de TODAS las leyes
            indexadas y tú eliges o preguntas cuál quería.

    Returns:
        El texto del artículo con su cita (ley, página y `ref`), o un
        aviso si ese artículo no existe en el corpus indexado.
    """
    ley_id = (ley or "").strip().upper() or None

    try:
        docs = await _en_hilo(lookup.obtener_articulo, numero, ley_id)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Fallo el lookup del artículo %s", numero)
        return f"Error consultando el artículo {numero}: {exc}", []

    return format_result(
        docs,
        vacio=(
            f"El artículo {numero}"
            + (f" de {ley_id}" if ley_id else "")
            + " no existe en el corpus indexado. Verifica el número, o "
            "consulta qué leyes están disponibles."
        ),
    )


@tool(response_format="content_and_artifact")
async def expandir_contexto(
    ref: str, direccion: str = "siguiente", cuantos: int = 1
) -> tuple[str, list[dict]]:
    """Trae los fragmentos vecinos a uno que ya recuperaste.

    Úsala cuando un fragmento se ve cortado a media frase, o cuando
    necesitas lo que viene inmediatamente antes o después en la ley
    (por ejemplo, las fracciones de un artículo largo).

    La `ref` viene en la cita de cada fragmento, con formato "LFC#12".

    Args:
        ref: La referencia del fragmento de partida, ej. "LFC#12".
        direccion: "siguiente", "anterior" o "ambos".
        cuantos: Cuántos fragmentos traer de ese lado (1 o 2 basta).

    Returns:
        Los fragmentos vecinos con su cita, o un aviso si la referencia
        no es válida.
    """
    try:
        ley_id, indice = lookup.parsear_ref(ref)
    except ValueError as exc:
        return str(exc), []

    cuantos = max(1, min(int(cuantos), 3))
    direccion = (direccion or "siguiente").strip().lower()

    if direccion == "anterior":
        desde, hasta = indice - cuantos, indice - 1
    elif direccion == "ambos":
        desde, hasta = indice - cuantos, indice + cuantos
    else:
        desde, hasta = indice + 1, indice + cuantos

    desde = max(0, desde)
    if hasta < desde:
        return f"No hay fragmentos {direccion} de {ref}.", []

    try:
        docs = await _en_hilo(lookup.obtener_rango, ley_id, desde, hasta)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Falló la expansión de contexto de %s", ref)
        return f"Error expandiendo el contexto de {ref}: {exc}", []

    return format_result(docs, vacio=f"No hay fragmentos {direccion} de {ref}.")


async def _en_hilo(func, *args):
    """Corre una función bloqueante sin trabar el event loop.

    El SDK de Azure Search que usamos aquí es síncrono; sin esto, cada
    lookup bloquearía el servidor de Chainlit para todos los usuarios.
    """
    import asyncio

    return await asyncio.to_thread(func, *args)
