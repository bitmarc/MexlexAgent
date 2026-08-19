"""Fase 5: consultas exactas por filtro (sin búsqueda por similitud).

Buscar "el artículo 19" es un problema de *lookup*, no de *similitud*:

- Para los embeddings, "artículo 19" y "artículo 18" son casi el mismo
  vector; el número apenas mueve la representación.
- Para BM25, "artículo" aparece en todos los chunks (no discrimina) y el
  "19" compite con fechas, fracciones y referencias cruzadas.

Por eso aquí no usamos vectores ni el semantic ranker: se consulta el
`SearchClient` con `search_text="*"` y un filtro OData sobre los campos
estructurales del índice. Es exacto, determinista y barato (no gasta
embeddings ni cuota de semantic ranker).
"""

from __future__ import annotations

import json
import logging

from langchain_core.documents import Document

from mexlex.config import settings
from mexlex.retrieval.vectorstore import get_vectorstore

logger = logging.getLogger(__name__)

# Campos que pedimos de vuelta. Excluimos content_vector: son miles de
# floats que no sirven de nada del lado del cliente.
CAMPOS = [
    "id",
    "content",
    "metadata",
    "ley",
    "ley_id",
    "articulos",
    "articulo_inicio",
    "chunk_index",
    "page",
    "source",
    "seccion",
    "ref",
]


def escapar_odata(valor: str) -> str:
    """Escapa una cadena para interpolarla en un filtro OData.

    En OData las comillas simples se escapan duplicándolas. Sin esto, un
    ley_id con apóstrofo rompería el filtro.
    """
    return valor.replace("'", "''")


def filtro_por_articulo(numero: int, ley_id: str | None = None) -> str:
    """Filtro OData para el chunk que contiene un artículo.

    `articulos` es una colección, así que se consulta con `any`. Un chunk
    que cubre los artículos 10-12 matchea para cualquiera de los tres.
    """
    filtro = f"articulos/any(a: a eq {int(numero)})"
    if ley_id:
        filtro = f"{filtro} and ley_id eq '{escapar_odata(ley_id)}'"
    return filtro


def filtro_por_rango(ley_id: str, desde: int, hasta: int) -> str:
    """Filtro OData para un rango de chunks contiguos de una misma ley."""
    return (
        f"ley_id eq '{escapar_odata(ley_id)}' "
        f"and chunk_index ge {int(desde)} and chunk_index le {int(hasta)}"
    )


def parsear_ref(ref: str) -> tuple[str, int]:
    """Interpreta una referencia "LFC#7" -> ("LFC", 7).

    Es el identificador que ve el agente en cada fragmento y el que le
    pasa a `expandir_contexto`.
    """
    texto = ref.strip().upper()
    if "#" not in texto:
        raise ValueError(f"Referencia inválida: {ref!r}. Formato esperado: LEY#7")

    ley_id, _, indice = texto.partition("#")
    if not ley_id or not indice.isdigit():
        raise ValueError(f"Referencia inválida: {ref!r}. Formato esperado: LEY#7")

    return ley_id, int(indice)


def _a_documento(resultado: dict) -> Document:
    """Convierte un resultado crudo de Azure Search en un Document."""
    # El campo `metadata` trae el dict completo serializado; los campos
    # sueltos son la versión filtrable. Partimos del JSON y dejamos que
    # los campos declarados manden.
    try:
        metadata = json.loads(resultado.get("metadata") or "{}")
    except json.JSONDecodeError:
        metadata = {}

    for campo in CAMPOS:
        if campo in ("content", "metadata"):
            continue
        if resultado.get(campo) is not None:
            metadata[campo] = resultado[campo]

    return Document(page_content=resultado.get("content", ""), metadata=metadata)


def _buscar(filtro: str, limite: int) -> list[Document]:
    """Ejecuta una consulta solo-filtro, ordenada por posición en la ley."""
    cliente = get_vectorstore().client
    resultados = cliente.search(
        search_text="*",  # sin término: no hay scoring ni cargo de semantic
        filter=filtro,
        select=CAMPOS,
        order_by=["chunk_index asc"],
        top=limite,
    )
    return [_a_documento(r) for r in resultados]


def obtener_articulo(numero: int, ley_id: str | None = None) -> list[Document]:
    """Devuelve los chunks que contienen ese artículo, en orden.

    Un artículo largo puede ocupar varios chunks: se devuelven todos y
    ordenados, para que el agente vea la disposición completa.
    """
    docs = _buscar(
        filtro_por_articulo(numero, ley_id), limite=settings.lookup_max_results
    )
    logger.info("lookup artículo %s (ley=%s): %d chunks", numero, ley_id, len(docs))
    return docs


def obtener_rango(ley_id: str, desde: int, hasta: int) -> list[Document]:
    """Devuelve chunks contiguos por posición dentro de una ley."""
    if desde > hasta:
        desde, hasta = hasta, desde
    return _buscar(
        filtro_por_rango(ley_id, desde, hasta), limite=settings.lookup_max_results
    )


def leyes_indexadas() -> list[tuple[str, str]]:
    """Lista (ley_id, nombre) de las leyes presentes en el índice.

    Sirve para que el agente sepa qué hay en el corpus en vez de adivinar.
    """
    cliente = get_vectorstore().client
    resultados = cliente.search(
        search_text="*",
        select=["ley_id", "ley"],
        facets=["ley_id"],
        top=0,
    )

    vistos: dict[str, str] = {}
    for faceta in (resultados.get_facets() or {}).get("ley_id", []):
        vistos[faceta["value"]] = faceta["value"]

    # El facet solo da el id; recuperamos un documento por ley para el
    # nombre legible.
    for ley_id in list(vistos):
        muestra = _buscar(f"ley_id eq '{escapar_odata(ley_id)}'", limite=1)
        if muestra:
            vistos[ley_id] = muestra[0].metadata.get("ley", ley_id)

    return sorted(vistos.items())
