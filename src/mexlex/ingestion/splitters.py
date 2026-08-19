"""División de documentos en chunks para indexar.

A partir de la Fase 5 el chunking es **consciente de la estructura**: en
vez de cortar cada página por tamaño, se detectan las fronteras de
artículo (ver `structure.py`) y se empacan artículos consecutivos sin
partir ninguno.

Este módulo orquesta ese pipeline y produce los `Document` finales con la
metadata que después se guarda como campos filtrables en Azure AI Search:

    ley, ley_id, articulos, articulo_inicio, chunk_index, page, source

Esa metadata es lo que permite que "dame el artículo 19" sea un filtro
exacto en vez de una búsqueda por similitud.
"""

from __future__ import annotations

import re
import unicodedata
from collections import OrderedDict
from pathlib import Path

from langchain_core.documents import Document

from mexlex.ingestion.structure import (
    construir_fragmentos,
    detectar_nombre_ley,
    detectar_vigencia,
    limpiar_paginas,
    paginas_de,
)

# Azure AI Search solo acepta letras, dígitos, guion bajo, guion y signo
# de igual en la key de un documento. Todo lo demás (puntos, espacios,
# acentos) lo normalizamos antes de construir el chunk_id.
_INVALID_KEY_CHARS = re.compile(r"[^A-Za-z0-9_\-=]")


def _sanitize_key_part(value: str) -> str:
    """Convierte un texto arbitrario en algo usable como key de Azure Search."""
    ascii_value = (
        unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    )
    return _INVALID_KEY_CHARS.sub("_", ascii_value)


def ley_id_de(source: str) -> str:
    """Identificador corto y estable de la ley, derivado del archivo.

    "LFC.pdf" -> "LFC". Se usa en los filtros y en las `ref` que el agente
    manda a `expandir_contexto`, así que conviene que sea corto.
    """
    return _sanitize_key_part(Path(source).stem).upper()


def _agrupar_por_documento(
    documents: list[Document],
) -> "OrderedDict[str, list[Document]]":
    """Agrupa las páginas por PDF de origen, conservando su orden.

    `load_pdfs` devuelve una lista plana de páginas de todos los PDFs. La
    estructura (artículos, encabezado corrido) solo tiene sentido dentro
    de UN documento, así que hay que reagrupar antes de parsear.
    """
    grupos: OrderedDict[str, list[Document]] = OrderedDict()
    for doc in documents:
        grupos.setdefault(doc.metadata.get("source", "doc"), []).append(doc)
    return grupos


def split_documents(
    documents: list[Document],
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
) -> list[Document]:
    """Convierte páginas de PDF en chunks con estructura legal."""
    chunks: list[Document] = []

    for source, paginas_doc in _agrupar_por_documento(documents).items():
        crudas = paginas_de(paginas_doc)
        ley = detectar_nombre_ley(crudas, respaldo=Path(source).stem)
        ley_id = ley_id_de(source)
        # Se lee ANTES de limpiar: la fecha vive en el encabezado corrido
        # que estamos por quitar.
        vigencia = detectar_vigencia(crudas)

        # Quitar el encabezado corrido es importante por dos razones: son
        # ~200 caracteres de ruido en cada página que diluyen el embedding,
        # y se cuelan a media frase cuando un artículo cruza de página.
        fragmentos = construir_fragmentos(
            limpiar_paginas(crudas), chunk_size, chunk_overlap
        )

        for i, frag in enumerate(fragmentos):
            metadata = {
                "source": source,
                "ley": ley,
                "ley_id": ley_id,
                "articulos": frag.articulos,
                # articulo_inicio permite ordenar y mostrar "art. 19" sin
                # tener que leer la colección completa.
                "articulo_inicio": frag.articulos[0] if frag.articulos else -1,
                "chunk_index": i,
                "page": frag.pagina,
                # chunk_id es la key en Azure: determinística, así que
                # reindexar sobreescribe en vez de duplicar.
                "chunk_id": f"{ley_id}-c{i:04d}",
                # ref es lo que ve el agente y lo que le pasa a
                # expandir_contexto para pedir los chunks vecinos.
                "ref": f"{ley_id}#{i}",
            }
            if frag.seccion:
                metadata["seccion"] = frag.seccion
            if vigencia:
                # Va solo en el JSON de metadata, no como campo del índice:
                # se usa para mostrar y advertir, nunca para filtrar, así
                # que no requiere recrear el índice.
                metadata["vigencia"] = vigencia

            chunks.append(Document(page_content=frag.texto, metadata=metadata))

    return chunks
