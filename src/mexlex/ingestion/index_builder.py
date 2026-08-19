"""Orquesta el pipeline completo de ingesta: PDFs -> chunks -> índice.

Este es el módulo que corre `scripts/run_ingestion.py`. Cada vez que
agregues un PDF nuevo a data/raw_pdfs/ y vuelvas a correr el script,
se añaden esos chunks al índice existente (no borra lo que ya había).
"""

from __future__ import annotations

import logging

from mexlex.config import RAW_PDFS_DIR, settings
from mexlex.ingestion.loaders import load_pdfs
from mexlex.ingestion.splitters import split_documents
from mexlex.retrieval.vectorstore import get_vectorstore

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def build_index() -> int:
    """Ejecuta el pipeline completo y regresa el número de chunks indexados."""
    logger.info("Cargando PDFs desde %s", RAW_PDFS_DIR)
    documents = load_pdfs(RAW_PDFS_DIR)
    logger.info("Total de páginas cargadas: %d", len(documents))

    chunks = split_documents(
        documents,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    logger.info("Total de chunks generados: %d", len(chunks))

    vectorstore = get_vectorstore()

    # Usamos chunk_id como key determinística: si vuelves a correr la
    # ingesta sobre el mismo PDF, se sobreescriben los chunks en vez de
    # duplicarse.
    keys = [chunk.metadata["chunk_id"] for chunk in chunks]
    vectorstore.add_documents(documents=chunks, keys=keys)

    logger.info(
        "Índice '%s' actualizado en Azure AI Search.", settings.azure_search_index_name
    )
    return len(chunks)


if __name__ == "__main__":
    build_index()
