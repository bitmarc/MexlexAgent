"""Carga de documentos PDF desde disco.

Empezamos con PyPDFLoader por simplicidad (funciona bien con PDFs de texto
nativo, como la mayoría de publicaciones oficiales de leyes mexicanas).
Si más adelante trabajas con PDFs escaneados o con tablas complejas, este
es el módulo que reemplazarías por Azure AI Document Intelligence, sin
tocar el resto del pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def load_pdfs(pdf_dir: Path) -> list[Document]:
    """Carga todos los PDFs de un directorio y regresa una lista de Document.

    Cada Document conserva metadata útil para citar la fuente después:
    - source: nombre del archivo (ej. "cpeum.pdf")
    - page: número de página (agregado automáticamente por PyPDFLoader)
    """
    pdf_dir = Path(pdf_dir)
    pdf_paths = sorted(pdf_dir.glob("*.pdf"))

    if not pdf_paths:
        raise FileNotFoundError(
            f"No se encontraron PDFs en {pdf_dir}. "
            "Coloca al menos un archivo (ej. la CPEUM) ahí antes de indexar."
        )

    documents: list[Document] = []
    for pdf_path in pdf_paths:
        logger.info("Cargando %s", pdf_path.name)
        loader = PyPDFLoader(str(pdf_path))
        pdf_documents = loader.load()

        # Normalizamos la metadata: queremos "source" limpio (solo el
        # nombre del archivo) para poder mostrarlo como cita al usuario.
        for doc in pdf_documents:
            doc.metadata["source"] = pdf_path.name

        documents.extend(pdf_documents)
        logger.info("  -> %d páginas cargadas", len(pdf_documents))

    return documents
