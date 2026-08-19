import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from langchain_core.documents import Document

from mexlex.ingestion.splitters import split_documents


def test_split_documents_assigns_chunk_id():
    docs = [
        Document(
            page_content="Artículo 1. Texto de prueba. " * 50,
            metadata={"source": "test.pdf", "page": 0},
        )
    ]
    chunks = split_documents(docs, chunk_size=200, chunk_overlap=20)

    assert len(chunks) > 1
    for chunk in chunks:
        assert "chunk_id" in chunk.metadata
        assert chunk.metadata["source"] == "test.pdf"


def test_metadata_estructural_para_el_indice():
    # Los campos que se vuelven filtrables en Azure AI Search.
    docs = [
        Document(
            page_content=(
                "ARTICULO 18.- Primera disposición.\n"
                "ARTICULO 19.- Los exhibidores reservarán el diez por ciento.\n"
            ),
            metadata={"source": "LFC.pdf", "page": 0},
        )
    ]
    chunks = split_documents(docs, chunk_size=1200, chunk_overlap=0)

    chunk = chunks[0]
    assert chunk.metadata["ley_id"] == "LFC"
    assert chunk.metadata["articulos"] == [18, 19]
    assert chunk.metadata["articulo_inicio"] == 18
    assert chunk.metadata["chunk_index"] == 0
    assert chunk.metadata["ref"] == "LFC#0"
    assert chunk.metadata["page"] == 0


def test_chunk_id_is_valid_azure_search_key():
    # Azure AI Search rechaza keys con puntos, espacios o acentos.
    docs = [
        Document(
            page_content="Artículo 1. Texto de prueba. " * 50,
            metadata={"source": "Ley Federal de Competencia (2024).pdf", "page": 3},
        )
    ]
    chunks = split_documents(docs, chunk_size=200, chunk_overlap=20)

    for chunk in chunks:
        assert re.fullmatch(r"[A-Za-z0-9_\-=]+", chunk.metadata["chunk_id"])


def test_chunk_ids_are_stable_per_document():
    # Agregar un PDF nuevo no debe recorrer los ids de los ya indexados.
    doc_a = Document(page_content="Artículo 1. " * 100, metadata={"source": "a.pdf", "page": 0})
    doc_b = Document(page_content="Artículo 2. " * 100, metadata={"source": "b.pdf", "page": 0})

    solo = [c.metadata["chunk_id"] for c in split_documents([doc_b], chunk_size=200)]
    con_vecino = [
        c.metadata["chunk_id"]
        for c in split_documents([doc_a, doc_b], chunk_size=200)
        if c.metadata["source"] == "b.pdf"
    ]

    assert solo == con_vecino


def test_split_documents_respects_chunk_size_roughly():
    docs = [
        Document(
            page_content="Palabra " * 500,
            metadata={"source": "test.pdf", "page": 0},
        )
    ]
    chunks = split_documents(docs, chunk_size=300, chunk_overlap=0)

    # Ningún chunk debería exceder por mucho el tamaño configurado
    assert all(len(c.page_content) <= 400 for c in chunks)
