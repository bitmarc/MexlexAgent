"""Vector store de Azure AI Search: embeddings + búsqueda híbrida.

`AzureSearch` de langchain_community crea el índice automáticamente la
primera vez que instancias la clase (usa la dimensión del modelo de
embeddings para definir el campo vectorial). No necesitas crear el índice
a mano en el portal de Azure.

search_type controla qué tipo de búsqueda hace el retriever:
- "similarity"      -> solo vectorial
- "hybrid"          -> vectorial + BM25 (keyword), combinados con RRF
- "semantic_hybrid" -> hybrid + semantic reranker de Azure AI Search
                        (requiere tener configurado un semantic config
                        y un tier Basic o superior del servicio)
"""

from __future__ import annotations

from functools import lru_cache

from langchain_community.vectorstores import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings

from mexlex.config import settings
from mexlex.retrieval.schema import build_index_fields


@lru_cache(maxsize=1)
def get_embeddings() -> AzureOpenAIEmbeddings:
    return AzureOpenAIEmbeddings(
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
        azure_deployment=settings.azure_openai_embedding_deployment,
    )


@lru_cache(maxsize=1)
def get_embedding_dimensions() -> int:
    """Dimensión del vector del modelo de embeddings configurado.

    Se detecta consultando al modelo (una llamada, cacheada) para que el
    esquema del índice no dependa de recordar si el deployment es
    text-embedding-3-large (3072) o -small (1536). Se puede fijar en el
    .env con EMBEDDING_DIMENSIONS para evitar esa llamada.
    """
    if settings.embedding_dimensions:
        return settings.embedding_dimensions
    return len(get_embeddings().embed_query("Text"))


@lru_cache(maxsize=1)
def get_vectorstore() -> AzureSearch:
    return AzureSearch(
        azure_search_endpoint=settings.azure_search_endpoint,
        azure_search_key=settings.azure_search_api_key,
        index_name=settings.azure_search_index_name,
        embedding_function=get_embeddings().embed_query,
        semantic_configuration_name=settings.azure_search_semantic_config_name,
        # Esquema explícito con campos filtrables (Fase 5). Solo tiene
        # efecto al CREAR el índice; sobre uno existente se ignora.
        fields=build_index_fields(get_embedding_dimensions()),
    )


def get_retriever(k: int | None = None):
    """Regresa un retriever configurado en modo híbrido.

    Con AZURE_SEARCH_USE_SEMANTIC_RANKER=true usa semantic_hybrid (reranker
    de Azure, mejor para preguntas temáticas). Si no, hybrid (vector + BM25).

    Ojo: el semantic ranker NO ayuda a encontrar un artículo por número —
    solo reordena los top 50 que ya devolvió el ranking base. Para eso está
    el lookup exacto por filtro (retrieval/lookup.py).
    """
    search_type = (
        "semantic_hybrid" if settings.azure_search_use_semantic_ranker else "hybrid"
    )
    # Ojo: a diferencia de otros vector stores, AzureSearchVectorStoreRetriever
    # expone `k` como campo propio y además expande `search_kwargs` al llamar
    # a la búsqueda. Si mandamos `k` dentro de search_kwargs, se pasa dos veces
    # y truena con "got multiple values for keyword argument 'k'".
    return get_vectorstore().as_retriever(
        search_type=search_type,
        k=k or settings.retrieval_k,
    )
