"""Fase 5: esquema explícito del índice de Azure AI Search.

Hasta la Fase 4 dejábamos que `AzureSearch` creara el índice solo. Ese
esquema default guarda TODA la metadata serializada como JSON en un
único campo `metadata`, que **no es filtrable**: sirve para leerla
después de recuperar un documento, pero no para buscar por ella.

Declarando los campos a mano ganamos filtros OData reales:

    articulos/any(a: a eq 19)        -> el chunk que contiene el art. 19
    ley_id eq 'LFC' and chunk_index ge 7 and chunk_index le 9

`AzureSearch.add_embeddings` escribe en estos campos las llaves de
`metadata` cuyo nombre coincida con un campo declarado, así que basta
con que el splitter las ponga en el Document.

⚠️ El esquema solo se aplica al CREAR el índice. Apuntar a un índice ya
existente con campos nuevos no lo migra: hay que usar un nombre nuevo.
"""

from __future__ import annotations

from azure.search.documents.indexes.models import (
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SimpleField,
)

# Nombre del perfil de búsqueda vectorial que crea langchain_community.
# Tiene que coincidir exactamente o el índice se crea sin vectores.
VECTOR_PROFILE = "myHnswProfile"


def build_index_fields(vector_dimensions: int) -> list[SearchField]:
    """Campos del índice: los 4 obligatorios + los estructurales.

    Los primeros cuatro replican lo que arma `AzureSearch` por dentro; si
    se omite alguno, la librería lanza error al crear el índice.
    """
    return [
        # --- obligatorios para langchain_community.AzureSearch ---
        SimpleField(
            name="id",
            type=SearchFieldDataType.String,
            key=True,
            filterable=True,
        ),
        SearchableField(
            name="content",
            type=SearchFieldDataType.String,
        ),
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=vector_dimensions,
            vector_search_profile_name=VECTOR_PROFILE,
        ),
        SearchableField(
            name="metadata",
            type=SearchFieldDataType.String,
        ),
        # --- estructura legal (Fase 5) ---
        SearchableField(
            name="ley",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True,
        ),
        SimpleField(
            name="ley_id",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True,
        ),
        # Colección porque un chunk puede cubrir varios artículos
        # consecutivos (ej. [10, 11, 12]).
        SimpleField(
            name="articulos",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Int32),
            filterable=True,
            facetable=True,
        ),
        SimpleField(
            name="articulo_inicio",
            type=SearchFieldDataType.Int32,
            filterable=True,
            sortable=True,
        ),
        # Posición dentro de la ley: permite pedir chunks vecinos por rango.
        SimpleField(
            name="chunk_index",
            type=SearchFieldDataType.Int32,
            filterable=True,
            sortable=True,
        ),
        SimpleField(
            name="page",
            type=SearchFieldDataType.Int32,
            filterable=True,
            sortable=True,
        ),
        SimpleField(
            name="source",
            type=SearchFieldDataType.String,
            filterable=True,
        ),
        SimpleField(
            name="seccion",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True,
        ),
        SimpleField(
            name="ref",
            type=SearchFieldDataType.String,
            filterable=True,
        ),
    ]
