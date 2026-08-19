"""Configuración centralizada del proyecto.

Todas las credenciales y nombres de recursos de Azure viven aquí, leídos
desde variables de entorno (.env en desarrollo local). Ningún otro módulo
debería leer os.environ directamente: todos importan `settings` desde aquí.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PDFS_DIR = PROJECT_ROOT / "data" / "raw_pdfs"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Azure OpenAI
    azure_openai_endpoint: str = Field(..., alias="AZURE_OPENAI_ENDPOINT")
    azure_openai_api_key: str = Field(..., alias="AZURE_OPENAI_API_KEY")
    azure_openai_api_version: str = Field("2024-10-21", alias="AZURE_OPENAI_API_VERSION")
    azure_openai_chat_deployment: str = Field(..., alias="AZURE_OPENAI_CHAT_DEPLOYMENT")
    azure_openai_embedding_deployment: str = Field(
        ..., alias="AZURE_OPENAI_EMBEDDING_DEPLOYMENT"
    )

    # Azure AI Search
    azure_search_endpoint: str = Field(..., alias="AZURE_SEARCH_ENDPOINT")
    azure_search_api_key: str = Field(..., alias="AZURE_SEARCH_API_KEY")
    azure_search_index_name: str = Field("mexlex-index", alias="AZURE_SEARCH_INDEX_NAME")
    # Nombre de la configuración semántica que se CREA junto con el índice.
    # Se deja siempre puesta: la configuración solo puede definirse al crear
    # el índice, así que si no se bakea ahora, activar el semantic ranker
    # después obligaría a reindexar todo otra vez.
    azure_search_semantic_config_name: str = Field(
        "mexlex-semantic-config", alias="AZURE_SEARCH_SEMANTIC_CONFIG_NAME"
    )
    # Si el retriever USA o no el semantic ranker (queryType=semantic).
    # Separado de lo anterior porque el ranker depende de que la región del
    # servicio lo soporte y consume cuota mensual. Actívalo con
    # AZURE_SEARCH_USE_SEMANTIC_RANKER=true cuando lo tengas disponible.
    azure_search_use_semantic_ranker: bool = Field(
        True, alias="AZURE_SEARCH_USE_SEMANTIC_RANKER"
    )

    # Tavily (Fase 4) — opcional: sin esta key el agente corre sin
    # búsqueda web, solo con el corpus indexado.
    tavily_api_key: str | None = Field(default=None, alias="TAVILY_API_KEY")

    # Chunking (ajustables sin tocar código)
    chunk_size: int = 1200
    chunk_overlap: int = 200
    retrieval_k: int = 4

    # Memoria conversacional (Fase 3): cuántos mensajes del historial se
    # mandan al LLM. Cuenta mensajes, no tokens (ver _trim en
    # chains/conversational_rag.py).
    memory_max_messages: int = 10

    # Agente (Fase 4)
    web_search_max_results: int = 5

    # Retrieval estructurado (Fase 5). Si se deja en None, la dimensión
    # del vector se detecta consultando al modelo de embeddings.
    embedding_dimensions: int | None = Field(
        default=None, alias="EMBEDDING_DIMENSIONS"
    )
    # Tope de chunks que devuelve una consulta por filtro exacto.
    lookup_max_results: int = 6

    # --- Persistencia del historial (Fase 7, opcional) ---
    # Sin endpoint, el checkpointer cae a MemorySaver: la app funciona
    # igual pero el historial se pierde al reiniciar el proceso.
    cosmos_endpoint: str | None = Field(default=None, alias="AZURE_COSMOS_ENDPOINT")
    # Sin key se usa DefaultAzureCredential (Managed Identity en Azure,
    # `az login` en local).
    cosmos_key: str | None = Field(default=None, alias="AZURE_COSMOS_KEY")
    cosmos_database: str = Field("mexlex", alias="AZURE_COSMOS_DATABASE")
    cosmos_container: str = Field("checkpoints", alias="AZURE_COSMOS_CONTAINER")
    # Contenedor de la UI (Fase 8): hilos, mensajes y usuarios de
    # Chainlit. Separado del de checkpoints porque son datos con otra
    # forma, otras consultas y otro índice — ver documentation/09.
    cosmos_threads_container: str = Field(
        "conversations", alias="AZURE_COSMOS_THREADS_CONTAINER"
    )
    # Retención de las conversaciones. None = para siempre.
    # Solo se aplica si el contenedor lo CREA este código: en Cosmos el
    # TTL por defecto no se puede cambiar después desde el SDK.
    cosmos_ttl_seconds: int | None = Field(
        default=None, alias="AZURE_COSMOS_TTL_SECONDS"
    )
    # Crear base y contenedor si no existen son operaciones de plano de
    # control: un identity con solo el rol de datos de Cosmos NO puede
    # ejecutarlas. Ponlo en false y aprovisiona el contenedor aparte.
    cosmos_create_if_missing: bool = Field(
        True, alias="AZURE_COSMOS_CREATE_IF_MISSING"
    )

    # --- Autenticación de la UI (Fase 8, opcional) ---
    # Chainlit solo persiste y reanuda conversaciones si sabe de QUIÉN
    # son, y para eso exige un usuario autenticado. Sin usuario y
    # contraseña aquí, la app corre sin login y sin barra lateral de
    # historial (el comportamiento de la Fase 7).
    #
    # ⚠️ Esto es una cuenta única de desarrollo, no un sistema de
    # usuarios: la contraseña viaja en el .env en claro. Para algo real,
    # Chainlit trae OAuth (incluido Entra ID) — ver documentation/09.
    auth_user: str | None = Field(default=None, alias="MEXLEX_AUTH_USER")
    auth_password: str | None = Field(default=None, alias="MEXLEX_AUTH_PASSWORD")
    # Secreto para firmar el JWT de sesión. Genéralo con
    # `chainlit create-secret`. Sin él, Chainlit se niega a arrancar con
    # autenticación activada.
    chainlit_auth_secret: str | None = Field(
        default=None, alias="CHAINLIT_AUTH_SECRET"
    )

    # --- LangSmith / trazabilidad (Fase 6, opcional) ---
    # Sin API key el tracing queda apagado y la app funciona igual.
    langsmith_tracing: bool = Field(False, alias="LANGSMITH_TRACING")
    langsmith_api_key: str | None = Field(default=None, alias="LANGSMITH_API_KEY")
    langsmith_project: str = Field("mexlex-agent", alias="LANGSMITH_PROJECT")
    langsmith_endpoint: str | None = Field(default=None, alias="LANGSMITH_ENDPOINT")


settings = Settings()  # type: ignore[call-arg]
