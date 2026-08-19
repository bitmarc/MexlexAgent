"""Construcción del cliente async de Cosmos DB, compartida.

El checkpointer (Fase 7) y el data layer (Fase 8) necesitan exactamente
el mismo cliente con la misma lógica de credenciales, así que vive aquí
en vez de duplicarse.

**Cada uno construye el suyo.** No es un singleton global a propósito:
el checkpointer se crea en un hook async (`on_app_startup`) mientras que
Chainlit pide el data layer desde una función *sync* y lo cachea por su
cuenta. Coordinar un único cliente entre esos dos ciclos de vida costaba
más de lo que ahorra: un `CosmosClient` es barato de construir (no hace
red al instanciarse) y cada dueño cierra el suyo.
"""

from __future__ import annotations

from typing import Any

from mexlex.config import settings

USER_AGENT = "mexlex-agent"

# El checkpointer de LangGraph escribe esta ruta en cada documento, así
# que no es configurable. La reusamos en el contenedor de la UI para que
# los dos contenedores se lean igual.
PARTITION_KEY_PATH = "/partition_key"


def cosmos_configurado() -> bool:
    """True si hay endpoint de Cosmos en el .env.

    Es el interruptor único de las fases 7 y 8: sin endpoint, ni el
    historial se persiste ni la UI muestra conversaciones anteriores.
    """
    return bool(settings.cosmos_endpoint)


def nuevo_cliente() -> tuple[Any, Any | None]:
    """Regresa `(cliente, credencial)`; la credencial también hay que cerrarla.

    Con `AZURE_COSMOS_KEY` la credencial es la key (un string, no hay
    nada que cerrar y se regresa `None`). Sin ella se usa
    `DefaultAzureCredential`, que sí sostiene recursos y por eso se
    devuelve: quien abre, cierra.
    """
    # Import diferido: sin Cosmos configurado no queremos ni cargar el
    # SDK de Azure.
    from azure.cosmos.aio import CosmosClient

    if not settings.cosmos_endpoint:
        raise RuntimeError(
            "AZURE_COSMOS_ENDPOINT no está configurado: usa cosmos_configurado() "
            "antes de llamar a nuevo_cliente()."
        )

    if settings.cosmos_key:
        cliente = CosmosClient(
            settings.cosmos_endpoint, settings.cosmos_key, user_agent=USER_AGENT
        )
        return cliente, None

    from azure.identity.aio import DefaultAzureCredential

    credencial = DefaultAzureCredential()
    cliente = CosmosClient(
        settings.cosmos_endpoint, credencial, user_agent=USER_AGENT
    )
    return cliente, credencial


async def obtener_contenedor(
    cliente: Any,
    nombre: str,
    *,
    indexing_policy: dict | None = None,
) -> Any:
    """Devuelve el contenedor, creándolo si hace falta y si se permite.

    `AZURE_COSMOS_CREATE_IF_MISSING=false` salta la creación porque crear
    bases y contenedores son operaciones de **plano de control**: un
    identity con solo el rol de datos de Cosmos no tiene permiso para
    ejecutarlas y la app fallaría al arrancar.
    """
    from azure.cosmos import PartitionKey

    if not settings.cosmos_create_if_missing:
        base = cliente.get_database_client(settings.cosmos_database)
        return base.get_container_client(nombre)

    base = await cliente.create_database_if_not_exists(settings.cosmos_database)
    extra = {"indexing_policy": indexing_policy} if indexing_policy else {}
    return await base.create_container_if_not_exists(
        id=nombre,
        partition_key=PartitionKey(path=PARTITION_KEY_PATH),
        # Ojo: Cosmos solo respeta el TTL al CREAR el contenedor. Si ya
        # existe, el valor se ignora en silencio.
        default_ttl=settings.cosmos_ttl_seconds,
        **extra,
    )
