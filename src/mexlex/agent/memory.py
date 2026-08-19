"""Fase 3 → 7: persistencia del historial conversacional (checkpointer).

Un *checkpointer* de LangGraph guarda el estado del grafo después de
cada paso, agrupado por `thread_id`. Eso es lo que convierte un grafo
sin estado en uno con memoria: al invocarlo con el mismo `thread_id`,
LangGraph rehidrata el estado anterior antes de ejecutar.

    config = {"configurable": {"thread_id": "abc-123"}}
    await graph.ainvoke({"messages": [...]}, config=config)

La Fase 3 usaba `MemorySaver`, un dict en memoria del proceso: el
historial se perdía al reiniciar el servidor y no se compartía entre
procesos. La **Fase 7** lo cambia por `CosmosDBSaver`, el checkpointer
oficial de LangGraph para Azure Cosmos DB NoSQL. El grafo no se entera:
solo cambia lo que devuelve este módulo.

Tres detalles que condicionan el diseño de este archivo:

1. **Va el saver async, no el sync.** `CosmosDBSaverSync` solo implementa
   `get_tuple`/`list`/`put`/`put_writes`; los métodos `a*` heredan el
   `raise NotImplementedError` de `BaseCheckpointSaver`. Como la app usa
   `astream`, con el sync reventaría en el primer turno.

2. **`CosmosDBSaver.from_conn_info` es un context manager** que cierra el
   `CosmosClient` al salir: sirve para un script, no para un servidor.
   Aquí construimos el contenedor a mano y sostenemos el cliente durante
   toda la vida del proceso, cerrándolo con `close_checkpointer()`.

3. **Si no hay Cosmos configurado, se cae a `MemorySaver`.** Mismo
   criterio que Tavily y LangSmith: la app arranca igual, solo pierde la
   funcionalidad (y lo dice en el log).
"""

from __future__ import annotations

import asyncio
import logging

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver

from mexlex.config import settings
from mexlex.persistence.cosmos import (
    cosmos_configurado,
    nuevo_cliente,
    obtener_contenedor,
)

logger = logging.getLogger(__name__)

_checkpointer: BaseCheckpointSaver | None = None
_client = None  # azure.cosmos.aio.CosmosClient
_credential = None  # azure.identity.aio.DefaultAzureCredential
# Sin lock, dos corrutinas que llamen a la vez pueden abrir dos clientes
# y filtrar uno. La app lo inicializa en el arranque, pero un singleton
# async sin lock es una bomba de tiempo barata de desactivar.
_lock = asyncio.Lock()


async def get_checkpointer() -> BaseCheckpointSaver:
    """Regresa el checkpointer compartido por todas las sesiones.

    Tiene que ser un singleton: si cada sesión creara el suyo, cada una
    abriría su propia conexión (y con `MemorySaver`, su propio dict). El
    aislamiento entre usuarios lo da el `thread_id`, no el tener
    checkpointers separados.

    Es `async` porque crear el cliente de Cosmos y asegurar la base y el
    contenedor son operaciones de red. Llámala una vez al arrancar y
    reparte el resultado; no la uses como fábrica por petición.
    """
    global _checkpointer, _client, _credential

    async with _lock:
        if _checkpointer is not None:
            return _checkpointer

        if not cosmos_configurado():
            logger.warning(
                "AZURE_COSMOS_ENDPOINT no configurado: el historial vive en "
                "memoria del proceso y se pierde al reiniciar."
            )
            _checkpointer = MemorySaver()
            return _checkpointer

        # Import diferido: sin Cosmos configurado no queremos ni cargar
        # el SDK de Azure (ni obligar a tenerlo instalado).
        from langchain_azure_cosmosdb.aio import CosmosDBSaver

        _client, _credential = nuevo_cliente()
        container = await obtener_contenedor(_client, settings.cosmos_container)

        _checkpointer = CosmosDBSaver(container)
        logger.info(
            "Checkpointer: Cosmos DB %s/%s (auth: %s)",
            settings.cosmos_database,
            settings.cosmos_container,
            "key" if settings.cosmos_key else "Entra ID",
        )
        return _checkpointer


async def borrar_checkpoints_del_hilo(thread_id: str) -> int:
    """Borra el estado del agente de una conversación. Regresa cuántos docs borró.

    Existe porque `CosmosDBSaver` **no implementa `adelete_thread`**: el
    `BaseCheckpointSaver` lo deja en `raise NotImplementedError` y el
    paquete no lo sobreescribe. Sin esto, borrar una conversación desde
    la UI (Fase 8) dejaría sus checkpoints huérfanos en Cosmos para
    siempre — o hasta que el TTL los alcanzara.

    El saver guarda `thread_id` como campo de primer nivel tanto en los
    documentos de checkpoint como en los de writes, así que una sola
    consulta *cross-partition* los encuentra todos. Es más cara que una
    consulta de partición única, pero borrar una conversación es una
    operación rara y manual.
    """
    checkpointer = await get_checkpointer()
    container = getattr(checkpointer, "container", None)
    if container is None:
        # MemorySaver: no hay nada persistido que limpiar.
        return 0

    consulta = "SELECT c.id, c.partition_key FROM c WHERE c.thread_id = @thread_id"
    parametros = [{"name": "@thread_id", "value": thread_id}]

    borrados = 0
    async for doc in container.query_items(query=consulta, parameters=parametros):
        await container.delete_item(item=doc["id"], partition_key=doc["partition_key"])
        borrados += 1

    logger.info("Checkpoints borrados del hilo %s: %d", thread_id, borrados)
    return borrados


async def close_checkpointer() -> None:
    """Cierra la conexión a Cosmos y olvida el singleton.

    Idempotente: llamarla sin haber abierto nada, o dos veces, no hace
    daño. Con `MemorySaver` no hay nada que cerrar, solo se descarta.
    """
    global _checkpointer, _client, _credential

    if _client is not None:
        await _client.close()
        _client = None
    if _credential is not None:
        await _credential.close()
        _credential = None
    _checkpointer = None
