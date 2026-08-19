"""Fase 8: data layer de Chainlit sobre Cosmos DB NoSQL.

Chainlit no trae integración con Cosmos DB (sus data layers oficiales son
SQLAlchemy, Literal AI y DynamoDB), así que implementamos `BaseDataLayer`
nosotros. El modelo está calcado del de DynamoDB, que es el que mejor
traduce a Cosmos: **un solo contenedor, varios tipos de documento**
distinguidos por prefijo.

    partition_key            id                 entity
    ---------------------------------------------------------
    THREAD_<thread_id>       THREAD             thread
    THREAD_<thread_id>       STEP_<step_id>     step
    THREAD_<thread_id>       ELEMENT_<elem_id>  element
    USER_<identifier>        USER               user

⚠️ El separador es `_`, no `#`. El data layer de DynamoDB usa `#` y es la
convención habitual para claves compuestas en NoSQL, pero **Cosmos
prohíbe `/`, `\`, `?` y `#` en el `id`**: esos ids viajan en la URI de
`read_item` y `delete_item`, y un `#` abre un fragmento que trunca la
ruta. El upsert sí funciona (el id va en el cuerpo), así que el fallo
aparece tarde y en otro lado — justo el tipo de bug que cuesta encontrar.

Que todos los documentos de una conversación compartan `partition_key` es
lo que hace barato lo que más se hace: abrir un hilo son todos sus
documentos en **una sola partición**, la consulta más eficiente que
existe en Cosmos. Lo caro es lo raro: listar los hilos de un usuario es
*cross-partition* (por eso hay un índice compuesto).

Ojo con `id`: en Cosmos es el identificador del documento **dentro de su
partición**, no un campo cualquiera. Por eso el id del hilo va en
`threadId` y el `id` del documento se usa como discriminador (`THREAD`,
`STEP_...`). Los dicts que Chainlit espera se guardan tal cual bajo
`data`, para poder devolvérselos sin traducción.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from chainlit.data.base import BaseDataLayer
from chainlit.data.utils import queue_until_user_message
from chainlit.types import (
    Feedback,
    PageInfo,
    PaginatedResponse,
    Pagination,
    ThreadDict,
    ThreadFilter,
)
from chainlit.user import PersistedUser, User

from mexlex.config import settings
from mexlex.persistence.cosmos import nuevo_cliente, obtener_contenedor

if TYPE_CHECKING:
    from chainlit.element import Element, ElementDict
    from chainlit.step import StepDict

logger = logging.getLogger(__name__)

# Cosmos necesita un índice compuesto para ordenar por una propiedad
# distinta de la que filtra. Sin esto, listar las conversaciones de un
# usuario funciona igual pero escanea de más.
INDEXING_POLICY = {
    "indexingMode": "consistent",
    "automatic": True,
    "includedPaths": [{"path": "/*"}],
    # Los dicts de Chainlit (`data`) no se consultan nunca: se leen
    # enteros. Excluirlos del índice ahorra RU en cada escritura, que es
    # la operación frecuente.
    "excludedPaths": [{"path": '/"data"/*'}, {"path": "/\"_etag\"/?"}],
    "compositeIndexes": [
        [
            {"path": "/entity", "order": "ascending"},
            {"path": "/userId", "order": "ascending"},
            {"path": "/createdAt", "order": "descending"},
        ]
    ],
}


def _ahora() -> str:
    return datetime.now(timezone.utc).isoformat()


# Cosmos prohíbe estos caracteres en el `id` de un documento: viajan en
# la URI de read/delete y la rompen. Ver el docstring del módulo.
CARACTERES_PROHIBIDOS = "/\\?#"


def _id_valido(doc_id: str) -> bool:
    return not any(c in doc_id for c in CARACTERES_PROHIBIDOS)


def _pk_hilo(thread_id: str) -> str:
    return f"THREAD_{thread_id}"


def _pk_usuario(identifier: str) -> str:
    return f"USER_{identifier}"


def _id_paso(paso_id: str) -> str:
    return f"STEP_{paso_id}"


def _si_no_cambio(doc: dict | None) -> dict:
    """Opciones de escritura condicional (`If-Match`) para un doc ya leído.

    Vacío si el documento no existía: entonces el upsert es una creación
    y no hay nada contra lo que comparar.
    """
    if not doc:
        return {}

    from azure.core import MatchConditions

    return {"etag": doc["_etag"], "match_condition": MatchConditions.IfNotModified}


class CosmosDataLayer(BaseDataLayer):
    """Persistencia de la UI de Chainlit en Cosmos DB NoSQL."""

    def __init__(self) -> None:
        # El cliente se construye aquí porque Chainlit pide el data layer
        # desde una función *sync*. Instanciar un CosmosClient no hace
        # red; asegurar el contenedor sí, y eso se hace perezosamente en
        # `_contenedor()`.
        self._cliente, self._credencial = nuevo_cliente()
        self._cont: Any | None = None
        self._lock = asyncio.Lock()

    async def _contenedor(self) -> Any:
        """Contenedor listo para usar, creado una sola vez."""
        if self._cont is not None:
            return self._cont
        async with self._lock:
            if self._cont is None:
                self._cont = await obtener_contenedor(
                    self._cliente,
                    settings.cosmos_threads_container,
                    indexing_policy=INDEXING_POLICY,
                )
        return self._cont

    async def _leer(self, particion: str, doc_id: str) -> dict | None:
        """Lee un documento; None si no existe."""
        from azure.cosmos.exceptions import CosmosResourceNotFoundError

        contenedor = await self._contenedor()
        try:
            return await contenedor.read_item(item=doc_id, partition_key=particion)
        except CosmosResourceNotFoundError:
            return None

    async def _borrar(self, particion: str, doc_id: str) -> None:
        """Borra un documento; no falla si ya no estaba."""
        from azure.cosmos.exceptions import CosmosResourceNotFoundError

        contenedor = await self._contenedor()
        try:
            await contenedor.delete_item(item=doc_id, partition_key=particion)
        except CosmosResourceNotFoundError:
            pass

    # --- usuarios ---------------------------------------------------- #

    async def get_user(self, identifier: str) -> PersistedUser | None:
        doc = await self._leer(_pk_usuario(identifier), "USER")
        if doc is None:
            return None
        return PersistedUser(
            id=doc["userId"],
            identifier=doc["identifier"],
            createdAt=doc["createdAt"],
            metadata=doc.get("metadata") or {},
        )

    async def create_user(self, user: User) -> PersistedUser | None:
        # El id del usuario ES su identifier. Chainlit compara el autor
        # de un hilo (`userIdentifier`) contra `user.identifier` para
        # autorizar, y filtra la lista de hilos por `user.id`: usar el
        # mismo valor en los dos evita un mapeo que solo daría errores
        # difíciles de ver (hilos que existen pero no se listan).
        creado = _ahora()
        contenedor = await self._contenedor()
        await contenedor.upsert_item(
            {
                "id": "USER",
                "partition_key": _pk_usuario(user.identifier),
                "entity": "user",
                "userId": user.identifier,
                "identifier": user.identifier,
                "createdAt": creado,
                "metadata": user.metadata or {},
            }
        )
        return PersistedUser(
            id=user.identifier,
            identifier=user.identifier,
            createdAt=creado,
            metadata=user.metadata or {},
        )

    # --- hilos ------------------------------------------------------- #

    async def update_thread(
        self,
        thread_id: str,
        name: str | None = None,
        user_id: str | None = None,
        metadata: dict | None = None,
        tags: list[str] | None = None,
    ) -> None:
        """Crea o actualiza el documento del hilo.

        Chainlit no tiene `create_thread`: esta misma función crea la
        primera vez y parchea después, y se llama con subconjuntos de
        campos distintos desde varios lados (el nombre al primer mensaje,
        el user_id al abrir la sesión HTTP). Por eso solo se pisan los
        campos que llegan con valor.

        Va con **concurrencia optimista**: dos de esas llamadas pueden
        cruzarse, y sin el `etag` la última en escribir borraría los
        campos que puso la otra. Cosmos rechaza el upsert con 412 si el
        documento cambió desde que lo leímos, y ahí reintentamos.
        """
        from azure.cosmos.exceptions import CosmosAccessConditionFailedError

        particion = _pk_hilo(thread_id)
        contenedor = await self._contenedor()

        for intento in range(3):
            actual = await self._leer(particion, "THREAD") or {}

            doc = {
                "id": "THREAD",
                "partition_key": particion,
                "entity": "thread",
                "threadId": thread_id,
                "createdAt": actual.get("createdAt") or _ahora(),
                "updatedAt": _ahora(),
                "name": name if name is not None else actual.get("name"),
                "userId": user_id if user_id is not None else actual.get("userId"),
                "userIdentifier": (
                    user_id if user_id is not None else actual.get("userIdentifier")
                ),
                "tags": tags if tags is not None else actual.get("tags"),
                "metadata": (
                    metadata if metadata is not None else actual.get("metadata") or {}
                ),
            }

            try:
                # Si el documento ya existía, exigimos que no haya cambiado.
                await contenedor.upsert_item(doc, **_si_no_cambio(actual))
                return
            except CosmosAccessConditionFailedError:
                logger.debug(
                    "update_thread: colisión en %s, reintento %d", thread_id, intento + 1
                )

        logger.warning(
            "update_thread: no se pudo escribir el hilo %s tras 3 intentos", thread_id
        )

    async def get_thread(self, thread_id: str) -> ThreadDict | None:
        """Arma el ThreadDict completo con una consulta de partición única."""
        contenedor = await self._contenedor()
        particion = _pk_hilo(thread_id)

        hilo: dict | None = None
        pasos: list[dict] = []
        elementos: list[dict] = []

        async for doc in contenedor.query_items(
            query="SELECT * FROM c WHERE c.partition_key = @pk",
            parameters=[{"name": "@pk", "value": particion}],
            partition_key=particion,
        ):
            if doc["entity"] == "thread":
                hilo = doc
            elif doc["entity"] == "step":
                pasos.append(doc["data"])
            elif doc["entity"] == "element":
                elementos.append(doc["data"])

        if hilo is None:
            if pasos or elementos:
                logger.warning("Documentos huérfanos para el hilo %s", thread_id)
            return None

        # El orden importa: la UI los pinta en el orden que le lleguen.
        pasos.sort(key=lambda p: p.get("createdAt") or "")

        return ThreadDict(
            id=thread_id,
            createdAt=hilo["createdAt"],
            name=hilo.get("name"),
            userId=hilo.get("userId"),
            userIdentifier=hilo.get("userIdentifier"),
            tags=hilo.get("tags"),
            metadata=hilo.get("metadata") or {},
            steps=pasos,  # type: ignore[arg-type]
            elements=elementos,  # type: ignore[arg-type]
        )

    async def get_thread_author(self, thread_id: str) -> str:
        """Quién es el dueño del hilo. Chainlit lo usa para autorizar."""
        doc = await self._leer(_pk_hilo(thread_id), "THREAD")
        if doc is None:
            raise ValueError(f"No hay hilo con id {thread_id}")
        # Devuelve el *identifier*: `is_thread_author` lo compara contra
        # `current_user.identifier`, no contra el id interno.
        return doc.get("userIdentifier") or ""

    async def list_threads(
        self, pagination: Pagination, filters: ThreadFilter
    ) -> PaginatedResponse[ThreadDict]:
        """La barra lateral. Consulta cross-partition, ordenada por fecha."""
        contenedor = await self._contenedor()

        if filters.feedback is not None:
            # Filtrar por feedback exigiría denormalizar el pulgar del
            # hilo; no vale la pena para el volumen de este proyecto.
            logger.info("list_threads: el filtro por feedback no está soportado")

        condiciones = ["c.entity = 'thread'", "c.userId = @userId"]
        parametros: list[dict[str, Any]] = [
            {"name": "@userId", "value": filters.userId}
        ]

        if filters.search:
            # CONTAINS con el tercer argumento en true = case-insensitive.
            condiciones.append("CONTAINS(c.name, @search, true)")
            parametros.append({"name": "@search", "value": filters.search})

        # El cursor es el offset ya consumido. Cosmos tiene tokens de
        # continuación "de verdad", pero para una lista de decenas de
        # hilos OFFSET/LIMIT es más simple y se comporta igual.
        desplazamiento = int(pagination.cursor) if pagination.cursor else 0
        # Pedimos uno de más para saber si hay página siguiente sin
        # tener que contar el total.
        parametros.append({"name": "@offset", "value": desplazamiento})
        parametros.append({"name": "@limit", "value": pagination.first + 1})

        consulta = (
            f"SELECT * FROM c WHERE {' AND '.join(condiciones)} "
            "ORDER BY c.createdAt DESC OFFSET @offset LIMIT @limit"
        )

        docs = [
            doc
            async for doc in contenedor.query_items(
                query=consulta, parameters=parametros
            )
        ]

        hay_mas = len(docs) > pagination.first
        docs = docs[: pagination.first]

        hilos = [
            ThreadDict(
                id=doc["threadId"],
                createdAt=doc["createdAt"],
                name=doc.get("name"),
                userId=doc.get("userId"),
                userIdentifier=doc.get("userIdentifier"),
                tags=doc.get("tags"),
                metadata=doc.get("metadata") or {},
                steps=[],
                elements=[],
            )
            for doc in docs
        ]

        return PaginatedResponse(
            data=hilos,
            pageInfo=PageInfo(
                hasNextPage=hay_mas,
                startCursor=pagination.cursor,
                endCursor=str(desplazamiento + len(hilos)) if hay_mas else None,
            ),
        )

    async def delete_thread(self, thread_id: str) -> None:
        """Borra la conversación entera: hilo, pasos y elementos."""
        contenedor = await self._contenedor()
        particion = _pk_hilo(thread_id)

        async for doc in contenedor.query_items(
            query="SELECT c.id FROM c WHERE c.partition_key = @pk",
            parameters=[{"name": "@pk", "value": particion}],
            partition_key=particion,
        ):
            await self._borrar(particion, doc["id"])

        # El estado del agente vive en OTRO contenedor (el checkpointer de
        # la Fase 7) y nadie más lo va a limpiar: si no se borra aquí, el
        # hilo desaparece de la UI pero el agente sigue recordándolo.
        from mexlex.agent.memory import borrar_checkpoints_del_hilo

        try:
            await borrar_checkpoints_del_hilo(thread_id)
        except Exception:  # noqa: BLE001
            # Que falle la limpieza del checkpointer no debe impedir que
            # la conversación desaparezca de la UI: el TTL la alcanzará.
            logger.exception("No se pudieron borrar los checkpoints de %s", thread_id)

    # --- pasos (mensajes) -------------------------------------------- #

    @queue_until_user_message()
    async def create_step(self, step_dict: StepDict) -> None:
        await self._guardar_paso(step_dict)

    @queue_until_user_message()
    async def update_step(self, step_dict: StepDict) -> None:
        # Un paso se crea vacío y se va actualizando mientras el LLM
        # streamea. Como el upsert reescribe el documento completo, crear
        # y actualizar son la misma operación.
        await self._guardar_paso(step_dict)

    async def _guardar_paso(self, step_dict: StepDict) -> None:
        thread_id = step_dict.get("threadId")
        paso_id = step_dict.get("id")
        if not thread_id or not paso_id:
            logger.warning("Paso sin threadId o id, se ignora: %s", step_dict.get("name"))
            return

        metadata = step_dict.get("metadata") or {}
        doc_id = _id_paso(paso_id)
        if not _id_valido(doc_id):
            raise ValueError(
                f"El id '{doc_id}' lleva alguno de los caracteres que Cosmos "
                f"prohíbe ({CARACTERES_PROHIBIDOS!r}). El upsert pasaría, pero "
                "read_item y delete_item se romperían después."
            )

        contenedor = await self._contenedor()
        # Chainlit persiste los mensajes con `asyncio.create_task(...)`, así
        # que si esto revienta el usuario NO ve nada: el chat sigue como si
        # todo hubiera ido bien y los mensajes simplemente no aparecen al
        # reabrirlo. Por eso lo registramos aquí, con traceback.
        try:
            await contenedor.upsert_item(
                {
                    "id": doc_id,
                    "partition_key": _pk_hilo(thread_id),
                    "entity": "step",
                    "threadId": thread_id,
                    # Los dos campos de abajo están duplicados dentro de
                    # `data`. Se sacan porque `data` está EXCLUIDO del índice
                    # (ver INDEXING_POLICY) y Cosmos rechaza filtrar u
                    # ordenar por una ruta que no indexó.
                    "createdAt": step_dict.get("createdAt") or _ahora(),
                    "favorite": bool(metadata.get("favorite")),
                    "data": dict(step_dict),
                }
            )
        except Exception:
            logger.exception(
                "No se pudo guardar el mensaje %s del hilo %s: no va a aparecer "
                "al reabrir la conversación.",
                paso_id,
                thread_id,
            )
            raise

    @queue_until_user_message()
    async def delete_step(self, step_id: str) -> None:
        from chainlit.context import context

        await self._borrar(_pk_hilo(context.session.thread_id), _id_paso(step_id))

    async def get_favorite_steps(self, user_id: str) -> list[StepDict]:
        """Pasos marcados como favoritos, en todos los hilos del usuario."""
        contenedor = await self._contenedor()

        ids = [
            doc["threadId"]
            async for doc in contenedor.query_items(
                query=(
                    "SELECT c.threadId FROM c "
                    "WHERE c.entity = 'thread' AND c.userId = @userId"
                ),
                parameters=[{"name": "@userId", "value": user_id}],
            )
        ]
        if not ids:
            return []

        # Una sola consulta cross-partition en vez de una por hilo.
        pasos = [
            doc["data"]
            async for doc in contenedor.query_items(
                query=(
                    "SELECT c.data FROM c WHERE c.entity = 'step' "
                    "AND ARRAY_CONTAINS(@ids, c.threadId) "
                    "AND c.favorite = true"
                ),
                parameters=[{"name": "@ids", "value": ids}],
            )
        ]
        pasos.sort(key=lambda p: p.get("createdAt") or "", reverse=True)
        return pasos  # type: ignore[return-value]

    # --- feedback ---------------------------------------------------- #

    async def upsert_feedback(self, feedback: Feedback) -> str:
        """El pulgar arriba/abajo se guarda dentro del paso que califica."""
        if not feedback.forId or not feedback.threadId:
            raise ValueError("El feedback necesita forId y threadId")

        particion = _pk_hilo(feedback.threadId)
        doc = await self._leer(particion, _id_paso(feedback.forId))
        if doc is None:
            raise ValueError(f"No hay paso {feedback.forId} en el hilo {feedback.threadId}")

        feedback.id = f"{feedback.threadId}::{feedback.forId}"
        doc["data"]["feedback"] = {
            "id": feedback.id,
            "forId": feedback.forId,
            "threadId": feedback.threadId,
            "value": feedback.value,
            "comment": feedback.comment,
        }

        contenedor = await self._contenedor()
        await contenedor.upsert_item(doc)
        return feedback.id

    async def delete_feedback(self, feedback_id: str) -> bool:
        thread_id, _, paso_id = feedback_id.partition("::")
        if not paso_id:
            return False

        particion = _pk_hilo(thread_id)
        doc = await self._leer(particion, _id_paso(paso_id))
        if doc is None:
            return False

        doc["data"].pop("feedback", None)
        contenedor = await self._contenedor()
        await contenedor.upsert_item(doc)
        return True

    # --- elementos --------------------------------------------------- #
    #
    # El panel de fuentes son elementos `cl.Text`, y Chainlit guarda su
    # contenido en un *storage provider* (S3, GCS, Azure Blob), no en el
    # data layer: el ElementDict solo lleva una URL. Sin blob storage
    # configurado no hay nada que persistir, así que no fingimos que sí:
    # guardar la metadata sin el contenido dejaría el panel roto al
    # reanudar. Ver las limitaciones en documentation/09.

    @queue_until_user_message()
    async def create_element(self, element: Element) -> None:
        return None

    async def get_element(self, thread_id: str, element_id: str) -> ElementDict | None:
        return None

    @queue_until_user_message()
    async def delete_element(self, element_id: str, thread_id: str | None = None) -> None:
        return None

    # --- ciclo de vida ----------------------------------------------- #

    async def build_debug_url(self) -> str:
        # Solo lo usa Literal AI, para enlazar a su UI de trazas.
        return ""

    async def close(self) -> None:
        await self._cliente.close()
        if self._credencial is not None:
            await self._credencial.close()
