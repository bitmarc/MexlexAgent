"""Un contenedor de Cosmos DB falso, en memoria, para probar el data layer.

El data layer de la Fase 8 es sobre todo lógica de consultas: armar el
`ThreadDict`, ordenar los pasos, paginar, meter el feedback dentro del
paso. Nada de eso necesita Azure para probarse — necesita algo que se
comporte como un contenedor.

Este doble implementa los cuatro métodos que usamos (`upsert_item`,
`read_item`, `delete_item`, `query_items`) con diccionarios, incluyendo
el `_etag` y el 412 de la concurrencia optimista.

`query_items` no interpreta SQL: reconoce las consultas concretas que el
data layer emite. Es deliberado — un motor SQL de juguete probaría el
motor, no el código. Si alguien cambia una consulta y no actualiza este
archivo, el test truena, que es exactamente lo que queremos.
"""

from __future__ import annotations

import itertools
from typing import Any

from azure.cosmos.exceptions import (
    CosmosAccessConditionFailedError,
    CosmosResourceNotFoundError,
)


class FakeContainer:
    """Contenedor en memoria indexado por (partition_key, id)."""

    def __init__(self) -> None:
        self.docs: dict[tuple[str, str], dict] = {}
        self._etags = itertools.count(1)
        # Para poder afirmar en los tests qué se consultó de verdad.
        self.consultas: list[str] = []

    # --- API que usa el data layer -------------------------------------

    async def upsert_item(
        self, body: dict, *, etag: str | None = None, match_condition: Any = None
    ) -> dict:
        clave = (body["partition_key"], body["id"])
        actual = self.docs.get(clave)

        if etag is not None and (actual is None or actual["_etag"] != etag):
            raise CosmosAccessConditionFailedError(
                status_code=412, message="etag mismatch"
            )

        doc = dict(body)
        doc["_etag"] = f"etag-{next(self._etags)}"
        self.docs[clave] = doc
        return doc

    async def read_item(self, item: str, partition_key: str) -> dict:
        doc = self.docs.get((partition_key, item))
        if doc is None:
            raise CosmosResourceNotFoundError(status_code=404, message="not found")
        return dict(doc)

    async def delete_item(self, item: str, partition_key: str) -> None:
        if self.docs.pop((partition_key, item), None) is None:
            raise CosmosResourceNotFoundError(status_code=404, message="not found")

    def query_items(
        self,
        query: str,
        parameters: list[dict] | None = None,
        partition_key: str | None = None,
        **_: Any,
    ):
        self.consultas.append(query)
        valores = {p["name"]: p["value"] for p in (parameters or [])}
        return _AsyncIter(self._resolver(query, valores))

    # --- resolución de las consultas conocidas -------------------------

    def _resolver(self, query: str, v: dict) -> list[dict]:
        docs = [dict(d) for d in self.docs.values()]

        # 1) Todo lo de una conversación (get_thread, delete_thread).
        if "c.partition_key = @pk" in query:
            return [d for d in docs if d["partition_key"] == v["@pk"]]

        # 2) Los hilos de un usuario (list_threads) o solo sus ids
        #    (get_favorite_steps).
        if "c.entity = 'thread'" in query:
            hilos = [
                d
                for d in docs
                if d.get("entity") == "thread" and d.get("userId") == v["@userId"]
            ]
            if "CONTAINS(c.name" in query:
                buscado = (v["@search"] or "").lower()
                hilos = [h for h in hilos if buscado in (h.get("name") or "").lower()]

            hilos.sort(key=lambda h: h.get("createdAt") or "", reverse=True)

            if "@offset" in v:
                inicio = v["@offset"]
                hilos = hilos[inicio : inicio + v["@limit"]]
            return hilos

        # 3) Los pasos favoritos de una lista de hilos.
        if "c.entity = 'step'" in query:
            return [
                d
                for d in docs
                if d.get("entity") == "step"
                and d.get("threadId") in v["@ids"]
                and d.get("favorite") is True
            ]

        raise AssertionError(f"El doble no conoce esta consulta:\n{query}")


class _AsyncIter:
    """Convierte una lista en el async iterator que devuelve Cosmos."""

    def __init__(self, items: list[dict]) -> None:
        self._items = iter(items)

    def __aiter__(self) -> _AsyncIter:
        return self

    async def __anext__(self) -> dict:
        try:
            return next(self._items)
        except StopIteration:
            raise StopAsyncIteration from None
