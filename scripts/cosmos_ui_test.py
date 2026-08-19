#!/usr/bin/env python
"""Uso: python scripts/cosmos_ui_test.py [thread_id | --limpiar-huerfanos]

Diagnóstico de la Fase 8: enseña qué hay REALMENTE en el contenedor de la
UI y simula la decisión que toma Chainlit al reabrir una conversación.

Existe porque el fallo típico de un data layer es invisible: Chainlit
persiste los mensajes con `asyncio.create_task(...)` y, si algo revienta,
el chat se ve normal y el problema solo aparece al reabrirlo, sin ningún
mensaje de error en la UI. Este script convierte "no aparece nada" en una
respuesta concreta.

    (sin argumento)        lista las conversaciones y cuántos mensajes tiene cada una
    <thread_id>            detalle de una y veredicto del resume
    --limpiar-huerfanos    BORRA los documentos con el formato de clave viejo
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mexlex.config import settings  # noqa: E402
from mexlex.persistence.cosmos import cosmos_configurado  # noqa: E402
from mexlex.persistence.data_layer import (  # noqa: E402
    CARACTERES_PROHIBIDOS,
    CosmosDataLayer,
    _id_valido,
    _pk_hilo,
)


async def listar(capa: CosmosDataLayer) -> None:
    """Todos los hilos del contenedor, con su conteo real de mensajes."""
    contenedor = await capa._contenedor()

    hilos = [
        doc
        async for doc in contenedor.query_items(
            query="SELECT * FROM c WHERE c.entity = 'thread'"
        )
    ]
    if not hilos:
        print("No hay ninguna conversación guardada en el contenedor.")
        print(
            "\nSi ya conversaste en la UI, revisa la terminal de Chainlit: los "
            "errores del data layer se registran ahí, no en el navegador."
        )
        return

    print(f"{len(hilos)} conversación(es):\n")
    for hilo in sorted(hilos, key=lambda h: h.get("createdAt") or "", reverse=True):
        thread_id = hilo["threadId"]
        pasos = [
            doc
            async for doc in contenedor.query_items(
                query=(
                    "SELECT c.id FROM c "
                    "WHERE c.partition_key = @pk AND c.entity = 'step'"
                ),
                parameters=[{"name": "@pk", "value": _pk_hilo(thread_id)}],
                partition_key=_pk_hilo(thread_id),
            )
        ]
        aviso = "  [!] SIN MENSAJES" if not pasos else ""
        print(f"  {thread_id}")
        print(f"    nombre         : {hilo.get('name')!r}")
        print(f"    userId         : {hilo.get('userId')!r}")
        print(f"    userIdentifier : {hilo.get('userIdentifier')!r}")
        print(f"    mensajes       : {len(pasos)}{aviso}")
        print()

    print("Para el detalle de una: python scripts/cosmos_ui_test.py <thread_id>")


async def detallar(capa: CosmosDataLayer, thread_id: str) -> None:
    """Todo lo que hay en la partición de un hilo, y el veredicto del resume."""
    contenedor = await capa._contenedor()
    particion = _pk_hilo(thread_id)

    docs = [
        doc
        async for doc in contenedor.query_items(
            query="SELECT * FROM c WHERE c.partition_key = @pk",
            parameters=[{"name": "@pk", "value": particion}],
            partition_key=particion,
        )
    ]

    print(f"Partición: {particion}")
    print(f"Documentos: {len(docs)}\n")

    for doc in docs:
        marca = "" if _id_valido(doc["id"]) else f"  [!] id ilegal ({CARACTERES_PROHIBIDOS!r})"
        print(f"  [{doc.get('entity'):<7}] id={doc['id']}{marca}")
    print()

    # Lo mismo que hace Chainlit al reabrir el hilo.
    hilo = await capa.get_thread(thread_id)
    if hilo is None:
        print("[X] get_thread devolvió None: falta el documento 'THREAD' de la")
        print("   partición. Chainlit mostraría 'Thread not found' y abriría un")
        print("   chat nuevo.")
        return

    print("[OK] get_thread reconstruyó el hilo:")
    print(f"   nombre         : {hilo['name']!r}")
    print(f"   userIdentifier : {hilo['userIdentifier']!r}")
    print(f"   mensajes       : {len(hilo['steps'])}")

    for paso in hilo["steps"]:
        # Chainlit solo repinta los pasos cuyo `type` contiene "message".
        se_pinta = "message" in (paso.get("type") or "")
        texto = str(paso.get("output") or "").replace("\n", " ")[:60]
        print(f"     - type={paso.get('type'):<20} pinta={se_pinta}  {texto}")

    print()
    if not hilo["steps"]:
        print("[X] El hilo existe pero NO tiene mensajes. Al abrirlo se ve la")
        print("   pantalla de conversación nueva. Los mensajes fallaron al")
        print("   guardarse: busca 'No se pudo guardar el mensaje' o")
        print("   'Error while flushing create_step' en la terminal de Chainlit.")
    elif not any("message" in (p.get("type") or "") for p in hilo["steps"]):
        print("[X] Hay pasos, pero ninguno es de tipo mensaje: la UI no pinta nada.")
    else:
        print("[OK] Este hilo debería reabrirse con su contenido.")

    if settings.auth_user and hilo["userIdentifier"] != settings.auth_user:
        print()
        print(f"[X] El hilo es de {hilo['userIdentifier']!r} pero MEXLEX_AUTH_USER")
        print(f"   es {settings.auth_user!r}. Chainlit no te dejaría reabrirlo.")


async def limpiar_huerfanos(capa: CosmosDataLayer) -> None:
    """Borra los documentos escritos con el formato de clave viejo (`#`).

    Las primeras versiones del data layer usaban `THREAD#<id>` y
    `USER#<id>` como partition key, copiando la convención de DynamoDB.
    Al cambiar el separador a `_` (porque Cosmos prohíbe `#` en el `id`),
    esos documentos quedaron inalcanzables: siguen apareciendo en la
    barra lateral, pero `get_thread` los busca en otra partición y nunca
    los encuentra, así que al hacer clic se abre un chat nuevo.
    """
    contenedor = await capa._contenedor()

    viejos = [
        doc
        async for doc in contenedor.query_items(
            query="SELECT c.id, c.partition_key, c.entity, c.name FROM c"
        )
        if "#" in (doc.get("partition_key") or "")
    ]

    if not viejos:
        print("No hay documentos con el formato viejo. Nada que limpiar.")
        return

    print(f"{len(viejos)} documento(s) con el formato viejo:\n")
    for doc in viejos:
        print(f"  [{doc.get('entity'):<7}] {doc['partition_key']}  {doc.get('name') or ''}")

    print("\nEstos documentos son inalcanzables desde la app. Borrando...")
    for doc in viejos:
        await contenedor.delete_item(item=doc["id"], partition_key=doc["partition_key"])
    print(f"Listo: {len(viejos)} borrados. El usuario se vuelve a crear al entrar.")


async def main() -> None:
    if not cosmos_configurado():
        print("AZURE_COSMOS_ENDPOINT no está configurado: no hay nada que revisar.")
        return

    print(f"Contenedor: {settings.cosmos_database}/{settings.cosmos_threads_container}\n")
    capa = CosmosDataLayer()
    try:
        argumento = sys.argv[1] if len(sys.argv) > 1 else None
        if argumento == "--limpiar-huerfanos":
            await limpiar_huerfanos(capa)
        elif argumento:
            await detallar(capa, argumento)
        else:
            await listar(capa)
    finally:
        await capa.close()


if __name__ == "__main__":
    asyncio.run(main())
