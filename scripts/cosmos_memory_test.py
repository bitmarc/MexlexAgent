#!/usr/bin/env python
"""Uso: python scripts/cosmos_memory_test.py [thread_id]

Verificación de la Fase 7: inspecciona lo que el checkpointer tiene
guardado para un `thread_id`, sin llamar al LLM ni a Azure AI Search.

Es la herramienta de depuración equivalente a `graph.get_state(config)`
de la Fase 3, pero mirando directo al checkpointer: te dice si el
historial está realmente en Cosmos DB o si te lo está inventando la
memoria del proceso.

Flujo típico para probar que la persistencia funciona:

    1. python scripts/agent_test.py 11111111-1111-1111-1111-111111111111
       (haz una pregunta, escribe 'salir' — el proceso muere)

    2. python scripts/cosmos_memory_test.py 11111111-1111-1111-1111-111111111111
       (proceso nuevo: si aparecen los mensajes, sobrevivieron)

Sin argumento usa un thread_id que no existe, lo cual también sirve:
debe reportar 0 checkpoints, confirmando el aislamiento entre hilos.
"""

import asyncio
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from langgraph.checkpoint.memory import MemorySaver  # noqa: E402

from mexlex.agent.memory import close_checkpointer, get_checkpointer  # noqa: E402
from mexlex.config import settings  # noqa: E402


async def main() -> None:
    thread_id = sys.argv[1] if len(sys.argv) > 1 else str(uuid.uuid4())
    checkpointer = await get_checkpointer()

    print(f"Checkpointer : {type(checkpointer).__name__}")
    if isinstance(checkpointer, MemorySaver):
        print(
            "\n⚠️  Es MemorySaver: no hay AZURE_COSMOS_ENDPOINT en el .env, así\n"
            "   que nada se persiste y este script no puede probar nada.\n"
        )
    else:
        print(f"Base/cont.   : {settings.cosmos_database}/{settings.cosmos_container}")
    print(f"thread_id    : {thread_id}\n")

    try:
        # El estado más reciente del hilo. None = nunca se guardó nada.
        tupla = await checkpointer.aget_tuple({"configurable": {"thread_id": thread_id}})

        if tupla is None:
            print("Sin checkpoints para este thread_id.")
            return

        mensajes = tupla.checkpoint["channel_values"].get("messages", [])
        print(f"Mensajes en el estado: {len(mensajes)}\n")
        for msg in mensajes:
            texto = str(msg.content).replace("\n", " ")[:100]
            # Las tool calls no traen content, así que hay que anunciarlas.
            llamadas = [tc["name"] for tc in getattr(msg, "tool_calls", []) or []]
            etiqueta = f" -> {llamadas}" if llamadas else ""
            print(f"  [{type(msg).__name__:<12}] {texto}{etiqueta}")

        # Cada super-step del grafo dejó su propio checkpoint: por eso son
        # varios por turno, no uno.
        historial = [
            c
            async for c in checkpointer.alist({"configurable": {"thread_id": thread_id}})
        ]
        print(f"\nCheckpoints guardados en el hilo: {len(historial)}")
        print(f"Último checkpoint_id: {tupla.config['configurable']['checkpoint_id']}")
    finally:
        await close_checkpointer()


if __name__ == "__main__":
    asyncio.run(main())
