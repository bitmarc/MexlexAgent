#!/usr/bin/env python
"""Uso: python scripts/chat_test.py [thread_id]

Igual que query_test.py, pero con memoria (Fase 3): usa el grafo
conversacional y mantiene un thread_id fijo durante toda la sesión, así
que puedes hacer preguntas de seguimiento.

Pruébalo con algo como:
    Tú: ¿qué dice el artículo 10 de la Ley Federal de Cinematografía?
    Tú: ¿y el 11?          <- sin memoria, esto no funcionaría

Escribe 'salir' para terminar.
"""

import asyncio
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from langchain_core.messages import HumanMessage  # noqa: E402

from mexlex.agent.memory import close_checkpointer, get_checkpointer  # noqa: E402
from mexlex.chains.conversational_rag import build_conversational_rag_graph  # noqa: E402


async def main() -> None:
    print("Cargando grafo conversacional... (esto crea las conexiones a Azure)")
    graph = build_conversational_rag_graph(await get_checkpointer())

    # Un thread_id por sesión de terminal. Si lo cambias, empiezas una
    # conversación nueva; si lo reutilizas, retomas la anterior.
    thread_id = sys.argv[1] if len(sys.argv) > 1 else str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print(f"Listo (thread_id={thread_id}). Escribe 'salir' para terminar.\n")

    try:
        while True:
            question = input("Tú: ").strip()
            if question.lower() in {"salir", "exit", "quit"}:
                break
            if not question:
                continue

            print("\nAsistente: ", end="", flush=True)
            async for chunk, metadata in graph.astream(
                {"messages": [HumanMessage(content=question)]},
                config=config,
                stream_mode="messages",
            ):
                # Solo nos interesan los tokens del nodo que redacta la
                # respuesta: el nodo contextualize también llama al LLM y no
                # queremos mostrar su salida.
                if metadata.get("langgraph_node") == "generate" and chunk.content:
                    print(chunk.content, end="", flush=True)
            print("\n")
    finally:
        await close_checkpointer()


if __name__ == "__main__":
    asyncio.run(main())
