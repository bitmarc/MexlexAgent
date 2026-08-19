#!/usr/bin/env python
"""Uso: python scripts/agent_test.py [thread_id]

Prueba el agente de la Fase 4 en terminal, mostrando qué tools decide
usar en cada turno. Es la mejor forma de entender el ciclo ReAct: vas a
ver que algunas preguntas disparan una búsqueda, otras dos, y un
"gracias" ninguna.

Con `thread_id` retoma una conversación anterior. Es la prueba de la
Fase 7: si el checkpointer de Cosmos DB está configurado, puedes salir,
volver a correr el script con el MISMO id y el agente se acuerda.

    python scripts/agent_test.py 11111111-1111-1111-1111-111111111111

Escribe 'salir' para terminar.
"""

import asyncio
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from langchain_core.messages import HumanMessage  # noqa: E402

from mexlex.agent.graph import build_agent  # noqa: E402
from mexlex.agent.memory import close_checkpointer, get_checkpointer  # noqa: E402


async def main() -> None:
    print("Cargando agente... (esto crea las conexiones a Azure)")
    agent = build_agent(await get_checkpointer())

    # Sin argumento, conversación nueva. Con argumento, se retoma la que
    # tenga ese id (si el checkpointer persiste, claro).
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

            print()
            # stream_mode=["updates", "messages"] nos da dos cosas a la vez:
            # - "updates": qué nodo terminó y con qué (para ver las tools)
            # - "messages": los tokens de la respuesta final
            async for modo, dato in agent.astream(
                {"messages": [HumanMessage(content=question)]},
                config=config,
                stream_mode=["updates", "messages"],
            ):
                if modo == "updates":
                    for nodo, salida in dato.items():
                        for msg in salida.get("messages", []):
                            # El LLM decidió llamar una o más tools
                            for tc in getattr(msg, "tool_calls", []) or []:
                                print(f"  [TOOL] {tc['name']}({tc['args']})")
                            # Resultado que regresó una tool
                            if nodo == "tools" and hasattr(msg, "content"):
                                preview = str(msg.content).replace("\n", " ")[:110]
                                print(f"  [RESULTADO] {preview}...")
                elif modo == "messages":
                    chunk, metadata = dato
                    if metadata.get("langgraph_node") == "agent" and chunk.content:
                        print(chunk.content, end="", flush=True)
            print("\n")
    finally:
        # Sin esto el cliente async de Cosmos se queja al salir.
        await close_checkpointer()


if __name__ == "__main__":
    asyncio.run(main())
