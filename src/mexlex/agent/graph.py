"""Fase 4: el agente ReAct con tools (LangGraph).

Diferencia con el grafo de la Fase 3:

    Fase 3 (flujo fijo):
        pregunta -> contextualize -> retrieve -> generate -> respuesta
        (SIEMPRE busca, exactamente una vez)

    Fase 4 (agente):
        pregunta -> [LLM decide] --tool--> ejecuta tool --+
                         ^                                |
                         +--------------------------------+
                         |
                         +--sin tool--> respuesta
        (busca 0, 1 o N veces, y elige QUÉ tool usar)

Ese ciclo es el patrón **ReAct** (Reasoning + Acting): el modelo alterna
entre razonar y ejecutar acciones hasta que decide que ya puede
responder. `create_react_agent` lo construye por nosotros; escribirlo a
mano con StateGraph sería un nodo "agent", un nodo "tools" y una arista
condicional entre ellos.
"""

from __future__ import annotations

import logging

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from mexlex.agent.prompts import AGENT_SYSTEM_PROMPT
from mexlex.chains.simple_rag_chain import get_llm
from mexlex.tools.article_tool import expandir_contexto, obtener_articulo
from mexlex.tools.legal_search_tool import buscar_en_leyes
from mexlex.tools.web_search_tool import get_web_search_tool

logger = logging.getLogger(__name__)


def get_tools() -> list:
    """Arma la lista de tools disponibles.

    Las tres del corpus siempre están; la web solo si hay TAVILY_API_KEY.
    El prompt del agente ya contempla ambos escenarios.
    """
    tools = [buscar_en_leyes, obtener_articulo, expandir_contexto]

    web_tool = get_web_search_tool()
    if web_tool is not None:
        tools.append(web_tool)

    logger.info("Tools del agente: %s", [t.name for t in tools])
    return tools


def build_agent(checkpointer: BaseCheckpointSaver | None = None):
    """Compila el agente ReAct con memoria.

    Igual que en la Fase 3, se invoca con un thread_id:

        config = {"configurable": {"thread_id": "sesion-123"}}
        agent.invoke({"messages": [HumanMessage("...")]}, config=config)

    Nota: el agente ya NO necesita el nodo `contextualize` de la Fase 3.
    Como el LLM ve el historial y él mismo redacta el argumento de la
    tool, la reformulación de preguntas de seguimiento ocurre sola (el
    system prompt se lo pide explícitamente).

    Desde la Fase 7 el checkpointer se **inyecta** en vez de importarse:
    construirlo abre una conexión a Cosmos DB y eso es async, mientras
    que esta función no lo es. Quien llama lo obtiene una vez con
    `await get_checkpointer()` y lo reparte.

    ⚠️ Sin argumento cae a un `MemorySaver` **nuevo por llamada**. Es lo
    correcto para tests y pruebas sueltas, y es una trampa si se llama
    una vez por sesión: cada sesión tendría su propia memoria aislada.
    """
    return create_react_agent(
        model=get_llm(),
        tools=get_tools(),
        prompt=AGENT_SYSTEM_PROMPT,
        checkpointer=checkpointer or MemorySaver(),
    )
