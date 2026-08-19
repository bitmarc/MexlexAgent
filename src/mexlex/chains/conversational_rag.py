"""Fase 3: RAG conversacional con memoria (LangGraph + checkpointer).

Diferencia con `simple_rag_chain.py` (Fase 1):

    Fase 1:  pregunta -> retrieve -> generate -> respuesta
    Fase 3:  pregunta -> contextualize -> retrieve -> generate -> respuesta
                              ^                          ^
                              |                          |
                        usa el historial          usa el historial

El nodo `contextualize` es la pieza que hace que la memoria realmente
sirva en un RAG. Sin él, una pregunta de seguimiento como "¿y el
siguiente artículo?" se mandaría tal cual al retriever, que no tiene
forma de saber a qué artículo te refieres: recuperaría basura y el LLM
respondería mal aunque "recordara" la conversación.

Usamos un grafo de LangGraph en vez de una chain LCEL porque el
checkpointer (la memoria) es una funcionalidad del grafo. Además deja
el terreno listo para la Fase 4: ahí este mismo grafo crece para que el
LLM *decida* si buscar o no, en vez de buscar siempre.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage, BaseMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from mexlex.chains.simple_rag_chain import SYSTEM_PROMPT, get_llm
from mexlex.config import settings
from mexlex.retrieval.formatting import format_docs
from mexlex.retrieval.vectorstore import get_retriever

CONTEXTUALIZE_PROMPT = """\
Dado el historial de conversación y la última pregunta del usuario, \
reformula la pregunta para que se entienda por sí sola, sin necesidad \
de leer el historial.

NO respondas la pregunta: solo reescríbela. Si ya se entiende por sí \
sola, devuélvela tal cual, sin cambios.

Ejemplo:
  Historial: "¿Qué dice el artículo 10 de la Ley de Cinematografía?"
  Pregunta:  "¿y el 11?"
  Salida:    "¿Qué dice el artículo 11 de la Ley Federal de Cinematografía?"\
"""

contextualize_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", CONTEXTUALIZE_PROMPT),
        MessagesPlaceholder("messages"),
    ]
)

# Reutilizamos el system prompt de la Fase 1 (mismas reglas de citación
# y de no inventar), pero ahora el historial entra como mensajes reales
# en vez de perderse.
generate_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("messages"),
    ]
)


class RagState(TypedDict):
    """El estado que viaja entre nodos y que el checkpointer persiste.

    `messages` lleva el reducer `add_messages`: cuando un nodo regresa
    {"messages": [x]}, LangGraph **agrega** x a la lista en vez de
    reemplazarla. Los otros campos sí se sobreescriben en cada turno.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    search_query: str
    context: str


def _trim(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Recorta el historial para que no crezca sin límite.

    Con `token_counter=len` contamos *mensajes*, no tokens: más simple
    de razonar para aprender. `strategy="last"` conserva los más
    recientes y `start_on="human"` evita empezar el historial recortado
    con una respuesta del asistente colgando sin su pregunta.
    """
    return trim_messages(
        messages,
        max_tokens=settings.memory_max_messages,
        strategy="last",
        token_counter=len,
        start_on="human",
        include_system=False,
        allow_partial=False,
    )


async def contextualize(state: RagState) -> dict:
    """Convierte la última pregunta en una consulta autónoma."""
    messages = state["messages"]
    ultima = messages[-1].content

    # En el primer turno no hay historial que resolver: nos ahorramos
    # una llamada al LLM (y su latencia) usando la pregunta tal cual.
    if len(messages) == 1:
        return {"search_query": ultima}

    chain = contextualize_prompt | get_llm()
    respuesta = await chain.ainvoke({"messages": _trim(messages)})
    return {"search_query": respuesta.content}


async def retrieve(state: RagState) -> dict:
    """Busca en Azure AI Search usando la consulta ya contextualizada."""
    docs = await get_retriever().ainvoke(state["search_query"])
    return {"context": format_docs(docs)}


async def generate(state: RagState) -> dict:
    """Redacta la respuesta con el contexto recuperado + el historial."""
    chain = generate_prompt | get_llm()

    # Usamos astream (en vez de ainvoke) para que los tokens salgan del
    # nodo conforme se generan: es lo que la UI consume con
    # stream_mode="messages".
    respuesta = None
    async for chunk in chain.astream(
        {"context": state["context"], "messages": _trim(state["messages"])}
    ):
        respuesta = chunk if respuesta is None else respuesta + chunk

    return {"messages": [respuesta]}


def build_conversational_rag_graph(checkpointer: BaseCheckpointSaver | None = None):
    """Compila el grafo con memoria.

    Se invoca SIEMPRE con un thread_id, que es lo que separa e
    identifica cada conversación:

        config = {"configurable": {"thread_id": "sesion-123"}}
        graph.invoke({"messages": [HumanMessage("...")]}, config=config)

    El checkpointer se inyecta (Fase 7): construir el de Cosmos DB es
    async y esta función no lo es. Sin argumento cae a un `MemorySaver`
    nuevo, que sirve para pruebas sueltas pero no persiste nada.
    """
    builder = StateGraph(RagState)

    builder.add_node("contextualize", contextualize)
    builder.add_node("retrieve", retrieve)
    builder.add_node("generate", generate)

    # Flujo lineal: en la Fase 3 el grafo todavía no decide nada, solo
    # nos da el checkpointer. Las aristas condicionales llegan en la
    # Fase 4.
    builder.add_edge(START, "contextualize")
    builder.add_edge("contextualize", "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)

    return builder.compile(checkpointer=checkpointer or MemorySaver())
