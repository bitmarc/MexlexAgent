"""Chain RAG simple (Fase 1): sin agente, sin memoria, sin UI.

El objetivo de este módulo es exclusivamente validar que la ingesta y el
retrieval híbrido funcionan bien, usando LCEL puro (prompt | llm | parser).
En la Fase 4 esta misma lógica de retrieval se reutiliza pero envuelta
como una `tool` dentro de un agente de LangGraph.
"""

from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import AzureChatOpenAI

from mexlex.config import settings
from mexlex.retrieval.formatting import format_docs
from mexlex.retrieval.vectorstore import get_retriever

SYSTEM_PROMPT = """\
Eres un asistente que responde dudas sobre las leyes mexicanas y la \
Constitución Política de los Estados Unidos Mexicanos (CPEUM), basándote \
únicamente en el CONTEXTO proporcionado.

Reglas:
1. Si el contexto no contiene información suficiente para responder, dilo \
   explícitamente. No inventes artículos ni contenido legal.
2. Cuando cites una disposición, menciona el artículo y la ley/documento \
   de origen (ej. "Artículo 4, CPEUM").
3. Sé claro y directo. No eres un abogado y esto no es asesoría legal: \
   si la pregunta requiere un caso concreto, recomienda consultar a un \
   profesional.

CONTEXTO:
{context}
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ]
)


def get_llm() -> AzureChatOpenAI:
    return AzureChatOpenAI(
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
        azure_deployment=settings.azure_openai_chat_deployment,
        temperature=0,
    )


def build_simple_rag_chain():
    """Arma la chain: retriever -> prompt -> llm -> string.

    Regresa un Runnable invocable como: chain.invoke("¿qué dice el artículo 4?")
    """
    retriever = get_retriever()
    llm = get_llm()

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain
