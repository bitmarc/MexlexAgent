"""Fase 6: trazabilidad con LangSmith.

LangSmith es el observability de LangChain: registra cada ejecución
(cada llamada al LLM, cada tool, cada nodo del grafo) con sus entradas,
salidas, latencia y tokens, y lo muestra como un árbol navegable.

Se vuelve especialmente útil desde la Fase 4: con un flujo fijo la traza
era predecible, pero un agente **decide** — y cuando responde algo raro,
lo que quieres saber es *por qué eligió esa tool y con qué argumentos*.

## Cómo se activa

No hay que instrumentar nada: LangChain y LangGraph ya emiten los
eventos. Solo hacen falta variables de entorno:

    LANGSMITH_TRACING=true
    LANGSMITH_API_KEY=lsv2_...        # de https://smith.langchain.com
    LANGSMITH_PROJECT=mexlex-agent

⚠️ Detalle importante en este proyecto: `pydantic-settings` lee el .env
por su cuenta, pero **no** lo exporta a `os.environ`, y el tracer de
LangChain lee de `os.environ`. Por eso `setup_tracing()` copia los
valores explícitamente; sin eso, poner las variables en el .env no
tendría ningún efecto.

Si no configuras nada, el tracing queda apagado y la app funciona igual.
"""

from __future__ import annotations

import logging
import os

from mexlex.config import settings

logger = logging.getLogger(__name__)


def setup_tracing() -> bool:
    """Activa el envío de trazas a LangSmith si está configurado.

    Regresa True si quedó activo. Es idempotente: llamarla varias veces
    no hace daño.
    """
    if not settings.langsmith_tracing:
        logger.info("LangSmith: tracing desactivado.")
        return False

    if not settings.langsmith_api_key:
        logger.warning(
            "LANGSMITH_TRACING=true pero falta LANGSMITH_API_KEY: "
            "las trazas no se van a enviar."
        )
        return False

    # El tracer de LangChain lee de os.environ, no de nuestro settings.
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    if settings.langsmith_endpoint:
        os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint

    logger.info("LangSmith: trazas activas en el proyecto '%s'.", settings.langsmith_project)
    return True


def run_config(thread_id: str, pregunta: str) -> dict:
    """Config de ejecución con metadata útil para buscar en LangSmith.

    Sin esto las trazas se ven como una lista de "RunnableSequence" sin
    contexto. Con `run_name`, `tags` y `metadata` puedes filtrar por
    conversación o por tipo de consulta desde la UI de LangSmith.

    El `thread_id` va en los dos lados a propósito:
    - dentro de `configurable`, porque es lo que usa el checkpointer;
    - dentro de `metadata`, porque es lo que hace la traza buscable.
    """
    return {
        "configurable": {"thread_id": thread_id},
        "run_name": "mexlex-agent",
        "tags": ["mexlex", "agente"],
        "metadata": {
            "thread_id": thread_id,
            # Guardar la pregunta como metadata permite encontrar la traza
            # de un caso concreto sin ir abriendo runs uno por uno.
            "pregunta": pregunta[:200],
        },
    }
