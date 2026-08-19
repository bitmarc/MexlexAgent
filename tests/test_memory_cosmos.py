"""Tests del checkpointer (Fase 7) que NO requieren Cosmos DB ni Azure.

Lo que se puede verificar sin una cuenta de Cosmos es justo lo que más
fácil se rompe al refactorizar: que sin configuración la app degrade a
memoria en vez de reventar, que el checkpointer sea uno solo para todas
las sesiones, y que los grafos usen el que se les inyecta.

Lo que NO se cubre aquí (necesita Cosmos de verdad): que el estado
sobreviva al proceso. Para eso está `scripts/cosmos_memory_test.py`.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from langgraph.checkpoint.memory import MemorySaver

from mexlex.agent import memory
from mexlex.config import settings


@pytest.fixture(autouse=True)
async def checkpointer_limpio():
    """Descarta el singleton antes y después de cada test.

    Sin esto el primer test que corra dejaría su checkpointer cacheado y
    los siguientes verificarían el estado del anterior, no el suyo.
    """
    await memory.close_checkpointer()
    yield
    await memory.close_checkpointer()


@pytest.fixture
def sin_cosmos(monkeypatch):
    """Simula un .env sin AZURE_COSMOS_ENDPOINT."""
    monkeypatch.setattr(settings, "cosmos_endpoint", None)


async def test_sin_endpoint_cae_a_memoria(sin_cosmos):
    # El criterio de toda la config opcional del proyecto (Tavily,
    # LangSmith): sin credenciales la app arranca, solo pierde la
    # funcionalidad.
    assert isinstance(await memory.get_checkpointer(), MemorySaver)


async def test_el_fallback_avisa_en_el_log(sin_cosmos, caplog):
    # El fallback es silencioso salvo por este warning. Si se pierde, se
    # puede desplegar creyendo que persiste cuando no.
    with caplog.at_level("WARNING", logger="mexlex.agent.memory"):
        await memory.get_checkpointer()
    assert "AZURE_COSMOS_ENDPOINT" in caplog.text


async def test_el_checkpointer_es_singleton(sin_cosmos):
    # Si cada sesión creara el suyo, no habría conversación que
    # recuperar: el aislamiento lo da el thread_id, no tener savers
    # separados.
    primero = await memory.get_checkpointer()
    segundo = await memory.get_checkpointer()
    assert primero is segundo


async def test_close_descarta_el_singleton(sin_cosmos):
    primero = await memory.get_checkpointer()
    await memory.close_checkpointer()
    assert await memory.get_checkpointer() is not primero


async def test_close_es_idempotente():
    # Se llama desde `finally` y desde el shutdown de Chainlit: tiene que
    # tolerar que no haya nada abierto.
    await memory.close_checkpointer()
    await memory.close_checkpointer()


async def test_el_agente_usa_el_checkpointer_inyectado():
    from mexlex.agent.graph import build_agent

    saver = MemorySaver()
    assert build_agent(saver).checkpointer is saver


async def test_sin_inyectar_cada_agente_tiene_su_propia_memoria():
    # Comportamiento deliberado y documentado en build_agent: sirve para
    # pruebas sueltas, pero llamar build_agent() una vez por sesión daría
    # a cada usuario una memoria distinta. La app siempre inyecta.
    from mexlex.agent.graph import build_agent

    assert build_agent().checkpointer is not build_agent().checkpointer


async def test_el_grafo_de_la_fase3_tambien_acepta_inyeccion():
    from mexlex.chains.conversational_rag import build_conversational_rag_graph

    saver = MemorySaver()
    assert build_conversational_rag_graph(saver).checkpointer is saver


def test_la_config_de_cosmos_es_opcional():
    # Ningún campo de Cosmos debe ser obligatorio, o un .env sin Cosmos
    # haría fallar el import de `settings` en todo el proyecto.
    campos = type(settings).model_fields
    for nombre in ("cosmos_endpoint", "cosmos_key", "cosmos_ttl_seconds"):
        assert not campos[nombre].is_required(), f"{nombre} no debería ser obligatorio"
