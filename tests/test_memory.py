"""Tests de la memoria conversacional que NO requieren Azure."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from langchain_core.messages import AIMessage, HumanMessage

from mexlex.chains.conversational_rag import _trim
from mexlex.config import settings


def _conversacion(n_turnos: int):
    """Genera n turnos de pregunta + respuesta."""
    mensajes = []
    for i in range(n_turnos):
        mensajes.append(HumanMessage(content=f"pregunta {i}"))
        mensajes.append(AIMessage(content=f"respuesta {i}"))
    return mensajes


def test_trim_no_recorta_conversaciones_cortas():
    mensajes = _conversacion(2)  # 4 mensajes
    assert _trim(mensajes) == mensajes


def test_trim_limita_conversaciones_largas():
    mensajes = _conversacion(20)  # 40 mensajes
    recortado = _trim(mensajes)
    assert len(recortado) <= settings.memory_max_messages


def test_trim_conserva_los_mensajes_mas_recientes():
    mensajes = _conversacion(20)
    recortado = _trim(mensajes)
    # El último mensaje siempre debe sobrevivir: es el turno actual.
    assert recortado[-1] is mensajes[-1]


def test_trim_empieza_en_un_mensaje_humano():
    # Un historial que empiece con una respuesta del asistente sin su
    # pregunta confunde al modelo (y algunas APIs lo rechazan).
    mensajes = _conversacion(20)
    recortado = _trim(mensajes)
    assert isinstance(recortado[0], HumanMessage)
