"""Fase 6: evaluación del agente sobre un set de casos conocidos.

Con un flujo fijo (fases 1-3) bastaba con probar a mano: siempre hacía
lo mismo. Un agente **decide**, y sus decisiones dependen del prompt y
del modelo, así que un cambio aparentemente inocuo en el system prompt
puede romper un caso que ya funcionaba sin que te enteres.

Esto es una suite de regresión de comportamiento: corre el agente sobre
`data/eval/casos.json` y verifica cuatro cosas que sí se pueden medir
sin un juez humano:

1. **Elección de tool** — ¿usó `obtener_articulo` cuando había un número?
2. **Artículo recuperado** — ¿las fuentes citadas incluyen el artículo
   pedido? (esto es lo que fallaba antes de la Fase 5)
3. **Contenido** — ¿la respuesta menciona los términos esperados?
4. **Honestidad** — ante un artículo inexistente, ¿admite que no lo
   tiene en vez de inventarlo?

Los evaluadores son deterministas a propósito: no usan un LLM como juez.
Son más limitados, pero no cuestan tokens ni introducen varianza — y
para regresión eso importa más que la sofisticación.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.messages import HumanMessage

from mexlex.config import PROJECT_ROOT
from mexlex.observability import run_config

CASOS_DEFAULT = PROJECT_ROOT / "data" / "eval" / "casos.json"

# Frases con las que el agente admite no tener algo. Si la respuesta a un
# artículo inexistente NO contiene ninguna, probablemente lo inventó.
SEÑALES_DE_IGNORANCIA = [
    "no encontré",
    "no encontre",
    "no existe",
    "no aparece",
    "no está",
    "no esta",
    "no tengo",
    "no figura",
    "no se encontró",
    "no se encontro",
]


@dataclass
class Resultado:
    """Lo que pasó al correr un caso, ya evaluado."""

    id: str
    pregunta: str
    respuesta: str
    tools: list[str] = field(default_factory=list)
    articulos: list[int] = field(default_factory=list)
    fallos: list[str] = field(default_factory=list)

    @property
    def paso(self) -> bool:
        return not self.fallos


def cargar_casos(ruta: Path | None = None) -> list[dict]:
    ruta = ruta or CASOS_DEFAULT
    return json.loads(ruta.read_text(encoding="utf-8"))


async def _ejecutar(agent, pregunta: str, config: dict) -> tuple[str, list[str], list[int]]:
    """Corre un turno y devuelve (respuesta, tools usadas, artículos citados)."""
    respuesta: list[str] = []
    tools: list[str] = []
    articulos: list[int] = []

    async for modo, dato in agent.astream(
        {"messages": [HumanMessage(content=pregunta)]},
        config=config,
        stream_mode=["updates", "messages"],
    ):
        if modo == "updates":
            for nodo, salida in dato.items():
                for msg in salida.get("messages", []) or []:
                    for tc in getattr(msg, "tool_calls", []) or []:
                        tools.append(tc["name"])
                    # Los artículos realmente recuperados vienen del
                    # artifact, no del texto: es la señal más confiable de
                    # si el retrieval trajo lo correcto.
                    if nodo == "tools":
                        for fuente in getattr(msg, "artifact", None) or []:
                            if isinstance(fuente, dict):
                                articulos.extend(fuente.get("articulos") or [])
        elif modo == "messages":
            chunk, meta = dato
            if meta.get("langgraph_node") == "agent" and chunk.content:
                respuesta.append(chunk.content)

    return "".join(respuesta), tools, sorted(set(articulos))


def _evaluar(caso: dict, respuesta: str, tools: list[str], articulos: list[int]) -> list[str]:
    """Aplica los criterios del caso y regresa la lista de fallos."""
    fallos = []
    texto = respuesta.lower()

    # 1. Elección de tool
    if "tools_esperadas" in caso:
        if sorted(tools) != sorted(caso["tools_esperadas"]):
            fallos.append(f"tools={tools}, esperadas={caso['tools_esperadas']}")
    elif caso.get("tool_esperada") and caso["tool_esperada"] not in tools:
        fallos.append(f"no usó {caso['tool_esperada']} (usó {tools or 'ninguna'})")

    # 2. Artículo recuperado
    esperado = caso.get("articulo_esperado")
    if esperado is not None and esperado not in articulos:
        fallos.append(f"no recuperó el artículo {esperado} (trajo {articulos})")

    # 3. Contenido esperado.
    #
    # Un LLM parafrasea: puede escribir "10%" donde la ley dice "diez por
    # ciento", y ambas respuestas son correctas. Por eso un término puede
    # ser una lista de alternativas equivalentes, y basta con que aparezca
    # una. Sin esto, el evaluador reporta fallos que no lo son y deja de
    # ser útil como señal de regresión.
    for termino in caso.get("debe_contener", []):
        alternativas = termino if isinstance(termino, list) else [termino]
        if not any(alt.lower() in texto for alt in alternativas):
            fallos.append(f"la respuesta no menciona ninguna de {alternativas}")

    # 4. Honestidad ante lo que no existe
    if caso.get("debe_admitir_ignorancia"):
        if not any(s in texto for s in SEÑALES_DE_IGNORANCIA):
            fallos.append("no admitió que no tiene la información")

    return fallos


async def correr_caso(agent, caso: dict) -> Resultado:
    """Corre un caso en su propio thread (sin contaminación entre casos)."""
    config = run_config(str(uuid.uuid4()), caso["pregunta"])

    # Algunos casos (los de seguimiento) necesitan un turno previo para
    # que exista historial que resolver.
    if caso.get("contexto_previo"):
        await _ejecutar(agent, caso["contexto_previo"], config)

    respuesta, tools, articulos = await _ejecutar(agent, caso["pregunta"], config)

    return Resultado(
        id=caso["id"],
        pregunta=caso["pregunta"],
        respuesta=respuesta,
        tools=tools,
        articulos=articulos,
        fallos=_evaluar(caso, respuesta, tools, articulos),
    )
