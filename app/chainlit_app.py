"""Fase 4: interfaz de chat sobre el agente ReAct con tools.

Esta capa sigue siendo *solo UI*. Lo que cambió respecto a la Fase 3:

- Invoca el agente (`mexlex.agent.graph`) en vez del grafo de flujo fijo.
- Muestra en la UI qué tool decidió usar el agente y con qué consulta,
  usando `cl.Step`. Sin eso, el usuario ve una pausa larga sin saber si
  el agente está buscando o si se colgó.

Fase 7: la memoria dejó de vivir en RAM. El checkpointer se abre una vez
por proceso (`on_app_startup`), se comparte entre todas las sesiones y se
cierra al apagar. Y el `thread_id` pasó a ser el de la *conversación*, no
el de la conexión — ver `on_chat_start`.

Fase 8: el historial se ve. Se agregan tres piezas que van juntas y no
sirven por separado:

    login (@cl.password_auth_callback)   -> Chainlit sabe de QUIÉN es el hilo
    data layer (@cl.data_layer)          -> guarda hilos y mensajes de la UI
    @cl.on_chat_resume                   -> rehidrata al abrir uno viejo

Las tres se registran **solo si están configuradas**: sin credenciales o
sin Cosmos, la app corre exactamente como en la Fase 7.

Uso:
    chainlit run app/chainlit_app.py -w
"""

import sys
from pathlib import Path

# app/ vive fuera de src/ a propósito (es interfaz, no lógica de negocio),
# así que agregamos src/ al path igual que hacen los scripts de CLI.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import chainlit as cl  # noqa: E402
from langchain_core.messages import HumanMessage  # noqa: E402

from mexlex.agent.graph import build_agent  # noqa: E402
from mexlex.agent.memory import close_checkpointer, get_checkpointer  # noqa: E402
from mexlex.observability import run_config, setup_tracing  # noqa: E402
from mexlex.persistence.auth import credenciales_validas, setup_auth  # noqa: E402
from mexlex.persistence.cosmos import cosmos_configurado  # noqa: E402

# Si hay LANGSMITH_API_KEY en el .env, cada conversación queda trazada.
# Si no, no pasa nada: la app funciona igual.
setup_tracing()

# Exporta CHAINLIT_AUTH_SECRET a os.environ, que es donde Chainlit lo
# busca. Regresa False (y no se registra el login) si falta algo.
AUTH_ACTIVA = setup_auth()

# Cuánto texto del resultado de una tool mostramos en el Step desplegable.
TOOL_OUTPUT_PREVIEW = 800

# El checkpointer sostiene una conexión a Cosmos DB, así que su vida es
# la del proceso, no la de una sesión de chat.
_checkpointer = None
_data_layer = None


# --- Fase 8: login e historial ------------------------------------------
#
# Los decoradores se aplican dentro de un `if` a propósito. Registrar
# `password_auth_callback` enciende la pantalla de login para TODO el
# mundo (`require_login()` en Chainlit solo mira si el callback existe),
# así que sin credenciales en el .env no debe registrarse.

if AUTH_ACTIVA:

    @cl.password_auth_callback
    async def auth_callback(usuario: str, password: str) -> cl.User | None:
        if credenciales_validas(usuario, password):
            return cl.User(identifier=usuario, metadata={"role": "user"})
        # None = credenciales rechazadas. Chainlit devuelve el error a la
        # pantalla de login sin decir cuál de los dos campos falló.
        return None


# El data layer necesita las dos cosas: Cosmos donde escribir y un
# usuario de quién colgar los hilos. Sin login, Chainlit nunca llamaría a
# `resume_thread` y solo acumularíamos documentos que nadie puede leer.
if AUTH_ACTIVA and cosmos_configurado():

    @cl.data_layer
    def data_layer():
        """Chainlit cachea el resultado: esto corre una sola vez."""
        global _data_layer
        from mexlex.persistence.data_layer import CosmosDataLayer

        _data_layer = CosmosDataLayer()
        return _data_layer


@cl.set_starters
async def starters() -> list[cl.Starter]:
    """Preguntas sugeridas que se muestran en la pantalla inicial."""
    return [
        cl.Starter(
            label="Artículo 4 constitucional",
            message="¿Qué derechos garantiza el artículo 4 de la CPEUM?",
        ),
        cl.Starter(
            label="Requisitos de una ley",
            message="¿Qué requisitos establece la ley para producir una película en México?",
        ),
        cl.Starter(
            label="Buscar por tema",
            message="¿Qué dice la ley sobre la libertad de expresión?",
        ),
    ]


@cl.on_app_startup
async def on_app_startup() -> None:
    """Se ejecuta una vez por proceso, antes de atender a nadie.

    Aquí se abre la conexión a Cosmos DB. Hacerlo en `on_chat_start`
    abriría un cliente por pestaña y ninguno se cerraría nunca.
    """
    global _checkpointer
    _checkpointer = await get_checkpointer()


@cl.on_app_shutdown
async def on_app_shutdown() -> None:
    """Cierra los clientes de Cosmos al apagar el servidor."""
    await close_checkpointer()
    if _data_layer is not None:
        await _data_layer.close()


def _preparar_sesion() -> None:
    """Deja lista la sesión: agente + thread_id.

    Es idéntico para una conversación nueva y para una reanudada, y esa
    es justo la gracia: como el `thread_id` que usa el checkpointer es el
    de la conversación, al reabrir un hilo viejo el agente rehidrata su
    estado solo. No hay que reinyectarle los mensajes a mano.
    """
    # El checkpointer es compartido a propósito: lo que aísla una
    # conversación de otra es el thread_id, no tener savers separados.
    cl.user_session.set("agent", build_agent(_checkpointer))
    # `session.thread_id` es el id de la CONVERSACIÓN; el de
    # `user_session.get("id")` es el de la conexión de websocket, que
    # cambia en cada recarga de la página. Con el segundo, Cosmos se
    # llenaría de hilos huérfanos de un solo turno que nadie puede
    # volver a abrir.
    cl.user_session.set("thread_id", cl.context.session.thread_id)


@cl.on_chat_start
async def on_chat_start() -> None:
    """Se ejecuta una vez por sesión de chat (cada pestaña del navegador)."""
    _preparar_sesion()


@cl.on_chat_resume
async def on_chat_resume(thread: cl.types.ThreadDict) -> None:
    """Se ejecuta al abrir una conversación de la barra lateral.

    Chainlit ya repintó los mensajes del hilo (los sacó del data layer)
    antes de llamarnos; lo que falta es la parte del agente, y eso es
    solo volver a armar la sesión. El `thread` llega por si hiciera falta
    inspeccionarlo, pero aquí no lo necesitamos: el `thread_id` de la
    sesión ya apunta a la conversación correcta.
    """
    _preparar_sesion()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """Se ejecuta con cada mensaje que escribe el usuario."""
    agent = cl.user_session.get("agent")
    # Además del thread_id que necesita el checkpointer, esto etiqueta la
    # traza en LangSmith para poder encontrarla después por conversación.
    config = run_config(cl.user_session.get("thread_id"), message.content)

    # La burbuja de respuesta se crea hasta el primer token, no antes.
    # Así los Steps de las tools (que ocurren primero) aparecen ARRIBA de
    # la respuesta y no debajo.
    answer: cl.Message | None = None
    pasos: dict[str, cl.Step] = {}
    # Fuentes citadas, deduplicadas por `ref` y en orden de aparición.
    fuentes: dict[str, dict] = {}

    try:
        # Dos modos de stream a la vez:
        # - "updates": qué produjo cada nodo -> para detectar tool calls
        # - "messages": los tokens del LLM   -> para la respuesta en vivo
        async for modo, dato in agent.astream(
            {"messages": [HumanMessage(content=message.content)]},
            config=config,
            stream_mode=["updates", "messages"],
        ):
            if modo == "updates":
                for nodo, salida in dato.items():
                    for msg in salida.get("messages", []) or []:
                        # a) El agente decidió llamar una tool
                        for tool_call in getattr(msg, "tool_calls", []) or []:
                            paso = cl.Step(name=tool_call["name"], type="tool")
                            paso.input = tool_call["args"]
                            await paso.send()
                            pasos[tool_call["id"]] = paso

                        # b) La tool regresó su resultado: lo colgamos del
                        #    Step correspondiente (se casan por tool_call_id).
                        if nodo == "tools":
                            paso = pasos.get(getattr(msg, "tool_call_id", ""))
                            if paso is not None:
                                paso.output = str(msg.content)[:TOOL_OUTPUT_PREVIEW]
                                await paso.update()

                            # c) Las tools de búsqueda devuelven las fuentes
                            #    como `artifact`: datos estructurados que el
                            #    LLM NO ve (no gastan tokens) y que aquí se
                            #    convierten en el panel de fuentes.
                            for fuente in getattr(msg, "artifact", None) or []:
                                clave = fuente.get("ref") or fuente.get("titulo", "")
                                if clave:
                                    fuentes.setdefault(clave, fuente)

            elif modo == "messages":
                chunk, metadata = dato
                # Solo los tokens del nodo que redacta la respuesta final.
                if metadata.get("langgraph_node") == "agent" and chunk.content:
                    if answer is None:
                        answer = cl.Message(content="")
                        await answer.send()
                    await answer.stream_token(chunk.content)

    except Exception as exc:  # noqa: BLE001
        # Sin esto, un fallo de Azure deja la UI en blanco y el stack
        # trace solo aparece en la terminal.
        if answer is None:
            answer = cl.Message(content="")
            await answer.send()
        answer.content = (
            "⚠️ Ocurrió un error al consultar el asistente.\n\n"
            f"```\n{type(exc).__name__}: {exc}\n```"
        )
        await answer.update()
        raise

    if answer is not None:
        _adjuntar_fuentes(answer, fuentes)
        await answer.update()


def _adjuntar_fuentes(answer: cl.Message, fuentes: dict[str, dict]) -> None:
    """Cuelga del mensaje los fragmentos exactos que sustentan la respuesta.

    Los elementos `display="side"` de Chainlit se abren en un panel lateral
    al hacer clic sobre su nombre EN EL TEXTO del mensaje, así que hay que
    listar los nombres al final de la respuesta para que sean clicables.
    """
    if not fuentes:
        return

    elementos = []
    lineas = []
    vigencias: set[str] = set()
    hubo_web = False

    for fuente in fuentes.values():
        if fuente.get("origen") == "web":
            hubo_web = True

        nombre = fuente.get("titulo") or fuente.get("cita", "fuente")
        elementos.append(
            cl.Text(name=nombre, content=fuente.get("texto", ""), display="side")
        )
        lineas.append(f"- {nombre}")

        if fuente.get("vigencia"):
            vigencias.add(fuente["vigencia"])

    answer.elements = elementos
    answer.content += "\n\n---\n**Fuentes consultadas**\n" + "\n".join(lineas)

    # Disclaimers con contenido, no genéricos: la fecha real de vigencia
    # de los textos citados y un aviso solo si se salió a internet.
    avisos = []
    if vigencias:
        avisos.append(
            "Textos vigentes al " + ", ".join(sorted(vigencias)) + "; "
            "pueden existir reformas posteriores."
        )
    if hubo_web:
        avisos.append(
            "Parte de la información proviene de búsqueda web, no del "
            "corpus oficial indexado."
        )
    avisos.append("Esto no sustituye asesoría legal profesional.")

    answer.content += "\n\n<small>⚠️ " + " ".join(avisos) + "</small>"
