"""Tests del data layer de Chainlit (Fase 8) que NO requieren Cosmos DB.

El data layer es sobre todo lógica: traducir entre los dicts que Chainlit
espera y los documentos que guardamos, ordenar, paginar y no perder
campos al actualizar. Todo eso se prueba contra `FakeContainer`, un doble
en memoria (ver `fake_cosmos.py`).

Lo que NO se cubre: que Cosmos acepte las consultas tal cual. El doble
las reconoce por forma, no las ejecuta.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chainlit.types import Feedback, Pagination, ThreadFilter
from chainlit.user import User

from fake_cosmos import FakeContainer  # noqa: E402  (tests/ está en el path)
from mexlex.agent import memory
from mexlex.config import settings
from mexlex.persistence.data_layer import (
    CARACTERES_PROHIBIDOS,
    CosmosDataLayer,
    _id_paso,
    _id_valido,
    _pk_hilo,
)


@pytest.fixture(autouse=True)
def contexto_chainlit():
    """Un contexto de Chainlit mínimo, o los métodos de paso no corren.

    `create_step`, `update_step` y compañía llevan `@queue_until_user_message`,
    que consulta `context.session` para decidir si escribe ya o si encola
    hasta el primer mensaje del usuario. Sin contexto lanza excepción.

    La sesión falsa **no** es una `WebsocketSession` a propósito: así el
    decorador ejecuta de inmediato y los tests ven el efecto sin tener
    que simular el ciclo de vida entero de una conversación.
    """
    from chainlit.context import context_var

    class _Sesion:
        thread_id = "t1"

    class _Contexto:
        session = _Sesion()

    token = context_var.set(_Contexto())
    yield
    context_var.reset(token)


@pytest.fixture
def capa():
    """Un CosmosDataLayer con el contenedor sustituido por el doble.

    Se saltea `__init__` con `__new__` para no construir un CosmosClient:
    no hay endpoint que darle y tampoco hace falta.
    """
    dl = CosmosDataLayer.__new__(CosmosDataLayer)
    dl._cont = FakeContainer()
    dl._cliente = None
    dl._credencial = None
    return dl


@pytest.fixture(autouse=True)
async def checkpointer_limpio():
    """`delete_thread` toca el checkpointer; que no herede el de otro test."""
    await memory.close_checkpointer()
    yield
    await memory.close_checkpointer()


def _paso(paso_id: str, thread_id: str, creado: str, salida: str = "") -> dict:
    return {
        "id": paso_id,
        "threadId": thread_id,
        "name": "assistant",
        "type": "assistant_message",
        "createdAt": creado,
        "output": salida,
        "metadata": {},
    }


# --- usuarios --------------------------------------------------------- #


async def test_usuario_va_y_vuelve(capa):
    creado = await capa.create_user(User(identifier="marco", metadata={"role": "user"}))
    leido = await capa.get_user("marco")

    assert leido is not None
    assert leido.identifier == "marco"
    assert leido.metadata == {"role": "user"}
    assert leido.createdAt == creado.createdAt


async def test_el_id_del_usuario_es_su_identifier(capa):
    # Chainlit filtra los hilos por `user.id` pero autoriza por
    # `user.identifier`. Si los dos valores no coinciden, los hilos se
    # guardan pero no se listan: un bug silencioso y muy difícil de ver.
    usuario = await capa.create_user(User(identifier="marco"))
    assert usuario.id == usuario.identifier


async def test_usuario_inexistente_es_none(capa):
    assert await capa.get_user("nadie") is None


# --- hilos ------------------------------------------------------------ #


async def test_update_thread_no_pisa_lo_que_ya_estaba(capa):
    # Chainlit llama update_thread desde varios lados con subconjuntos
    # distintos de campos: primero el user_id al abrir la sesión, después
    # el nombre al primer mensaje. Si el segundo borrara el primero, el
    # hilo se quedaría sin dueño y desaparecería de la lista.
    await capa.update_thread("t1", user_id="marco")
    await capa.update_thread("t1", name="¿Qué dice el artículo 4?")

    hilo = await capa.get_thread("t1")
    assert hilo["userId"] == "marco"
    assert hilo["name"] == "¿Qué dice el artículo 4?"


async def test_update_thread_conserva_la_fecha_de_creacion(capa):
    await capa.update_thread("t1", user_id="marco")
    creado = (await capa.get_thread("t1"))["createdAt"]

    await capa.update_thread("t1", name="otro nombre")
    assert (await capa.get_thread("t1"))["createdAt"] == creado


async def test_hilo_inexistente_es_none(capa):
    assert await capa.get_thread("no-existe") is None


async def test_get_thread_ordena_los_pasos_por_fecha(capa):
    # La UI los pinta en el orden que le lleguen: si el orden se pierde,
    # la conversación se lee al revés.
    await capa.update_thread("t1", user_id="marco")
    await capa.create_step(_paso("s2", "t1", "2026-01-02T00:00:00Z"))
    await capa.create_step(_paso("s1", "t1", "2026-01-01T00:00:00Z"))
    await capa.create_step(_paso("s3", "t1", "2026-01-03T00:00:00Z"))

    hilo = await capa.get_thread("t1")
    assert [p["id"] for p in hilo["steps"]] == ["s1", "s2", "s3"]


async def test_get_thread_devuelve_los_pasos_tal_cual_los_dio_chainlit(capa):
    # El StepDict se guarda anidado y debe volver intacto: la UI lo
    # deserializa con `Message.from_dict`.
    original = _paso("s1", "t1", "2026-01-01T00:00:00Z", salida="El artículo 4...")
    await capa.update_thread("t1", user_id="marco")
    await capa.create_step(original)

    assert (await capa.get_thread("t1"))["steps"][0] == original


async def test_update_step_reemplaza_al_paso_creado(capa):
    # Un mensaje se crea vacío y se va llenando mientras el LLM streamea.
    await capa.update_thread("t1", user_id="marco")
    await capa.create_step(_paso("s1", "t1", "2026-01-01T00:00:00Z", salida=""))
    await capa.update_step(_paso("s1", "t1", "2026-01-01T00:00:00Z", salida="completo"))

    pasos = (await capa.get_thread("t1"))["steps"]
    assert len(pasos) == 1
    assert pasos[0]["output"] == "completo"


async def test_get_thread_author_devuelve_el_identifier(capa):
    # Chainlit lo compara contra `current_user.identifier` para autorizar.
    await capa.update_thread("t1", user_id="marco")
    assert await capa.get_thread_author("t1") == "marco"


async def test_get_thread_author_falla_si_no_hay_hilo(capa):
    with pytest.raises(ValueError):
        await capa.get_thread_author("no-existe")


# --- lista de conversaciones (la barra lateral) ----------------------- #


async def _tres_hilos(capa):
    for i, nombre in enumerate(["primero", "segundo", "tercero"], start=1):
        await capa.update_thread(f"t{i}", user_id="marco", name=nombre)
        # Fechas explícitas: sin esto los tres se crean en el mismo
        # milisegundo y el orden queda indefinido.
        doc = await capa._leer(_pk_hilo(f"t{i}"), "THREAD")
        doc["createdAt"] = f"2026-01-0{i}T00:00:00Z"
        await capa._cont.upsert_item(doc)


async def test_list_threads_devuelve_lo_mas_reciente_primero(capa):
    await _tres_hilos(capa)

    pagina = await capa.list_threads(
        Pagination(first=10), ThreadFilter(userId="marco")
    )
    assert [h["id"] for h in pagina.data] == ["t3", "t2", "t1"]
    assert pagina.pageInfo.hasNextPage is False


async def test_list_threads_solo_ve_los_hilos_del_usuario(capa):
    await capa.update_thread("mio", user_id="marco", name="mío")
    await capa.update_thread("ajeno", user_id="otra-persona", name="ajeno")

    pagina = await capa.list_threads(
        Pagination(first=10), ThreadFilter(userId="marco")
    )
    assert [h["id"] for h in pagina.data] == ["mio"]


async def test_list_threads_pagina_y_avisa_que_hay_mas(capa):
    await _tres_hilos(capa)

    primera = await capa.list_threads(
        Pagination(first=2), ThreadFilter(userId="marco")
    )
    assert [h["id"] for h in primera.data] == ["t3", "t2"]
    assert primera.pageInfo.hasNextPage is True

    segunda = await capa.list_threads(
        Pagination(first=2, cursor=primera.pageInfo.endCursor),
        ThreadFilter(userId="marco"),
    )
    assert [h["id"] for h in segunda.data] == ["t1"]
    assert segunda.pageInfo.hasNextPage is False


async def test_list_threads_filtra_por_texto(capa):
    await _tres_hilos(capa)

    pagina = await capa.list_threads(
        Pagination(first=10), ThreadFilter(userId="marco", search="segun")
    )
    assert [h["id"] for h in pagina.data] == ["t2"]


async def test_list_threads_no_carga_los_mensajes(capa):
    # La barra lateral solo necesita el título. Traer los pasos de cada
    # hilo la volvería carísima en RU.
    await _tres_hilos(capa)
    await capa.create_step(_paso("s1", "t1", "2026-01-01T00:00:00Z"))

    pagina = await capa.list_threads(
        Pagination(first=10), ThreadFilter(userId="marco")
    )
    assert all(h["steps"] == [] for h in pagina.data)


# --- borrado ---------------------------------------------------------- #


async def test_delete_thread_se_lleva_todo_el_hilo(capa):
    await capa.update_thread("t1", user_id="marco")
    await capa.create_step(_paso("s1", "t1", "2026-01-01T00:00:00Z"))
    await capa.create_step(_paso("s2", "t1", "2026-01-02T00:00:00Z"))

    await capa.delete_thread("t1")

    assert await capa.get_thread("t1") is None
    assert capa._cont.docs == {}


async def test_delete_thread_no_toca_otras_conversaciones(capa):
    await capa.update_thread("t1", user_id="marco")
    await capa.create_step(_paso("s1", "t1", "2026-01-01T00:00:00Z"))
    await capa.update_thread("t2", user_id="marco")

    await capa.delete_thread("t1")

    assert await capa.get_thread("t2") is not None


async def test_delete_thread_intenta_limpiar_los_checkpoints(capa, monkeypatch):
    # Sin esto el hilo desaparece de la UI pero el agente lo sigue
    # recordando: el estado vive en OTRO contenedor y nadie más lo borra.
    llamado = []

    async def espia(thread_id):
        llamado.append(thread_id)
        return 0

    monkeypatch.setattr(memory, "borrar_checkpoints_del_hilo", espia)

    await capa.update_thread("t1", user_id="marco")
    await capa.delete_thread("t1")

    assert llamado == ["t1"]


async def test_delete_thread_sobrevive_a_un_fallo_del_checkpointer(capa, monkeypatch):
    # Que no se pueda limpiar el estado no debe impedir que la
    # conversación desaparezca de la UI: para eso está el TTL.
    async def explota(_):
        raise RuntimeError("Cosmos caído")

    monkeypatch.setattr(memory, "borrar_checkpoints_del_hilo", explota)

    await capa.update_thread("t1", user_id="marco")
    await capa.delete_thread("t1")

    assert await capa.get_thread("t1") is None


# --- feedback --------------------------------------------------------- #


async def test_el_feedback_se_guarda_dentro_del_paso(capa):
    await capa.update_thread("t1", user_id="marco")
    await capa.create_step(_paso("s1", "t1", "2026-01-01T00:00:00Z"))

    fid = await capa.upsert_feedback(Feedback(forId="s1", threadId="t1", value=1))

    paso = (await capa.get_thread("t1"))["steps"][0]
    assert paso["feedback"]["value"] == 1
    assert fid == "t1::s1"


async def test_delete_feedback_lo_quita_del_paso(capa):
    await capa.update_thread("t1", user_id="marco")
    await capa.create_step(_paso("s1", "t1", "2026-01-01T00:00:00Z"))
    fid = await capa.upsert_feedback(Feedback(forId="s1", threadId="t1", value=0))

    assert await capa.delete_feedback(fid) is True
    assert "feedback" not in (await capa.get_thread("t1"))["steps"][0]


async def test_feedback_sobre_un_paso_que_no_existe_falla_claro(capa):
    with pytest.raises(ValueError):
        await capa.upsert_feedback(Feedback(forId="fantasma", threadId="t1", value=1))


# --- favoritos -------------------------------------------------------- #


async def test_get_favorite_steps_solo_trae_los_marcados(capa):
    await capa.update_thread("t1", user_id="marco")
    await capa.create_step(_paso("s1", "t1", "2026-01-01T00:00:00Z"))
    await capa.set_step_favorite(_paso("s2", "t1", "2026-01-02T00:00:00Z"), True)

    assert [p["id"] for p in await capa.get_favorite_steps("marco")] == ["s2"]


async def test_el_favorito_se_indexa_fuera_de_data(capa):
    # `data` está excluido del índice para abaratar las escrituras, así
    # que Cosmos rechazaría filtrar por `c.data.metadata.favorite`. El
    # flag tiene que estar en la raíz del documento.
    await capa.update_thread("t1", user_id="marco")
    await capa.set_step_favorite(_paso("s1", "t1", "2026-01-01T00:00:00Z"), True)

    doc = capa._cont.docs[(_pk_hilo("t1"), _id_paso("s1"))]
    assert doc["favorite"] is True


async def test_get_favorite_steps_sin_hilos_no_consulta_de_mas(capa):
    # Sin hilos no hay nada que buscar: la segunda consulta
    # (cross-partition, la cara) debe ni siquiera lanzarse.
    assert await capa.get_favorite_steps("marco") == []
    assert not any("c.entity = 'step'" in q for q in capa._cont.consultas)


# --- concurrencia ----------------------------------------------------- #


async def test_update_thread_reintenta_si_alguien_escribio_antes(capa):
    await capa.update_thread("t1", user_id="marco")

    # Simula que otra corrutina escribió entre nuestro read y nuestro
    # upsert: el _etag que llevábamos deja de ser válido.
    upsert_real = capa._cont.upsert_item
    fallos = {"quedan": 1}

    async def upsert_con_colision(body, **kwargs):
        if fallos["quedan"] and kwargs.get("etag"):
            fallos["quedan"] -= 1
            kwargs["etag"] = "etag-viejo"
        return await upsert_real(body, **kwargs)

    capa._cont.upsert_item = upsert_con_colision
    await capa.update_thread("t1", name="sobrevivió")

    assert (await capa.get_thread("t1"))["name"] == "sobrevivió"


# --- elementos (limitación conocida) ---------------------------------- #


async def test_los_elementos_no_se_persisten(capa):
    # El panel de fuentes son elementos cl.Text, cuyo contenido Chainlit
    # guarda en un blob storage, no aquí. Sin blob configurado no hay
    # nada que persistir; guardar solo la metadata dejaría el panel roto
    # al reanudar. Ver documentation/09.
    assert await capa.get_element("t1", "e1") is None
    assert capa._cont.docs == {}


# --- ids legales para Cosmos ------------------------------------------ #
#
# Cosmos prohíbe / \ ? # en el `id`. El upsert los acepta (el id va en el
# cuerpo) pero read_item y delete_item los meten en la URI y ahí se
# rompen, así que el síntoma aparece tarde y lejos de la causa. El data
# layer de DynamoDB usa `#` como separador y copiar ese patrón introdujo
# exactamente ese bug.


@pytest.mark.parametrize("prohibido", list(CARACTERES_PROHIBIDOS))
def test_los_helpers_de_id_no_generan_caracteres_prohibidos(prohibido):
    assert prohibido not in _id_paso("abc-123")


async def test_ningun_documento_escrito_lleva_un_id_ilegal(capa):
    await capa.update_thread("t1", user_id="marco", name="hola")
    await capa.create_user(User(identifier="marco"))
    await capa.create_step(_paso("s1", "t1", "2026-01-01T00:00:00Z"))

    for _, doc_id in capa._cont.docs:
        assert _id_valido(doc_id), f"'{doc_id}' rompería read_item y delete_item"


async def test_guardar_un_paso_con_id_ilegal_falla_de_inmediato(capa):
    # Vale más un error al escribir que un documento que se guarda y
    # después no se puede ni leer ni borrar.
    with pytest.raises(ValueError, match="prohíbe"):
        await capa.create_step(_paso("con#gato", "t1", "2026-01-01T00:00:00Z"))


async def test_el_paso_se_puede_releer_y_borrar_por_id(capa):
    # La prueba de que el id sirve para las operaciones que lo ponen en la
    # URI: si volviera el `#`, esto seguiría pasando en el doble pero
    # fallaría contra Cosmos. Por eso además va el test de arriba.
    await capa.update_thread("t1", user_id="marco")
    await capa.create_step(_paso("s1", "t1", "2026-01-01T00:00:00Z"))

    assert await capa._leer(_pk_hilo("t1"), _id_paso("s1")) is not None
    await capa.delete_step("s1")
    assert await capa._leer(_pk_hilo("t1"), _id_paso("s1")) is None


# --- configuración ---------------------------------------------------- #


def test_el_contenedor_de_la_ui_no_es_el_de_los_checkpoints():
    # Comparten cuenta y base, pero no contenedor: distinta forma de
    # documento, distintas consultas, distinto índice.
    assert settings.cosmos_threads_container != settings.cosmos_container
