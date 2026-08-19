"""Formateo de Documents recuperados para inyectarlos en un prompt.

Vive aquí (y no dentro de una chain concreta) porque todas las tools de
búsqueda necesitan exactamente el mismo formato de contexto.

Desde la Fase 5 la cita usa la metadata estructural en vez del nombre
del archivo:

    antes:  [Fuente: LFC.pdf, página 9]
    ahora:  [Ley Federal de Cinematografía · art. 19 · p. 3 · ref LFC#12]

El `ref` es lo que el agente le pasa a `expandir_contexto` cuando un
fragmento se quedó corto.
"""

from __future__ import annotations

from langchain_core.documents import Document


def _etiqueta_articulos(articulos: list[int] | None, seccion: str | None) -> str:
    """Describe qué artículos cubre un fragmento."""
    if not articulos:
        return seccion.lower() if seccion else "preámbulo"

    if len(articulos) == 1:
        return f"art. {articulos[0]}"

    # Los artículos de un chunk son consecutivos por construcción, así que
    # un rango se lee mejor que la lista completa.
    return f"arts. {articulos[0]}-{articulos[-1]}"


# Palabras que quedan en minúscula al normalizar el nombre de una ley.
_MINUSCULAS = {"de", "del", "la", "las", "los", "y", "en", "para", "por", "a"}


def nombre_legible(ley: str) -> str:
    """Convierte "LEY FEDERAL DE CINEMATOGRAFÍA" en "Ley Federal de Cinematografía".

    Los PDFs oficiales traen los títulos en mayúsculas. Citarlos así en el
    chat se lee como si el asistente estuviera gritando.
    """
    if not ley or not ley.isupper():
        return ley

    palabras = []
    for i, palabra in enumerate(ley.split()):
        minuscula = palabra.lower()
        # La primera palabra siempre va capitalizada, aunque sea "de".
        palabras.append(
            minuscula if i > 0 and minuscula in _MINUSCULAS else minuscula.capitalize()
        )
    return " ".join(palabras)


def titulo_de(doc: Document) -> str:
    """Cita legible para el usuario final, sin la `ref` interna."""
    m = doc.metadata
    ley = m.get("ley") or m.get("source") or "documento desconocido"
    partes = [
        nombre_legible(str(ley)),
        _etiqueta_articulos(m.get("articulos"), m.get("seccion")),
    ]

    if m.get("page") is not None:
        partes.append(f"p. {m['page']}")

    return " · ".join(partes)


def cita_de(doc: Document) -> str:
    """Encabezado citable de un fragmento (lo que ve el LLM).

    Incluye la `ref`, que el agente necesita para pedir contexto vecino.
    El usuario no la ve: para eso está `titulo_de`.
    """
    m = doc.metadata
    partes = [titulo_de(doc)]

    # La vigencia va en la cita que ve el LLM para que pueda advertir al
    # usuario si la pregunta es sobre algo que pudo cambiar después.
    if m.get("vigencia"):
        partes.append(f"vigente al {m['vigencia']}")
    if m.get("ref"):
        partes.append(f"ref {m['ref']}")

    return " · ".join(partes)


def fuente_de(doc: Document) -> dict:
    """Versión estructurada de la cita, para la UI.

    Viaja como `artifact` de la tool: el LLM no la ve (no gasta tokens),
    pero Chainlit la usa para armar el panel de fuentes.
    """
    m = doc.metadata
    return {
        # Marca de procedencia: el corpus oficial indexado, no la web.
        # La UI la usa para advertir cuando una respuesta se apoyó en
        # fuentes de internet.
        "origen": "corpus",
        "ley": nombre_legible(str(m.get("ley") or m.get("source", ""))),
        "ley_id": m.get("ley_id", ""),
        "articulos": m.get("articulos") or [],
        "seccion": m.get("seccion"),
        "vigencia": m.get("vigencia"),
        "page": m.get("page"),
        "ref": m.get("ref", ""),
        "cita": cita_de(doc),
        # Sin la ref: es lo que se muestra en el panel de fuentes.
        "titulo": titulo_de(doc),
        "texto": doc.page_content,
    }


def format_docs(docs: list[Document]) -> str:
    """Aplana una lista de Documents a un solo string citable."""
    return "\n\n---\n\n".join(f"[{cita_de(doc)}]\n{doc.page_content}" for doc in docs)


def format_result(docs: list[Document], vacio: str) -> tuple[str, list[dict]]:
    """Salida estándar de una tool de búsqueda: (texto, fuentes).

    Con `response_format="content_and_artifact"`, el primer elemento va al
    LLM y el segundo a la UI.
    """
    if not docs:
        return vacio, []
    return format_docs(docs), [fuente_de(doc) for doc in docs]
