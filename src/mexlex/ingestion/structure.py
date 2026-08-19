"""Fase 5: lectura de la estructura de un documento legal.

Las fases 1-4 trataban el PDF como texto plano: se cortaba cada página
por tamaño, sin saber dónde empieza o termina un artículo. Eso hacía
imposible responder "dame el artículo 19" de forma determinista.

Este módulo extrae la estructura real del documento:

    páginas del PDF
        -> limpiar encabezado corrido (se repite en cada página)
        -> unir en un solo texto + mapa offset->página
        -> partir por frontera de ARTICULO
        -> empacar artículos consecutivos sin partir ninguno

El resultado son `Fragmento`s que saben a qué artículo(s) y página
pertenecen, que es lo que después se guarda como metadata filtrable en
Azure AI Search.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Encabezado de artículo. Tolerante a las variantes reales que produce la
# extracción de PDF de la Cámara de Diputados. En un mismo documento
# conviven varias, según la época de la reforma que lo introdujo:
#   "ARTICULO 1o.-"   "ARTICULO 4 o.-"   "ARTICULO 11. - "
#   "ARTICULO 10.-"   "ARTICULO 47."  <- sin guion
#
# Dos condiciones lo hacen específico:
#  1. Dígitos justo después de ARTICULO, así que NO captura las notas de
#     reforma ("Artículo adicionado DOF 05-01-1999").
#  2. Puntuación (. - :) justo después del número, así que NO captura las
#     referencias cruzadas ("...se refiere el artículo 24 de la presente
#     Ley", "...el artículo 41, fracción I"), donde al número le sigue
#     una palabra o una coma.
ENCABEZADO_ARTICULO = re.compile(
    r"(?im)^[ \t]*ART[IÍ]CULO\s+(\d+)\s*[oº°]?\s*(?:\.[ \t]*[-–]?|[-–]|:)"
)

# Secciones finales que numeran con palabras ("PRIMERO.", "ARTICULO UNICO")
# en vez de dígitos. No les inventamos número de artículo.
#
# Exigimos que el encabezado ocupe su PROPIA línea ($ al final). Sin eso,
# el salto de línea del PDF deja palabras como "Transitorio" al inicio de
# una línea a media prosa ("...\nTransitorio se destinarán en términos de
# la Ley de Ingresos...") y se cortaba el documento en lugares absurdos.
INICIO_SECCION = re.compile(
    r"(?im)^[ \t]*(TRANSITORIOS?|Transitorio del Decreto|DISPOSICIONES TRANSITORIAS)"
    r"[ \t]*[:.]?[ \t]*$"
)

# Marcador de paginación del documento: "10 de 31"
MARCADOR_PAGINA = re.compile(r"^\s*\d+\s+de\s+\d+\s*$")

# Fecha de la última reforma, que la Cámara de Diputados imprime en el
# encabezado corrido: "Última Reforma DOF 22-03-2021".
#
# Es el dato que convierte el disclaimer legal en algo verificable: en vez
# de un genérico "puede estar desactualizado", se puede decir "texto
# vigente al 22-03-2021".
ULTIMA_REFORMA = re.compile(
    r"(?i)[úu]ltima\s+reforma\s+(?:publicada\s+)?DOF[\s:]+(\d{2}-\d{2}-\d{4})"
)

# Con qué palabra empieza el nombre de una ley mexicana
INICIO_NOMBRE_LEY = re.compile(
    r"^(LEY|C[OÓ]DIGO|CONSTITUCI[OÓ]N|REGLAMENTO|DECRETO)\b", re.IGNORECASE
)

# Un encabezado corrido aparece en casi todas las páginas. No exigimos el
# 100% porque la portada suele traer una línea de más o de menos.
UMBRAL_ENCABEZADO = 0.6

# Cuántas líneas del inicio de cada página se consideran zona de encabezado.
# Acotarlo evita borrar texto del cuerpo que casualmente se repita.
LINEAS_ZONA_ENCABEZADO = 12


@dataclass
class Fragmento:
    """Un trozo de documento con la estructura que le corresponde."""

    texto: str
    articulos: list[int] = field(default_factory=list)
    pagina: int = 0
    seccion: str | None = None


def _lineas_de_encabezado(paginas: list[str]) -> set[str]:
    """Detecta las líneas del encabezado corrido que se repite por página.

    En los PDFs de la Cámara de Diputados son ~5 líneas (nombre de la ley,
    "CÁMARA DE DIPUTADOS...", "Secretaría General", etc.) más el marcador
    "N de 31". Son ~200 caracteres de ruido en CADA página: contaminan los
    embeddings y, peor, se cuelan a media frase cuando un artículo cruza
    un salto de página.
    """
    if not paginas:
        return set()

    conteo: Counter[str] = Counter()
    for pagina in paginas:
        # Solo miramos la zona superior: una línea del cuerpo que se repita
        # por casualidad no debería considerarse encabezado.
        lineas_iniciales = pagina.splitlines()[:LINEAS_ZONA_ENCABEZADO]
        for linea in set(ln.strip() for ln in lineas_iniciales if ln.strip()):
            conteo[linea] += 1

    minimo = max(2, int(len(paginas) * UMBRAL_ENCABEZADO))
    return {linea for linea, veces in conteo.items() if veces >= minimo}


def limpiar_paginas(paginas: list[str]) -> list[str]:
    """Quita el encabezado corrido del inicio de cada página."""
    encabezado = _lineas_de_encabezado(paginas)

    limpias = []
    for pagina in paginas:
        lineas = pagina.splitlines()
        i = 0
        # Avanzamos mientras sigamos viendo ruido de encabezado. Paramos en
        # la primera línea de contenido real para no comernos el cuerpo.
        while i < len(lineas) and i < LINEAS_ZONA_ENCABEZADO:
            desnuda = lineas[i].strip()
            if not desnuda or desnuda in encabezado or MARCADOR_PAGINA.match(desnuda):
                i += 1
                continue
            break
        limpias.append("\n".join(lineas[i:]))

    return limpias


def detectar_nombre_ley(paginas: list[str], respaldo: str) -> str:
    """Deduce el nombre de la ley del encabezado corrido.

    En estos PDFs el nombre aparece como primera línea de cada página, así
    que es la línea repetida más frecuente que parece nombre de ley.
    `respaldo` (normalmente el nombre del archivo) se usa si no se detecta.
    """
    candidatos = [
        linea
        for linea in _lineas_de_encabezado(paginas)
        if INICIO_NOMBRE_LEY.match(linea)
    ]
    if not candidatos:
        return respaldo

    # El nombre de la ley suele ser el candidato más largo y descriptivo
    # ("LEY FEDERAL DE CINEMATOGRAFÍA" vs. "LEY").
    return max(candidatos, key=len).strip()


def detectar_vigencia(paginas: list[str]) -> str | None:
    """Extrae la fecha de la última reforma del documento (dd-mm-aaaa).

    Se busca en las primeras páginas: aparece tanto en el encabezado
    corrido como en la portada. Regresa None si el documento no la trae.
    """
    for pagina in paginas[:3]:
        m = ULTIMA_REFORMA.search(pagina)
        if m:
            return m.group(1)
    return None


def unir_paginas(paginas: list[str]) -> tuple[str, list[int]]:
    """Une las páginas en un texto continuo y regresa dónde empieza cada una.

    El mapa de offsets es lo que permite saber después en qué página cae
    cada fragmento, aunque un artículo cruce el salto de página.
    """
    partes: list[str] = []
    offsets: list[int] = []
    acumulado = 0

    for pagina in paginas:
        offsets.append(acumulado)
        partes.append(pagina)
        acumulado += len(pagina) + 1  # +1 por el "\n" que las une

    return "\n".join(partes), offsets


def pagina_de_offset(offsets: list[int], offset: int) -> int:
    """Regresa el índice de página (0-based) que contiene ese offset."""
    pagina = 0
    for i, inicio in enumerate(offsets):
        if offset >= inicio:
            pagina = i
        else:
            break
    return pagina


def partir_por_articulo(texto: str) -> list[tuple[int | None, int, int]]:
    """Corta el texto en las fronteras de artículo.

    Regresa tuplas (numero_de_articulo, offset_inicio, offset_fin). El
    número es None para el preámbulo (todo lo anterior al artículo 1) y
    para las secciones de transitorios, que numeran con palabras.
    """
    cortes: list[tuple[int | None, int]] = []

    for m in ENCABEZADO_ARTICULO.finditer(texto):
        cortes.append((int(m.group(1)), m.start()))
    for m in INICIO_SECCION.finditer(texto):
        cortes.append((None, m.start()))

    cortes.sort(key=lambda c: c[1])

    # Todo lo que va antes del primer corte es preámbulo.
    if not cortes or cortes[0][1] > 0:
        cortes.insert(0, (None, 0))

    tramos = []
    for i, (numero, inicio) in enumerate(cortes):
        fin = cortes[i + 1][1] if i + 1 < len(cortes) else len(texto)
        if texto[inicio:fin].strip():
            tramos.append((numero, inicio, fin))

    return tramos


def _partir_articulo_largo(
    texto: str, numero: int | None, chunk_size: int, chunk_overlap: int
) -> list[str]:
    """Parte un artículo que por sí solo excede el tamaño de chunk.

    Es el único caso en que cortamos a media disposición. Todas las partes
    conservan el mismo número de artículo, así que el lookup exacto las
    sigue encontrando completas.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(texto)


def construir_fragmentos(
    paginas: list[str],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Fragmento]:
    """Pipeline completo: páginas limpias -> fragmentos empacados.

    Empaca artículos consecutivos hasta llenar `chunk_size` sin partir
    ninguno. Un chunk puede cubrir los artículos 10, 11 y 12; los tres
    quedan registrados en `articulos` para que el filtro exacto los
    encuentre.
    """
    texto, offsets = unir_paginas(paginas)
    tramos = partir_por_articulo(texto)

    fragmentos: list[Fragmento] = []

    # Buffer del empaquetado
    buffer_texto = ""
    buffer_articulos: list[int] = []
    buffer_offset = 0
    buffer_seccion: str | None = None

    def vaciar() -> None:
        nonlocal buffer_texto, buffer_articulos, buffer_offset, buffer_seccion
        if buffer_texto.strip():
            fragmentos.append(
                Fragmento(
                    texto=buffer_texto.strip(),
                    articulos=list(buffer_articulos),
                    pagina=pagina_de_offset(offsets, buffer_offset),
                    seccion=buffer_seccion,
                )
            )
        buffer_texto = ""
        buffer_articulos = []
        buffer_seccion = None

    for numero, inicio, fin in tramos:
        cuerpo = texto[inicio:fin].strip()
        seccion = None if numero is not None else _nombre_seccion(cuerpo)

        # Caso 1: el artículo solo ya no cabe -> se parte aparte.
        if len(cuerpo) > chunk_size:
            vaciar()
            for parte in _partir_articulo_largo(
                cuerpo, numero, chunk_size, chunk_overlap
            ):
                fragmentos.append(
                    Fragmento(
                        texto=parte.strip(),
                        articulos=[numero] if numero is not None else [],
                        pagina=pagina_de_offset(offsets, inicio),
                        seccion=seccion,
                    )
                )
            continue

        # Caso 2: cambio de sección (p. ej. del articulado a TRANSITORIOS).
        # Nunca empacamos juntas cosas de secciones distintas, aunque quepan:
        # el chunk perdería la etiqueta de sección y mezclaría disposiciones
        # que no se leen juntas.
        if buffer_texto and seccion != buffer_seccion:
            vaciar()

        # Caso 3: no cabe en el buffer actual -> cerramos y empezamos otro.
        if buffer_texto and len(buffer_texto) + len(cuerpo) + 2 > chunk_size:
            vaciar()

        if not buffer_texto:
            buffer_offset = inicio
            buffer_seccion = seccion

        buffer_texto = f"{buffer_texto}\n\n{cuerpo}" if buffer_texto else cuerpo
        if numero is not None:
            buffer_articulos.append(numero)

    vaciar()
    return fragmentos


def _nombre_seccion(cuerpo: str) -> str | None:
    """Etiqueta las secciones sin numeración de artículo (transitorios)."""
    m = INICIO_SECCION.match(cuerpo)
    if m:
        return m.group(1).upper()
    return None


def paginas_de(documentos: list[Document]) -> list[str]:
    """Extrae el texto de una lista de Documents (una por página)."""
    return [doc.page_content for doc in documentos]
