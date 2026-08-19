"""Tests del parser de estructura legal. No requieren Azure.

Los casos de encabezado son los formatos REALES que produce la extracción
de LFC.pdf, incluidas sus rarezas ("ARTICULO 4 o.-", "ARTICULO 11. - ").
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mexlex.ingestion.structure import (
    ENCABEZADO_ARTICULO,
    construir_fragmentos,
    detectar_nombre_ley,
    limpiar_paginas,
    pagina_de_offset,
    partir_por_articulo,
    unir_paginas,
)

ENCABEZADO = (
    "LEY FEDERAL DE CINEMATOGRAFÍA \n"
    " \n"
    "CÁMARA DE DIPUTADOS DEL H. CONGRESO DE LA UNIÓN \n"
    "Secretaría General \n"
    "Secretaría de Servicios Parlamentarios  \n"
    "Última Reforma DOF 22-03-2021 \n"
    " \n"
    " \n"
)


def _pagina(numero: int, cuerpo: str) -> str:
    return f"{ENCABEZADO}{numero} de 31 \n \n{cuerpo}"


# --- detección de encabezados de artículo ---------------------------------


def test_detecta_las_variantes_reales_de_encabezado():
    casos = {
        "ARTICULO 1o.- Las disposiciones de esta Ley": 1,
        "ARTICULO 4 o.- La industria cinematográfica": 4,
        "ARTICULO 10.- Quienes produzcan películas": 10,
        "ARTICULO 11. - Toda persona podrá participar": 11,
        "ARTÍCULO 19.- Los exhibidores reservarán": 19,
        # Formato sin guion: convive con los demás en el mismo documento
        # (lo introdujeron reformas posteriores). Omitirlo hacía que los
        # artículos 43-58 se absorbieran dentro del tramo del artículo 42.
        "ARTICULO 47. Las medidas de aseguramiento consistirán en:": 47,
    }
    for texto, esperado in casos.items():
        m = ENCABEZADO_ARTICULO.search(texto)
        assert m is not None, f"no detectó: {texto!r}"
        assert int(m.group(1)) == esperado


def test_ignora_notas_de_reforma_y_referencias_cruzadas():
    # Estas líneas aparecen por todo el documento y NO son encabezados.
    assert ENCABEZADO_ARTICULO.search("Artículo reformado DOF 05-01-1999") is None
    assert ENCABEZADO_ARTICULO.search("Artículo adicionado DOF 28-04-2010") is None
    assert (
        ENCABEZADO_ARTICULO.search(
            "la autorización a que se refiere el artículo 24 de la presente Ley"
        )
        is None
    )
    # Al partir líneas, el PDF puede dejar una referencia cruzada al inicio
    # de renglón. Se distinguen porque al número le sigue palabra o coma,
    # no puntuación de encabezado.
    assert ENCABEZADO_ARTICULO.search("artículo 24 de la presente Ley;") is None
    assert ENCABEZADO_ARTICULO.search("artículo 41, fracción I") is None


def test_marca_de_seccion_debe_ocupar_su_propia_linea():
    from mexlex.ingestion.structure import INICIO_SECCION

    assert INICIO_SECCION.search("TRANSITORIOS \n") is not None
    assert INICIO_SECCION.search("Transitorios \n") is not None
    # El salto de línea del PDF deja esta palabra iniciando renglón a media
    # prosa; no es un encabezado y no debe cortar el documento.
    assert (
        INICIO_SECCION.search("Transitorio se destinarán en términos de la Ley")
        is None
    )


# --- limpieza del encabezado corrido --------------------------------------


def test_limpia_el_encabezado_repetido_en_cada_pagina():
    paginas = [_pagina(i, f"ARTICULO {i}.- Contenido del artículo {i}.") for i in range(1, 6)]
    limpias = limpiar_paginas(paginas)

    for limpia in limpias:
        assert "CÁMARA DE DIPUTADOS" not in limpia
        assert "Secretaría General" not in limpia
        assert "de 31" not in limpia
        assert limpia.lstrip().startswith("ARTICULO")


def test_no_borra_contenido_del_cuerpo():
    paginas = [_pagina(i, f"ARTICULO {i}.- Texto único {i}.") for i in range(1, 6)]
    limpias = limpiar_paginas(paginas)
    for i, limpia in enumerate(limpias, start=1):
        assert f"Texto único {i}" in limpia


# --- nombre de la ley ------------------------------------------------------


def test_detecta_el_nombre_de_la_ley_del_encabezado():
    paginas = [_pagina(i, f"ARTICULO {i}.- x") for i in range(1, 6)]
    assert detectar_nombre_ley(paginas, "respaldo") == "LEY FEDERAL DE CINEMATOGRAFÍA"


def test_usa_el_respaldo_si_no_hay_encabezado_reconocible():
    paginas = ["texto suelto sin encabezado", "otro texto distinto"]
    assert detectar_nombre_ley(paginas, "LFC") == "LFC"


# --- mapa de páginas -------------------------------------------------------


def test_articulo_a_caballo_entre_paginas_conserva_su_pagina_de_inicio():
    # El artículo 7 empieza al final de la página 0 y sigue en la 1.
    paginas = [
        "relleno inicial\nARTICULO 7.- Primera mitad del artículo",
        "segunda mitad del artículo siete.",
    ]
    texto, offsets = unir_paginas(paginas)
    inicio = texto.index("ARTICULO 7")

    assert pagina_de_offset(offsets, inicio) == 0
    assert pagina_de_offset(offsets, texto.index("segunda mitad")) == 1


# --- partido y empaquetado -------------------------------------------------


def test_partir_por_articulo_separa_cada_articulo():
    texto = (
        "preámbulo del documento\n"
        "ARTICULO 1o.- Primero.\n"
        "ARTICULO 2o.- Segundo.\n"
        "ARTICULO 3o.- Tercero.\n"
    )
    tramos = partir_por_articulo(texto)
    numeros = [n for n, _, _ in tramos]

    assert numeros == [None, 1, 2, 3]  # el None es el preámbulo


def test_ningun_chunk_parte_un_articulo():
    # Artículos cortos: varios deben caber juntos en un chunk.
    paginas = [
        "\n".join(f"ARTICULO {i}.- Disposición número {i}." for i in range(1, 21))
    ]
    fragmentos = construir_fragmentos(paginas, chunk_size=200, chunk_overlap=0)

    for frag in fragmentos:
        # Si un artículo apareció en un chunk, su texto está completo ahí.
        for numero in frag.articulos:
            assert f"Disposición número {numero}." in frag.texto


def test_empaqueta_articulos_consecutivos():
    paginas = [
        "\n".join(f"ARTICULO {i}.- Disposición número {i}." for i in range(1, 21))
    ]
    fragmentos = construir_fragmentos(paginas, chunk_size=200, chunk_overlap=0)

    # Con artículos de ~35 chars y chunk_size 200, deben agruparse varios.
    assert any(len(f.articulos) > 1 for f in fragmentos)
    # Y ninguno debe exceder el tamaño pedido.
    assert all(len(f.texto) <= 200 for f in fragmentos)


def test_articulo_gigante_se_parte_conservando_su_numero():
    cuerpo = "ARTICULO 5o.- " + ("palabra " * 400)
    fragmentos = construir_fragmentos([cuerpo], chunk_size=300, chunk_overlap=0)

    partes = [f for f in fragmentos if 5 in f.articulos]
    assert len(partes) > 1  # se partió
    assert all(f.articulos == [5] for f in partes)  # todas siguen siendo el 5


def test_todos_los_articulos_quedan_cubiertos():
    paginas = [
        "\n".join(f"ARTICULO {i}.- Disposición número {i}." for i in range(1, 21))
    ]
    fragmentos = construir_fragmentos(paginas, chunk_size=200, chunk_overlap=0)

    cubiertos = {n for f in fragmentos for n in f.articulos}
    assert cubiertos == set(range(1, 21))


def test_transitorios_no_inventan_numero_de_articulo():
    paginas = [
        "ARTICULO 1o.- Disposición.\n"
        "TRANSITORIOS\n"
        "PRIMERO.- El presente decreto entrará en vigor.\n"
    ]
    fragmentos = construir_fragmentos(paginas, chunk_size=500, chunk_overlap=0)

    transitorios = [f for f in fragmentos if f.seccion == "TRANSITORIOS"]
    assert transitorios
    assert all(f.articulos == [] for f in transitorios)
