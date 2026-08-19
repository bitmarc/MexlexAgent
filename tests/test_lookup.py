"""Tests de la construcción de filtros OData. No requieren Azure."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mexlex.retrieval.lookup import (
    escapar_odata,
    filtro_por_articulo,
    filtro_por_rango,
    parsear_ref,
)


def test_filtro_por_articulo_usa_any_sobre_la_coleccion():
    # `articulos` es una colección: un chunk que cubre 10-12 debe matchear
    # para cualquiera de los tres.
    assert filtro_por_articulo(19) == "articulos/any(a: a eq 19)"


def test_filtro_por_articulo_acota_por_ley():
    filtro = filtro_por_articulo(19, "LFC")
    assert filtro == "articulos/any(a: a eq 19) and ley_id eq 'LFC'"


def test_filtro_por_rango():
    filtro = filtro_por_rango("LFC", 7, 9)
    assert filtro == "ley_id eq 'LFC' and chunk_index ge 7 and chunk_index le 9"


def test_los_numeros_se_fuerzan_a_entero():
    # Blindaje contra interpolar texto arbitrario en el filtro.
    assert filtro_por_articulo("19") == "articulos/any(a: a eq 19)"
    with pytest.raises(ValueError):
        filtro_por_articulo("19; drop")


def test_escapa_comillas_simples():
    assert escapar_odata("O'Brien") == "O''Brien"
    assert "''" in filtro_por_articulo(1, "O'X")


# --- referencias que ve el agente -----------------------------------------


def test_parsear_ref():
    assert parsear_ref("LFC#7") == ("LFC", 7)
    assert parsear_ref(" lfc#0 ") == ("LFC", 0)


@pytest.mark.parametrize("malo", ["LFC", "LFC#", "#7", "LFC#abc", ""])
def test_parsear_ref_rechaza_formatos_invalidos(malo):
    with pytest.raises(ValueError):
        parsear_ref(malo)
