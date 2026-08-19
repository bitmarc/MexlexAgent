"""Tests del login mínimo de la Fase 8. No requieren Azure ni Chainlit corriendo.

La autenticación aquí no es una funcionalidad que quisiéramos, es una que
Chainlit exige para poder persistir y reanudar conversaciones. Lo que se
verifica es que el interruptor esté bien puesto: sin credenciales en el
.env la app NO debe pedir login, y con ellas la comparación no debe tener
atajos.
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mexlex.config import settings
from mexlex.persistence import auth


@pytest.fixture
def con_credenciales(monkeypatch):
    monkeypatch.setattr(settings, "auth_user", "marco")
    monkeypatch.setattr(settings, "auth_password", "contraseña-larga")
    monkeypatch.setattr(settings, "chainlit_auth_secret", "un-secreto")
    monkeypatch.delenv("CHAINLIT_AUTH_SECRET", raising=False)


@pytest.fixture
def sin_credenciales(monkeypatch):
    monkeypatch.setattr(settings, "auth_user", None)
    monkeypatch.setattr(settings, "auth_password", None)


def test_sin_credenciales_no_hay_auth(sin_credenciales):
    # Es el interruptor de la fase: registrar el callback de login en
    # Chainlit enciende la pantalla de acceso para todo el mundo, así que
    # sin credenciales no debe activarse nada.
    assert auth.auth_configurada() is False
    assert auth.setup_auth() is False


def test_con_credenciales_se_activa(con_credenciales):
    assert auth.auth_configurada() is True
    assert auth.setup_auth() is True


def test_setup_auth_exporta_el_secreto_a_os_environ(con_credenciales):
    # pydantic-settings lee el .env pero no lo exporta, y Chainlit busca
    # CHAINLIT_AUTH_SECRET en os.environ. Sin este puente la app se cae
    # al arrancar con un ValueError.
    auth.setup_auth()
    assert os.environ["CHAINLIT_AUTH_SECRET"] == "un-secreto"


def test_sin_secreto_no_se_activa(con_credenciales, monkeypatch):
    # Mejor arrancar sin login que reventar: Chainlit lanza ValueError si
    # hay callback de auth y no hay secreto con qué firmar el JWT.
    monkeypatch.setattr(settings, "chainlit_auth_secret", None)
    assert auth.setup_auth() is False


def test_las_credenciales_correctas_pasan(con_credenciales):
    assert auth.credenciales_validas("marco", "contraseña-larga") is True


@pytest.mark.parametrize(
    "usuario,password",
    [
        ("marco", "otra-cosa"),  # contraseña mal
        ("otro", "contraseña-larga"),  # usuario mal
        ("", ""),  # vacío
        ("marco", "contraseña-larg"),  # prefijo de la buena
    ],
)
def test_las_credenciales_incorrectas_se_rechazan(con_credenciales, usuario, password):
    assert auth.credenciales_validas(usuario, password) is False


def test_sin_auth_configurada_nadie_entra(sin_credenciales):
    # Si el .env se queda sin credenciales pero alguien conserva una
    # cookie vieja, no debe poder pasar con cadenas vacías.
    assert auth.credenciales_validas("", "") is False
    assert auth.credenciales_validas("marco", "contraseña-larga") is False
