"""Fase 8: autenticación mínima para poder tener historial.

No es una funcionalidad que quisiéramos, es una que Chainlit **exige**:
`resume_thread` corta de inmediato si `session.user` es None, y la lista
de conversaciones se filtra por `user.id`. Sin login no hay historial,
por muy persistido que esté todo.

Lo que hay aquí es una cuenta única leída del `.env`, pensada para
desarrollo. Para algo real, Chainlit trae OAuth (Google, GitHub, Entra
ID) con `@cl.oauth_callback`, y ahí el `.env` deja de guardar
contraseñas.
"""

from __future__ import annotations

import hmac
import logging
import os

from mexlex.config import settings

logger = logging.getLogger(__name__)


def auth_configurada() -> bool:
    """True si hay credenciales en el .env.

    Es el interruptor de la fase: registrar el callback de login en
    Chainlit activa la pantalla de acceso para todo el mundo, así que
    solo se registra si de verdad hay con qué entrar.
    """
    return bool(settings.auth_user and settings.auth_password)


def setup_auth() -> bool:
    """Exporta el secreto de firma a os.environ. Regresa True si quedó lista.

    Mismo detalle que con LangSmith en `observability.py`:
    `pydantic-settings` lee el .env por su cuenta pero **no** lo exporta,
    y Chainlit busca `CHAINLIT_AUTH_SECRET` en `os.environ`. Sin este
    puente, poner el secreto en el .env no tendría ningún efecto y la app
    se caería al arrancar con un `ValueError`.
    """
    if not auth_configurada():
        logger.info(
            "Autenticación desactivada: sin MEXLEX_AUTH_USER no hay historial "
            "de conversaciones en la UI."
        )
        return False

    if not settings.chainlit_auth_secret:
        logger.warning(
            "MEXLEX_AUTH_USER está configurado pero falta CHAINLIT_AUTH_SECRET. "
            "Genéralo con `chainlit create-secret`."
        )
        return False

    os.environ["CHAINLIT_AUTH_SECRET"] = settings.chainlit_auth_secret
    return True


def credenciales_validas(usuario: str, password: str) -> bool:
    """Compara contra la cuenta del .env en tiempo constante.

    `compare_digest` en vez de `==` para no filtrar por cuánto tarda la
    comparación en qué carácter falló. Es barato y es la forma correcta
    de comparar un secreto.

    Se comparan **bytes**, no str: sobre cadenas, `compare_digest` exige
    que sean ASCII y lanza `TypeError` si no lo son. Una contraseña con
    una `ñ` o un acento tumbaría el login con un 500 en vez de
    rechazarlo — y esas son justo las contraseñas de este proyecto.
    """
    if not auth_configurada():
        return False

    usuario_ok = hmac.compare_digest(
        usuario.encode("utf-8"), (settings.auth_user or "").encode("utf-8")
    )
    password_ok = hmac.compare_digest(
        password.encode("utf-8"), (settings.auth_password or "").encode("utf-8")
    )
    # Se evalúan las dos SIEMPRE (nada de `and` que corte antes) para que
    # el tiempo de respuesta no dependa de si el usuario existe.
    return usuario_ok and password_ok
