#!/usr/bin/env python
"""Uso: python scripts/run_eval.py [--verbose]

Corre el agente sobre los casos de data/eval/casos.json y reporta cuáles
pasan. Es una suite de regresión de COMPORTAMIENTO: úsala después de
tocar el system prompt, las descripciones de las tools o el chunking.

Si tienes LangSmith configurado en el .env, cada caso queda además
trazado allá con su id como metadata, para poder abrir el que falló y
ver exactamente qué decidió el agente.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mexlex.agent.graph import build_agent  # noqa: E402
from mexlex.evaluation import cargar_casos, correr_caso  # noqa: E402
from mexlex.observability import setup_tracing  # noqa: E402


async def main(verbose: bool) -> int:
    trazando = setup_tracing()
    if trazando:
        print("LangSmith: trazas activas.\n")

    casos = cargar_casos()
    # A propósito SIN checkpointer de Cosmos: cada caso corre en su propio
    # thread desechable y no tiene por qué dejar basura en la base de
    # producción. La memoria en RAM le sobra para los casos de seguimiento.
    agent = build_agent()

    print(f"Evaluando {len(casos)} casos...\n")
    resultados = []

    for caso in casos:
        resultado = await correr_caso(agent, caso)
        resultados.append(resultado)

        marca = "PASA" if resultado.paso else "FALLA"
        print(f"[{marca}] {resultado.id}")
        print(f"        tools={resultado.tools} articulos={resultado.articulos}")
        for fallo in resultado.fallos:
            print(f"        -> {fallo}")
        if verbose:
            print(f"        respuesta: {resultado.respuesta[:200]}...")
        print()

    pasaron = sum(1 for r in resultados if r.paso)
    print("=" * 60)
    print(f"{pasaron}/{len(resultados)} casos pasaron")

    # Exit code distinto de cero si algo falló: así se puede encadenar en CI.
    return 0 if pasaron == len(resultados) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main("--verbose" in sys.argv)))
