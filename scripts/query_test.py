#!/usr/bin/env python
"""Uso: python scripts/query_test.py

Loop interactivo de terminal para probar la chain RAG de la Fase 1,
antes de conectarla a Chainlit. Escribe 'salir' para terminar.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mexlex.chains.simple_rag_chain import build_simple_rag_chain  # noqa: E402

if __name__ == "__main__":
    print("Cargando chain RAG... (esto crea las conexiones a Azure)")
    chain = build_simple_rag_chain()
    print("Listo. Escribe tu pregunta (o 'salir' para terminar).\n")

    while True:
        question = input("Tú: ").strip()
        if question.lower() in {"salir", "exit", "quit"}:
            break
        if not question:
            continue

        respuesta = chain.invoke(question)
        print(f"\nAsistente: {respuesta}\n")
