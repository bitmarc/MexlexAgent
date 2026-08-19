#!/usr/bin/env python
"""Uso: python scripts/run_ingestion.py

Carga los PDFs de data/raw_pdfs/, los chunkea y los sube a Azure AI Search.
Corre esto cada vez que agregues un documento legal nuevo.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mexlex.ingestion.index_builder import build_index  # noqa: E402

if __name__ == "__main__":
    n_chunks = build_index()
    print(f"\n✅ Listo. {n_chunks} chunks indexados.")
