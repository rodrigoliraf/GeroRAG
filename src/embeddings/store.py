"""
store.py
--------
Persistência dos embeddings gerados: salva a matriz de vetores (`.npy`),
os metadados de cada chunk alinhados por posição (`.jsonl`) e informações
do embedder usado (`.json`), dentro de `data/processed/embeddings/`.

O alinhamento é por índice: a linha `i` de `embeddings.npy` corresponde à
linha `i` de `chunk_ids.jsonl`. Isso evita duplicar o texto do chunk dentro
do array numérico e mantém os arquivos fáceis de inspecionar.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def save_embeddings(
    vectors: np.ndarray,
    chunk_records: list[dict[str, Any]],
    output_dir: Path,
    embedder_info: dict[str, Any],
    vectorizer: Any = None,
) -> None:
    """Salva `vectors` (N, dim) e os metadados dos N chunks correspondentes.

    `chunk_records` deve ter o mesmo tamanho e ordem de `vectors` — cada
    item é um dicionário com, no mínimo, `id`, `source`, `source_type` e
    `text`.
    """
    if len(vectors) != len(chunk_records):
        raise ValueError(
            f"vectors ({len(vectors)}) e chunk_records ({len(chunk_records)}) "
            "precisam ter o mesmo tamanho"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "embeddings.npy", vectors.astype(np.float32))

    with (output_dir / "chunk_ids.jsonl").open("w", encoding="utf-8") as f:
        for record in chunk_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    info = {
        **embedder_info,
        "n_vectors": int(len(vectors)),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with (output_dir / "info.json").open("w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    if vectorizer is not None:
        import joblib

        joblib.dump(vectorizer, output_dir / "vectorizer.pkl")


def load_embeddings(
    output_dir: Path,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    """Carrega de volta o que foi salvo por `save_embeddings`."""
    vectors = np.load(output_dir / "embeddings.npy")

    chunk_records = []
    with (output_dir / "chunk_ids.jsonl").open(encoding="utf-8") as f:
        for line in f:
            chunk_records.append(json.loads(line))

    with (output_dir / "info.json").open(encoding="utf-8") as f:
        info = json.load(f)

    return vectors, chunk_records, info
