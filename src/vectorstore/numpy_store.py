"""
numpy_store.py
---------------
Banco vetorial baseado em NumPy puro (similaridade por cosseno via produto
interno de vetores normalizados). Não precisa de nenhuma dependência extra
além do que o projeto já usa — é o backend padrão e o fallback 100% offline,
no mesmo espírito do `TfidfEmbedder` em `src.embeddings`.

Para poucas dezenas de milhares de chunks (o caso do GeroRAG: ~9 documentos
+ até ~580 pacientes), uma busca por força bruta em NumPy é rápida o
suficiente e evita a complexidade operacional de um banco vetorial externo.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .base import BaseVectorStore, SearchResult


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # evita divisão por zero em vetores nulos
    return matrix / norms


class NumpyVectorStore(BaseVectorStore):
    name = "numpy"

    def __init__(self) -> None:
        self._vectors: np.ndarray | None = None
        self._records: list[dict[str, Any]] = []

    def build(self, vectors: np.ndarray, chunk_records: list[dict[str, Any]]) -> "NumpyVectorStore":
        if len(vectors) != len(chunk_records):
            raise ValueError(
                f"vectors ({len(vectors)}) e chunk_records ({len(chunk_records)}) "
                "precisam ter o mesmo tamanho"
            )
        self._vectors = _l2_normalize(vectors.astype(np.float32))
        self._records = list(chunk_records)
        return self

    def query(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        source_type: str | None = None,
    ) -> list[SearchResult]:
        if self._vectors is None or len(self._records) == 0:
            return []

        q = _l2_normalize(query_vector.reshape(1, -1).astype(np.float32))[0]

        if source_type is not None:
            idx = [i for i, r in enumerate(self._records) if r.get("source_type") == source_type]
        else:
            idx = list(range(len(self._records)))
        if not idx:
            return []

        candidate_vectors = self._vectors[idx]
        scores = candidate_vectors @ q  # cosseno, pois ambos já normalizados

        k = min(top_k, len(idx))
        top_local = np.argpartition(-scores, k - 1)[:k]
        top_local = top_local[np.argsort(-scores[top_local])]  # ordena por score desc

        results = []
        for local_i in top_local:
            global_i = idx[local_i]
            r = self._records[global_i]
            results.append(
                SearchResult(
                    id=r.get("id", str(global_i)),
                    text=r["text"],
                    source=r.get("source", ""),
                    source_type=r.get("source_type", ""),
                    score=float(scores[local_i]),
                    metadata={k2: v for k2, v in r.items() if k2 not in {"id", "text", "source", "source_type"}},
                )
            )
        return results

    def persist(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        if self._vectors is None:
            raise RuntimeError("Nada para salvar — chame build() antes de persist().")
        np.save(path / "vectorstore_vectors.npy", self._vectors)
        with (path / "vectorstore_records.jsonl").open("w", encoding="utf-8") as f:
            for r in self._records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def load(self, path: Path) -> "NumpyVectorStore":
        vectors_path = path / "vectorstore_vectors.npy"
        records_path = path / "vectorstore_records.jsonl"
        if not vectors_path.exists() or not records_path.exists():
            raise FileNotFoundError(
                f"Índice não encontrado em '{path}'. Rode antes: "
                "python -m src.vectorstore.run_build_index"
            )
        self._vectors = np.load(vectors_path)
        self._records = []
        with records_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self._records.append(json.loads(line))
        return self

    def count(self) -> int:
        return 0 if self._vectors is None else len(self._vectors)
