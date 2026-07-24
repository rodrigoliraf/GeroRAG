"""base.py — interface comum que todo banco vetorial do GeroRAG deve implementar.

Segue o mesmo padrão de `src.embeddings.base.BaseEmbedder`: uma interface
mínima para que novos backends (numpy em memória, ChromaDB, FAISS, etc.)
funcionem com `run_build_index.py` e com a etapa de recuperação (retrieval)
sem que o restante do código precise saber qual foi escolhido.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class SearchResult:
    """Um resultado de busca semântica, já pronto para exibição/uso no prompt."""

    id: str
    text: str
    source: str
    source_type: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseVectorStore(ABC):
    """Contrato mínimo para armazenar vetores + metadados e buscar por similaridade."""

    name: str = "base"

    @abstractmethod
    def build(self, vectors: np.ndarray, chunk_records: list[dict[str, Any]]) -> "BaseVectorStore":
        """Indexa `vectors` (N, dim) junto dos metadados de cada chunk
        (mesmo formato salvo por `src.embeddings.store.save_embeddings`:
        precisa ter `id`, `source`, `source_type`, `text`)."""
        raise NotImplementedError

    @abstractmethod
    def query(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        source_type: str | None = None,
    ) -> list[SearchResult]:
        """Retorna os `top_k` chunks mais similares a `query_vector`.

        `source_type` permite filtrar (ex.: só "documento", só "paciente").
        """
        raise NotImplementedError

    @abstractmethod
    def persist(self, path: Path) -> None:
        """Salva o índice em disco, em `path`."""
        raise NotImplementedError

    @abstractmethod
    def load(self, path: Path) -> "BaseVectorStore":
        """Carrega um índice previamente salvo por `persist()`."""
        raise NotImplementedError

    @abstractmethod
    def count(self) -> int:
        """Número de vetores atualmente indexados."""
        raise NotImplementedError

    def info(self) -> dict[str, Any]:
        return {"backend": self.name, "n_vectors": self.count()}
