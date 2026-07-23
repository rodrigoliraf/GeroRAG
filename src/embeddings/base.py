"""base.py — interface comum que todo embedder do GeroRAG deve implementar."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseEmbedder(ABC):
    """Contrato mínimo para transformar textos em vetores numéricos.

    Qualquer novo backend de embeddings (ex.: OpenAI, Cohere, um modelo
    local diferente) deve implementar esta interface para funcionar com
    `run_embeddings.py` e com a futura etapa de recuperação (retrieval).
    """

    name: str = "base"

    @abstractmethod
    def fit(self, texts: list[str]) -> "BaseEmbedder":
        """Ajusta o embedder ao corpus, quando aplicável (ex.: TF-IDF
        precisa aprender o vocabulário). Modelos pré-treinados (como
        sentence-transformers) podem simplesmente retornar `self`."""
        raise NotImplementedError

    @abstractmethod
    def embed(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Retorna uma matriz (N, dim) de embeddings, um vetor por texto."""
        raise NotImplementedError

    @property
    @abstractmethod
    def dim(self) -> int:
        """Dimensionalidade dos vetores gerados."""
        raise NotImplementedError

    def info(self) -> dict:
        return {"backend": self.name, "dim": self.dim}
