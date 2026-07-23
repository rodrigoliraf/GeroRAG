"""
sentence_transformer_embedder.py
---------------------------------
Embedder semântico/denso baseado na biblioteca `sentence-transformers`.
É o backend recomendado para o GeroRAG (mencionado no README como parte da
stack), pois captura similaridade de significado entre frases — essencial
para perguntas em linguagem natural como "este paciente tem sinais de
fragilidade?" encontrarem trechos relevantes mesmo sem sobreposição exata
de palavras.

Requer a dependência opcional `sentence-transformers` (e, na primeira
execução, acesso à internet para baixar os pesos do modelo do Hugging Face).
Instale com:
    pip install sentence-transformers
"""

from __future__ import annotations

import numpy as np

from .base import BaseEmbedder

DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


class SentenceTransformerEmbedder(BaseEmbedder):
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "O backend 'sentence-transformers' requer a dependência "
                "opcional 'sentence-transformers' (e 'torch'). Instale com:\n"
                "    pip install sentence-transformers\n"
                "Na primeira execução, os pesos do modelo "
                f"'{model_name}' serão baixados do Hugging Face "
                "(é necessário acesso à internet nesse momento)."
            ) from exc

        self.model_name = model_name
        self.name = f"sentence-transformers::{model_name}"
        self._model = SentenceTransformer(model_name, device=device)
        self._dim = self._model.get_sentence_embedding_dimension()

    def fit(self, texts: list[str]) -> "SentenceTransformerEmbedder":
        # Modelo pré-treinado: não há ajuste ao corpus.
        return self

    def embed(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        vectors = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 50,
            normalize_embeddings=True,  # facilita usar similaridade de cosseno via produto interno
            convert_to_numpy=True,
        )
        return vectors.astype(np.float32)

    @property
    def dim(self) -> int:
        return self._dim
