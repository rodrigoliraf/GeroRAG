"""
tfidf_embedder.py
------------------
Embedder esparso baseado em TF-IDF (scikit-learn). Não captura similaridade
semântica tão bem quanto embeddings densos (não entende sinônimos, por
exemplo), mas roda 100% offline, sem baixar nenhum modelo — útil como
baseline para comparação e como alternativa quando não há internet
disponível para baixar o modelo do sentence-transformers.

Diferente de um modelo pré-treinado, o TF-IDF precisa ser "ajustado"
(`fit`) ao vocabulário do próprio corpus antes de gerar vetores — por isso
ele deve ser persistido (ver `store.py`) para poder vetorizar novas
perguntas de forma consistente na etapa de recuperação.
"""

from __future__ import annotations

import numpy as np

from .base import BaseEmbedder


class TfidfEmbedder(BaseEmbedder):
    name = "tfidf"

    def __init__(self, max_features: int = 20_000, ngram_range: tuple[int, int] = (1, 2)):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            strip_accents="unicode",
            lowercase=True,
            sublinear_tf=True,
        )
        self._fitted = False
        self._dim = 0

    def fit(self, texts: list[str]) -> "TfidfEmbedder":
        self._vectorizer.fit(texts)
        self._fitted = True
        self._dim = len(self._vectorizer.vocabulary_)
        return self

    def embed(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Chame fit(texts) antes de embed() para o backend TF-IDF.")
        matrix = self._vectorizer.transform(texts)
        return matrix.toarray().astype(np.float32)

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def vectorizer(self):
        return self._vectorizer
