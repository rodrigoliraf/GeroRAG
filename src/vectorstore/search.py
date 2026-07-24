"""
search.py
---------
Ponto único de busca semântica: vetoriza a pergunta do usuário com o MESMO
embedder usado para indexar os chunks (lido de `info.json`, salvo por
`src.embeddings.store.save_embeddings`) e consulta o banco vetorial.

Importante: para o backend TF-IDF, a pergunta precisa ser transformada com
o `vectorizer.pkl` já ajustado ao corpus (nunca re-treinado do zero),
senão as dimensões do vetor da pergunta não baterão com as do índice.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..embeddings.base import BaseEmbedder
from ..ingestion import config
from .base import BaseVectorStore, SearchResult
from .numpy_store import NumpyVectorStore


def load_query_embedder(embeddings_dir: Path = config.EMBEDDINGS_DIR) -> BaseEmbedder:
    """Reconstrói o embedder usado para gerar o índice, a partir de
    `info.json`. Garante que a pergunta do usuário seja vetorizada de forma
    compatível com os vetores já indexados."""
    info_path = embeddings_dir / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(
            f"'{info_path}' não encontrado. Rode antes: "
            "python -m src.embeddings.run_embeddings"
        )
    with info_path.open(encoding="utf-8") as f:
        info = json.load(f)

    backend = info["backend"]

    if backend.startswith("sentence-transformers::"):
        model_name = backend.split("::", 1)[1]
        from ..embeddings.sentence_transformer_embedder import SentenceTransformerEmbedder

        return SentenceTransformerEmbedder(model_name=model_name)

    elif backend == "tfidf":
        import joblib

        vectorizer_path = embeddings_dir / "vectorizer.pkl"
        if not vectorizer_path.exists():
            raise FileNotFoundError(
                f"'{vectorizer_path}' não encontrado — o backend TF-IDF precisa "
                "do vectorizer.pkl salvo junto dos embeddings para vetorizar "
                "novas perguntas de forma consistente."
            )
        from ..embeddings.tfidf_embedder import TfidfEmbedder

        embedder = TfidfEmbedder()
        embedder._vectorizer = joblib.load(vectorizer_path)  # reaproveita o já ajustado
        embedder._fitted = True
        embedder._dim = len(embedder._vectorizer.vocabulary_)
        return embedder

    else:
        raise ValueError(f"Backend desconhecido em info.json: {backend!r}")


def load_vector_store(
    vectorstore_dir: Path = config.VECTORSTORE_DIR,
    backend: str = config.DEFAULT_VECTORSTORE_BACKEND,
) -> BaseVectorStore:
    if backend == "numpy":
        return NumpyVectorStore().load(vectorstore_dir)
    elif backend == "chroma":
        from .chroma_store import ChromaVectorStore

        return ChromaVectorStore().load(vectorstore_dir)
    else:
        raise ValueError(f"Backend de vector store desconhecido: {backend!r}")


def semantic_search(
    query: str,
    top_k: int = config.DEFAULT_TOP_K,
    source_type: str | None = "documento",
    embedder: BaseEmbedder | None = None,
    store: BaseVectorStore | None = None,
) -> list[SearchResult]:
    """Função principal de busca semântica: pergunta em texto -> top-k chunks.

    `embedder` e `store` podem ser passados prontos (ex.: pela interface
    Streamlit, para evitar recarregar o modelo a cada pergunta); se
    omitidos, são carregados a partir da configuração padrão.
    """
    if embedder is None:
        embedder = load_query_embedder()
    if store is None:
        store = load_vector_store()

    query_vector = embedder.embed([query])[0]
    return store.query(query_vector, top_k=top_k, source_type=source_type)
