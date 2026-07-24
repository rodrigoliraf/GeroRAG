"""
src.vectorstore
-----------------
Etapa de banco vetorial do pipeline RAG do GeroRAG (Semana 2): indexa os
vetores gerados por `src.embeddings` para permitir busca semântica
(RF07/RF08) e retorna os trechos + metadados (documento, fonte, score)
usados para montar o prompt do LLM (RF09).

Dois backends disponíveis (ver `run_build_index.py --help`):
    - "numpy": banco vetorial em memória via NumPy, sem dependências extras
      (recomendado para começar / rodar offline).
    - "chroma": ChromaDB, com persistência e filtragem por metadados nativas
      (recomendado no relatório de tecnologias da Semana 1).
"""

from .base import BaseVectorStore, SearchResult
from .numpy_store import NumpyVectorStore
from .search import load_query_embedder, load_vector_store, semantic_search

__all__ = [
    "BaseVectorStore",
    "SearchResult",
    "NumpyVectorStore",
    "load_query_embedder",
    "load_vector_store",
    "semantic_search",
]
