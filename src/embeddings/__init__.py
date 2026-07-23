"""
src.embeddings
---------------
Etapa de geração de embeddings do pipeline RAG do GeroRAG (Semana 2):
transforma os chunks gerados por `src.ingestion` em vetores numéricos,
prontos para indexação em um banco vetorial.

Dois backends disponíveis (ver `run_embeddings.py --help`):
    - "sentence-transformers": embeddings densos/semânticos (recomendado).
    - "tfidf": embeddings esparsos clássicos, 100% offline, sem download
      de modelo — útil como baseline/fallback quando não há internet.
"""

from .base import BaseEmbedder
from .store import save_embeddings, load_embeddings

__all__ = ["BaseEmbedder", "save_embeddings", "load_embeddings"]
