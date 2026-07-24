"""
chroma_store.py
-----------------
Banco vetorial usando ChromaDB (a tecnologia recomendada no relatório de
análise de tecnologias da Semana 1, por já persistir metadados nativamente
— útil para exibir "fontes utilizadas", RF11). Backend opcional: só é
carregado se `chromadb` estiver instalado.

Requer a dependência opcional `chromadb`:
    pip install chromadb
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .base import BaseVectorStore, SearchResult

COLLECTION_NAME = "gerorag_chunks"


class ChromaVectorStore(BaseVectorStore):
    name = "chroma"

    def __init__(self) -> None:
        try:
            import chromadb  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "O backend 'chroma' requer a dependência opcional 'chromadb'. "
                "Instale com:\n    pip install chromadb"
            ) from exc
        self._client = None
        self._collection = None
        self._path: Path | None = None

    def _open(self, path: Path):
        import chromadb

        self._path = path
        path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(path))
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        return self._collection

    def build(self, vectors: np.ndarray, chunk_records: list[dict[str, Any]]) -> "ChromaVectorStore":
        raise NotImplementedError(
            "ChromaVectorStore.build() requer um caminho para persistir "
            "(ChromaDB é sempre baseado em disco). Use "
            "`build_and_persist(vectors, chunk_records, path)` em vez disso, "
            "chamado por run_build_index.py."
        )

    def build_and_persist(
        self, vectors: np.ndarray, chunk_records: list[dict[str, Any]], path: Path
    ) -> "ChromaVectorStore":
        if len(vectors) != len(chunk_records):
            raise ValueError(
                f"vectors ({len(vectors)}) e chunk_records ({len(chunk_records)}) "
                "precisam ter o mesmo tamanho"
            )
        collection = self._open(path)

        ids = [str(r["id"]) for r in chunk_records]
        documents = [r["text"] for r in chunk_records]
        metadatas = [
            {"source": r.get("source", ""), "source_type": r.get("source_type", "")}
            for r in chunk_records
        ]

        # Chroma recomenda inserir em lotes para coleções grandes
        batch = 512
        for start in range(0, len(ids), batch):
            end = start + batch
            collection.upsert(
                ids=ids[start:end],
                embeddings=vectors[start:end].tolist(),
                documents=documents[start:end],
                metadatas=metadatas[start:end],
            )
        return self

    def query(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        source_type: str | None = None,
    ) -> list[SearchResult]:
        if self._collection is None:
            raise RuntimeError("Índice não carregado — chame load(path) antes de query().")

        where = {"source_type": source_type} if source_type else None
        res = self._collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=top_k,
            where=where,
        )
        results = []
        ids = res["ids"][0]
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]  # menor = mais similar (espaço cosine -> distância = 1 - similaridade)
        for _id, doc, meta, dist in zip(ids, docs, metas, dists):
            results.append(
                SearchResult(
                    id=_id,
                    text=doc,
                    source=meta.get("source", ""),
                    source_type=meta.get("source_type", ""),
                    score=float(1 - dist),
                    metadata=meta,
                )
            )
        return results

    def persist(self, path: Path) -> None:
        # ChromaDB com PersistentClient já grava em disco a cada operação;
        # nada adicional a fazer aqui além de garantir que abrimos nesse path.
        if self._path is None or self._path != path:
            self._open(path)

    def load(self, path: Path) -> "ChromaVectorStore":
        self._open(path)
        return self

    def count(self) -> int:
        return 0 if self._collection is None else self._collection.count()
