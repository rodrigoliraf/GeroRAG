"""
run_build_index.py
--------------------
Constrói o banco vetorial a partir dos embeddings já gerados por
`src.embeddings.run_embeddings` (RF07: "Criar o banco vetorial").

Uso:
    # backend padrão (numpy, sem dependências extras)
    python -m src.vectorstore.run_build_index

    # usando ChromaDB (requer `pip install chromadb`)
    python -m src.vectorstore.run_build_index --backend chroma

    # apontando para outra pasta de embeddings/índice
    python -m src.vectorstore.run_build_index --embeddings-dir data/processed/embeddings --output-dir data/processed/vectorstore
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..embeddings.store import load_embeddings
from ..ingestion import config
from .numpy_store import NumpyVectorStore

BACKENDS = ("numpy", "chroma")


def run_pipeline(embeddings_dir: Path, output_dir: Path, backend: str) -> None:
    print(f"Carregando embeddings de '{embeddings_dir}'...")
    vectors, chunk_records, info = load_embeddings(embeddings_dir)
    print(f"  {vectors.shape[0]} vetores de dimensão {vectors.shape[1]} "
          f"(backend do embedder: {info.get('backend')})")

    if vectors.shape[0] == 0:
        print("Nenhum vetor encontrado — rode antes o chunking e os embeddings.")
        return

    print(f"Construindo índice (backend='{backend}')...")
    if backend == "numpy":
        store = NumpyVectorStore().build(vectors, chunk_records)
        store.persist(output_dir)
    elif backend == "chroma":
        from .chroma_store import ChromaVectorStore

        store = ChromaVectorStore().build_and_persist(vectors, chunk_records, output_dir)
    else:
        raise ValueError(f"Backend desconhecido: {backend!r}. Use um de {BACKENDS}.")

    # grava qual backend/dir de embeddings foi usado, para o search.py saber recarregar
    (output_dir / "vectorstore_info.json").write_text(
        __import__("json").dumps(
            {"vectorstore_backend": backend, "embeddings_dir": str(embeddings_dir)},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Índice construído com {store.count()} vetores.")
    print(f"Salvo em: {output_dir}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Construção do banco vetorial do GeroRAG")
    parser.add_argument("--backend", choices=BACKENDS, default=config.DEFAULT_VECTORSTORE_BACKEND)
    parser.add_argument("--embeddings-dir", type=Path, default=config.EMBEDDINGS_DIR)
    parser.add_argument("--output-dir", type=Path, default=config.VECTORSTORE_DIR)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    print("GeroRAG — Construção do Banco Vetorial (Semana 2)")
    print("=" * 50)
    try:
        run_pipeline(args.embeddings_dir, args.output_dir, args.backend)
    except (ImportError, FileNotFoundError, ValueError) as exc:
        print(f"\nErro: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
