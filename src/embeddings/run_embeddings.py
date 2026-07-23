"""
run_embeddings.py
------------------
Gera embeddings para os chunks produzidos por `src.ingestion.run_chunking`.

Uso:
    # backend recomendado (requer `pip install sentence-transformers`)
    python -m src.embeddings.run_embeddings

    # backend 100% offline, sem download de modelo
    python -m src.embeddings.run_embeddings --backend tfidf

    # outro modelo sentence-transformers, ou outro arquivo de entrada
    python -m src.embeddings.run_embeddings --model all-MiniLM-L6-v2
    python -m src.embeddings.run_embeddings --input data/processed/chunks_documentos.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from ..ingestion import config
from .base import BaseEmbedder
from .store import save_embeddings

BACKENDS = ("sentence-transformers", "tfidf")


def _load_chunks(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"'{path}' não encontrado. Rode antes: python -m src.ingestion.run_chunking"
        )
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _build_embedder(backend: str, model: str | None) -> BaseEmbedder:
    if backend == "sentence-transformers":
        from .sentence_transformer_embedder import DEFAULT_MODEL, SentenceTransformerEmbedder

        return SentenceTransformerEmbedder(model_name=model or DEFAULT_MODEL)
    elif backend == "tfidf":
        from .tfidf_embedder import TfidfEmbedder

        return TfidfEmbedder()
    else:
        raise ValueError(f"Backend desconhecido: {backend!r}. Use um de {BACKENDS}.")


def run_pipeline(
    input_path: Path,
    output_dir: Path,
    backend: str = config.DEFAULT_EMBEDDING_BACKEND,
    model: str | None = None,
    batch_size: int = config.EMBEDDING_BATCH_SIZE,
) -> None:
    print(f"Carregando chunks de '{input_path}'...")
    chunks = _load_chunks(input_path)
    if not chunks:
        print("Nenhum chunk encontrado — nada a fazer.")
        return
    print(f"  {len(chunks)} chunks carregados.")

    texts = [c["text"] for c in chunks]

    print(f"Inicializando embedder (backend='{backend}'" + (f", model='{model}'" if model else "") + ")...")
    embedder = _build_embedder(backend, model)

    t0 = time.time()
    embedder.fit(texts)
    vectors = embedder.embed(texts, batch_size=batch_size)
    elapsed = time.time() - t0
    print(f"  {vectors.shape[0]} vetores de dimensão {vectors.shape[1]} gerados em {elapsed:.1f}s.")

    chunk_records = [
        {
            "id": c["id"],
            "source": c["source"],
            "source_type": c["source_type"],
            "chunk_index": c["chunk_index"],
            "text": c["text"],
        }
        for c in chunks
    ]

    vectorizer = getattr(embedder, "vectorizer", None)
    save_embeddings(
        vectors=vectors,
        chunk_records=chunk_records,
        output_dir=output_dir,
        embedder_info={"backend": embedder.name, "dim": embedder.dim, "input_file": str(input_path)},
        vectorizer=vectorizer,
    )
    print(f"Salvo em: {output_dir}")
    print("  - embeddings.npy   (matriz N x dim)")
    print("  - chunk_ids.jsonl  (metadados alinhados por linha)")
    print("  - info.json        (backend, dimensão, timestamp)")
    if vectorizer is not None:
        print("  - vectorizer.pkl   (necessário para vetorizar novas perguntas com TF-IDF)")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline de geração de embeddings do GeroRAG")
    parser.add_argument("--backend", choices=BACKENDS, default=config.DEFAULT_EMBEDDING_BACKEND)
    parser.add_argument("--model", default=None, help="Nome do modelo (apenas para --backend sentence-transformers)")
    parser.add_argument("--input", type=Path, default=config.CHUNKS_ALL_PATH)
    parser.add_argument("--output-dir", type=Path, default=config.EMBEDDINGS_DIR)
    parser.add_argument("--batch-size", type=int, default=config.EMBEDDING_BATCH_SIZE)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    print("GeroRAG — Pipeline de Embeddings (Semana 2)")
    print("=" * 50)
    try:
        run_pipeline(
            input_path=args.input,
            output_dir=args.output_dir,
            backend=args.backend,
            model=args.model,
            batch_size=args.batch_size,
        )
    except (ImportError, FileNotFoundError) as exc:
        print(f"\nErro: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
