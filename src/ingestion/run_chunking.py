"""
run_chunking.py
----------------
Orquestra a etapa de chunking do GeroRAG (Semana 2 do roadmap):

    1. Carrega os documentos de diretrizes/escalas em `docs/`.
    2. Carrega e limpa a base de pacientes (`data/raw/DATASET_IDOSOS.csv`)
       e converte cada paciente em um resumo clínico textual.
    3. Divide ambos os conjuntos em chunks (`RecursiveCharacterChunker`).
    4. Salva os chunks em `data/processed/*.jsonl`, prontos para a próxima
       etapa do pipeline (geração de embeddings).

Uso:
    python -m src.ingestion.run_chunking
    python -m src.ingestion.run_chunking --chunk-size 600 --chunk-overlap 100
    python -m src.ingestion.run_chunking --only docs
    python -m src.ingestion.run_chunking --only pacientes
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from . import config
from .document_loader import load_documents
from .patient_chunker import build_patient_documents, load_and_clean_patients
from .text_chunker import Chunk, RecursiveCharacterChunker


def chunk_documents(chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    chunker = RecursiveCharacterChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_chunk_chars=config.MIN_CHUNK_CHARS,
    )
    chunks: list[Chunk] = []
    for doc in load_documents():
        chunks.extend(
            chunker.chunk_document(
                text=doc.text,
                source=doc.source,
                source_type="documento",
                metadata=doc.metadata,
                id_prefix=doc.source,
            )
        )
    return chunks


def chunk_patients(chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    chunker = RecursiveCharacterChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_chunk_chars=1,
    )
    df = load_and_clean_patients()
    chunks: list[Chunk] = []
    for doc in build_patient_documents(df):
        chunks.extend(
            chunker.chunk_document(
                text=doc.text,
                source=doc.source,
                source_type="paciente",
                metadata=doc.metadata,
                id_prefix=doc.source,
            )
        )
    return chunks


def _write_jsonl(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")


def _print_stats(label: str, chunks: list[Chunk]) -> None:
    if not chunks:
        print(f"  {label}: nenhum chunk gerado.")
        return
    sizes = [c.n_chars for c in chunks]
    n_sources = len({c.source for c in chunks})
    print(
        f"  {label}: {len(chunks)} chunks de {n_sources} fonte(s) | "
        f"tamanho médio {statistics.mean(sizes):.0f} chars "
        f"(min {min(sizes)}, max {max(sizes)})"
    )


def run_pipeline(
    chunk_size: int = config.DOC_CHUNK_SIZE,
    chunk_overlap: int = config.DOC_CHUNK_OVERLAP,
    patient_chunk_size: int = config.PATIENT_CHUNK_SIZE,
    patient_chunk_overlap: int = config.PATIENT_CHUNK_OVERLAP,
    only: str | None = None,
) -> dict[str, list[Chunk]]:
    """Executa o pipeline de chunking e grava os arquivos JSONL de saída.
    Retorna um dicionário {"documentos": [...], "pacientes": [...]}."""
    result: dict[str, list[Chunk]] = {"documentos": [], "pacientes": []}

    if only in (None, "docs", "documentos"):
        result["documentos"] = chunk_documents(chunk_size, chunk_overlap)
        _write_jsonl(result["documentos"], config.CHUNKS_DOCS_PATH)

    if only in (None, "pacientes", "patients"):
        result["pacientes"] = chunk_patients(patient_chunk_size, patient_chunk_overlap)
        _write_jsonl(result["pacientes"], config.CHUNKS_PATIENTS_PATH)

    all_chunks = result["documentos"] + result["pacientes"]
    if all_chunks:
        _write_jsonl(all_chunks, config.CHUNKS_ALL_PATH)

    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline de chunking do GeroRAG")
    parser.add_argument("--chunk-size", type=int, default=config.DOC_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=config.DOC_CHUNK_OVERLAP)
    parser.add_argument(
        "--patient-chunk-size", type=int, default=config.PATIENT_CHUNK_SIZE
    )
    parser.add_argument(
        "--patient-chunk-overlap", type=int, default=config.PATIENT_CHUNK_OVERLAP
    )
    parser.add_argument(
        "--only",
        choices=["docs", "documentos", "pacientes", "patients"],
        default=None,
        help="Processa apenas documentos ou apenas pacientes (padrão: ambos).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    print("GeroRAG — Pipeline de Chunking (Semana 2)")
    print("=" * 50)

    result = run_pipeline(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        patient_chunk_size=args.patient_chunk_size,
        patient_chunk_overlap=args.patient_chunk_overlap,
        only=args.only,
    )

    print("\nResumo:")
    _print_stats("Documentos (docs/)", result["documentos"])
    _print_stats("Pacientes (DATASET_IDOSOS.csv)", result["pacientes"])

    total = len(result["documentos"]) + len(result["pacientes"])
    print(f"\nTotal de chunks gerados: {total}")
    print(f"Arquivos salvos em: {config.DATA_PROCESSED_DIR}")


if __name__ == "__main__":
    main()
