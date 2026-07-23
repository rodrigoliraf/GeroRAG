"""
src.ingestion
--------------
Módulo responsável pela etapa de ingestão e chunking do pipeline RAG do
GeroRAG (Semana 2 do roadmap): carregamento de documentos da base
documental (`docs/`), carregamento/limpeza dos dados clínicos estruturados
(`data/raw/DATASET_IDOSOS.csv`) e divisão de ambos em "chunks" prontos para
a etapa seguinte de geração de embeddings.
"""

from .text_chunker import Chunk, RecursiveCharacterChunker
from .document_loader import Document, load_documents
from .patient_chunker import build_patient_documents

__all__ = [
    "Chunk",
    "RecursiveCharacterChunker",
    "Document",
    "load_documents",
    "build_patient_documents",
]
