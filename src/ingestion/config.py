"""Configurações centrais da etapa de ingestão/chunking do GeroRAG."""

from pathlib import Path

# Raiz do repositório (este arquivo fica em <raiz>/src/ingestion/config.py)
ROOT_DIR = Path(__file__).resolve().parents[2]

DOCS_DIR = ROOT_DIR / "docs"
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"

PATIENTS_CSV = DATA_RAW_DIR / "DATASET_IDOSOS.csv"

# Saídas do pipeline de chunking
CHUNKS_DOCS_PATH = DATA_PROCESSED_DIR / "chunks_documentos.jsonl"
CHUNKS_PATIENTS_PATH = DATA_PROCESSED_DIR / "chunks_pacientes.jsonl"
CHUNKS_ALL_PATH = DATA_PROCESSED_DIR / "chunks.jsonl"

# ---------------------------------------------------------------------------
# Parâmetros de chunking
# ---------------------------------------------------------------------------
# Tamanhos em caracteres (aprox. 1 token ~ 4 caracteres em português).
# 800/120 gera chunks de ~200 tokens com ~15% de sobreposição, um valor
# comumente usado como ponto de partida para RAG com sentence-transformers.
DOC_CHUNK_SIZE = 800
DOC_CHUNK_OVERLAP = 120

# Registros de pacientes já nascem curtos e semanticamente coesos (um
# "resumo clínico" por paciente); o tamanho é maior para evitar fragmentar
# um mesmo paciente em vários chunks sem necessidade.
PATIENT_CHUNK_SIZE = 1500
PATIENT_CHUNK_OVERLAP = 0

MIN_CHUNK_CHARS = 40  # descarta fragmentos residuais praticamente vazios

# ---------------------------------------------------------------------------
# Parâmetros de embeddings (usados por src.embeddings.run_embeddings)
# ---------------------------------------------------------------------------
EMBEDDINGS_DIR = DATA_PROCESSED_DIR / "embeddings"

# "sentence-transformers" dá melhor qualidade semântica, mas requer
# `pip install sentence-transformers` e baixa o modelo na primeira execução.
# "tfidf" funciona 100% offline, sem downloads.
DEFAULT_EMBEDDING_BACKEND = "sentence-transformers"

EMBEDDING_BATCH_SIZE = 32

# --- Vetor store (Semana 2) ---
VECTORSTORE_DIR = EMBEDDINGS_DIR.parent / "vectorstore"
DEFAULT_VECTORSTORE_BACKEND = "numpy"   # ou "chroma"
DEFAULT_TOP_K = 5