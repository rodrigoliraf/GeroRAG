"""
document_loader.py
-------------------
Carrega os arquivos da base documental do RAG (pasta `docs/`): diretrizes,
manuais de escalas e critérios clínicos usados para fundamentar as respostas
do assistente. Suporta `.txt`/`.md` nativamente e `.pdf` caso a biblioteca
`pypdf` esteja instalada (dependência opcional — ver requirements.txt).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import config

SUPPORTED_TEXT_EXTENSIONS = {".txt", ".md"}
SUPPORTED_PDF_EXTENSIONS = {".pdf"}


@dataclass
class Document:
    """Um documento bruto carregado da base documental, antes do chunking."""

    source: str          # nome do arquivo (usado como identificador)
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_pdf_file(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError(
            "Leitura de PDF requer a dependência opcional 'pypdf'. "
            "Instale com: pip install pypdf"
        ) from exc

    reader = PdfReader(str(path))
    pages_text = []
    for page in reader.pages:
        pages_text.append(page.extract_text() or "")
    return "\n\n".join(pages_text)


def load_documents(docs_dir: Path | None = None) -> list[Document]:
    """Carrega todos os documentos suportados dentro de `docs_dir`
    (por padrão, `<raiz-do-repo>/docs`). Arquivos com extensão não suportada
    são ignorados silenciosamente."""
    docs_dir = docs_dir or config.DOCS_DIR
    documents: list[Document] = []

    if not docs_dir.exists():
        return documents

    for path in sorted(docs_dir.rglob("*")):
        if not path.is_file():
            continue

        suffix = path.suffix.lower()
        try:
            if suffix in SUPPORTED_TEXT_EXTENSIONS:
                text = _read_text_file(path)
            elif suffix in SUPPORTED_PDF_EXTENSIONS:
                text = _read_pdf_file(path)
            else:
                continue
        except Exception as exc:  # noqa: BLE001 - relatamos e seguimos
            print(f"[document_loader] Falha ao ler '{path.name}': {exc}")
            continue

        if not text.strip():
            continue

        documents.append(
            Document(
                source=path.relative_to(docs_dir).as_posix(),
                text=text,
                metadata={
                    "filetype": suffix.lstrip("."),
                    "n_chars_original": len(text),
                },
            )
        )

    return documents
