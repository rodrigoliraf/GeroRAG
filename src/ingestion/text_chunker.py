"""
text_chunker.py
----------------
Implementação de um "recursive character splitter": divide um texto longo em
pedaços (chunks) menores, respeitando, sempre que possível, as fronteiras
naturais do texto (parágrafos > linhas > sentenças > palavras), aplicando
sobreposição (overlap) entre chunks consecutivos para preservar contexto na
fronteira entre eles.

Esta é a mesma estratégia usada por bibliotecas como o
`RecursiveCharacterTextSplitter` do LangChain, reimplementada aqui sem
dependências externas para manter o projeto leve (apenas biblioteca padrão).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    """Um pedaço de texto pronto para ser embeddado e indexado."""

    id: str
    text: str
    source: str
    source_type: str  # "documento" | "paciente"
    chunk_index: int
    total_chunks: int
    char_start: int
    char_end: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_chars(self) -> int:
        return len(self.text)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "source": self.source,
            "source_type": self.source_type,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "n_chars": self.n_chars,
            "metadata": self.metadata,
        }


class RecursiveCharacterChunker:
    """Divide texto em chunks de tamanho ~`chunk_size`, com `chunk_overlap`
    caracteres de sobreposição entre chunks consecutivos.

    A divisão tenta, em ordem de preferência, quebrar em:
        1. parágrafos duplos ("\n\n")
        2. quebras de linha simples ("\n")
        3. sentenças (após ., !, ?)
        4. espaços entre palavras
        5. caractere-a-caractere (último recurso, para tokens muito longos)

    Isso evita cortar frases ou palavras no meio sempre que uma fronteira
    "melhor" estiver disponível dentro do tamanho-alvo do chunk.
    """

    # Separadores em ordem de prioridade (do mais "estrutural" ao mais fino)
    _SEPARATORS = ["\n\n", "\n", r"(?<=[.!?])\s+", " "]

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
        min_chunk_chars: int = 1,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size deve ser positivo")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap deve estar entre 0 e chunk_size-1")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_chars = min_chunk_chars

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------
    def split_text(self, text: str) -> list[str]:
        """Retorna a lista de strings (chunks) para um texto bruto."""
        text = self._normalize_whitespace(text)
        if not text:
            return []

        pieces = self._split_recursive(text, list(self._SEPARATORS))
        merged = self._merge_pieces(pieces)
        return [c for c in merged if len(c.strip()) >= self.min_chunk_chars]

    def chunk_document(
        self,
        text: str,
        source: str,
        source_type: str,
        metadata: dict[str, Any] | None = None,
        id_prefix: str | None = None,
    ) -> list[Chunk]:
        """Divide `text` e devolve uma lista de objetos `Chunk` já com
        metadados de posição (índice do chunk, offsets no texto original)."""
        normalized = self._normalize_whitespace(text)
        pieces = self._merge_pieces(
            self._split_recursive(normalized, list(self._SEPARATORS))
        )
        pieces = [p for p in pieces if len(p.strip()) >= self.min_chunk_chars]

        chunks: list[Chunk] = []
        cursor = 0
        prefix = id_prefix or source
        for i, piece in enumerate(pieces):
            start = normalized.find(piece, cursor)
            if start == -1:  # fallback defensivo (não deveria ocorrer)
                start = cursor
            end = start + len(piece)
            cursor = max(start + 1, end - self.chunk_overlap)

            chunks.append(
                Chunk(
                    id=f"{prefix}::chunk_{i:04d}",
                    text=piece.strip(),
                    source=source,
                    source_type=source_type,
                    chunk_index=i,
                    total_chunks=len(pieces),
                    char_start=start,
                    char_end=end,
                    metadata=dict(metadata or {}),
                )
            )
        return chunks

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # colapsa 3+ quebras de linha em uma quebra de parágrafo dupla
        text = re.sub(r"\n{3,}", "\n\n", text)
        # remove espaços/tabs no fim de cada linha
        text = re.sub(r"[ \t]+\n", "\n", text)
        return text.strip()

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        """Quebra `text` usando o primeiro separador da lista; para
        qualquer fragmento resultante que ainda exceda `chunk_size`,
        aplica recursivamente o próximo separador da lista."""
        if len(text) <= self.chunk_size:
            return [text]

        if not separators:
            # último recurso: corta por caracteres em blocos de chunk_size
            return [
                text[i : i + self.chunk_size]
                for i in range(0, len(text), self.chunk_size)
            ]

        sep, *rest = separators
        parts = re.split(sep, text) if sep.startswith("(?") else text.split(sep)
        parts = [p for p in parts if p != ""]

        if len(parts) <= 1:
            # separador não encontrado neste nível; tenta o próximo
            return self._split_recursive(text, rest)

        result: list[str] = []
        for part in parts:
            if len(part) > self.chunk_size:
                result.extend(self._split_recursive(part, rest))
            else:
                result.append(part)
        return result

    def _merge_pieces(self, pieces: list[str]) -> list[str]:
        """Reagrupa fragmentos pequenos em chunks próximos de `chunk_size`,
        aplicando sobreposição (`chunk_overlap`) entre chunks consecutivos."""
        if not pieces:
            return []

        merged: list[str] = []
        current = ""

        for piece in pieces:
            candidate = f"{current}\n{piece}".strip() if current else piece
            if len(candidate) <= self.chunk_size or not current:
                current = candidate
            else:
                merged.append(current)
                overlap_text = self._tail(current, self.chunk_overlap)
                current = f"{overlap_text}\n{piece}".strip() if overlap_text else piece

        if current:
            merged.append(current)
        return merged

    @staticmethod
    def _tail(text: str, n_chars: int) -> str:
        """Retorna os últimos `n_chars` de `text`, ajustando para não
        cortar uma palavra ao meio (recua até o espaço anterior)."""
        if n_chars <= 0 or not text:
            return ""
        tail = text[-n_chars:]
        space_idx = tail.find(" ")
        if space_idx > 0:
            tail = tail[space_idx + 1 :]
        return tail.strip()
