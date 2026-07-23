"""
Testes simples (sem pytest) para o RecursiveCharacterChunker.
Executar com: python3 -m src.ingestion.tests.test_chunking
"""

from src.ingestion.text_chunker import RecursiveCharacterChunker


def test_no_split_when_short():
    chunker = RecursiveCharacterChunker(chunk_size=200, chunk_overlap=20)
    texto = "Um texto curto que cabe em um único chunk."
    chunks = chunker.split_text(texto)
    assert len(chunks) == 1, f"esperado 1 chunk, obtido {len(chunks)}"
    assert chunks[0] == texto


def test_splits_long_text():
    chunker = RecursiveCharacterChunker(chunk_size=100, chunk_overlap=20)
    paragrafo = "Frase número {}. " * 1
    texto = "\n\n".join(f"Parágrafo {i}. " + "Conteúdo de exemplo. " * 5 for i in range(6))
    chunks = chunker.split_text(texto)
    assert len(chunks) > 1, "texto longo deveria gerar múltiplos chunks"
    for c in chunks:
        assert len(c) <= chunker.chunk_size + chunker.chunk_overlap + 50


def test_chunk_document_metadata_consistency():
    chunker = RecursiveCharacterChunker(chunk_size=80, chunk_overlap=15)
    texto = "Frase A. Frase B. Frase C. " * 10
    chunks = chunker.chunk_document(texto, source="doc_teste.md", source_type="documento")
    assert len(chunks) > 1
    for i, c in enumerate(chunks):
        assert c.chunk_index == i
        assert c.total_chunks == len(chunks)
        assert c.source == "doc_teste.md"
        assert c.source_type == "documento"
        assert c.text.strip() != ""


def test_empty_text_returns_no_chunks():
    chunker = RecursiveCharacterChunker(chunk_size=100, chunk_overlap=10)
    assert chunker.split_text("") == []
    assert chunker.split_text("   \n\n  ") == []


def test_overlap_preserves_context_between_chunks():
    chunker = RecursiveCharacterChunker(chunk_size=60, chunk_overlap=20)
    texto = " ".join(f"palavra{i}" for i in range(40))
    chunks = chunker.split_text(texto)
    assert len(chunks) > 1
    # ao menos uma palavra do fim do chunk N deve reaparecer no início do chunk N+1
    for a, b in zip(chunks, chunks[1:]):
        ultima_palavra_a = a.split()[-1]
        assert ultima_palavra_a in b or b.startswith(a.split()[-2:][0])


def run_all():
    tests = [
        test_no_split_when_short,
        test_splits_long_text,
        test_chunk_document_metadata_consistency,
        test_empty_text_returns_no_chunks,
        test_overlap_preserves_context_between_chunks,
    ]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"OK   {t.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} testes passaram.")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    run_all()
