"""
Testes simples (sem pytest) para a etapa de embeddings.
Usa o backend TF-IDF de propósito, pois ele não depende de baixar nenhum
modelo — mantém os testes rápidos e executáveis em qualquer máquina/CI.

Executar com: python3 -m src.embeddings.tests.test_embeddings
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np

from src.embeddings.store import load_embeddings, save_embeddings
from src.embeddings.tfidf_embedder import TfidfEmbedder


def test_tfidf_embedder_shapes():
    embedder = TfidfEmbedder(max_features=100)
    textos = [
        "Paciente com fragilidade e sarcopenia.",
        "Índice de Katz avalia atividades básicas de vida diária.",
        "Escala de Lawton-Brody avalia atividades instrumentais.",
    ]
    embedder.fit(textos)
    vectors = embedder.embed(textos)
    assert vectors.shape[0] == len(textos)
    assert vectors.shape[1] == embedder.dim
    assert embedder.dim > 0


def test_tfidf_requires_fit_before_embed():
    embedder = TfidfEmbedder()
    try:
        embedder.embed(["texto qualquer"])
        raise AssertionError("deveria ter levantado RuntimeError")
    except RuntimeError:
        pass


def test_save_and_load_embeddings_roundtrip():
    embedder = TfidfEmbedder(max_features=50)
    textos = ["frase um sobre idosos", "frase dois sobre fragilidade", "frase três sobre nutrição"]
    embedder.fit(textos)
    vectors = embedder.embed(textos)

    records = [
        {"id": f"doc::chunk_{i:04d}", "source": "doc_teste.md", "source_type": "documento",
         "chunk_index": i, "text": t}
        for i, t in enumerate(textos)
    ]

    tmp_dir = Path(tempfile.mkdtemp())
    try:
        save_embeddings(
            vectors=vectors,
            chunk_records=records,
            output_dir=tmp_dir,
            embedder_info={"backend": embedder.name, "dim": embedder.dim},
            vectorizer=embedder.vectorizer,
        )
        loaded_vectors, loaded_records, info = load_embeddings(tmp_dir)

        assert np.allclose(loaded_vectors, vectors)
        assert len(loaded_records) == len(records)
        assert loaded_records[0]["id"] == records[0]["id"]
        assert info["n_vectors"] == len(records)
        assert (tmp_dir / "vectorizer.pkl").exists()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_mismatched_lengths_raise():
    vectors = np.zeros((3, 4), dtype=np.float32)
    records = [{"id": "a", "source": "x", "source_type": "documento", "text": "t"}]  # só 1, não 3
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        try:
            save_embeddings(vectors, records, tmp_dir, {"backend": "x", "dim": 4})
            raise AssertionError("deveria ter levantado ValueError")
        except ValueError:
            pass
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def run_all():
    tests = [
        test_tfidf_embedder_shapes,
        test_tfidf_requires_fit_before_embed,
        test_save_and_load_embeddings_roundtrip,
        test_mismatched_lengths_raise,
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
