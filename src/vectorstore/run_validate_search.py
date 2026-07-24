"""
run_validate_search.py
------------------------
Valida a busca semântica (checagem manual + automática) rodando um
conjunto de perguntas de exemplo contra o índice já construído por
`run_build_index.py`, e imprimindo os top-k trechos recuperados com o
score de similaridade e a fonte — para inspeção humana da qualidade da
recuperação (RF08).

Uso:
    python -m src.vectorstore.run_validate_search
    python -m src.vectorstore.run_validate_search --top-k 3 --source-type documento
    python -m src.vectorstore.run_validate_search --query "o paciente tem risco de sarcopenia?"
"""

from __future__ import annotations

import argparse
import sys
import time

from ..ingestion import config
from .search import load_query_embedder, load_vector_store, semantic_search

EXEMPLOS_PERGUNTAS = [
    "Este paciente apresenta sinais de fragilidade?",
    "Como interpretar o escore de Katz?",
    "O paciente apresenta risco funcional?",
    "Como está a qualidade do sono?",
    "O paciente apresenta risco de sarcopenia?",
    "Como interpretar os resultados da Escala de Lawton?",
    "Quais fatores merecem maior atenção?",
]


def _print_result(rank: int, result) -> None:
    preview = result.text.strip().replace("\n", " ")
    if len(preview) > 160:
        preview = preview[:160] + "..."
    print(f"    [{rank}] score={result.score:.3f}  fonte={result.source}  tipo={result.source_type}")
    print(f"        {preview}")


def run_validation(queries: list[str], top_k: int, source_type: str | None) -> bool:
    """Retorna True se a validação passou nas checagens automáticas mínimas."""
    print("Carregando embedder da pergunta e índice vetorial...")
    embedder = load_query_embedder()
    store = load_vector_store()
    n_indexed = store.count()
    print(f"  Índice carregado com {n_indexed} chunks.")

    if n_indexed == 0:
        print(
            "\n⚠️  O índice está vazio. Rode antes:\n"
            "    python -m src.ingestion.run_chunking\n"
            "    python -m src.embeddings.run_embeddings\n"
            "    python -m src.vectorstore.run_build_index"
        )
        return False

    ok = True
    print(f"\nRodando {len(queries)} pergunta(s) de validação"
          + (f" (filtro source_type={source_type!r})" if source_type else " (sem filtro de tipo)")
          + "...\n")

    for query in queries:
        t0 = time.time()
        results = semantic_search(
            query, top_k=top_k, source_type=source_type, embedder=embedder, store=store
        )
        elapsed_ms = (time.time() - t0) * 1000

        print(f"❓ {query}")
        if not results:
            print("    (nenhum resultado — verifique se há chunks desse source_type indexados)")
            ok = False
        else:
            # checagem automática mínima: scores devem vir em ordem decrescente
            scores = [r.score for r in results]
            if scores != sorted(scores, reverse=True):
                print("    ⚠️  ATENÇÃO: resultados não estão ordenados por score decrescente!")
                ok = False
            for i, r in enumerate(results, start=1):
                _print_result(i, r)
        print(f"    (busca em {elapsed_ms:.0f} ms)\n")

    return ok


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validação da busca semântica do GeroRAG")
    parser.add_argument("--top-k", type=int, default=config.DEFAULT_TOP_K)
    parser.add_argument(
        "--source-type", default=None,
        help="Filtra por 'documento' ou 'paciente'. Padrão: sem filtro (busca em tudo).",
    )
    parser.add_argument(
        "--query", action="append", dest="queries",
        help="Pergunta customizada (pode repetir a flag várias vezes). "
             "Se omitido, usa a lista de perguntas de exemplo do projeto.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    queries = args.queries or EXEMPLOS_PERGUNTAS

    print("GeroRAG — Validação da Busca Semântica (Semana 2)")
    print("=" * 50)
    try:
        passed = run_validation(queries, top_k=args.top_k, source_type=args.source_type)
    except (ImportError, FileNotFoundError, ValueError) as exc:
        print(f"\nErro: {exc}", file=sys.stderr)
        sys.exit(1)

    print("=" * 50)
    if passed:
        print("✅ Validação concluída sem problemas automáticos detectados.")
        print("   Revise manualmente se os trechos recuperados fazem sentido para cada pergunta.")
    else:
        print("⚠️  Validação encontrou problemas — veja os avisos acima.")
        sys.exit(1)


if __name__ == "__main__":
    main()
