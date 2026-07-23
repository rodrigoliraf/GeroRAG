# GeroRAG

Assistente Inteligente para Avaliação Geriátrica utilizando LLM e
Retrieval-Augmented Generation (RAG).

## Sobre o projeto

O GeroRAG é um assistente que analisa dados clínicos de idosos
(`DATASET_IDOSOS.csv`), responde perguntas em linguagem natural e gera
avaliações geriátricas estruturadas, fundamentadas em diretrizes clínicas e
instrumentos validados (Katz, Lawton-Brody, Critérios de Fried, SARC-F, PSQI,
IPAQ, entre outros), combinando dados estruturados do paciente com
recuperação semântica de documentos (RAG).

Projeto desenvolvido para a disciplina de Tópicos Especiais em Computação, do curso de
Ciência da Computação — UEPB.

**Dataset:** 581 registros, 372 variáveis (579 pacientes com ID único — ver
detalhes e problemas de qualidade dos dados em `reports/entrega_semana1.md`).

## Integrantes

| Nome | GitHub |
| Rodrigo Lira | rodrigoliraf |
| Antônio Neri | NeriProg |
| Eric Natan | EricNatanbt |
| Rodrigo Daniel | rodrigo-ot |

## Estrutura de pastas

```
GeroRAG/
├── app/                    # Aplicação Streamlit
│   ├── app.py
│   └── requirements.txt
├── data/
│   ├── raw/                 # DATASET_IDOSOS.csv original (581 linhas, 372 colunas)
│   └── processed/           # Dataset tratado (limpeza descrita em reports/entrega_semana1.md)
├── docs/                    # Base documental do RAG (manuais/escalas em .md; aceita .pdf)
├── notebooks/                # Análise exploratória (EDA)
├── src/
│   ├── ingestion/            # Carregamento e chunking (documentos + pacientes) ✅
│   │   ├── document_loader.py
│   │   ├── patient_chunker.py
│   │   ├── text_chunker.py
│   │   ├── run_chunking.py
│   │   └── tests/
<<<<<<<< HEAD:docs/README.md
│   ├── embeddings/           # Geração de embeddings
========
│   ├── embeddings/           # Geração de embeddings ✅
│   │   ├── base.py
│   │   ├── sentence_transformer_embedder.py
│   │   ├── tfidf_embedder.py
│   │   ├── store.py
│   │   ├── run_embeddings.py
│   │   └── tests/
>>>>>>>> 81786dc (Geração de Embeddings):README.md
│   ├── vectorstore/          # Configuração do banco vetorial
│   ├── rag/                  # Pipeline de recuperação + prompt
│   └── llm/                  # Integração com o LLM
├── reports/                  # Relatório técnico e material de apresentação
├── README.md
└── requirements.txt
```

## Como executar (protótipo atual)

```bash
cd app
pip install -r requirements.txt
streamlit run app.py
```

A tela **🧩 Chunking (prévia)** do app permite visualizar, com sliders de
tamanho/sobreposição, como os documentos de `docs/` e o resumo clínico de
cada paciente são divididos em chunks.

## Pipeline de ingestão / chunking (Semana 2)

O módulo `src/ingestion/` implementa a etapa de chunking do RAG:

- **`document_loader.py`** — carrega os documentos de `docs/` (`.txt`,
  `.md`, e `.pdf` via `pypdf`, se instalado): diretrizes e manuais das
  escalas (Fried, Katz, Lawton-Brody, SARC-F).
- **`patient_chunker.py`** — carrega e limpa `DATASET_IDOSOS.csv` (mesma
  limpeza usada em `app/app.py`) e converte cada paciente em um resumo
  clínico textual (idade, IMC, escalas, comorbidades, medicamentos), para
  que os dados estruturados também possam ser recuperados semanticamente.
- **`text_chunker.py`** — `RecursiveCharacterChunker`: divide textos longos
  em chunks de tamanho configurável, respeitando fronteiras de parágrafo /
  linha / sentença / palavra sempre que possível, com sobreposição
  (overlap) entre chunks consecutivos.
- **`run_chunking.py`** — orquestra o pipeline e grava os chunks em
  `data/processed/*.jsonl` (`chunks_documentos.jsonl`,
  `chunks_pacientes.jsonl`, `chunks.jsonl`).

Para rodar o pipeline via linha de comando (a partir da raiz do repo):

```bash
pip install -r requirements.txt
python -m src.ingestion.run_chunking
# parâmetros opcionais:
python -m src.ingestion.run_chunking --chunk-size 600 --chunk-overlap 100
python -m src.ingestion.run_chunking --only docs        # só documentos
python -m src.ingestion.run_chunking --only pacientes    # só pacientes
```

Testes unitários do chunker:

```bash
python -m src.ingestion.tests.test_chunking
```

<<<<<<<< HEAD:docs/README.md
========
## Geração de embeddings (Semana 2)

O módulo `src/embeddings/` transforma os chunks em vetores numéricos:

- **`sentence_transformer_embedder.py`** — backend recomendado, embeddings
  densos/semânticos via `sentence-transformers` (modelo multilíngue
  `paraphrase-multilingual-MiniLM-L12-v2`, 384 dimensões). Captura
  similaridade de significado, não só de palavras — essencial para
  perguntas em linguagem natural.
- **`tfidf_embedder.py`** — backend alternativo (TF-IDF, via scikit-learn),
  100% offline e sem download de modelo. Útil como baseline de comparação
  ou quando não há internet disponível.
- **`store.py`** — salva/carrega os vetores (`embeddings.npy`) e os
  metadados de cada chunk alinhados por posição (`chunk_ids.jsonl`).
- **`run_embeddings.py`** — orquestra tudo.

```bash
pip install -r requirements.txt

# backend recomendado (baixa o modelo na primeira execução — requer internet)
python -m src.embeddings.run_embeddings

# backend 100% offline, sem download de modelo
python -m src.embeddings.run_embeddings --backend tfidf

# outro modelo / outro arquivo de entrada
python -m src.embeddings.run_embeddings --model all-MiniLM-L6-v2
python -m src.embeddings.run_embeddings --input data/processed/chunks_documentos.jsonl
```

Saída em `data/processed/embeddings/`: `embeddings.npy` (matriz N × dim),
`chunk_ids.jsonl` (metadados alinhados por linha) e `info.json` (backend,
dimensão, timestamp).

Testes unitários (usam o backend TF-IDF, sem downloads):

```bash
python -m src.embeddings.tests.test_embeddings
```

>>>>>>>> 81786dc (Geração de Embeddings):README.md
## Tecnologias

Python · Streamlit · Pandas · LangChain/LlamaIndex · Sentence Transformers ·
ChromaDB/FAISS · GPT-4o-mini/Gemini/Llama 3 · Plotly

Veja a análise crítica das tecnologias em `reports/`.

## Roadmap

- [x] Semana 1 — Estudo do dataset, base documental, arquitetura, protótipo de interface
<<<<<<<< HEAD:docs/README.md
- [~] Semana 2 — Chunking ✅ (`src/ingestion/`); embeddings e banco vetorial ainda pendentes
========
- [~] Semana 2 — Chunking ✅ e embeddings ✅ (`src/ingestion/`, `src/embeddings/`); banco vetorial ainda pendente
>>>>>>>> 81786dc (Geração de Embeddings):README.md
- [ ] Semana 3 — Integração RAG + LLM
- [ ] Semana 4 — Interface final, testes, documentação e apresentação
