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
├── docs/                    # Base documental do RAG (PDFs, manuais, escalas)
├── notebooks/                # Análise exploratória (EDA)
├── src/
│   ├── ingestion/            # Carregamento e chunking dos documentos
│   ├── embeddings/           # Geração de embeddings
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

## Tecnologias

Python · Streamlit · Pandas · LangChain/LlamaIndex · Sentence Transformers ·
ChromaDB/FAISS · GPT-4o-mini/Gemini/Llama 3 · Plotly

Veja a análise crítica das tecnologias em `reports/`.

## Roadmap

- [x] Semana 1 — Estudo do dataset, base documental, arquitetura, protótipo de interface
- [ ] Semana 2 — Chunking, embeddings, banco vetorial
- [ ] Semana 3 — Integração RAG + LLM
- [ ] Semana 4 — Interface final, testes, documentação e apresentação
