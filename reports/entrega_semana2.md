# GeroRAG — Roteiro de Apresentação (Semana 2)
### Chunking → Embeddings → Banco Vetorial

---

## 1. Chunking

**Como foi implementado.** Um *chunker* recursivo por caracteres: tenta
cortar respeitando a estrutura natural do texto (primeiro por parágrafo,
depois por linha, depois por frase, só corta no meio de uma palavra se não
tiver outro jeito). Aplica sobreposição entre chunks vizinhos, para não
perder contexto que caiu bem na fronteira do corte.

**Duas fontes, dois tratamentos:**
- **Documentos** (diretrizes e manuais das escalas — Fried, Katz,
  Lawton-Brody, SARC-F) → chunks de **~800 caracteres, com 120 de
  sobreposição**.
- **Base de pacientes** (`DATASET_IDOSOS.csv`) → cada paciente vira um
  **resumo clínico em linguagem natural** (idade, IMC, escores das
  escalas, comorbidades, medicamentos) e, em geral, **um único chunk**
  por paciente — já é uma unidade de informação coesa, não faz sentido
  misturar pedaço de um paciente com outro.

**Resultado:** 602 chunks no total — **21 de documentos** + **581 de
pacientes** (um por paciente) — salvos em arquivos JSON, prontos para a
etapa de embeddings.

**Prévia na interface:** aba de chunking no Streamlit, com sliders de
tamanho e sobreposição, mostrando o corte em tempo real (documentos e
pacientes).

**Como rodar:**
```bash
pip install -r requirements.txt
python -m src.ingestion.run_chunking
```
Saída em `data/processed/`.

---

## 2. Embeddings

**Dois backends, no módulo `src/embeddings/`:**
- **`sentence-transformers`** (principal) — embeddings densos/semânticos,
  modelo multilíngue com suporte a português, gera vetores de **384
  posições**. Entende relação de significado (ex.: "perda de força
  muscular" ≈ "fraqueza", mesmo sem palavras em comum).
- **`tfidf`** (alternativa) — método clássico por frequência de palavras;
  não entende sinônimos tão bem, mas roda 100% offline, sem baixar
  modelo — útil sem internet e para testar o pipeline mais rápido durante
  o desenvolvimento.

**Como validaram (teste de sanidade):** pegaram o vetor de um chunk sobre
os critérios de Fried e buscaram os vetores mais próximos dele. O
resultado mais próximo foi outro trecho do próprio documento de Fried; o
segundo foi um trecho do SARC-F — que também fala de fraqueza muscular e
sarcopenia. Isso mostra que o espaço vetorial captura relação de
significado, não coincidência de palavras.

**Saída:** ao rodar, carrega os 602 chunks da etapa anterior e salva três
arquivos — a matriz de vetores, os metadados de cada chunk alinhados por
posição, e as informações do modelo usado.

**Como rodar:**
```bash
pip install -r requirements.txt

# backend recomendado (baixa o modelo na 1ª execução — requer internet)
python -m src.embeddings.run_embeddings

# backend 100% offline, sem download de modelo
python -m src.embeddings.run_embeddings --backend tfidf
```
Saída em `data/processed/embeddings`.

---

## 3. Banco Vetorial + Busca Semântica

**Dois backends, no módulo `src/vectorstore/`** (mesmo padrão usado nos
embeddings — abstração comum + múltiplos backends):
- **`numpy`** (padrão) — busca por similaridade de cosseno em NumPy puro,
  sem nenhuma dependência nova. Para o volume do projeto (602 chunks), a
  busca por força bruta já é praticamente instantânea, e evita a
  complexidade de configurar um banco externo.
- **`chroma`** (opcional) — ChromaDB, tecnologia já recomendada no
  relatório de tecnologias da Semana 1, com persistência e filtragem por
  metadados nativas. Trocar de backend é uma linha no `config.py`.

**Ponto técnico importante:** a pergunta do usuário precisa ser
vetorizada com o **mesmo embedder** usado para indexar os chunks — por
isso o vectorizer do TF-IDF é salvo (`vectorizer.pkl`) e reaproveitado
(nunca re-treinado do zero), e para o sentence-transformers é usado o
mesmo modelo registrado no `info.json` do índice.

**Como validaram (RF08):** um script (`run_validate_search.py`) roda a
lista de perguntas de exemplo do projeto contra o índice e imprime os
top-k trechos recuperados com o *score* de similaridade. Fazem duas
checagens automáticas — resultados vêm ordenados por score decrescente, e
o índice não está vazio — e o script falha com erro se algo estiver
errado, para poder rodar antes de cada entrega.

**Prévia na interface:** aba **"🔍 Busca Semântica (prévia)"** no
Streamlit — digita uma pergunta, escolhe `top-k` e filtro por tipo
(documento/paciente/todos), e vê os trechos recuperados com o score. Ex.:
a pergunta "Como está a qualidade do sono?" traz o trecho do PSQI em
primeiro lugar.

**Como rodar:**
```bash
python -m src.vectorstore.run_build_index          # constrói o índice (RF07)
python -m src.vectorstore.run_validate_search       # valida a busca (RF08)

# variações úteis
python -m src.vectorstore.run_validate_search --query "o paciente tem risco de sarcopenia?"
python -m src.vectorstore.run_validate_search --source-type documento
```
Índice salvo em `data/processed/vectorstore`.
