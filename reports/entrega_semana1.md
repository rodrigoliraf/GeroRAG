# GeroRAG — Entrega Semana 1

## 1. Estudo do Dataset (DATASET_IDOSOS.csv)

> ✅ **Atualizado com dados reais** a partir da leitura do `DATASET_IDOSOS.csv`
> (detalhes da extração na Seção 2).

- **Objetivo da base de dados:** reunir informações de Avaliação Geriátrica
  Ampla (AGA) de pessoas idosas, permitindo caracterizar aspectos de
  envelhecimento saudável, funcionalidade, fragilidade e qualidade de vida.
- **Origem dos dados:** dados fictícios (sintéticos), gerados a partir de
  padrões observados em dados reais de estudos sobre envelhecimento — usados
  aqui para fins exclusivamente educacionais, sem risco de identificação de
  pacientes reais.
- **Número de pacientes:** 581 linhas no arquivo. Há 2 IDs de participante
  duplicados (574 e 575, cada um aparecendo em duas linhas com idades
  diferentes), logo o número de **identificadores únicos é 579** — isso entra
  na lista de problemas de qualidade de dados (Seção 2).
- **Quantidade de variáveis:** **372 colunas**, sendo aproximadamente:
  - 54 colunas de comorbidades (prefixo `CD_`)
  - ~200 colunas de medicamentos em uso (prefixo `MEDICAMENTO_`, com algumas
    variações de grafia: `MEDICAMETO_`, `MEDICAMENBTO_`)
  - ~118 colunas de dados sociodemográficos, antropométricos, escalas e escores
- **Principais grupos de variáveis:**
  - Dados sociodemográficos (idade, sexo, escolaridade, estado civil, raça/cor, renda, arranjo familiar)
  - Doenças crônicas / comorbidades (54 doenças, ex.: hipertensão, diabetes, artrose)
  - Medicamentos em uso (~200 fármacos, ex.: losartana, hidroclorotiazida, sinvastatina)
  - Índice de Katz (atividades básicas de vida diária) — `INDICE_DE_KATz_TOTAL`
  - Escala de Lawton-Brody (atividades instrumentais) — `INDICE_DE_LAWTON_TOTAL`
  - Critérios de Fragilidade de Fried — colunas `FF_*` e `CLASSIFICACAO_FRAGILIDADE`
  - SARC-F (rastreio de sarcopenia) — colunas `SARC_F_*` e `SARC-F_SOMATORIO`
  - Escala de Leganés (função cognitiva) — colunas `LEGANES_*`
  - Escala de Depressão — `ESCALA_DE_DEPRESSAO_ESCORE_TOTAL`
  - SPPB (desempenho físico) — colunas `SPPB_*`
  - PSQI — Pittsburgh Sleep Quality Index (qualidade do sono) — colunas `PSQI_*` (7 domínios + escore global)
  - IPAQ (nível de atividade física) — colunas `IPAQ_*`
  - LASA (nível de atividade física, versão alternativa) — colunas `LASA_*`
  - Variáveis antropométricas (peso, altura, IMC, circunferências abdominal/cintura/panturrilha)
  - Hábitos de vida e uso de tecnologia (smartphone, tempo de tela, sono diário, COVID)
  - Questões de usabilidade do estudo original (`USABILIDADE_*`) — não são dados clínicos do paciente
- **Possíveis aplicações do dataset:** triagem de fragilidade, apoio à
  avaliação multidimensional do idoso, estudos epidemiológicos sobre
  envelhecimento, treinamento de modelos educacionais de apoio à decisão
  clínica (não diagnóstica).

### Variáveis que serão utilizadas na aplicação

Subconjunto confirmado após a EDA (nomes exatos das colunas no CSV):

| Grupo | Colunas |
|---|---|
| Identificação/demografia | `PARTICIPANTE`, `IDADE_ANOS`, `SEXO`, `ESCOLARIDADE_ANOS DE ESTUDO`, `ESTADO CIVIL`, `RACA/COR` |
| Antropometria | `PESO (kg)`, `ALTURA (m)`, `IMC (kg/m2)` |
| Funcionalidade básica | `INDICE_DE_KATz_TOTAL` |
| Funcionalidade instrumental | `INDICE_DE_LAWTON_TOTAL` |
| Fragilidade | `CLASSIFICACAO_FRAGILIDADE` (+ componentes `FF_*` para detalhamento) |
| Sarcopenia | `SARC-F_SOMATORIO` |
| Cognição | `LEGANES_TOTAL_GERAL` |
| Sono | `PSQI-ESCORE GLOBAL CLASSIFICAÇÃO PONTUAÇÃO` |
| Atividade física | `IPAQ_CLASSIFICACAO` |
| Comorbidades | as 54 colunas `CD_*` + `NUMERO_COMORBIDADES_APRESENTADAS` |
| Medicamentos | as ~200 colunas `MEDICAMENTO_*` + `NUMERO_MEDICAMENTOS (DIARIO)` |

As colunas de usabilidade do estudo (`USABILIDADE_*`), LASA e as métricas
diárias de sono (`SONO_1DIA...7DIA`) **não serão usadas** na aplicação, pois
são instrumentos redundantes com o IPAQ/PSQI ou específicos da coleta
original de dados, sem relação direta com a avaliação geriátrica pedida.

---

## 2. Exploração e Preparação dos Dados

> ✅ Análise exploratória executada sobre o `DATASET_IDOSOS.csv` real.

### Visão geral

- **Registros:** 581 linhas (579 participantes únicos — ver duplicidade de ID abaixo).
- **Atributos:** 372 colunas.
- **Tipos de dados:** 336 colunas lidas como texto (`object`/`string`), 35 como `float64` e 1 como `int64`. Grande parte das colunas numéricas está sendo lida como texto por dois motivos identificados (ver "Problemas de qualidade").
- **Total de células ausentes (NaN puro, sem contar marcador `.`):** 17.926 de 216.132 células (**8,3%**). Esse número **subestima** os ausentes reais, pois em muitas colunas o valor ausente foi registrado como o caractere `"."` em vez de célula vazia (ver abaixo).

### Distribuição das principais variáveis

| Variável | Resultado |
|---|---|
| `IDADE_ANOS` | média 70,9 anos (DP 6,6); mínimo 55, máximo 93; mediana 70 |
| `SEXO` | valor `1.0`: 403 registros (69,4%) · valor `2.0`: 177 (30,5%) · sem codebook disponível para confirmar qual valor é "feminino"/"masculino" |
| `ESCOLARIDADE_ANOS DE ESTUDO` | 20,7% ausente; entre os preenchidos, "0 anos" é o valor mais comum (74 casos) |
| `INDICE_DE_KATz_TOTAL` | 27% ausente (`.` ou vazio); entre os preenchidos, 88 pacientes com escore 6 (independência total) e 22 com escore 5 |
| `INDICE_DE_LAWTON_TOTAL` | 27,2% ausente; entre os preenchidos, escore 27 (máximo/independência total) é o mais comum (41 casos) |
| `CLASSIFICACAO_FRAGILIDADE` | classe `1`: 261 (45%) · classe `2`: 218 (37,5%) · classe `0`: 97 (16,7%) — sem legenda no arquivo para saber a que cada classe corresponde (robusto/pré-frágil/frágil) |
| `SARC-F_SOMATORIO` | média 4,3 (DP 4,6); varia de 0 a 20 (escore ≥ 4 costuma indicar risco de sarcopenia na literatura) |
| `IPAQ_CLASSIFICACAO` | classe `4`: 236 (40,6%) · classe `2`: 130 (22,4%) · classe `3`: 76 · classe `1`: 72 · classe `5`: 60 — também sem legenda no arquivo |
| `NUMERO_COMORBIDADES_APRESENTADAS` | mais comuns: 1 comorbidade (111 casos) e 2 comorbidades (111 casos); 79 pacientes sem nenhuma |
| `NUMERO_MEDICAMENTOS (DIARIO)` | mais comuns: 2 medicamentos/dia (128 casos), 1 (107), 3 (104); 100 pacientes sem nenhum medicamento |
| Comorbidades mais prevalentes | Hipertensão 43,7% · Diabetes Mellitus 20,1% · Reumatismo 10,5% · Osteoporose 8,1% · Cardiopatia 7,9% |
| Medicamentos mais citados | Losartana (205) · Hidroclorotiazida (128) · Sinvastatina (112) · Glifage (66) · Metformina (63) |

### Problemas de qualidade dos dados identificados

1. **Marcador de ausência inconsistente:** 335 colunas usam o caractere `"."`
   como valor para "não se aplica"/ausente, além das células realmente
   vazias (`NaN`). Isso faz com que `df.isna().sum()` **subestime** a real
   taxa de dados faltantes — é necessário tratar `"."` (e também `"-"`,
   encontrado pontualmente) como ausente antes de qualquer análise ou uso no
   RAG/prompt.
2. **Decimais em formato brasileiro (vírgula):** 23 colunas numéricas (ex.:
   `PESO (kg)`, `ALTURA (m)`, `IMC (kg/m2)`, circunferências, velocidade de
   marcha) usam vírgula como separador decimal (`"28,2"`), fazendo com que o
   pandas as leia como texto em vez de número. É necessário converter
   (`str.replace(',', '.').astype(float)`) antes de qualquer cálculo.
3. **IDs de participante duplicados:** os valores `574` e `575` em
   `PARTICIPANTE` aparecem cada um em duas linhas com idades diferentes (ex.:
   574 → 75 anos e 574 → 66 anos), indicando erro de digitação/geração do ID
   — não são registros duplicados do mesmo paciente, mas um problema de
   unicidade que exige gerar um novo identificador interno (ex.: índice da
   linha) para uso na aplicação.
4. **Nome de coluna malformado:** a coluna do escore global do PSQI vem com
   quebra de linha no próprio nome
   (`"PSQI-ESCORE GLOBAL CLASSIFICAÇÃO PONTUAÇÃO\n"`), o que exige
   `str.strip()` nos nomes de todas as colunas ao carregar o arquivo.
5. **Ausência de dicionário de dados (codebook):** variáveis categóricas
   codificadas numericamente (`SEXO`, `CLASSIFICACAO_FRAGILIDADE`,
   `IPAQ_CLASSIFICACAO`, `ESTADO CIVIL`, `RACA/COR`) não têm a legenda dos
   códigos no próprio arquivo. É necessário obter ou inferir (a partir da
   literatura dos instrumentos originais — Fried, IPAQ etc.) o significado de
   cada valor antes de exibi-los ao usuário final.
6. **Variação de grafia em nomes de colunas de medicamentos:** três colunas
   usam `MEDICAMETO_` ou `MEDICAMENBTO_` (erros de digitação) em vez de
   `MEDICAMENTO_`, o que pode causar perda dessas colunas em buscas
   automatizadas por prefixo se não forem tratadas manualmente.
7. **Alto percentual de ausência em blocos inteiros de colunas:** os blocos
   `SONO_1DIA...7DIA` (~99,7% ausente) e `LASA_*` (~87% ausente) têm
   qualidade muito baixa para uso confiável — reforça a decisão de não
   utilizá-los no projeto (ver tabela de variáveis selecionadas acima).

### Atributos selecionados para o projeto

Ver tabela "Variáveis que serão utilizadas na aplicação" acima. A limpeza
necessária antes do uso (RF01–RF03) inclui: normalizar nomes de colunas,
tratar `"."`/`"-"` como ausente, converter decimais com vírgula para ponto,
e gerar um identificador único por paciente independente do `PARTICIPANTE`
original.

---

## 3. Seleção da Base Documental para o RAG

| Documento | Origem | Finalidade no sistema | Formato |
|---|---|---|---|
| Manual de Avaliação Multidimensional da Pessoa Idosa para a APS (CONASS/Ministério da Saúde) | Portal CONASS / BVS Ministério da Saúde | Avaliação geriátrica geral | PDF |
| Cadernos de Atenção Básica — Saúde da Pessoa Idosa | Ministério da Saúde (BVS/DAB) | Recomendações clínicas no SUS | PDF |
| Manual/artigo sobre a Escala de Katz | Literatura científica / manuais clínicos oficiais | Interpretação da capacidade funcional (AVDs básicas) | PDF |
| Escala de Lawton-Brody | Literatura científica / manuais clínicos oficiais | Avaliação das atividades instrumentais (AIVDs) | PDF |
| Critérios de Fragilidade de Fried | Artigo científico original (Fried et al.) / manuais de geriatria | Identificação de fragilidade | PDF |
| Manual SARC-F | Literatura científica / manuais clínicos oficiais | Avaliação de risco de sarcopenia | PDF |
| PSQI — Pittsburgh Sleep Quality Index | Literatura científica / manuais validados em português | Avaliação da qualidade do sono | PDF |
| IPAQ | Manual oficial do questionário (versão curta/longa) | Avaliação do nível de atividade física | PDF |
| MEEM — Mini Exame do Estado Mental | Literatura científica / manuais clínicos oficiais | Avaliação cognitiva (quando aplicável) | PDF |

Observações:
- Todos os documentos escolhidos são **públicos e de origem oficial/científica**
  (Ministério da Saúde, CONASS, ou publicações científicas amplamente
  utilizadas na prática clínica), evitando materiais protegidos por direitos
  autorais restritivos.
- Nesta etapa os documentos apenas foram **selecionados e organizados** em
  `docs/` — o chunking, geração de embeddings e indexação vetorial ficam
  para a Semana 2, conforme o cronograma do projeto.
- Próximo passo: baixar os PDFs, conferir se o texto é extraível (não
  escaneado) e registrar a URL de origem de cada um em uma planilha de
  controle (`docs/fontes.csv`), para fins de citação na avaliação gerada.

---

## 4. Arquitetura da Solução (detalhada)

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CAMADA DE APRESENTAÇÃO                       │
│                         (Streamlit — app/app.py)                     │
│  Início | Seleção de Paciente | Perguntas | Fontes Recuperadas       │
└───────────────┬───────────────────────────────┬──────────────────────┘
                │                                │
                ▼                                ▼
┌───────────────────────────────┐   ┌─────────────────────────────────┐
│   CAMADA DE DADOS DO PACIENTE  │   │     CAMADA DE ORQUESTRAÇÃO       │
│  src/ingestion/patient_loader  │   │        (src/rag/pipeline)        │
│  - Lê DATASET_IDOSOS.csv       │   │  1. Recebe pergunta do usuário   │
│  - Filtra paciente selecionado │──▶│  2. Busca semântica no vetorial  │
│  - Normaliza escalas/escores   │   │  3. Monta prompt (dados +        │
└───────────────┬─────────────────┘   │     pergunta + contexto RAG)     │
                │                     │  4. Chama o LLM                  │
                │                     │  5. Formata resposta estruturada │
                │                     └───────────────┬───────────────────┘
                │                                     │
                ▼                                     ▼
┌───────────────────────────────┐   ┌─────────────────────────────────┐
│   BASE DOCUMENTAL (RAG)        │   │        CAMADA DE LLM             │
│  docs/*.pdf                    │   │   src/llm/client.py              │
│  src/ingestion/chunker.py      │   │   - GPT-4o-mini / Gemini /       │
│  src/embeddings/encoder.py     │   │     Llama 3 (Ollama)             │
│  src/vectorstore/index.py      │   │   - Prompt template versionado   │
│  (ChromaDB / FAISS)            │◀──│   - Parsing da resposta em       │
└───────────────────────────────┘   │     seções estruturadas          │
                                     └─────────────────────────────────┘
```

**Fluxo de comunicação:**

1. O **Streamlit** (camada de apresentação) captura a seleção do paciente e a
   pergunta do usuário.
2. O **carregador de pacientes** lê o CSV (ou, futuramente, um banco de
   dados leve) e retorna os dados clínicos estruturados do paciente
   selecionado.
3. O **orquestrador RAG** recebe a pergunta, consulta o **banco vetorial**
   (que já contém os documentos indexados via chunking + embeddings) e
   recupera os *top-k* trechos mais relevantes, junto de metadados
   (documento, página/seção, similaridade).
4. O orquestrador monta o **prompt final**, combinando: dados do paciente +
   pergunta + trechos recuperados + instruções de formatação.
5. O **cliente do LLM** envia o prompt ao modelo escolhido e recebe a
   resposta.
6. A resposta é **parseada** em seções (resumo clínico, achados,
   interpretação, recomendações, fontes, aviso) e devolvida à interface.
7. A interface exibe a avaliação estruturada e a lista de fontes utilizadas.

**Separação em módulos** (facilita testes e trocas de tecnologia):
`ingestion` (dados + documentos) → `embeddings` → `vectorstore` → `rag`
(orquestração) → `llm` (cliente do modelo) → `app` (interface).

---

## 5. Análise das Tecnologias

| Camada | Sugestão do projeto | Avaliação crítica | Decisão do grupo |
|---|---|---|---|
| Linguagem | Python | Adequada — ecossistema maduro para NLP/RAG, integra bem com Streamlit e bibliotecas de ML. | Manter Python |
| Interface | Streamlit | Ótima para prototipagem rápida em times pequenos/acadêmicos; limitação: menos controle fino de UI que uma stack web tradicional (React), mas isso não é prioridade aqui. | Manter Streamlit |
| Manipulação de dados | Pandas | Padrão de mercado para datasets tabulares de até centenas de milhares de linhas; 580 pacientes é um volume trivial para Pandas. | Manter Pandas |
| Framework RAG | LangChain **ou** LlamaIndex | Ambos funcionam bem; LangChain tem mais integrações e comunidade, LlamaIndex é mais enxuto e focado em indexação/recuperação. Para o escopo do projeto (RAG simples, um único banco vetorial), **LlamaIndex tende a ter curva de aprendizado menor**; LangChain compensa se o grupo quiser adicionar memória de contexto (desafio extra) ou agentes. | Escolher **um** dos dois logo na Semana 1 e não trocar depois — recomenda-se LlamaIndex pela simplicidade, ou LangChain se já houver familiaridade da equipe |
| Embeddings | Sentence Transformers | Boa escolha: modelos como `paraphrase-multilingual-MiniLM-L12-v2` funcionam bem em português e rodam localmente (sem custo de API). | Manter, especificando um modelo multilíngue/PT-BR |
| Banco vetorial | ChromaDB **ou** FAISS | ChromaDB é mais simples de usar e já persiste metadados nativamente (documento, página) — vantagem para o requisito RF11 (exibir fontes). FAISS é mais rápido em escala mas exige gerenciar metadados manualmente. Para ~9 documentos e poucas dezenas de milhares de chunks, a diferença de performance é irrelevante. | **ChromaDB**, pela simplicidade de metadados |
| LLM | GPT-4o-mini / Gemini / Qwen / Gemma / Llama 3 (Ollama) | Modelos via API (GPT-4o-mini, Gemini Flash) dão melhor qualidade de resposta com pouco esforço de infraestrutura, mas têm custo e dependem de internet/chave de API. Modelos locais via Ollama (Llama 3, Qwen, Gemma) evitam custo e problemas de privacidade de dados de saúde, mas exigem hardware razoável e podem ter qualidade textual inferior. | Recomenda-se **GPT-4o-mini ou Gemini Flash como modelo principal** (melhor qualidade/custo) **com um modelo via Ollama como alternativa/fallback** para testes sem custo e para a demonstração de comparação com/sem RAG |
| Visualização | Plotly **ou** Matplotlib | Plotly é mais interativo e se integra melhor ao Streamlit (`st.plotly_chart`); Matplotlib é mais simples para gráficos estáticos no relatório técnico. | **Plotly** na interface, **Matplotlib** no relatório/notebook de EDA |
| Versionamento | GitHub | Adequado e obrigatório para entrega. | Manter |

**Conclusão:** o conjunto de tecnologias sugerido no enunciado é **viável e
adequado** ao escopo do projeto (~580 pacientes, 9 documentos). As únicas
decisões que exigem definição explícita do grupo, para evitar retrabalho, são:
(1) LangChain vs. LlamaIndex, (2) ChromaDB vs. FAISS e (3) LLM via API vs.
via Ollama — recomenda-se registrar essas decisões no README do repositório.

---

## 7. Protótipo Inicial da Interface

- **Tela inicial** — visão geral do projeto e status do pipeline.
- **Seleção do paciente** — escolha de um paciente simulado e visualização
  dos principais dados clínicos (Katz, Lawton, Fried, SARC-F, PSQI, IPAQ).
- **Avaliação / Perguntas** — campo de texto para a pergunta em linguagem
  natural e área de resposta.
- **Fontes Recuperadas** — área de exibição dos documentos
  que fundamentariam a resposta.