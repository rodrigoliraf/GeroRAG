"""
GeroRAG - Protótipo Inicial de Interface (Semana 1)
----------------------------------------------------
Este protótipo já carrega e limpa o DATASET_IDOSOS.csv real (RF01-RF03),
mas ainda NÃO implementa o pipeline RAG nem o LLM (RF04 em diante).
A área de resposta e a área de fontes recuperadas seguem como placeholders,
para validar apenas a organização das telas e a navegação.

Para executar:
    pip install streamlit pandas
    streamlit run app.py
(o arquivo data/raw/DATASET_IDOSOS.csv deve estar presente na estrutura do repositório)
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import sys
from pathlib import Path

# Permite importar o pacote src/ (irmão de app/) independentemente de onde
# o Streamlit for iniciado.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion import config as ingestion_config
from src.ingestion.text_chunker import RecursiveCharacterChunker
from src.ingestion.document_loader import load_documents
from src.ingestion.patient_chunker import build_patient_documents

# ---------------------------------------------------------------------------
# Configuração geral da página
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="GeroRAG - Avaliação Geriátrica Inteligente",
    page_icon="🩺",
    layout="wide",
)

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "DATASET_IDOSOS.csv"

# Colunas selecionadas para exibição na aplicação (ver relatório da Semana 1)
COLS_DISPLAY = {
    "IDADE_ANOS": "Idade",
    "SEXO": "Sexo (código)",
    "ESCOLARIDADE_ANOS DE ESTUDO": "Escolaridade (anos de estudo)",
    "ESTADO CIVIL": "Estado civil (código)",
    "IMC (kg/m2)": "IMC (kg/m²)",
    "INDICE_DE_KATz_TOTAL": "Índice de Katz",
    "INDICE_DE_LAWTON_TOTAL": "Escala de Lawton-Brody",
    "CLASSIFICACAO_FRAGILIDADE": "Classificação de Fragilidade (código)",
    "SARC-F_SOMATORIO": "SARC-F (somatório)",
    "IPAQ_CLASSIFICACAO": "IPAQ (código)",
    "NUMERO_COMORBIDADES_APRESENTADAS": "Nº de comorbidades",
    "NUMERO_MEDICAMENTOS (DIARIO)": "Nº de medicamentos/dia",
}


# ---------------------------------------------------------------------------
# Carregamento e limpeza dos dados (RF01-RF03)
# ---------------------------------------------------------------------------
@st.cache_data
def load_patients() -> pd.DataFrame:
    """Carrega o DATASET_IDOSOS.csv e aplica a limpeza mínima necessária,
    conforme os problemas de qualidade identificados no relatório da
    Semana 1: nomes de coluna malformados, marcador '.' como ausente,
    decimais com vírgula e IDs de paciente duplicados."""
    df = pd.read_csv(DATA_PATH)

    # 1) normaliza nomes de coluna (remove \n e espaços nas pontas)
    df.columns = [c.strip() for c in df.columns]

    # 2) trata '.' e '-' como ausente em todo o dataframe
    df = df.replace({".": np.nan, "-": np.nan})

    # 3) converte colunas numéricas com vírgula decimal (ex.: "28,2")
    def to_float_comma(series: pd.Series) -> pd.Series:
        return pd.to_numeric(
            series.astype(str).str.replace(",", ".", regex=False), errors="coerce"
        )

    for col in ["IMC (kg/m2)", "PESO (kg)", "ALTURA (m)"]:
        if col in df.columns:
            df[col] = to_float_comma(df[col])

    for col in ["INDICE_DE_KATz_TOTAL", "INDICE_DE_LAWTON_TOTAL", "SARC-F_SOMATORIO",
                "NUMERO_COMORBIDADES_APRESENTADAS", "NUMERO_MEDICAMENTOS (DIARIO)",
                "CLASSIFICACAO_FRAGILIDADE", "IPAQ_CLASSIFICACAO"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 4) gera um identificador único interno (PARTICIPANTE tem duplicidades)
    df = df.reset_index(drop=True)
    df["ID_INTERNO"] = ["P" + str(i + 1).zfill(3) for i in df.index]

    return df


@st.cache_data
def get_comorbidities(df: pd.DataFrame, row_idx: int) -> list:
    cd_cols = [c for c in df.columns if c.startswith("CD_")]
    row = df.loc[row_idx, cd_cols]
    presentes = [c.replace("CD_", "").title() for c in cd_cols if str(row[c]) == "1"]
    return presentes


@st.cache_data
def get_medications(df: pd.DataFrame, row_idx: int) -> list:
    med_cols = [c for c in df.columns if c.startswith(("MEDICAMENTO_", "MEDICAMETO_", "MEDICAMENBTO_"))]
    row = df.loc[row_idx, med_cols]
    presentes = [c.split("_", 1)[1].title() for c in med_cols if str(row[c]) == "1"]
    return presentes


try:
    df_pacientes = load_patients()
    DATA_OK = True
except Exception as e:
    DATA_OK = False
    DATA_ERROR = str(e)


@st.cache_data
def run_chunking_preview(chunk_size: int, chunk_overlap: int):
    """Executa o pipeline de chunking (documentos + pacientes) para a
    página de prévia. Cacheado por combinação de parâmetros."""
    doc_chunker = RecursiveCharacterChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_chunk_chars=ingestion_config.MIN_CHUNK_CHARS,
    )
    doc_chunks = []
    for doc in load_documents():
        doc_chunks.extend(
            doc_chunker.chunk_document(
                text=doc.text, source=doc.source, source_type="documento",
                metadata=doc.metadata, id_prefix=doc.source,
            )
        )

    patient_chunker = RecursiveCharacterChunker(
        chunk_size=ingestion_config.PATIENT_CHUNK_SIZE,
        chunk_overlap=ingestion_config.PATIENT_CHUNK_OVERLAP,
        min_chunk_chars=1,
    )
    patient_chunks = []
    for doc in build_patient_documents(df_pacientes):
        patient_chunks.extend(
            patient_chunker.chunk_document(
                text=doc.text, source=doc.source, source_type="paciente",
                metadata=doc.metadata, id_prefix=doc.source,
            )
        )
    return doc_chunks, patient_chunks

EXEMPLOS_PERGUNTAS = [
    "Este paciente apresenta sinais de fragilidade?",
    "Como interpretar o escore de Katz?",
    "O paciente apresenta risco funcional?",
    "Como está a qualidade do sono?",
    "Quais fatores merecem maior atenção?",
]

MOCK_FONTES = [
    {"documento": "Critérios de Fragilidade de Fried", "trecho": "Definição dos 5 componentes da síndrome de fragilidade...", "pagina": 3},
    {"documento": "Manual da Escala de Katz", "trecho": "Interpretação das pontuações de independência funcional...", "pagina": 7},
    {"documento": "Escala de Lawton-Brody", "trecho": "Classificação das atividades instrumentais de vida diária...", "pagina": 2},
]

# ---------------------------------------------------------------------------
# Estado da sessão
# ---------------------------------------------------------------------------
if "paciente_selecionado" not in st.session_state:
    st.session_state.paciente_selecionado = None
if "historico" not in st.session_state:
    st.session_state.historico = []

# ---------------------------------------------------------------------------
# Barra lateral - navegação
# ---------------------------------------------------------------------------
st.sidebar.title("🩺 GeroRAG")
st.sidebar.caption("Assistente de Avaliação Geriátrica (protótipo)")
pagina = st.sidebar.radio(
    "Navegação",
    [
        "🏠 Início",
        "👤 Seleção do Paciente",
        "💬 Avaliação / Perguntas",
        "📚 Fontes Recuperadas",
        "🧩 Chunking (prévia)",
    ],
)

if not DATA_OK:
    st.error(
        f"Não foi possível carregar `data/raw/DATASET_IDOSOS.csv`: {DATA_ERROR}\n\n"
        "Verifique se o arquivo está na estrutura de pastas do repositório."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Tela 1 - Início
# ---------------------------------------------------------------------------
if pagina == "🏠 Início":
    st.title("Assistente Inteligente para Avaliação Geriátrica")
    st.markdown(
        """
        Bem-vindo ao **GeroRAG**, um assistente que combina dados clínicos de
        pacientes idosos com diretrizes e instrumentos validados de avaliação
        geriátrica, usando **RAG (Retrieval-Augmented Generation)**.
        """
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Pacientes na base", f"{len(df_pacientes)}")
    col2.metric("Documentos indexados", "7 (a organizar em docs/)")
    col3.metric("Status do pipeline RAG", "Ainda não implementado")

    st.subheader("Como usar")
    st.markdown(
        """
        1. Vá até **Seleção do Paciente** e escolha um paciente da base.
        2. Acesse **Avaliação / Perguntas** para digitar uma pergunta em
           linguagem natural sobre o paciente.
        3. Consulte **Fontes Recuperadas** para ver quais documentos
           fundamentariam a resposta.
        """
    )

# ---------------------------------------------------------------------------
# Tela 2 - Seleção do Paciente
# ---------------------------------------------------------------------------
elif pagina == "👤 Seleção do Paciente":
    st.title("Seleção do Paciente")
    st.caption(f"Base real ({len(df_pacientes)} pacientes) — DATASET_IDOSOS.csv, com limpeza básica aplicada")

    id_escolhido = st.selectbox("Selecione o ID do paciente:", df_pacientes["ID_INTERNO"])
    st.session_state.paciente_selecionado = id_escolhido

    row_idx = df_pacientes.index[df_pacientes["ID_INTERNO"] == id_escolhido][0]
    paciente = df_pacientes.loc[row_idx]

    st.subheader(f"Dados clínicos — {id_escolhido}")

    c1, c2, c3 = st.columns(3)
    idade = paciente["IDADE_ANOS"]
    c1.metric("Idade", f"{idade:.0f} anos" if pd.notna(idade) else "—")
    c2.metric("Sexo (código)", f"{paciente['SEXO']:.0f}" if pd.notna(paciente["SEXO"]) else "—")
    esc = paciente.get("ESCOLARIDADE_ANOS DE ESTUDO")
    c3.metric("Escolaridade (anos)", esc if pd.notna(esc) else "não informado")

    st.caption("⚠️ Códigos de SEXO/estado civil/raça ainda não têm dicionário de dados (codebook) confirmado — ver relatório da Semana 1.")

    st.markdown("#### Instrumentos e escalas")
    e1, e2, e3 = st.columns(3)
    katz = paciente["INDICE_DE_KATz_TOTAL"]
    lawton = paciente["INDICE_DE_LAWTON_TOTAL"]
    frag = paciente["CLASSIFICACAO_FRAGILIDADE"]
    e1.metric("Índice de Katz", f"{katz:.0f}" if pd.notna(katz) else "não avaliado")
    e2.metric("Escala de Lawton-Brody", f"{lawton:.0f}" if pd.notna(lawton) else "não avaliado")
    e3.metric("Fragilidade (código)", f"{frag:.0f}" if pd.notna(frag) else "não avaliado")

    e4, e5, e6 = st.columns(3)
    sarcf = paciente["SARC-F_SOMATORIO"]
    ipaq = paciente["IPAQ_CLASSIFICACAO"]
    ncom = paciente["NUMERO_COMORBIDADES_APRESENTADAS"]
    e4.metric("SARC-F", f"{sarcf:.0f}" if pd.notna(sarcf) else "não avaliado")
    e5.metric("IPAQ (código)", f"{ipaq:.0f}" if pd.notna(ipaq) else "não avaliado")
    e6.metric("Nº comorbidades", f"{ncom:.0f}" if pd.notna(ncom) else "não informado")

    with st.expander("🩺 Comorbidades registradas"):
        comorbidades = get_comorbidities(df_pacientes, row_idx)
        st.write(", ".join(comorbidades) if comorbidades else "Nenhuma comorbidade marcada.")

    with st.expander("💊 Medicamentos em uso"):
        medicamentos = get_medications(df_pacientes, row_idx)
        st.write(", ".join(medicamentos) if medicamentos else "Nenhum medicamento marcado.")

    with st.expander("Ver dados brutos selecionados (tabela)"):
        cols_existentes = [c for c in COLS_DISPLAY if c in df_pacientes.columns]
        tabela = paciente[cols_existentes].rename(index=COLS_DISPLAY)
        st.dataframe(tabela.to_frame(name="Valor"), use_container_width=True)

    st.success(f"Paciente **{id_escolhido}** selecionado. Vá até 'Avaliação / Perguntas' para continuar.")

# ---------------------------------------------------------------------------
# Tela 3 - Avaliação / Perguntas
# ---------------------------------------------------------------------------
elif pagina == "💬 Avaliação / Perguntas":
    st.title("Avaliação Geriátrica — Perguntas em Linguagem Natural")

    if not st.session_state.paciente_selecionado:
        st.warning("Nenhum paciente selecionado. Volte em 'Seleção do Paciente'.")
    else:
        st.caption(f"Paciente atual: **{st.session_state.paciente_selecionado}**")

        with st.expander("💡 Exemplos de perguntas"):
            for p in EXEMPLOS_PERGUNTAS:
                st.markdown(f"- {p}")

        pergunta = st.text_area(
            "Digite sua pergunta:",
            placeholder="Ex.: Este paciente apresenta sinais de fragilidade?",
            height=100,
        )

        if st.button("Gerar avaliação", type="primary", disabled=not pergunta):
            st.session_state.historico.append(
                {"paciente": st.session_state.paciente_selecionado, "pergunta": pergunta}
            )
            with st.spinner("Buscando nas diretrizes e gerando avaliação... (simulado)"):
                time.sleep(1.2)

            st.subheader("📋 Resposta")
            st.markdown(
                """
                > **Este é um espaço reservado.**
                > Nas próximas semanas, esta área exibirá a avaliação
                > estruturada gerada pelo LLM com base nos dados do paciente
                > e nos trechos recuperados pelo RAG, contendo:
                > - Resumo clínico
                > - Principais achados
                > - Interpretação baseada nas diretrizes
                > - Justificativa
                > - Recomendações educacionais
                > - Fontes utilizadas
                > - Aviso de segurança
                """
            )
            st.caption("👉 Consulte a aba 'Fontes Recuperadas' para ver os documentos que fundamentariam esta resposta.")

        if st.session_state.historico:
            with st.expander("🕑 Histórico de perguntas (nesta sessão)"):
                for h in reversed(st.session_state.historico):
                    st.markdown(f"- **{h['paciente']}**: {h['pergunta']}")

# ---------------------------------------------------------------------------
# Tela 4 - Fontes Recuperadas
# ---------------------------------------------------------------------------
elif pagina == "📚 Fontes Recuperadas":
    st.title("Documentos e Trechos Recuperados")
    st.caption("Placeholder — a busca semântica real será implementada nas próximas semanas.")

    if not st.session_state.historico:
        st.info("Faça uma pergunta na aba 'Avaliação / Perguntas' para simular a recuperação de fontes.")
    else:
        st.markdown(f"Última pergunta: *\"{st.session_state.historico[-1]['pergunta']}\"*")
        st.divider()
        for fonte in MOCK_FONTES:
            with st.container(border=True):
                st.markdown(f"**📄 {fonte['documento']}** — página {fonte['pagina']}")
                st.caption(fonte["trecho"])

# ---------------------------------------------------------------------------
# Tela 5 - Chunking (prévia) — Semana 2
# ---------------------------------------------------------------------------
elif pagina == "🧩 Chunking (prévia)":
    st.title("Pipeline de Chunking (Semana 2)")
    st.caption(
        "Divide os documentos de `docs/` e o resumo clínico de cada paciente "
        "em chunks prontos para a próxima etapa (embeddings + banco vetorial)."
    )

    c1, c2 = st.columns(2)
    chunk_size = c1.slider("Tamanho do chunk (caracteres)", 300, 1500, ingestion_config.DOC_CHUNK_SIZE, step=50)
    chunk_overlap = c2.slider("Sobreposição (caracteres)", 0, 300, ingestion_config.DOC_CHUNK_OVERLAP, step=10)

    doc_chunks, patient_chunks = run_chunking_preview(chunk_size, chunk_overlap)

    m1, m2, m3 = st.columns(3)
    m1.metric("Chunks de documentos", len(doc_chunks))
    m2.metric("Chunks de pacientes", len(patient_chunks))
    m3.metric("Total de chunks", len(doc_chunks) + len(patient_chunks))

    st.divider()
    st.subheader("📄 Documentos (docs/)")
    if not doc_chunks:
        st.info(
            "Nenhum documento encontrado em `docs/`. Adicione arquivos `.txt`, "
            "`.md` ou `.pdf` com diretrizes e manuais de escalas para vê-los "
            "aqui divididos em chunks."
        )
    else:
        fontes = sorted({c.source for c in doc_chunks})
        fonte_sel = st.selectbox("Fonte:", fontes)
        chunks_da_fonte = [c for c in doc_chunks if c.source == fonte_sel]
        st.caption(f"{len(chunks_da_fonte)} chunk(s) gerados para este documento.")
        for c in chunks_da_fonte:
            with st.expander(f"Chunk {c.chunk_index + 1}/{c.total_chunks} — {c.n_chars} caracteres"):
                st.text(c.text)

    st.divider()
    st.subheader("👤 Pacientes (resumo clínico → chunk)")
    if patient_chunks:
        ids = sorted({c.source for c in patient_chunks})
        id_sel = st.selectbox("Paciente:", ids)
        for c in [c for c in patient_chunks if c.source == id_sel]:
            with st.expander(f"Chunk {c.chunk_index + 1}/{c.total_chunks} — {c.n_chars} caracteres", expanded=True):
                st.text(c.text)

    st.divider()
    st.caption(
        "Para gerar os arquivos `data/processed/chunks*.jsonl` usados pela "
        "etapa de embeddings, rode: `python -m src.ingestion.run_chunking`"
    )
