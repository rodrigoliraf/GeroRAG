"""
patient_chunker.py
-------------------
A base de pacientes (`DATASET_IDOSOS.csv`) é tabular, mas o RAG do GeroRAG
combina recuperação semântica de documentos com dados clínicos do paciente
(ver README). Para que os dados estruturados também possam ser indexados e
recuperados semanticamente (ex.: "pacientes com fragilidade e polifarmácia"),
cada linha do CSV é convertida em um pequeno "documento" textual em
linguagem natural — um resumo clínico do paciente — que passa pelo mesmo
pipeline de chunking usado para os documentos de diretrizes.

A limpeza aplicada ao CSV replica a lógica já usada em `app/app.py`
(colunas malformadas, marcador '.' como ausente, decimais com vírgula),
mantendo as duas partes do projeto consistentes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .document_loader import Document

CODEBOOK_FRAGILIDADE = {
    0: "não frágil (robusto)",
    1: "pré-frágil",
    2: "frágil",
}

CODEBOOK_IPAQ = {
    1: "baixo nível de atividade física",
    2: "nível moderado de atividade física",
    3: "alto nível de atividade física",
}


def load_and_clean_patients(csv_path=None) -> pd.DataFrame:
    """Carrega e limpa `DATASET_IDOSOS.csv` (mesma lógica de `app/app.py`,
    reimplementada aqui sem depender do Streamlit)."""
    csv_path = csv_path or config.PATIENTS_CSV
    df = pd.read_csv(csv_path)

    df.columns = [c.strip() for c in df.columns]
    df = df.replace({".": np.nan, "-": np.nan})

    def to_float_comma(series: pd.Series) -> pd.Series:
        return pd.to_numeric(
            series.astype(str).str.replace(",", ".", regex=False), errors="coerce"
        )

    for col in ["IMC (kg/m2)", "PESO (kg)", "ALTURA (m)"]:
        if col in df.columns:
            df[col] = to_float_comma(df[col])

    for col in [
        "INDICE_DE_KATz_TOTAL",
        "INDICE_DE_LAWTON_TOTAL",
        "SARC-F_SOMATORIO",
        "NUMERO_COMORBIDADES_APRESENTADAS",
        "NUMERO_MEDICAMENTOS (DIARIO)",
        "CLASSIFICACAO_FRAGILIDADE",
        "IPAQ_CLASSIFICACAO",
        "IDADE_ANOS",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.reset_index(drop=True)
    df["ID_INTERNO"] = ["P" + str(i + 1).zfill(3) for i in df.index]
    return df


def _comorbidities_for_row(df: pd.DataFrame, row_idx: int) -> list[str]:
    cd_cols = [c for c in df.columns if c.startswith("CD_")]
    row = df.loc[row_idx, cd_cols]
    return [c.replace("CD_", "").title() for c in cd_cols if str(row[c]) == "1"]


def _medications_for_row(df: pd.DataFrame, row_idx: int) -> list[str]:
    med_cols = [
        c
        for c in df.columns
        if c.startswith(("MEDICAMENTO_", "MEDICAMETO_", "MEDICAMENBTO_"))
    ]
    row = df.loc[row_idx, med_cols]
    return [c.split("_", 1)[1].title() for c in med_cols if str(row[c]) == "1"]


def _fmt(value, unit: str = "", ndigits: int | None = 0, default: str = "não informado") -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    if ndigits is not None and isinstance(value, (int, float)):
        value = f"{value:.{ndigits}f}"
    return f"{value}{unit}"


def build_patient_text(df: pd.DataFrame, row_idx: int) -> str:
    """Gera um resumo clínico em linguagem natural para um paciente,
    combinando dados demográficos, antropométricos, escalas funcionais,
    comorbidades e medicamentos em uso."""
    p = df.loc[row_idx]
    pid = p["ID_INTERNO"]

    idade = _fmt(p.get("IDADE_ANOS"), " anos")
    imc = _fmt(p.get("IMC (kg/m2)"), " kg/m²", ndigits=1)
    escolaridade = _fmt(p.get("ESCOLARIDADE_ANOS DE ESTUDO"), " anos de estudo")

    katz = _fmt(p.get("INDICE_DE_KATz_TOTAL"), "/6")
    lawton = _fmt(p.get("INDICE_DE_LAWTON_TOTAL"))
    sarcf = _fmt(p.get("SARC-F_SOMATORIO"), "/10")
    n_comorb = _fmt(p.get("NUMERO_COMORBIDADES_APRESENTADAS"))
    n_med = _fmt(p.get("NUMERO_MEDICAMENTOS (DIARIO)"))

    frag_cod = p.get("CLASSIFICACAO_FRAGILIDADE")
    frag_desc = (
        CODEBOOK_FRAGILIDADE.get(int(frag_cod)) if pd.notna(frag_cod) else None
    )
    frag_txt = frag_desc or "classificação de fragilidade não avaliada"

    ipaq_cod = p.get("IPAQ_CLASSIFICACAO")
    ipaq_desc = CODEBOOK_IPAQ.get(int(ipaq_cod)) if pd.notna(ipaq_cod) else None
    ipaq_txt = ipaq_desc or "nível de atividade física não avaliado"

    comorbidades = _comorbidities_for_row(df, row_idx)
    medicamentos = _medications_for_row(df, row_idx)

    comorb_txt = (
        "Comorbidades registradas: " + ", ".join(comorbidades) + "."
        if comorbidades
        else "Nenhuma comorbidade registrada."
    )
    med_txt = (
        "Medicamentos em uso: " + ", ".join(medicamentos) + "."
        if medicamentos
        else "Nenhum medicamento em uso registrado."
    )

    partes = [
        f"Paciente {pid}, {idade}, com {escolaridade} e IMC de {imc}.",
        f"Classificação de fragilidade: {frag_txt}. "
        f"Índice de Katz (ABVD): {katz}. "
        f"Escala de Lawton-Brody (AIVD): {lawton}. "
        f"SARC-F (triagem de sarcopenia): {sarcf}.",
        f"Nível de atividade física (IPAQ): {ipaq_txt}. "
        f"Número de comorbidades: {n_comorb}. "
        f"Número de medicamentos de uso diário: {n_med}.",
        comorb_txt,
        med_txt,
    ]
    return "\n".join(partes)


def build_patient_documents(df: pd.DataFrame | None = None) -> list[Document]:
    """Converte cada paciente da base em um `Document` textual, pronto para
    ser dividido em chunks pelo `RecursiveCharacterChunker`."""
    if df is None:
        df = load_and_clean_patients()

    documents: list[Document] = []
    for row_idx in df.index:
        pid = df.loc[row_idx, "ID_INTERNO"]
        text = build_patient_text(df, row_idx)
        documents.append(
            Document(
                source=pid,
                text=text,
                metadata={"row_idx": int(row_idx)},
            )
        )
    return documents
