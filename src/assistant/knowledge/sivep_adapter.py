"""Adaptação do SIVEP-Gripe para a base estruturada do assistente.

O projeto original da Fase 3 usava 40 prontuários sintéticos. Este módulo permite
usar registros reais do OpenDataSUS sem mudar o restante do assistente: os campos
necessários são selecionados, códigos são convertidos para texto e um novo
``patient_id`` é criado. Nome, CPF, CNS e endereço não entram nessa base.

Para o repositório deixamos uma amostra anonimizada de 8 mil registros. Em uma
execução local, a mesma função também aceita um ou vários CSVs/Parquets anuais do
SIVEP e grava tudo em SQLite, sem manter todos os anos na memória ao mesmo tempo.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


PATIENT_COLUMNS = [
    "NU_IDADE_N", "TP_IDADE", "CS_SEXO",
    "FEBRE", "TOSSE", "GARGANTA", "DISPNEIA", "DESC_RESP", "SATURACAO",
    "DIARREIA", "VOMITO", "DOR_ABD", "FADIGA", "PERD_OLFT", "PERD_PALA",
    "FATOR_RISC", "PUERPERA", "CARDIOPATI", "HEPATICA", "ASMA", "DIABETES",
    "NEUROLOGIC", "PNEUMOPATI", "IMUNODEPRE", "RENAL", "OBESIDADE",
    "HOSPITAL", "UTI", "SUPORT_VEN", "RAIOX_RES", "TOMO_RES", "AMOSTRA",
    "PCR_RESUL", "VACINA_COV", "NOSOCOMIAL", "CLASSI_FIN", "EVOLUCAO",
]

YES_NO_COLUMNS = {
    "FEBRE", "TOSSE", "GARGANTA", "DISPNEIA", "DESC_RESP", "SATURACAO",
    "DIARREIA", "VOMITO", "DOR_ABD", "FADIGA", "PERD_OLFT", "PERD_PALA",
    "FATOR_RISC", "PUERPERA", "CARDIOPATI", "HEPATICA", "ASMA", "DIABETES",
    "NEUROLOGIC", "PNEUMOPATI", "IMUNODEPRE", "RENAL", "OBESIDADE",
    "HOSPITAL", "UTI", "VACINA_COV", "NOSOCOMIAL",
}

YES_NO_MAP = {1: "sim", 2: "não", 9: "ignorado"}
SUPPORT_MAP = {1: "invasivo", 2: "não invasivo", 3: "não utilizou", 9: "ignorado"}
XRAY_MAP = {
    1: "normal", 2: "infiltrado intersticial", 3: "consolidação",
    4: "misto", 5: "outro", 6: "não realizado", 9: "ignorado",
}
CT_MAP = {
    1: "típico para covid-19", 2: "indeterminado para covid-19",
    3: "atípico para covid-19", 4: "negativo para pneumonia",
    5: "outro", 6: "não realizado", 9: "ignorado",
}
SAMPLE_MAP = {
    1: "secreção de naso-orofaringe", 2: "lavado broncoalveolar",
    3: "tecido post-mortem", 4: "outra", 5: "LCR", 9: "ignorado",
}
PCR_MAP = {
    1: "detectável", 2: "não detectável", 3: "inconclusivo",
    4: "não realizado", 5: "aguardando resultado", 9: "ignorado",
}
EVOLUTION_MAP = {1: "cura", 2: "óbito", 3: "óbito por outras causas", 9: "ignorado"}
CLASSIFICATION_MAP = {
    1: "SRAG por influenza",
    2: "SRAG por outro vírus respiratório",
    3: "SRAG por outro agente etiológico",
    4: "SRAG não especificado",
    5: "SRAG por covid-19",
}


def _year_from_filename(path: Path) -> int | None:
    match = re.search(r"INFLUD(\d{2})", path.name.upper())
    return 2000 + int(match.group(1)) if match else None


def _map_code(series: pd.Series, mapping: dict[int, str]) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.map(mapping).fillna("não informado")


def _age_in_years(age: pd.Series, age_type: pd.Series) -> pd.Series:
    age_num = pd.to_numeric(age, errors="coerce")
    type_num = pd.to_numeric(age_type, errors="coerce").fillna(3)
    result = age_num.copy()
    result = result.where(type_num != 1, age_num / 365.25)
    result = result.where(type_num != 2, age_num / 12.0)
    return result


def _read_raw_sivep(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
        return df[[c for c in PATIENT_COLUMNS if c in df.columns]].copy()

    header = pd.read_csv(path, sep=";", encoding="latin-1", nrows=0).columns
    usecols = [c for c in PATIENT_COLUMNS if c in header]
    try:
        return pd.read_csv(
            path, sep=";", encoding="latin-1", usecols=usecols, low_memory=False
        )
    except pd.errors.ParserError:
        # Alguns snapshots do SIVEP têm uma linha malformada. O parser Python
        # permite pular só essa linha sem perder o arquivo inteiro.
        return pd.read_csv(
            path, sep=";", encoding="latin-1", usecols=usecols,
            engine="python", on_bad_lines="skip",
        )


def _transform_raw(df: pd.DataFrame, source: Path, start_id: int) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["patient_id"] = [f"SIVEP-{n:07d}" for n in range(start_id, start_id + len(df))]
    out["ano_fonte"] = _year_from_filename(source)
    out["arquivo_fonte"] = source.name
    out["idade"] = _age_in_years(df.get("NU_IDADE_N"), df.get("TP_IDADE"))
    out["sexo"] = df.get("CS_SEXO", "não informado").fillna("não informado")

    rename = {
        "FEBRE": "febre", "TOSSE": "tosse", "GARGANTA": "garganta",
        "DISPNEIA": "dispneia", "DESC_RESP": "desc_resp", "SATURACAO": "saturacao",
        "DIARREIA": "diarreia", "VOMITO": "vomito", "DOR_ABD": "dor_abd",
        "FADIGA": "fadiga", "PERD_OLFT": "perd_olft", "PERD_PALA": "perd_pala",
        "FATOR_RISC": "fator_risc", "PUERPERA": "puerpera", "CARDIOPATI": "cardiopati",
        "HEPATICA": "hepatica", "ASMA": "asma", "DIABETES": "diabetes",
        "NEUROLOGIC": "neurologic", "PNEUMOPATI": "pneumopati",
        "IMUNODEPRE": "imunodepre", "RENAL": "renal", "OBESIDADE": "obesidade",
        "HOSPITAL": "hospital", "UTI": "uti", "VACINA_COV": "vacina_cov",
        "NOSOCOMIAL": "nosocomial",
    }
    for original, new_name in rename.items():
        out[new_name] = _map_code(df[original], YES_NO_MAP) if original in df else "não informado"

    out["suporte_ventilatorio"] = _map_code(df["SUPORT_VEN"], SUPPORT_MAP) if "SUPORT_VEN" in df else "não informado"
    out["raio_x_torax"] = _map_code(df["RAIOX_RES"], XRAY_MAP) if "RAIOX_RES" in df else "não informado"
    out["tomografia"] = _map_code(df["TOMO_RES"], CT_MAP) if "TOMO_RES" in df else "não informado"
    out["tipo_amostra"] = _map_code(df["AMOSTRA"], SAMPLE_MAP) if "AMOSTRA" in df else "não informado"
    out["resultado_pcr"] = _map_code(df["PCR_RESUL"], PCR_MAP) if "PCR_RESUL" in df else "não informado"
    out["classificacao_final"] = _map_code(df["CLASSI_FIN"], CLASSIFICATION_MAP) if "CLASSI_FIN" in df else "não informado"
    out["desfecho_registrado"] = _map_code(df["EVOLUCAO"], EVOLUTION_MAP) if "EVOLUCAO" in df else "não informado"
    return out


def _normalise_paths(paths: str | Path | Sequence[str | Path]) -> list[Path]:
    if isinstance(paths, (str, Path)):
        paths = [paths]
    found: list[Path] = []
    for item in paths:
        path = Path(item)
        if path.is_dir():
            found.extend(sorted(path.glob("INFLUD*.csv")))
            found.extend(sorted(path.glob("INFLUD*.parquet")))
        else:
            found.append(path)
    return found


def _add_derived_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Cria somente campos necessários ao fluxo, sem inventar medidas clínicas."""
    df = df.copy()
    yes = lambda col: df.get(col, pd.Series("não", index=df.index)).astype(str).str.lower().eq("sim")
    support = df.get("suporte_ventilatorio", pd.Series("", index=df.index)).astype(str).str.lower()

    severe = yes("uti") | support.isin(["invasivo", "não invasivo"])
    warning = yes("dispneia") | yes("desc_resp") | yes("saturacao") | yes("fator_risc")
    df["classificacao_risco"] = "verde"
    df.loc[warning, "classificacao_risco"] = "amarelo"
    df.loc[severe, "classificacao_risco"] = "vermelho"

    # O SIVEP não informa uma lista genérica de "exames pendentes". Para não
    # transformar "não realizado" em "pendente", só marcamos PCR quando o próprio
    # registro informa que o resultado está aguardando.
    df["exames_pendentes"] = ""
    waiting_pcr = df.get("resultado_pcr", pd.Series("", index=df.index)).astype(str).str.lower().eq("aguardando resultado")
    df.loc[waiting_pcr, "exames_pendentes"] = "PCR"
    return df


def prepare_sqlite_from_prepared_csv(csv_path: str | Path, db_path: str | Path) -> int:
    """Importa a amostra anonimizada já preparada para SQLite."""
    df = pd.read_csv(csv_path, low_memory=False)
    df = _add_derived_fields(df)
    return _write_sqlite(df, db_path, replace=True)


def prepare_sqlite_from_sivep(
    paths: str | Path | Sequence[str | Path],
    db_path: str | Path,
    max_rows_per_file: int | None = None,
) -> int:
    """Converte um ou vários arquivos anuais reais do SIVEP diretamente para SQLite."""
    files = _normalise_paths(paths)
    if not files:
        raise FileNotFoundError("Nenhum arquivo do SIVEP foi informado.")

    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    total = 0
    for source in files:
        if not source.exists():
            raise FileNotFoundError(f"Arquivo do SIVEP não encontrado: {source}")
        raw = _read_raw_sivep(source)
        if max_rows_per_file:
            raw = raw.head(max_rows_per_file).copy()
        prepared = _add_derived_fields(_transform_raw(raw, source, total + 1))
        total += _write_sqlite(prepared, db_path, replace=(total == 0))
    return total


def _write_sqlite(df: pd.DataFrame, db_path: str | Path, replace: bool = False) -> int:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        df.to_sql("patients", conn, if_exists="replace" if replace else "append", index=False)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_patients_id ON patients(patient_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_patients_year ON patients(ano_fonte)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_patients_risk ON patients(classificacao_risco)")
    return len(df)
