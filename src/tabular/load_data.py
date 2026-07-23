"""
load_data.py
------------
Funções para carregamento e inspeção inicial do dataset SIVEP-Gripe (INFLUD24).

O dataset é um CSV separado por ponto-e-vírgula (;) com encoding latin-1,
contendo registros de Síndrome Respiratória Aguda Grave (SRAG) hospitalizados.
"""

import pandas as pd
import numpy as np
from pathlib import Path


# Caminho padrão esperado para o dataset bruto
RAW_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "INFLUD24-26-06-2025.csv"


def carregar_dataset(caminho: str | Path = RAW_DATA_PATH, nrows: int = None) -> pd.DataFrame:
    """
    Carrega o dataset SIVEP-Gripe a partir de um arquivo CSV.

    Parâmetros
    ----------
    caminho : str ou Path
        Caminho para o arquivo CSV. Por padrão usa data/raw/INFLUD24-26-06-2025.csv.
    nrows : int, opcional
        Número de linhas a carregar (útil para testes rápidos). None carrega tudo.

    Retorna
    -------
    pd.DataFrame
        DataFrame com os dados brutos.
    """
    caminho = Path(caminho)
    if not caminho.exists():
        raise FileNotFoundError(
            f"Dataset não encontrado em: {caminho}\n"
            "Copie o arquivo INFLUD24-26-06-2025.csv para data/raw/"
        )

    df = pd.read_csv(
        caminho,
        sep=";",
        encoding="latin-1",
        low_memory=False,
        nrows=nrows,
    )
    print(f"Dataset carregado: {df.shape[0]:,} registros, {df.shape[1]} colunas")
    return df


def inspecionar_dataset(df: pd.DataFrame) -> None:
    """
    Exibe um resumo inicial do DataFrame: shape, tipos, valores nulos e amostra.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame a ser inspecionado.
    """
    print("=" * 60)
    print(f"Shape: {df.shape}")
    print(f"\nTipos de dados:\n{df.dtypes.value_counts()}")
    print(f"\nValores nulos por coluna (top 20):")
    nulos = df.isnull().sum().sort_values(ascending=False)
    pct_nulos = (nulos / len(df) * 100).round(2)
    resumo_nulos = pd.DataFrame({"Nulos": nulos, "% Nulos": pct_nulos})
    print(resumo_nulos.head(20).to_string())
    print(f"\nAmostra (primeiras 3 linhas):")
    print(df.head(3).to_string())
    print("=" * 60)


def resumo_target(df: pd.DataFrame, coluna_target: str = "EVOLUCAO") -> pd.DataFrame:
    """
    Exibe a distribuição da variável alvo EVOLUCAO.

    Valores esperados:
      1 = Cura
      2 = Óbito
      3 = Óbito por outras causas
      9 = Ignorado

    Parâmetros
    ----------
    df : pd.DataFrame
    coluna_target : str

    Retorna
    -------
    pd.DataFrame com contagem e percentual por categoria.
    """
    mapa_evolucao = {1: "Cura", 2: "Óbito", 3: "Óbito outras causas", 9: "Ignorado"}
    contagem = df[coluna_target].value_counts(dropna=False).rename("Contagem")
    pct = (df[coluna_target].value_counts(dropna=False, normalize=True) * 100).round(2).rename("% Total")
    resumo = pd.concat([contagem, pct], axis=1)
    resumo.index = resumo.index.map(lambda x: f"{x} — {mapa_evolucao.get(x, 'NaN/Outro')}")
    print(f"\nDistribuição de '{coluna_target}':")
    print(resumo.to_string())
    return resumo
