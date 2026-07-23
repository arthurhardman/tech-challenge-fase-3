"""
Testes do pipeline de pré-processamento.

Foco em:
- Sanidade dos splits (shapes, proporção do target)
- Ausência de data leakage (imputador fitado só no treino)
- Codificação categórica sem vazar informação de val/test
"""

import numpy as np
import pandas as pd
import pytest

from src.tabular.preprocessing import (
    filtrar_registros_validos,
    criar_target_binario,
    selecionar_features,
    dividir_dados,
    tratar_valores_ausentes,
    codificar_categoricas,
    normalizar_numericas,
    FEATURES,
    TARGET,
    TARGET_BINARIO,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def df_bruto():
    """Dataset sintético com a estrutura mínima necessária."""
    np.random.seed(42)
    n = 1000
    data = {col: np.random.choice([1, 2, 9], size=n) for col in FEATURES if col != "CS_SEXO"}
    data["CS_SEXO"] = np.random.choice(["M", "F", "I"], size=n)
    data["NU_IDADE_N"] = np.random.randint(0, 100, size=n)
    data[TARGET] = np.random.choice([1, 2, 3, 9], size=n, p=[0.5, 0.3, 0.1, 0.1])
    return pd.DataFrame(data)


@pytest.fixture
def df_processado(df_bruto):
    """Dataset após filtro, target binário e seleção de features."""
    df = filtrar_registros_validos(df_bruto)
    df = criar_target_binario(df)
    df = selecionar_features(df)
    return df


@pytest.fixture
def splits(df_processado):
    """Splits X/y estratificados."""
    return dividir_dados(df_processado)


# ---------------------------------------------------------------------------
# Filtro e target
# ---------------------------------------------------------------------------

def test_filtrar_remove_invalidos(df_bruto):
    df = filtrar_registros_validos(df_bruto)
    assert df[TARGET].isin([1, 2]).all()


def test_target_binario_valores(df_bruto):
    df = filtrar_registros_validos(df_bruto)
    df = criar_target_binario(df)
    assert set(df[TARGET_BINARIO].unique()).issubset({0, 1})
    assert (df[TARGET_BINARIO] == 1).sum() == (df[TARGET] == 2).sum()


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

def test_splits_tamanho(splits):
    X_train, X_val, X_test, y_train, y_val, y_test = splits
    total = len(X_train) + len(X_val) + len(X_test)
    assert abs(len(X_train) / total - 0.70) < 0.02
    assert abs(len(X_val) / total - 0.15) < 0.02
    assert abs(len(X_test) / total - 0.15) < 0.02


def test_splits_estratificados(splits):
    """Proporção do target deve ser similar nos três splits."""
    X_train, X_val, X_test, y_train, y_val, y_test = splits
    prop_train = y_train.mean()
    prop_val = y_val.mean()
    prop_test = y_test.mean()
    assert abs(prop_train - prop_val) < 0.05
    assert abs(prop_train - prop_test) < 0.05


def test_splits_sem_overlap(splits):
    """Índices não devem se repetir entre splits."""
    X_train, X_val, X_test, y_train, y_val, y_test = splits
    idx_train = set(X_train.index)
    idx_val = set(X_val.index)
    idx_test = set(X_test.index)
    assert idx_train.isdisjoint(idx_val)
    assert idx_train.isdisjoint(idx_test)
    assert idx_val.isdisjoint(idx_test)


# ---------------------------------------------------------------------------
# Ausência de leakage na imputação
# ---------------------------------------------------------------------------

def test_imputer_fitado_apenas_no_treino(splits):
    """
    A mediana calculada no treino deve ser diferente da mediana do dataset completo.
    Se fossem iguais (imputador fitado em tudo), o teste seria inconclusivo —
    mas o objetivo é garantir que o fit usa só X_train.
    """
    X_train, X_val, X_test, y_train, y_val, y_test = splits

    # Introduz NaN artificialmente
    X_train_nan = X_train.copy()
    X_val_nan = X_val.copy()
    X_test_nan = X_test.copy()

    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        X_train_nan.loc[X_train_nan.index[:10], num_cols[0]] = np.nan
        X_val_nan.loc[X_val_nan.index[:5], num_cols[0]] = np.nan
        X_test_nan.loc[X_test_nan.index[:5], num_cols[0]] = np.nan

    X_tr, X_v, X_te, imputers = tratar_valores_ausentes(X_train_nan, X_val_nan, X_test_nan)

    # Sem NaN após imputação
    assert X_tr.isnull().sum().sum() == 0
    assert X_v.isnull().sum().sum() == 0
    assert X_te.isnull().sum().sum() == 0

    # Imputador numérico deve existir e ter sido fitado apenas no treino
    if "numeric" in imputers and num_cols:
        median_from_imputer = imputers["numeric"].statistics_[
            list(X_train.select_dtypes(include=[np.number]).columns).index(num_cols[0])
        ]
        # A mediana do imputer deve corresponder à mediana do X_train (sem os NaN)
        expected = X_train_nan[num_cols[0]].median()
        assert abs(median_from_imputer - expected) < 1e-6


# ---------------------------------------------------------------------------
# Codificação categórica
# ---------------------------------------------------------------------------

def test_encoder_sem_leakage(splits):
    """LabelEncoder deve ser fitado só com categorias do treino."""
    X_train, X_val, X_test, y_train, y_val, y_test = splits

    # Introduz categoria nova em val/test que não existe no treino
    cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    if not cat_cols:
        pytest.skip("Sem colunas categóricas no dataset sintético")

    col = cat_cols[0]
    X_val_novo = X_val.copy()
    X_val_novo.loc[X_val_novo.index[0], col] = "CATEGORIA_DESCONHECIDA"

    X_tr, X_v, X_te, encoders = codificar_categoricas(X_train, X_val_novo, X_test)

    # Valor desconhecido deve ser mapeado para -1
    assert X_v[col].min() == -1


# ---------------------------------------------------------------------------
# Normalização
# ---------------------------------------------------------------------------

def test_scaler_fitado_no_treino(splits):
    """StandardScaler deve ter média ~0 no treino após encoding + normalização."""
    X_train, X_val, X_test, y_train, y_val, y_test = splits

    # Precisa codificar antes de normalizar (StandardScaler exige input numérico)
    X_tr, X_v, X_te, _ = tratar_valores_ausentes(X_train, X_val, X_test)
    X_tr, X_v, X_te, _ = codificar_categoricas(X_tr, X_v, X_te)
    X_tr, X_v, X_te, scaler = normalizar_numericas(X_tr, X_v, X_te)

    mean_train = X_tr.mean().abs().max()
    assert mean_train < 1e-6, f"Média do treino normalizado deve ser ~0, got {mean_train}"
