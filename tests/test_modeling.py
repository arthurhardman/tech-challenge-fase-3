"""
Testes do pipeline de modelagem.

Foco em:
- Definição correta de modelos e hiperparâmetros
- scale_pos_weight aplicado ao XGBoost para balanceamento de classes
- Treinamento sem grid search (rápido) retorna estimator válido
- Modelos produzem predições no shape correto
"""

import numpy as np
import pandas as pd
import pytest

from src.tabular.modeling import definir_modelos, treinar_modelo, treinar_todos_modelos


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dados_treino():
    """Dataset sintético pequeno para testar treinamento rapidamente."""
    np.random.seed(42)
    n = 300
    X = pd.DataFrame(np.random.randn(n, 10), columns=[f"f{i}" for i in range(10)])
    # Desbalanceado: ~70% cura (0), ~30% óbito (1)
    y = pd.Series(np.random.choice([0, 1], size=n, p=[0.7, 0.3]), name="OBITO")
    return X, y


# ---------------------------------------------------------------------------
# definir_modelos
# ---------------------------------------------------------------------------

def test_definir_modelos_retorna_quatro_modelos():
    modelos = definir_modelos()
    assert len(modelos) == 4
    assert set(modelos.keys()) == {
        "logistic_regression", "decision_tree", "random_forest", "xgboost"
    }


def test_definir_modelos_estrutura():
    """Cada entrada deve ter 'model' e 'params'."""
    for nome, config in definir_modelos().items():
        assert "model" in config, f"[{nome}] faltando chave 'model'"
        assert "params" in config, f"[{nome}] faltando chave 'params'"
        assert isinstance(config["params"], dict)


def test_xgboost_scale_pos_weight_aplicado():
    """XGBClassifier deve receber scale_pos_weight passado à função."""
    spw = 2.5
    modelos = definir_modelos(scale_pos_weight=spw)
    xgb_model = modelos["xgboost"]["model"]
    assert xgb_model.scale_pos_weight == spw, (
        f"Esperado scale_pos_weight={spw}, obtido {xgb_model.scale_pos_weight}"
    )


def test_xgboost_scale_pos_weight_padrao_um():
    """Sem argumento, scale_pos_weight deve ser 1.0 (sem balanceamento extra)."""
    modelos = definir_modelos()
    xgb_model = modelos["xgboost"]["model"]
    assert xgb_model.scale_pos_weight == 1.0


def test_modelos_sklearn_tem_class_weight_balanced():
    """LR, DT e RF devem ter class_weight='balanced'."""
    modelos = definir_modelos()
    for nome in ["logistic_regression", "decision_tree", "random_forest"]:
        modelo = modelos[nome]["model"]
        assert getattr(modelo, "class_weight", None) == "balanced", (
            f"[{nome}] class_weight deve ser 'balanced'"
        )


# ---------------------------------------------------------------------------
# treinar_modelo (sem grid search para rapidez)
# ---------------------------------------------------------------------------

def test_treinar_modelo_logistic_regression(dados_treino):
    X, y = dados_treino
    config = definir_modelos()["logistic_regression"]
    modelo = treinar_modelo("logistic_regression", config, X, y, usar_grid_search=False)
    assert hasattr(modelo, "predict")
    preds = modelo.predict(X)
    assert preds.shape == (len(X),)
    assert set(np.unique(preds)).issubset({0, 1})


def test_treinar_modelo_decision_tree(dados_treino):
    X, y = dados_treino
    config = definir_modelos()["decision_tree"]
    modelo = treinar_modelo("decision_tree", config, X, y, usar_grid_search=False)
    preds = modelo.predict(X)
    assert preds.shape == (len(X),)


# ---------------------------------------------------------------------------
# treinar_todos_modelos
# ---------------------------------------------------------------------------

def test_treinar_todos_modelos_retorna_quatro(dados_treino):
    """treinar_todos_modelos deve retornar dict com 4 modelos treinados."""
    X, y = dados_treino
    modelos = treinar_todos_modelos(X, y, usar_grid_search=False, salvar=False)
    assert len(modelos) == 4


def test_treinar_todos_modelos_predizem_corretamente(dados_treino):
    """Todos os modelos treinados devem predizer arrays de shape (n,)."""
    X, y = dados_treino
    modelos = treinar_todos_modelos(X, y, usar_grid_search=False, salvar=False)
    for nome, modelo in modelos.items():
        preds = modelo.predict(X)
        assert preds.shape == (len(X),), f"[{nome}] shape incorreto: {preds.shape}"
        assert set(np.unique(preds)).issubset({0, 1}), f"[{nome}] predições fora de {{0,1}}"


def test_scale_pos_weight_calculado_automaticamente(dados_treino):
    """treinar_todos_modelos deve calcular scale_pos_weight igual a n_neg/n_pos."""
    X, y = dados_treino
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    spw_esperado = n_neg / n_pos

    modelos = treinar_todos_modelos(X, y, usar_grid_search=False, salvar=False)
    xgb = modelos["xgboost"]
    # Após treinamento, scale_pos_weight deve corresponder ao calculado
    assert abs(xgb.scale_pos_weight - spw_esperado) < 1e-6, (
        f"scale_pos_weight esperado {spw_esperado:.4f}, obtido {xgb.scale_pos_weight:.4f}"
    )
