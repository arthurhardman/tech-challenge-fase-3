"""
modeling.py
-----------
Definição, treinamento e ajuste de hiperparâmetros dos modelos de classificação.

Modelos implementados:
  1. Regressão Logística  — baseline linear
  2. Árvore de Decisão    — interpretável, sem necessidade de normalização
  3. Random Forest        — ensemble de árvores, robusto a outliers
  4. XGBoost              — gradient boosting, geralmente melhor performance

Todos os modelos são treinados com class_weight='balanced' ou equivalente
para lidar com possível desbalanceamento entre Cura e Óbito.
"""

import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score
from xgboost import XGBClassifier


MODELS_DIR = Path(__file__).resolve().parents[1] / "results" / "models"

# Semente global para reprodutibilidade
RANDOM_STATE = 42

# GridSearchCV paralelo + RF/XGB com n_jobs=-1 = paralelismo aninhado que estoura RAM
# e derruba workers (joblib/loky) em máquinas com pouca memória. Use 1 por padrão;
# export SRAG_GRID_N_JOBS=4 para acelerar em máquinas com RAM suficiente (e reduza n_jobs do RF).
GRID_N_JOBS = int(os.environ.get("SRAG_GRID_N_JOBS", "1"))


def definir_modelos(scale_pos_weight: float = 1.0) -> dict:
    """
    Retorna um dicionário com os modelos e suas grades de hiperparâmetros para busca.

    Parâmetros
    ----------
    scale_pos_weight : float
        Razão entre negativos e positivos no treino (n_cura / n_obito).
        Passada ao XGBClassifier para compensar o desbalanceamento de classes,
        equivalente ao class_weight='balanced' dos demais modelos.
        Valor padrão 1.0 (sem balanceamento); calcule com:
            scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    Estrutura de retorno
    --------------------
      { nome: {"model": estimator, "params": param_grid} }
    """
    modelos = {
        "logistic_regression": {
            "model": LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=RANDOM_STATE,
            ),
            "params": {
                "C": [0.01, 0.1, 1.0, 10.0],
                "solver": ["lbfgs", "liblinear"],
            },
        },
        "decision_tree": {
            "model": DecisionTreeClassifier(
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            "params": {
                "max_depth": [5, 10, 20, None],
                "min_samples_split": [2, 10, 50],
                "criterion": ["gini", "entropy"],
            },
        },
        "random_forest": {
            "model": RandomForestClassifier(
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "params": {
                "n_estimators": [100, 300],
                "max_depth": [10, 20, None],
                "min_samples_split": [2, 10],
            },
        },
        "xgboost": {
            # XGBClassifier não aceita class_weight; usa scale_pos_weight
            # como mecanismo equivalente de balanceamento de classes.
            "model": XGBClassifier(
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                scale_pos_weight=scale_pos_weight,
            ),
            "params": {
                "n_estimators": [100, 300],
                "max_depth": [4, 6, 8],
                "learning_rate": [0.05, 0.1, 0.2],
                "subsample": [0.8, 1.0],
            },
        },
    }
    return modelos


def treinar_modelo(
    nome: str,
    modelo_config: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    usar_grid_search: bool = True,
    cv_folds: int = 5,
) -> object:
    """
    Treina um único modelo, com ou sem GridSearchCV.

    Parâmetros
    ----------
    nome : str — identificador do modelo
    modelo_config : dict — {"model": estimator, "params": param_grid}
    X_train, y_train — dados de treino
    usar_grid_search : bool — se True, realiza busca de hiperparâmetros
    cv_folds : int — número de folds para cross-validation

    Retorna
    -------
    Estimator treinado (melhor estimator se GridSearchCV foi usado)
    """
    modelo = modelo_config["model"]
    params = modelo_config["params"]

    print(f"\n[{nome}] Iniciando treinamento...")

    if usar_grid_search and params and GRID_N_JOBS > 1 and hasattr(modelo, "n_jobs"):
        modelo = clone(modelo)
        modelo.set_params(n_jobs=1)

    if usar_grid_search and params:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
        grid = GridSearchCV(
            estimator=modelo,
            param_grid=params,
            scoring="f1",
            cv=cv,
            n_jobs=GRID_N_JOBS,
            verbose=1,
        )
        grid.fit(X_train, y_train)
        melhor = grid.best_estimator_
        print(f"[{nome}] Melhores parâmetros: {grid.best_params_}")
        print(f"[{nome}] F1 no CV (treino): {grid.best_score_:.4f}")
        return melhor
    else:
        modelo.fit(X_train, y_train)
        y_pred_train = modelo.predict(X_train)
        f1 = f1_score(y_train, y_pred_train)
        print(f"[{nome}] F1 no treino: {f1:.4f}")
        return modelo


def treinar_todos_modelos(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    usar_grid_search: bool = True,
    salvar: bool = True,
) -> dict:
    """
    Treina todos os modelos definidos em definir_modelos() e opcionalmente salva os artefatos.

    Calcula automaticamente scale_pos_weight para o XGBoost a partir do y_train,
    garantindo tratamento equivalente ao class_weight='balanced' dos demais modelos.

    Parâmetros
    ----------
    X_train, y_train — dados de treino
    usar_grid_search : bool — ativa busca de hiperparâmetros
    salvar : bool — salva modelos treinados em results/models/

    Retorna
    -------
    dict { nome_modelo: estimator_treinado }
    """
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    spw = n_neg / n_pos if n_pos > 0 else 1.0
    print(f"Desbalanceamento — Cura: {n_neg:,} | Óbito: {n_pos:,} | scale_pos_weight XGB: {spw:.2f}")

    modelos_config = definir_modelos(scale_pos_weight=spw)
    modelos_treinados = {}

    for nome, config in modelos_config.items():
        modelo = treinar_modelo(nome, config, X_train, y_train, usar_grid_search)
        modelos_treinados[nome] = modelo

    if salvar:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        for nome, modelo in modelos_treinados.items():
            joblib.dump(modelo, MODELS_DIR / f"{nome}.pkl")
            print(f"Modelo salvo: results/models/{nome}.pkl")

    return modelos_treinados


def carregar_modelos() -> dict:
    """
    Carrega todos os modelos serializados de results/models/.

    Retorna
    -------
    dict { nome_modelo: estimator }
    """
    modelos = {}
    for arquivo in MODELS_DIR.glob("*.pkl"):
        nome = arquivo.stem
        modelos[nome] = joblib.load(arquivo)
        print(f"Modelo carregado: {nome}")
    return modelos
