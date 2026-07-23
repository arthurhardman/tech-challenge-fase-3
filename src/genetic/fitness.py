"""
fitness.py
----------
Função fitness do Algoritmo Genético.

Avalia um indivíduo (conjunto de hiperparâmetros) treinando um XGBoost com
validação cruzada estratificada sobre (X_train, y_train) e retornando a média
da métrica escolhida — por padrão **F1**, com opção de **recall** (custo clínico
de falso-negativo de óbito é alto).

A avaliação de cada indivíduo é independente, o que permite paralelizar a
população (ver `ga_optimizer`).
"""

from __future__ import annotations

from typing import Dict, Iterable
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

from .chromosome import decode

Individual = Dict[str, float | int]

# Métricas suportadas → nome de scorer do scikit-learn.
_SCORERS = {
    "f1": "f1",
    "recall": "recall",
    "precision": "precision",
    "accuracy": "accuracy",
    "roc_auc": "roc_auc",
}


class FitnessEvaluator:
    """
    Avaliador de fitness reutilizável e determinístico.

    Parâmetros
    ----------
    X_train, y_train : dados de treino (a CV é feita internamente).
    metric : str
        Métrica primária ("f1", "recall", ...). Maior = melhor.
    cv_folds : int
        Número de folds do StratifiedKFold.
    scale_pos_weight : float | None
        Se informado, fixa o balanceamento (caso contrário, é um gene evoluído).
    complexity_penalty : float
        Peso de uma penalização opcional por complexidade do modelo
        (n_estimators × max_depth normalizado), favorecendo modelos mais enxutos.
        0.0 = desativada.
    random_state : int
        Semente para a CV e o modelo.
    """

    def __init__(
        self,
        X_train,
        y_train,
        metric: str = "f1",
        cv_folds: int = 3,
        scale_pos_weight: float | None = None,
        complexity_penalty: float = 0.0,
        random_state: int = 42,
    ):
        if metric not in _SCORERS:
            raise ValueError(f"Métrica '{metric}' não suportada. Use uma de {list(_SCORERS)}.")
        self.X_train = X_train
        self.y_train = y_train
        self.metric = metric
        self.scorer = _SCORERS[metric]
        self.cv_folds = cv_folds
        self.scale_pos_weight = scale_pos_weight
        self.complexity_penalty = complexity_penalty
        self.random_state = random_state

    def evaluate(self, individual: Individual) -> float:
        """Retorna o fitness (float) de um indivíduo. Falhas → fitness 0.0."""
        kwargs = decode(
            individual,
            scale_pos_weight=self.scale_pos_weight,
            random_state=self.random_state,
        )
        try:
            model = XGBClassifier(**kwargs)
            cv = StratifiedKFold(
                n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
            )
            scores = cross_val_score(
                model, self.X_train, self.y_train, scoring=self.scorer, cv=cv, n_jobs=1
            )
            fitness = float(np.mean(scores))
        except Exception:
            # Combinações inválidas de hiperparâmetros não devem derrubar a evolução.
            return 0.0

        if self.complexity_penalty > 0.0:
            fitness -= self.complexity_penalty * self._complexity(kwargs)
        return fitness

    @staticmethod
    def _complexity(kwargs: dict) -> float:
        """Complexidade normalizada ~[0,1] a partir de n_estimators e max_depth."""
        n_est = kwargs.get("n_estimators", 100) / 600.0
        depth = kwargs.get("max_depth", 6) / 12.0
        return float(n_est * depth)

    def evaluate_population(self, population: Iterable[Individual]) -> list[float]:
        """Avalia uma população em sequência (versão simples; o AG paraleliza)."""
        return [self.evaluate(ind) for ind in population]
