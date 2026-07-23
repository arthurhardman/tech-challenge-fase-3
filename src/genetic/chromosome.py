"""
chromosome.py
-------------
Codificação dos genes (representação do indivíduo) para o Algoritmo Genético.

Cada **indivíduo** é um dicionário `{nome_do_gene: valor}`, onde cada gene corresponde
a um hiperparâmetro do XGBoost. O espaço de busca (`GENES`) define, para cada gene, o
tipo (int/float) e o domínio [low, high].

Funções principais:
  random_individual(rng)        — gera um indivíduo aleatório dentro dos domínios
  clip_individual(ind)          — garante que todo gene respeita tipo e limites
  decode(ind, scale_pos_weight) — converte o indivíduo em kwargs do XGBClassifier
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import random


@dataclass(frozen=True)
class GeneSpec:
    """Especificação de um gene: nome, tipo e domínio fechado [low, high]."""
    name: str
    kind: str          # "int" ou "float"
    low: float
    high: float

    def random_value(self, rng: random.Random) -> float | int:
        if self.kind == "int":
            return rng.randint(int(self.low), int(self.high))
        return rng.uniform(self.low, self.high)

    def clip(self, value: float) -> float | int:
        """Recorta o valor ao domínio e ao tipo do gene."""
        value = max(self.low, min(self.high, value))
        if self.kind == "int":
            return int(round(value))
        return float(value)


# Espaço de busca — hiperparâmetros do XGBoost (modelo tabular principal da Fase 1).
# scale_pos_weight tem teto generoso para acomodar o desbalanceamento Cura/Óbito;
# o ga_optimizer pode estreitar esse domínio com base no ratio real do dataset.
GENES: List[GeneSpec] = [
    GeneSpec("n_estimators",     "int",   50,   600),
    GeneSpec("max_depth",        "int",    2,    12),
    GeneSpec("learning_rate",    "float", 0.01,  0.3),
    GeneSpec("subsample",        "float", 0.5,   1.0),
    GeneSpec("colsample_bytree", "float", 0.5,   1.0),
    GeneSpec("min_child_weight", "int",    1,    10),
    GeneSpec("gamma",            "float", 0.0,   5.0),
    GeneSpec("scale_pos_weight", "float", 1.0,  12.0),
]

# Índice por nome para acesso rápido nos operadores.
GENES_BY_NAME: Dict[str, GeneSpec] = {g.name: g for g in GENES}


def random_individual(rng: random.Random | None = None) -> Dict[str, float | int]:
    """Gera um indivíduo aleatório respeitando os domínios de cada gene."""
    rng = rng or random.Random()
    return {g.name: g.random_value(rng) for g in GENES}


def clip_individual(individual: Dict[str, float | int]) -> Dict[str, float | int]:
    """Recorta cada gene ao seu domínio/tipo. Genes desconhecidos são preservados."""
    out: Dict[str, float | int] = {}
    for name, value in individual.items():
        spec = GENES_BY_NAME.get(name)
        out[name] = spec.clip(value) if spec else value
    return out


def decode(
    individual: Dict[str, float | int],
    scale_pos_weight: float | None = None,
    random_state: int = 42,
) -> dict:
    """
    Converte um indivíduo em kwargs prontos para o `XGBClassifier`.

    Parâmetros
    ----------
    individual : dict
        Indivíduo (genes → valores).
    scale_pos_weight : float | None
        Se informado, sobrescreve o gene `scale_pos_weight` por um valor fixo
        (útil quando se quer balanceamento determinístico em vez de evoluído).
    random_state : int
        Semente para reprodutibilidade do modelo.
    """
    ind = clip_individual(individual)
    kwargs = dict(ind)
    if scale_pos_weight is not None:
        kwargs["scale_pos_weight"] = float(scale_pos_weight)
    kwargs.update(
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=1,  # paralelismo fica a cargo do AG (avaliação da população)
    )
    return kwargs
