"""
Pacote `genetic`
----------------
Algoritmo Genético (implementação própria) para otimização dos hiperparâmetros
do modelo tabular principal da Fase 1 (XGBoost).

Módulos:
  chromosome   — codificação dos genes (hiperparâmetros) e decode para kwargs do XGBoost
  operators    — seleção por torneio, crossover uniforme e mutação
  fitness      — função fitness (F1/recall via cross-validation)
  ga_optimizer — loop evolutivo com elitismo e histórico por geração
"""

from .chromosome import GENES, random_individual, clip_individual, decode
from .operators import tournament_selection, uniform_crossover, mutate
from .fitness import FitnessEvaluator
from .ga_optimizer import GeneticOptimizer, GAConfig

__all__ = [
    "GENES",
    "random_individual",
    "clip_individual",
    "decode",
    "tournament_selection",
    "uniform_crossover",
    "mutate",
    "FitnessEvaluator",
    "GeneticOptimizer",
    "GAConfig",
]
