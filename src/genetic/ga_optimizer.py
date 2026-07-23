"""
ga_optimizer.py
---------------
Loop evolutivo do Algoritmo Genético para otimização de hiperparâmetros do XGBoost.

Junta os componentes: codificação (`chromosome`), operadores (`operators`) e
fitness (`fitness`). Registra o histórico por geração (melhor/média/desvio) e
suporta paralelização da avaliação da população via `joblib`.

Exemplo
-------
    from src.genetic import GeneticOptimizer, GAConfig
    from src.genetic.fitness import FitnessEvaluator

    evaluator = FitnessEvaluator(X_train, y_train, metric="f1", cv_folds=3)
    ga = GeneticOptimizer(evaluator, GAConfig(population_size=20, generations=10))
    result = ga.run()
    print(result.best_individual, result.best_fitness)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional
import random

from joblib import Parallel, delayed

from .chromosome import random_individual
from .operators import (
    tournament_selection,
    uniform_crossover,
    mutate,
    elitism,
)

Individual = Dict[str, float | int]


@dataclass
class GAConfig:
    """Configuração de um experimento de AG."""
    population_size: int = 20
    generations: int = 10
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    mutation_sigma: float = 0.15
    tournament_k: int = 3
    elitism_size: int = 1
    n_jobs: int = 1                  # paralelismo da avaliação (joblib)
    early_stopping_patience: int = 0  # 0 = desativado; >0 = para após N gerações sem melhora
    random_state: int = 42
    name: str = "experimento"


@dataclass
class GenerationRecord:
    """Métricas de uma geração (para monitoramento/convergência)."""
    generation: int
    best_fitness: float
    mean_fitness: float
    std_fitness: float
    best_individual: Individual


@dataclass
class GAResult:
    """Resultado final da otimização."""
    best_individual: Individual
    best_fitness: float
    history: List[GenerationRecord] = field(default_factory=list)
    config: Optional[GAConfig] = None


class GeneticOptimizer:
    """
    Otimizador genético.

    Parâmetros
    ----------
    evaluator : objeto com método `evaluate(individual) -> float`
        Tipicamente um `FitnessEvaluator`.
    config : GAConfig
    on_generation : callable(GenerationRecord) | None
        Callback opcional chamado ao fim de cada geração (ex.: logging/tracker).
    """

    def __init__(
        self,
        evaluator,
        config: GAConfig | None = None,
        on_generation: Optional[Callable[[GenerationRecord], None]] = None,
    ):
        self.evaluator = evaluator
        self.config = config or GAConfig()
        self.on_generation = on_generation
        self.rng = random.Random(self.config.random_state)

    # ---- avaliação (com paralelização opcional) -------------------------------
    def _evaluate_population(self, population: List[Individual]) -> List[float]:
        if self.config.n_jobs == 1:
            return [self.evaluator.evaluate(ind) for ind in population]
        return Parallel(n_jobs=self.config.n_jobs)(
            delayed(self.evaluator.evaluate)(ind) for ind in population
        )

    # ---- construção da próxima geração ----------------------------------------
    def _next_generation(
        self, population: List[Individual], fitnesses: List[float]
    ) -> List[Individual]:
        cfg = self.config
        nova: List[Individual] = elitism(population, fitnesses, cfg.elitism_size)
        while len(nova) < cfg.population_size:
            p1 = tournament_selection(population, fitnesses, cfg.tournament_k, self.rng)
            p2 = tournament_selection(population, fitnesses, cfg.tournament_k, self.rng)
            f1, f2 = uniform_crossover(p1, p2, cfg.crossover_rate, self.rng)
            f1 = mutate(f1, cfg.mutation_rate, cfg.mutation_sigma, self.rng)
            f2 = mutate(f2, cfg.mutation_rate, cfg.mutation_sigma, self.rng)
            nova.append(f1)
            if len(nova) < cfg.population_size:
                nova.append(f2)
        return nova[: cfg.population_size]

    # ---- loop principal -------------------------------------------------------
    def run(self, verbose: bool = True) -> GAResult:
        cfg = self.config
        population = [random_individual(self.rng) for _ in range(cfg.population_size)]

        best_individual: Individual = {}
        best_fitness = float("-inf")
        history: List[GenerationRecord] = []
        sem_melhora = 0

        for gen in range(cfg.generations):
            fitnesses = self._evaluate_population(population)

            gen_best_idx = max(range(len(fitnesses)), key=lambda i: fitnesses[i])
            gen_best_fit = fitnesses[gen_best_idx]
            mean_fit = sum(fitnesses) / len(fitnesses)
            var = sum((f - mean_fit) ** 2 for f in fitnesses) / len(fitnesses)
            std_fit = var ** 0.5

            if gen_best_fit > best_fitness:
                best_fitness = gen_best_fit
                best_individual = dict(population[gen_best_idx])
                sem_melhora = 0
            else:
                sem_melhora += 1

            record = GenerationRecord(
                generation=gen,
                best_fitness=gen_best_fit,
                mean_fitness=mean_fit,
                std_fitness=std_fit,
                best_individual=dict(population[gen_best_idx]),
            )
            history.append(record)
            if self.on_generation:
                self.on_generation(record)
            if verbose:
                print(
                    f"[{cfg.name}] geração {gen + 1}/{cfg.generations} "
                    f"| melhor={gen_best_fit:.4f} | média={mean_fit:.4f} | desvio={std_fit:.4f}"
                )

            if (
                cfg.early_stopping_patience > 0
                and sem_melhora >= cfg.early_stopping_patience
            ):
                if verbose:
                    print(f"[{cfg.name}] early stopping na geração {gen + 1} "
                          f"(sem melhora há {sem_melhora} gerações).")
                break

            # Não evolui após a última geração avaliada.
            if gen < cfg.generations - 1:
                population = self._next_generation(population, fitnesses)

        return GAResult(
            best_individual=best_individual,
            best_fitness=best_fitness,
            history=history,
            config=cfg,
        )
