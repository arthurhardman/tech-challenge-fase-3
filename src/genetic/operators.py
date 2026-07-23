"""
operators.py
------------
Operadores genéticos: seleção, cruzamento (crossover) e mutação.

Todos operam sobre indivíduos no formato `{nome_do_gene: valor}` e devolvem
indivíduos válidos (recortados aos domínios via `clip_individual`).
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple
import random

from .chromosome import GENES, GENES_BY_NAME, clip_individual

Individual = Dict[str, float | int]


def tournament_selection(
    population: Sequence[Individual],
    fitnesses: Sequence[float],
    k: int = 3,
    rng: random.Random | None = None,
) -> Individual:
    """
    Seleção por torneio: sorteia `k` indivíduos e retorna (cópia do) melhor.

    Maior fitness = melhor. Robusto a `k` maior que a população (limita ao tamanho).
    """
    rng = rng or random.Random()
    if not population:
        raise ValueError("População vazia na seleção por torneio.")
    k = max(1, min(k, len(population)))
    aspirantes = rng.sample(range(len(population)), k)
    vencedor = max(aspirantes, key=lambda i: fitnesses[i])
    return dict(population[vencedor])


def uniform_crossover(
    parent1: Individual,
    parent2: Individual,
    crossover_rate: float = 0.8,
    rng: random.Random | None = None,
) -> Tuple[Individual, Individual]:
    """
    Crossover uniforme: com probabilidade `crossover_rate`, cada gene é trocado
    independentemente entre os pais (p=0.5). Caso contrário, os filhos são cópias
    dos pais (sem recombinação).
    """
    rng = rng or random.Random()
    child1, child2 = dict(parent1), dict(parent2)
    if rng.random() < crossover_rate:
        for g in GENES:
            if rng.random() < 0.5:
                child1[g.name], child2[g.name] = child2[g.name], child1[g.name]
    return clip_individual(child1), clip_individual(child2)


def mutate(
    individual: Individual,
    mutation_rate: float = 0.1,
    sigma: float = 0.15,
    rng: random.Random | None = None,
) -> Individual:
    """
    Mutação gene a gene. Cada gene muta com probabilidade `mutation_rate`:

      - genes float: perturbação gaussiana proporcional à amplitude do domínio
        (`sigma` é a fração do range usada como desvio-padrão);
      - genes int: perturbação inteira de ±1..±2, também escalada por `sigma`.

    O resultado é sempre recortado ao domínio/tipo do gene.
    """
    rng = rng or random.Random()
    mutant = dict(individual)
    for g in GENES:
        if rng.random() >= mutation_rate:
            continue
        amplitude = g.high - g.low
        if g.kind == "int":
            passo = max(1, int(round(sigma * amplitude)))
            delta = rng.randint(-passo, passo)
            mutant[g.name] = mutant[g.name] + delta
        else:
            mutant[g.name] = mutant[g.name] + rng.gauss(0.0, sigma * amplitude)
    return clip_individual(mutant)


def elitism(
    population: Sequence[Individual],
    fitnesses: Sequence[float],
    n_elite: int,
) -> List[Individual]:
    """Retorna cópias dos `n_elite` melhores indivíduos (maior fitness primeiro)."""
    if n_elite <= 0:
        return []
    ordem = sorted(range(len(population)), key=lambda i: fitnesses[i], reverse=True)
    return [dict(population[i]) for i in ordem[:n_elite]]
