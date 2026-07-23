"""
Testes dos operadores e da mecânica do Algoritmo Genético.

Estes testes NÃO treinam XGBoost (rápidos, sem dependência de dados): usam um
avaliador de fitness fake e validam codificação, operadores e o loop evolutivo.
"""

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.genetic.chromosome import (
    GENES,
    GENES_BY_NAME,
    random_individual,
    clip_individual,
    decode,
)
from src.genetic.operators import (
    tournament_selection,
    uniform_crossover,
    mutate,
    elitism,
)
from src.genetic.ga_optimizer import GeneticOptimizer, GAConfig


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _within_bounds(individual) -> bool:
    for g in GENES:
        v = individual[g.name]
        if not (g.low <= v <= g.high):
            return False
        if g.kind == "int" and not isinstance(v, int):
            return False
    return True


class FakeEvaluator:
    """Fitness determinístico e barato: favorece learning_rate alto."""
    def evaluate(self, individual):
        return float(individual["learning_rate"])


# --------------------------------------------------------------------------- #
# Codificação
# --------------------------------------------------------------------------- #
def test_random_individual_respeita_dominios():
    rng = random.Random(0)
    for _ in range(50):
        assert _within_bounds(random_individual(rng))


def test_clip_recorta_valores_fora_do_dominio():
    bad = {g.name: g.high * 10 for g in GENES}
    bad["learning_rate"] = -5.0
    clipped = clip_individual(bad)
    assert _within_bounds(clipped)
    assert clipped["learning_rate"] == GENES_BY_NAME["learning_rate"].low


def test_decode_gera_kwargs_validos_para_xgboost():
    ind = random_individual(random.Random(1))
    kwargs = decode(ind, scale_pos_weight=3.0, random_state=7)
    assert kwargs["scale_pos_weight"] == 3.0
    assert kwargs["random_state"] == 7
    assert kwargs["n_jobs"] == 1
    assert "eval_metric" in kwargs
    assert isinstance(kwargs["n_estimators"], int)


# --------------------------------------------------------------------------- #
# Operadores
# --------------------------------------------------------------------------- #
def test_tournament_seleciona_dentro_da_populacao():
    rng = random.Random(2)
    pop = [random_individual(rng) for _ in range(10)]
    fits = [i for i in range(10)]  # melhor é o índice 9
    # Com k = tamanho da população, sempre sai o melhor.
    vencedor = tournament_selection(pop, fits, k=10, rng=rng)
    assert vencedor == pop[9]


def test_crossover_produz_filhos_validos():
    rng = random.Random(3)
    p1 = random_individual(rng)
    p2 = random_individual(rng)
    c1, c2 = uniform_crossover(p1, p2, crossover_rate=1.0, rng=rng)
    assert _within_bounds(c1) and _within_bounds(c2)
    assert set(c1.keys()) == set(p1.keys())


def test_crossover_rate_zero_preserva_pais():
    rng = random.Random(4)
    p1 = random_individual(rng)
    p2 = random_individual(rng)
    c1, c2 = uniform_crossover(p1, p2, crossover_rate=0.0, rng=rng)
    assert c1 == clip_individual(p1)
    assert c2 == clip_individual(p2)


def test_mutacao_mantem_validade_e_pode_alterar_genes():
    rng = random.Random(5)
    ind = random_individual(rng)
    mutante = mutate(ind, mutation_rate=1.0, sigma=0.3, rng=rng)
    assert _within_bounds(mutante)
    # Com taxa 1.0 e sigma alto, ao menos um gene deve mudar.
    assert any(mutante[g.name] != ind[g.name] for g in GENES)


def test_mutacao_taxa_zero_nao_altera():
    rng = random.Random(6)
    ind = random_individual(rng)
    assert mutate(ind, mutation_rate=0.0, rng=rng) == clip_individual(ind)


def test_elitismo_preserva_os_melhores():
    rng = random.Random(7)
    pop = [random_individual(rng) for _ in range(5)]
    fits = [0.1, 0.9, 0.5, 0.95, 0.2]
    elite = elitism(pop, fits, n_elite=2)
    assert elite[0] == pop[3]  # fitness 0.95
    assert elite[1] == pop[1]  # fitness 0.90


# --------------------------------------------------------------------------- #
# Loop evolutivo (com avaliador fake)
# --------------------------------------------------------------------------- #
def test_ga_run_melhora_ou_mantem_fitness():
    cfg = GAConfig(population_size=12, generations=8, random_state=42, name="teste")
    ga = GeneticOptimizer(FakeEvaluator(), cfg)
    result = ga.run(verbose=False)

    assert len(result.history) == cfg.generations
    assert _within_bounds(result.best_individual)
    # Fitness final >= fitness da primeira geração (elitismo garante não-regressão).
    assert result.best_fitness >= result.history[0].best_fitness
    # Como o fitness fake = learning_rate, o ótimo tende ao teto do domínio.
    assert result.best_fitness > 0.2


def test_ga_early_stopping():
    cfg = GAConfig(
        population_size=8, generations=50, early_stopping_patience=3,
        random_state=1, name="early",
    )
    ga = GeneticOptimizer(FakeEvaluator(), cfg)
    result = ga.run(verbose=False)
    # Deve parar antes das 50 gerações.
    assert len(result.history) < 50


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
