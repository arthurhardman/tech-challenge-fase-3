"""
tracker.py
----------
Monitoramento e logging dos experimentos de Algoritmo Genético.

Responsabilidades:
  - registrar, geração a geração, melhor/média/desvio do fitness e tempo decorrido;
  - persistir o histórico em CSV e JSON (em `experiments/<nome>/`);
  - gerar o gráfico de convergência (melhor e média por geração).

Uso típico com o `GeneticOptimizer` via callback `on_generation`:

    tracker = ExperimentTracker("exp_A")
    ga = GeneticOptimizer(evaluator, cfg, on_generation=tracker.log_generation)
    result = ga.run()
    tracker.save(result)
    tracker.plot_convergence()
"""

from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_EXPERIMENTS_DIR = _PROJECT_ROOT / "experiments"
_DEFAULT_FIGURES_DIR = _PROJECT_ROOT / "results" / "figures"


class ExperimentTracker:
    """Coleta e persiste métricas de um experimento de AG."""

    def __init__(
        self,
        name: str,
        experiments_dir: Optional[Path] = None,
        figures_dir: Optional[Path] = None,
    ):
        self.name = name
        self.experiments_dir = Path(experiments_dir or _DEFAULT_EXPERIMENTS_DIR) / name
        self.figures_dir = Path(figures_dir or _DEFAULT_FIGURES_DIR)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        self.records: List[dict] = []
        self._start = time.perf_counter()

        self.logger = logging.getLogger(f"ga.{name}")
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            handler = logging.FileHandler(self.experiments_dir / "run.log")
            handler.setFormatter(
                logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
            )
            self.logger.addHandler(handler)

    # ---- callback usado pelo GeneticOptimizer ---------------------------------
    def log_generation(self, record) -> None:
        """Registra uma geração. Aceita um `GenerationRecord` (dataclass)."""
        row = asdict(record) if hasattr(record, "__dataclass_fields__") else dict(record)
        row["elapsed_s"] = round(time.perf_counter() - self._start, 3)
        # best_individual vira string JSON para caber numa célula de CSV.
        row["best_individual"] = json.dumps(row.get("best_individual", {}))
        self.records.append(row)
        self.logger.info(
            "gen=%s best=%.4f mean=%.4f std=%.4f elapsed=%.1fs",
            row["generation"], row["best_fitness"], row["mean_fitness"],
            row["std_fitness"], row["elapsed_s"],
        )

    # ---- persistência ---------------------------------------------------------
    def save(self, result=None) -> dict:
        """Grava histórico (CSV) e resumo (JSON). Retorna o dict de resumo."""
        if self.records:
            csv_path = self.experiments_dir / "history.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(self.records[0].keys()))
                writer.writeheader()
                writer.writerows(self.records)

        summary = {
            "name": self.name,
            "generations": len(self.records),
            "total_elapsed_s": round(time.perf_counter() - self._start, 3),
        }
        if result is not None:
            summary["best_fitness"] = result.best_fitness
            summary["best_individual"] = result.best_individual
            if getattr(result, "config", None) is not None:
                summary["config"] = asdict(result.config)

        with open(self.experiments_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        return summary

    # ---- visualização ---------------------------------------------------------
    def plot_convergence(self, show: bool = False) -> Optional[Path]:
        """Gera o gráfico de convergência (melhor e média por geração)."""
        if not self.records:
            return None
        import matplotlib
        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        gens = [r["generation"] + 1 for r in self.records]
        best = [r["best_fitness"] for r in self.records]
        mean = [r["mean_fitness"] for r in self.records]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(gens, best, marker="o", label="Melhor fitness")
        ax.plot(gens, mean, marker="s", linestyle="--", label="Fitness médio")
        ax.set_xlabel("Geração")
        ax.set_ylabel("Fitness")
        ax.set_title(f"Convergência do AG — {self.name}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        out_path = self.figures_dir / f"ga_convergence_{self.name}.png"
        fig.savefig(out_path, dpi=120)
        if show:
            plt.show()
        plt.close(fig)
        return out_path
