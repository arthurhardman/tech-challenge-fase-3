"""
Pacote `monitoring`
-------------------
Tracking de experimentos do Algoritmo Genético: registra métricas por geração,
persiste em CSV/JSON e gera gráficos de convergência.
"""

from .tracker import ExperimentTracker

__all__ = ["ExperimentTracker"]
