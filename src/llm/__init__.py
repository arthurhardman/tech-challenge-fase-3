"""
Pacote `llm`
------------
Integração com LLM para gerar explicações em linguagem natural dos diagnósticos
produzidos pelo modelo tabular (XGBoost) da Fase 1.

Backends:
  - `ollama` : LLM local via Ollama (principal, demonstrado no vídeo);
  - `mock`   : respostas por template/cache, sem LLM (garante que notebooks e
               testes rodem em qualquer máquina; é o fallback automático).

Componentes:
  client    — seleção e chamada do backend (com fallback ollama → mock)
  prompts   — templates de prompt engineering para o contexto médico
  explainer — orquestra: predição + features/SHAP → explicação em PT-BR
"""

from .client import LLMClient, get_client
from .explainer import DiagnosisExplainer, PatientPrediction

__all__ = [
    "LLMClient",
    "get_client",
    "DiagnosisExplainer",
    "PatientPrediction",
]
