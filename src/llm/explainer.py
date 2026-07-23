"""
explainer.py
------------
Orquestra a geração de explicações clínicas: recebe a predição de um paciente
(classe, probabilidade e principais fatores/SHAP) e produz uma explicação em
linguagem natural via `LLMClient`.

Projetado para encaixar nos artefatos da Fase 1: as contribuições por feature
podem vir diretamente dos valores SHAP já calculados no pipeline tabular.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .client import LLMClient, get_client
from .prompts import SYSTEM_PROMPT, build_diagnosis_prompt


# Rótulos do alvo binário da Fase 1 (0 = Cura, 1 = Óbito).
CLASS_LABELS = {0: "Cura", 1: "Óbito"}


@dataclass
class PatientPrediction:
    """
    Predição de um paciente, pronta para explicação.

    Atributos
    ---------
    predicted_class : int          — 0 (Cura) ou 1 (Óbito)
    probability : float            — probabilidade da classe predita [0,1]
    factors : lista de dicts        — top fatores no formato
        {"feature": str, "value": any, "direction": "up"|"down", "contribution": float}
    patient_id : str | None        — identificador opcional (apenas para logging)
    """
    predicted_class: int
    probability: float
    factors: List[Dict] = field(default_factory=list)
    patient_id: Optional[str] = None

    @property
    def label(self) -> str:
        return CLASS_LABELS.get(self.predicted_class, str(self.predicted_class))


class DiagnosisExplainer:
    """Gera explicações clínicas a partir de `PatientPrediction`."""

    def __init__(self, client: Optional[LLMClient] = None, prompt_version: str = "v2"):
        self.client = client or get_client()
        self.prompt_version = prompt_version

    @property
    def backend(self) -> str:
        return self.client.active_backend

    def explain(self, prediction: PatientPrediction) -> str:
        """Retorna a explicação textual para uma predição."""
        prompt = build_diagnosis_prompt(
            prediction_label=prediction.label,
            probability=prediction.probability,
            factors=prediction.factors,
            version=self.prompt_version,
        )
        return self.client.generate(prompt, system=SYSTEM_PROMPT)

    def explain_batch(self, predictions: Sequence[PatientPrediction]) -> List[Dict]:
        """Explica várias predições; retorna [{patient_id, label, explanation}]."""
        out: List[Dict] = []
        for p in predictions:
            out.append(
                {
                    "patient_id": p.patient_id,
                    "label": p.label,
                    "probability": p.probability,
                    "explanation": self.explain(p),
                }
            )
        return out


def prediction_from_shap(
    predicted_class: int,
    probability: float,
    feature_names: Sequence[str],
    feature_values: Sequence,
    shap_values: Sequence[float],
    top_k: int = 5,
    patient_id: Optional[str] = None,
) -> PatientPrediction:
    """
    Constrói um `PatientPrediction` a partir de vetores SHAP da Fase 1.

    Seleciona as `top_k` features por magnitude de contribuição SHAP e define a
    direção (up/down) pelo sinal do valor SHAP.
    """
    fatores = []
    for nome, valor, shap in zip(feature_names, feature_values, shap_values):
        fatores.append(
            {
                "feature": nome,
                "value": valor,
                "direction": "up" if shap >= 0 else "down",
                "contribution": float(shap),
            }
        )
    fatores.sort(key=lambda f: abs(f["contribution"]), reverse=True)
    return PatientPrediction(
        predicted_class=predicted_class,
        probability=probability,
        factors=fatores[:top_k],
        patient_id=patient_id,
    )
