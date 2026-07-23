"""
Testes da camada de LLM.

Rodam inteiramente no backend `mock` — sem exigir Ollama nem rede — garantindo
que notebooks/CI funcionem em qualquer máquina. Também validam o contrato do
`explainer` e a construção de fatores a partir de SHAP.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llm.client import LLMClient, get_client
from src.llm.prompts import build_diagnosis_prompt, SYSTEM_PROMPT, PROMPT_VERSIONS
from src.llm.explainer import (
    DiagnosisExplainer,
    PatientPrediction,
    prediction_from_shap,
    CLASS_LABELS,
)


# --------------------------------------------------------------------------- #
# Client / backend
# --------------------------------------------------------------------------- #
def test_mock_backend_sempre_disponivel():
    client = LLMClient(backend="mock")
    assert client.active_backend == "mock"


def test_auto_faz_fallback_para_mock_sem_ollama():
    # Aponta para uma porta sem servidor → deve cair em mock.
    client = LLMClient(backend="auto", ollama_url="http://localhost:1")
    assert client.active_backend == "mock"


def test_get_client_respeita_env(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "mock")
    assert get_client().active_backend == "mock"


def test_mock_generate_extrai_predicao_e_fatores():
    prompt = build_diagnosis_prompt(
        prediction_label="Óbito",
        probability=0.78,
        factors=[
            {"feature": "SATURACAO", "value": 88, "direction": "up", "contribution": 0.31},
            {"feature": "IDADE", "value": 72, "direction": "up", "contribution": 0.22},
        ],
        version="v2",
    )
    resp = LLMClient(backend="mock").generate(prompt, system=SYSTEM_PROMPT)
    assert "Óbito" in resp
    assert "78" in resp
    assert "SATURACAO" in resp
    assert "apoio" in resp.lower()  # disclaimer presente


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #
def test_prompt_versions_diferem():
    args = dict(prediction_label="Cura", probability=0.9, factors=[])
    v1 = build_diagnosis_prompt(**args, version="v1")
    v2 = build_diagnosis_prompt(**args, version="v2")
    assert v1 != v2
    assert "EXEMPLO" in v2 and "EXEMPLO" not in v1
    assert set(PROMPT_VERSIONS) == {"v1", "v2"}


# --------------------------------------------------------------------------- #
# Explainer
# --------------------------------------------------------------------------- #
def test_explainer_contrato_de_saida():
    explainer = DiagnosisExplainer(client=LLMClient(backend="mock"))
    pred = PatientPrediction(
        predicted_class=1,
        probability=0.81,
        factors=[{"feature": "IDADE", "value": 70, "direction": "up", "contribution": 0.4}],
        patient_id="P1",
    )
    texto = explainer.explain(pred)
    assert isinstance(texto, str) and len(texto) > 0
    assert pred.label == "Óbito"


def test_explain_batch_estrutura():
    explainer = DiagnosisExplainer(client=LLMClient(backend="mock"))
    preds = [
        PatientPrediction(0, 0.9, [], "A"),
        PatientPrediction(1, 0.6, [], "B"),
    ]
    out = explainer.explain_batch(preds)
    assert len(out) == 2
    assert {"patient_id", "label", "probability", "explanation"} <= set(out[0].keys())
    assert out[0]["label"] == CLASS_LABELS[0]


def test_prediction_from_shap_ordena_por_magnitude():
    pred = prediction_from_shap(
        predicted_class=1,
        probability=0.7,
        feature_names=["A", "B", "C"],
        feature_values=[1, 2, 3],
        shap_values=[0.05, -0.40, 0.10],
        top_k=2,
    )
    # Maior magnitude primeiro: B (0.40) depois C (0.10).
    assert [f["feature"] for f in pred.factors] == ["B", "C"]
    assert pred.factors[0]["direction"] == "down"  # shap negativo


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
