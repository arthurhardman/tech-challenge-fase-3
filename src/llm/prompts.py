"""
prompts.py
----------
Templates de prompt engineering para interpretação de diagnósticos no contexto médico.

Diretrizes aplicadas:
  - papel claro (assistente clínico que EXPLICA, não decide);
  - entrada estruturada (predição, probabilidade, principais fatores/SHAP);
  - restrições de segurança (não inventar dados, citar fatores, apoio à decisão);
  - duas versões para avaliação comparativa: v1 (direta) e v2 (com few-shot).
"""

from __future__ import annotations

from typing import Dict, List, Sequence


SYSTEM_PROMPT = (
    "Você é um assistente clínico de apoio à decisão. Sua função é EXPLICAR, em "
    "linguagem clara e acessível, as predições de risco geradas por um modelo de "
    "machine learning para pacientes com Síndrome Respiratória Aguda Grave (SRAG). "
    "Regras: (1) baseie-se SOMENTE nos dados e fatores fornecidos, nunca invente "
    "informações; (2) cite explicitamente os fatores que mais influenciaram a predição; "
    "(3) deixe claro que é uma ferramenta de apoio e NÃO substitui o julgamento médico; "
    "(4) seja conciso e objetivo; (5) responda em português do Brasil."
)


def _format_factors(factors: Sequence[Dict]) -> str:
    """Formata a lista de fatores (features/SHAP) em bullets legíveis."""
    linhas: List[str] = []
    for f in factors:
        nome = f.get("feature", "?")
        valor = f.get("value", "")
        direcao = f.get("direction", "")
        seta = {"up": "↑ aumenta o risco", "down": "↓ reduz o risco"}.get(direcao, "")
        contrib = f.get("contribution")
        peso = f" (peso {contrib:+.3f})" if isinstance(contrib, (int, float)) else ""
        linhas.append(f"- {nome} = {valor} → {seta}{peso}".rstrip())
    return "\n".join(linhas) if linhas else "- (sem fatores destacados)"


def build_diagnosis_prompt(
    prediction_label: str,
    probability: float,
    factors: Sequence[Dict],
    version: str = "v2",
) -> str:
    """
    Monta o prompt de usuário para explicar uma predição.

    Parâmetros
    ----------
    prediction_label : str   — ex.: "Óbito" ou "Cura"
    probability : float      — probabilidade da classe predita [0,1]
    factors : sequência de dicts — top fatores: {feature, value, direction, contribution}
    version : "v1" | "v2"    — v2 inclui um exemplo (few-shot) para guiar o formato.
    """
    bloco_fatores = _format_factors(factors)
    pct = f"{probability * 100:.1f}%"

    base = (
        f"Predição do modelo: **{prediction_label}** "
        f"(probabilidade estimada: {pct}).\n\n"
        f"Principais fatores que influenciaram a predição:\n{bloco_fatores}\n\n"
        "Tarefa: escreva uma explicação clínica curta (3 a 5 frases) que: "
        "(a) resuma o que o modelo prevê e com qual confiança; "
        "(b) explique, em linguagem acessível, os principais fatores listados; "
        "(c) sugira pontos de atenção para a equipe; "
        "(d) reforce que é apoio à decisão, não diagnóstico definitivo."
    )

    if version == "v1":
        return base

    # v2: few-shot — um exemplo curto para fixar tom e formato.
    exemplo = (
        "EXEMPLO (formato esperado):\n"
        "Predição: Óbito (78%).\n"
        "Explicação: O modelo estima risco elevado de desfecho desfavorável, "
        "principalmente pela saturação baixa e pela idade avançada do paciente, "
        "fatores que mais pesaram na predição. A presença de comorbidade reforça o "
        "alerta. Recomenda-se monitorização intensiva e reavaliação frequente. "
        "Esta é uma estimativa de apoio e não substitui a avaliação clínica.\n\n"
    )
    return exemplo + base


# Conjunto de versões disponíveis (para avaliação comparativa documentada no relatório).
PROMPT_VERSIONS = ("v1", "v2")
