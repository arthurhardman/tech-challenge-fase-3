"""Guardrails de entrada e saída do assistente médico.

A regra é simples: o assistente pode localizar e explicar um protocolo, mas não
pode transformar a resposta em uma prescrição direta. A checagem acontece antes
e depois da geração, porque uma LLM ainda pode produzir uma instrução indevida
mesmo quando a pergunta original parecia segura.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List


_PADROES_PRESCRICAO_ENTRADA = [
    r"\bprescrev\w*",
    r"\breceit\w*",
    r"\bqual\s+(?:a\s+)?dose\b",
    r"\bdose\s+exata\b",
    r"\bposologia\b",
    r"\baument\w*\s+a\s+(?:dose|medica\w*)",
    r"\bque\s+medicamento\s+(?:dar|administrar|usar)\b",
    r"\bquantos?\s+(?:mg|g|mcg|µg|ml)\b",
]
_RE_PRESCRICAO_ENTRADA = re.compile("|".join(_PADROES_PRESCRICAO_ENTRADA), re.IGNORECASE)

# Na saída buscamos frases imperativas com quantidade. O padrão cobre variações
# comuns em português (tome/tomar/administre/usar/aplique) sem bloquear uma frase
# explicativa como "o protocolo cita dose" quando não há instrução ao paciente.
_RE_INSTRUCAO_DOSE = re.compile(
    r"\b(?:tome|tomar|administre|administrar|use|usar|aplique|aplicar|ingerir|ingira)\b"
    r"[^.!?\n]{0,80}?\b\d+(?:[.,]\d+)?\s*(?:mg|g|mcg|µg|ml|mL|comprimidos?|gotas?)\b",
    re.IGNORECASE,
)

AVISO_VALIDACAO = (
    "\n\n⚠️ Este conteúdo é apoio à consulta dos protocolos. "
    "Condutas, prescrições e doses precisam de avaliação e validação do médico responsável."
)

RESPOSTA_BLOQUEIO = (
    "Não posso definir prescrição, dose ou posologia diretamente. "
    "Posso localizar os critérios e orientações descritos nos protocolos, "
    "mas a decisão terapêutica precisa ser feita e validada pelo médico responsável."
)

RESPOSTA_SAIDA_INSEGURA = (
    "A resposta gerada continha uma instrução de dose/prescrição e foi bloqueada pelo guardrail. "
    "Consulte os protocolos citados e valide a conduta com o médico responsável."
)


@dataclass
class ResultadoGuardrail:
    permitido: bool
    motivo: str = ""
    categorias: List[str] = None

    def __post_init__(self):
        if self.categorias is None:
            self.categorias = []


def checar_entrada(pergunta: str) -> ResultadoGuardrail:
    if _RE_PRESCRICAO_ENTRADA.search(pergunta or ""):
        return ResultadoGuardrail(
            permitido=False,
            motivo="pedido de prescrição/dose direta",
            categorias=["prescricao_direta"],
        )
    return ResultadoGuardrail(permitido=True)


def detectar_instrucao_prescritiva(resposta: str) -> bool:
    return bool(_RE_INSTRUCAO_DOSE.search(resposta or ""))


def sanitizar_saida(resposta: str) -> str:
    """Bloqueia instruções de dose e sempre inclui o aviso de validação humana."""
    resposta = (resposta or "").strip()
    if detectar_instrucao_prescritiva(resposta):
        resposta = RESPOSTA_SAIDA_INSEGURA
    if "valida" in resposta.lower() and "médic" in resposta.lower():
        return resposta
    return resposta.rstrip() + AVISO_VALIDACAO
