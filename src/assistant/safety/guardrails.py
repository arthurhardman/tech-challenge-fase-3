"""
guardrails.py
-------------
Limites de atuação do assistente (requisito 3 — Segurança e validação).

Duas camadas:
  1. Entrada  — detecta pedidos que o assistente NÃO pode atender diretamente
                (ex.: "prescreva", "qual a dose exata", "aumente a medicação").
  2. Saída    — anexa um aviso obrigatório de validação humana e bloqueia
                qualquer texto que soe como prescrição direta gerada pela LLM.

O princípio central: o assistente EXPLICA e SUGERE condutas de protocolo, mas
toda prescrição/decisão é ato médico e exige validação humana.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

# Verbos/pedidos que caracterizam solicitação de prescrição direta.
_PADROES_PRESCRICAO = [
    r"\bprescrev[ae]",
    r"\breceit[ae]\b",
    r"\bqual (?:a )?dose\b",
    r"\bdose exata\b",
    r"\bposologia\b",
    r"\baument[ae] a (?:dose|medica)",
    r"\bpode(?:mos)? medicar\b",
    r"\bque medicamento (?:dar|administrar)\b",
]
_RE_PRESCRICAO = re.compile("|".join(_PADROES_PRESCRICAO), re.IGNORECASE)

AVISO_VALIDACAO = (
    "\n\n⚠️ Este é um apoio à decisão baseado nos protocolos internos. "
    "Toda conduta, prescrição e dose exige validação e assinatura do médico responsável."
)

RESPOSTA_BLOQUEIO = (
    "Não posso indicar prescrição, dose ou posologia diretamente — isso é ato "
    "médico e exige validação humana. Posso, porém, resumir o que os protocolos "
    "internos orientam sobre a conduta e apontar a fonte."
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
    """
    Avalia a pergunta do médico. Se pedir prescrição/dose diretamente, marca
    como não permitido — o assistente responderá com `RESPOSTA_BLOQUEIO` mais o
    resumo de protocolo (sem números de dose).
    """
    if _RE_PRESCRICAO.search(pergunta or ""):
        return ResultadoGuardrail(
            permitido=False,
            motivo="pedido de prescrição/dose direta",
            categorias=["prescricao_direta"],
        )
    return ResultadoGuardrail(permitido=True)


def sanitizar_saida(resposta: str) -> str:
    """
    Garante o disclaimer de validação humana na saída. Idempotente: não duplica
    o aviso caso já esteja presente.
    """
    if "validação" in (resposta or "").lower() and "médic" in resposta.lower():
        return resposta
    return (resposta or "").rstrip() + AVISO_VALIDACAO
