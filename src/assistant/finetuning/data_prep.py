"""
data_prep.py
------------
Pré-processamento, ANONIMIZAÇÃO e curadoria dos dados médicos usados no
fine-tuning (requisito 1 do enunciado).

Os dados de origem já são sintéticos, mas as funções de anonimização são reais e
aplicáveis a dados de produção: removem identificadores diretos (nomes, CPF,
telefone, e-mail, RG, cartão SUS, datas de nascimento) por regex antes de
qualquer treinamento. A curadoria remove exemplos vazios/curtos e duplicados.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, Iterable, List

# --- Padrões de PII (identificadores diretos) ------------------------------- #
_PII_PATTERNS = {
    "CPF": re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b"),
    "CNS": re.compile(r"\b\d{15}\b"),                         # cartão SUS
    "TELEFONE": re.compile(r"\b\(?\d{2}\)?\s?9?\d{4}-?\d{4}\b"),
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "DATA": re.compile(r"\b\d{2}/\d{2}/\d{4}\b"),
    "RG": re.compile(r"\bRG[:\s]*\d[\d.\-]{5,}\b", re.IGNORECASE),
    # Nomes só são removidos quando aparecem depois de um rótulo explícito.
    # Isso evita apagar nomes de doenças/localidades no meio de um protocolo.
    "NOME_ROTULADO": re.compile(
        r"\b(?:paciente|nome(?:\s+do\s+paciente)?|sr\.?|sra\.?|dr\.?|dra\.?)"
        r"\s*[:\-]?\s+[A-ZÀ-ÖØ-Ý][A-Za-zÀ-ÖØ-öø-ÿ]+"
        r"(?:\s+(?:da|de|do|das|dos|e)?\s*[A-ZÀ-ÖØ-Ý][A-Za-zÀ-ÖØ-öø-ÿ]+){1,5}",
        re.IGNORECASE,
    ),
}


def anonimizar(texto: str) -> str:
    """
    Substitui identificadores diretos por marcadores genéricos.

    Retorna o texto com PII mascarada (ex.: '[CPF]', '[EMAIL]'). É idempotente:
    aplicar duas vezes não altera o resultado.
    """
    for rotulo, padrao in _PII_PATTERNS.items():
        texto = padrao.sub(f"[{rotulo}]", texto)
    return texto


def normalizar_espacos(texto: str) -> str:
    """Colapsa espaços/quebras redundantes e faz trim."""
    texto = unicodedata.normalize("NFC", texto)
    texto = re.sub(r"[ \t]+", " ", texto)
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    return texto.strip()


def preprocessar(texto: str) -> str:
    """Pipeline de limpeza aplicado a qualquer campo textual: anonimizar + normalizar."""
    return normalizar_espacos(anonimizar(texto))


def curar_exemplos(
    exemplos: Iterable[Dict],
    campo_prompt: str = "prompt",
    campo_resposta: str = "response",
    min_chars_resposta: int = 20,
) -> List[Dict]:
    """
    Curadoria: remove exemplos com resposta muito curta/vazia e deduplica por
    (prompt, resposta). Aplica `preprocessar` em ambos os campos.

    Retorna a lista curada, preservando a ordem de entrada.
    """
    vistos = set()
    curados: List[Dict] = []
    for ex in exemplos:
        prompt = preprocessar(str(ex.get(campo_prompt, "")))
        resposta = preprocessar(str(ex.get(campo_resposta, "")))
        if len(resposta) < min_chars_resposta or not prompt:
            continue
        chave = (prompt, resposta)
        if chave in vistos:
            continue
        vistos.add(chave)
        novo = dict(ex)
        novo[campo_prompt] = prompt
        novo[campo_resposta] = resposta
        curados.append(novo)
    return curados
