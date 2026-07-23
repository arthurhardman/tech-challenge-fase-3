"""
medical_assistant.py
--------------------
Chain principal do assistente médico (requisito 2), construída com LangChain.

Fluxo da chain (RetrievalQA com contextualização por paciente):

    pergunta (+ paciente_id opcional)
        │
        ├─ guardrail de entrada ......... bloqueia pedido de prescrição direta
        ├─ retriever .................... trechos de protocolo + FONTES (RAG)
        ├─ patient_db ................... resumo clínico do paciente (contexto)
        ├─ PromptTemplate → LLM ......... resposta fundamentada
        └─ guardrail de saída ........... anexa aviso de validação humana

Retorna um objeto com resposta, fontes citadas e metadados (explainability).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from langchain_core.prompts import PromptTemplate

from ..knowledge.retriever import ProtocolRetriever, Trecho
from ..knowledge.patient_db import PatientDB
from ..safety import guardrails
from .llm_backend import CustomMedicalLLM

SYSTEM_PROMPT = (
    "Você é um assistente clínico do hospital, especializado em SRAG. Responda de "
    "forma objetiva e SOMENTE com base no CONTEXTO de protocolos fornecido. Cite a "
    "fonte (código do protocolo) ao final. Nunca prescreva medicação nem indique "
    "doses; conduta é ato médico. Responda em português do Brasil."
)

_TEMPLATE = PromptTemplate.from_template(
    "CONTEXTO (protocolos internos):\n{contexto}\n\n"
    "{bloco_paciente}"
    "PERGUNTA DO MÉDICO:\n{pergunta}\n\n"
    "Escreva uma resposta curta e fundamentada apenas no contexto acima. "
    "Ao final, liste as fontes usadas."
)


@dataclass
class RespostaAssistente:
    """Saída estruturada do assistente (base da explainability e da auditoria)."""
    resposta: str
    fontes: List[str] = field(default_factory=list)
    paciente_id: Optional[str] = None
    backend: str = ""
    bloqueado_guardrail: bool = False
    categorias_guardrail: List[str] = field(default_factory=list)


class MedicalAssistant:
    """Assistente médico: RAG + contexto do paciente + guardrails, via LangChain."""

    def __init__(
        self,
        retriever: Optional[ProtocolRetriever] = None,
        patient_db: Optional[PatientDB] = None,
        llm: Optional[CustomMedicalLLM] = None,
        top_k: int = 3,
    ):
        self.retriever = retriever or ProtocolRetriever()
        self.patient_db = patient_db or PatientDB()
        self.llm = llm or CustomMedicalLLM(system_prompt=SYSTEM_PROMPT)
        self.top_k = top_k

    def responder(self, pergunta: str, paciente_id: Optional[str] = None) -> RespostaAssistente:
        # 1) Guardrail de entrada
        g = guardrails.checar_entrada(pergunta)

        # 2) RAG — recupera trechos e fontes (sempre, para citar protocolo)
        trechos: List[Trecho] = self.retriever.buscar(pergunta, top_k=self.top_k)
        contexto = self.retriever.contexto_formatado(trechos)
        fontes = sorted({t.protocolo for t in trechos})

        # 3) Contexto do paciente (se informado)
        bloco_paciente = ""
        if paciente_id:
            resumo = self.patient_db.resumo_clinico(paciente_id)
            bloco_paciente = f"DADOS DO PACIENTE (base estruturada):\n{resumo}\n\n"

        # Caminho bloqueado: não gera prescrição; devolve resumo de protocolo.
        if not g.permitido:
            corpo = (
                guardrails.RESPOSTA_BLOQUEIO
                + "\n\nResumo dos protocolos pertinentes:\n"
                + contexto
            )
            resposta = guardrails.sanitizar_saida(corpo)
            if fontes:
                resposta += f"\n\nFontes: {', '.join(fontes)}."
            return RespostaAssistente(
                resposta=resposta,
                fontes=fontes,
                paciente_id=paciente_id,
                backend=self.llm.backend,
                bloqueado_guardrail=True,
                categorias_guardrail=g.categorias,
            )

        # 4) Monta o prompt e chama a LLM via LangChain
        prompt = _TEMPLATE.format(
            contexto=contexto, bloco_paciente=bloco_paciente, pergunta=pergunta
        )
        bruto = self.llm.invoke(prompt, system=SYSTEM_PROMPT)

        # 5) Guardrail de saída (aviso obrigatório) + fontes explícitas
        resposta = guardrails.sanitizar_saida(bruto)
        if fontes and not any(f in resposta for f in fontes):
            resposta += f"\n\nFontes: {', '.join(fontes)}."

        return RespostaAssistente(
            resposta=resposta,
            fontes=fontes,
            paciente_id=paciente_id,
            backend=self.llm.backend,
        )
