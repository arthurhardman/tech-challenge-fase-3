"""Assistente médico: RAG + paciente estruturado + guardrails.

Com LangChain instalado, o backend continua usando a interface ``LLM`` do
framework. O restante do fluxo fica em Python simples para facilitar leitura e
testes. As fontes retornam arquivo/página quando o contexto vem dos protocolos
oficiais.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from ..knowledge.retriever import ProtocolRetriever, Trecho
from ..knowledge.patient_db import PatientDB
from ..safety import guardrails
from .llm_backend import CustomMedicalLLM

SYSTEM_PROMPT = (
    "Você é um assistente clínico especializado em SRAG. Responda somente com base "
    "no contexto recuperado e nos dados estruturados apresentados. Não invente dados "
    "do paciente. Nunca prescreva medicação nem indique dose; a decisão final é do "
    "médico responsável. Responda em português do Brasil e cite as fontes usadas."
)

_TEMPLATE = (
    "CONTEXTO CLÍNICO RECUPERADO:\n{contexto}\n\n"
    "{bloco_paciente}"
    "PERGUNTA DO MÉDICO:\n{pergunta}\n\n"
    "Responda de forma curta, baseada somente no contexto acima. "
    "Se o contexto não for suficiente, diga isso claramente."
)


@dataclass
class RespostaAssistente:
    resposta: str
    fontes: List[str] = field(default_factory=list)
    paciente_id: Optional[str] = None
    backend: str = ""
    bloqueado_guardrail: bool = False
    categorias_guardrail: List[str] = field(default_factory=list)


class MedicalAssistant:
    def __init__(
        self,
        retriever: Optional[ProtocolRetriever] = None,
        patient_db: Optional[PatientDB] = None,
        llm: Optional[CustomMedicalLLM] = None,
        top_k: int = 5,
    ):
        self.retriever = retriever or ProtocolRetriever()
        self.patient_db = patient_db or PatientDB()
        self.llm = llm or CustomMedicalLLM(system_prompt=SYSTEM_PROMPT)
        self.top_k = top_k

    def responder(self, pergunta: str, paciente_id: Optional[str] = None) -> RespostaAssistente:
        # 1) Primeiro barramos pedidos que o assistente não deve atender diretamente.
        g = guardrails.checar_entrada(pergunta)

        # 2) O RAG roda mesmo no caminho bloqueado, pois ainda podemos mostrar onde
        # estão os critérios/protocolos sem devolver uma prescrição.
        trechos: List[Trecho] = self.retriever.buscar(pergunta, top_k=self.top_k)
        contexto = self.retriever.contexto_formatado(trechos)
        fontes = []
        for trecho in trechos:
            cit = trecho.citacao()
            if cit not in fontes:
                fontes.append(cit)

        # 3) Quando há patient_id, só fatos vindos da base entram no prompt.
        bloco_paciente = ""
        if paciente_id:
            resumo = self.patient_db.resumo_clinico(paciente_id)
            bloco_paciente = f"DADOS DO PACIENTE (base estruturada):\n{resumo}\n\n"

        if not g.permitido:
            resposta = guardrails.sanitizar_saida(guardrails.RESPOSTA_BLOQUEIO)
            if fontes:
                resposta += "\n\nFontes consultadas: " + "; ".join(fontes) + "."
            return RespostaAssistente(
                resposta=resposta,
                fontes=fontes,
                paciente_id=paciente_id,
                backend=self.llm.backend,
                bloqueado_guardrail=True,
                categorias_guardrail=g.categorias,
            )

        # 4) Geração: adapter LoRA/Ollama quando disponível; em ambiente sem modelo,
        # o backend extrativo usa o mesmo contexto recuperado em vez de simular loss.
        prompt = _TEMPLATE.format(
            contexto=contexto,
            bloco_paciente=bloco_paciente,
            pergunta=pergunta,
        )
        bruto = self.llm.invoke(prompt, system=SYSTEM_PROMPT)

        # 5) A saída também passa pelo guardrail. Se a LLM produzir uma dose imperativa,
        # o conteúdo é substituído por uma mensagem segura antes de chegar ao usuário.
        resposta = guardrails.sanitizar_saida(bruto)
        if fontes and not any(f in resposta for f in fontes):
            resposta += "\n\nFontes consultadas: " + "; ".join(fontes) + "."

        return RespostaAssistente(
            resposta=resposta,
            fontes=fontes,
            paciente_id=paciente_id,
            backend=self.llm.backend,
        )
