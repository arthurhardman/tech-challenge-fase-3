"""
graph.py
--------
Fluxo de decisão automatizado com **LangGraph** (requisito central da Fase 3).

Modela o cenário do enunciado: ao receber informações de um paciente, o sistema
aciona etapas coordenadas — verifica exames pendentes, sugere conduta com base
nos protocolos e emite alerta para a equipe quando há risco alto.

Grafo de estados:

        entrada
           │
      [triagem] ── carrega prontuário + classificação de risco
           │
   [verificar_exames] ── lista exames pendentes
           │
    (roteamento por risco)
        ┌──┴───────────────┐
     vermelho            demais
        │                   │
  [emitir_alerta]     [sugerir_conduta] ── assistente (RAG + guardrails)
        │                   │
        └────────┬──────────┘
            [consolidar] → resumo final + auditoria
                 │
                END

Cada nó registra um `AuditEvent`, tornando o fluxo inteiro auditável.
"""

from __future__ import annotations

from typing import List, Optional, TypedDict

from langgraph.graph import StateGraph, END

from ..knowledge.patient_db import PatientDB
from ..safety.audit_log import AuditLogger, AuditEvent
from .medical_assistant import MedicalAssistant


class EstadoAtendimento(TypedDict, total=False):
    """Estado compartilhado entre os nós do grafo."""
    paciente_id: str
    pergunta: str
    risco: str
    exames_pendentes: List[str]
    conduta: str
    alerta: Optional[str]
    fontes: List[str]
    backend: str
    trilha: List[str]        # sequência de nós percorridos (para explainability)
    resumo_final: str


class FluxoAtendimento:
    """Encapsula o `StateGraph` do LangGraph e suas dependências."""

    def __init__(
        self,
        assistant: Optional[MedicalAssistant] = None,
        patient_db: Optional[PatientDB] = None,
        auditor: Optional[AuditLogger] = None,
    ):
        self.assistant = assistant or MedicalAssistant()
        self.patient_db = patient_db or self.assistant.patient_db
        self.auditor = auditor or AuditLogger()
        self.app = self._compilar()

    # ---- nós ------------------------------------------------------------- #
    def _no_triagem(self, estado: EstadoAtendimento) -> EstadoAtendimento:
        reg = self.patient_db.get(estado["paciente_id"])
        risco = reg["classificacao_risco"] if reg else "desconhecido"
        trilha = estado.get("trilha", []) + ["triagem"]
        return {**estado, "risco": risco, "trilha": trilha}

    def _no_verificar_exames(self, estado: EstadoAtendimento) -> EstadoAtendimento:
        pend = self.patient_db.exames_pendentes(estado["paciente_id"])
        trilha = estado.get("trilha", []) + ["verificar_exames"]
        return {**estado, "exames_pendentes": pend, "trilha": trilha}

    def _no_sugerir_conduta(self, estado: EstadoAtendimento) -> EstadoAtendimento:
        pergunta = estado.get("pergunta") or "Qual a conduta recomendada para este paciente?"
        r = self.assistant.responder(pergunta, paciente_id=estado["paciente_id"])
        trilha = estado.get("trilha", []) + ["sugerir_conduta"]
        self.auditor.registrar(AuditEvent(
            pergunta=pergunta, resposta=r.resposta, backend_llm=r.backend,
            paciente_id=estado["paciente_id"], fontes=r.fontes,
            guardrail_bloqueou=r.bloqueado_guardrail,
            guardrail_categorias=r.categorias_guardrail, fluxo_no="sugerir_conduta",
        ))
        return {**estado, "conduta": r.resposta, "fontes": r.fontes,
                "backend": r.backend, "trilha": trilha}

    def _no_emitir_alerta(self, estado: EstadoAtendimento) -> EstadoAtendimento:
        pend = estado.get("exames_pendentes", [])
        alerta = (
            f"🚨 ALERTA — paciente {estado['paciente_id']} classificado como RISCO "
            f"VERMELHO. Acionar equipe de resposta rápida (PROT-SRAG-01). "
            + (f"Exames pendentes: {', '.join(pend)}." if pend else "Exames em dia.")
        )
        # No risco alto também geramos a sugestão de conduta.
        r = self.assistant.responder(
            estado.get("pergunta") or "Conduta imediata para risco alto de SRAG?",
            paciente_id=estado["paciente_id"],
        )
        trilha = estado.get("trilha", []) + ["emitir_alerta"]
        self.auditor.registrar(AuditEvent(
            pergunta="[fluxo] risco vermelho", resposta=alerta, backend_llm=r.backend,
            paciente_id=estado["paciente_id"], fontes=r.fontes, fluxo_no="emitir_alerta",
        ))
        return {**estado, "alerta": alerta, "conduta": r.resposta,
                "fontes": r.fontes, "backend": r.backend, "trilha": trilha}

    def _no_consolidar(self, estado: EstadoAtendimento) -> EstadoAtendimento:
        partes = [f"Paciente {estado['paciente_id']} | risco: {estado.get('risco')}"]
        pend = estado.get("exames_pendentes", [])
        partes.append(
            "Exames pendentes: " + (", ".join(pend) if pend else "nenhum")
        )
        if estado.get("alerta"):
            partes.append(estado["alerta"])
        partes.append("Conduta sugerida:\n" + estado.get("conduta", "(sem conduta)"))
        if estado.get("fontes"):
            partes.append("Fontes: " + ", ".join(estado["fontes"]))
        partes.append("Trilha do fluxo: " + " → ".join(estado.get("trilha", []) + ["consolidar"]))
        resumo = "\n\n".join(partes)
        return {**estado, "resumo_final": resumo,
                "trilha": estado.get("trilha", []) + ["consolidar"]}

    # ---- roteamento condicional ------------------------------------------ #
    @staticmethod
    def _rota_por_risco(estado: EstadoAtendimento) -> str:
        return "emitir_alerta" if estado.get("risco") == "vermelho" else "sugerir_conduta"

    # ---- compilação ------------------------------------------------------ #
    def _compilar(self):
        g = StateGraph(EstadoAtendimento)
        g.add_node("triagem", self._no_triagem)
        g.add_node("verificar_exames", self._no_verificar_exames)
        g.add_node("sugerir_conduta", self._no_sugerir_conduta)
        g.add_node("emitir_alerta", self._no_emitir_alerta)
        g.add_node("consolidar", self._no_consolidar)

        g.set_entry_point("triagem")
        g.add_edge("triagem", "verificar_exames")
        g.add_conditional_edges(
            "verificar_exames",
            self._rota_por_risco,
            {"emitir_alerta": "emitir_alerta", "sugerir_conduta": "sugerir_conduta"},
        )
        g.add_edge("emitir_alerta", "consolidar")
        g.add_edge("sugerir_conduta", "consolidar")
        g.add_edge("consolidar", END)
        return g.compile()

    def executar(self, paciente_id: str, pergunta: Optional[str] = None) -> EstadoAtendimento:
        """Roda o fluxo completo para um paciente e retorna o estado final."""
        entrada: EstadoAtendimento = {"paciente_id": paciente_id, "trilha": []}
        if pergunta:
            entrada["pergunta"] = pergunta
        return self.app.invoke(entrada)
