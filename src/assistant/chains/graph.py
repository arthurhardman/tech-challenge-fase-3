"""Fluxo automatizado de atendimento com LangGraph.

O fluxo segue o exemplo do challenge: consulta o paciente, verifica exames
pendentes, separa casos de maior risco e então gera uma orientação baseada no
RAG. Quando LangGraph não está instalado, a classe executa a mesma sequência em
Python puro para permitir testes locais; com a dependência instalada, o mesmo
conjunto de nós é compilado em ``StateGraph``.
"""

from __future__ import annotations

from typing import List, Optional, TypedDict

try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    StateGraph = None
    END = "END"
    LANGGRAPH_AVAILABLE = False

from ..knowledge.patient_db import PatientDB
from ..safety.audit_log import AuditLogger, AuditEvent
from .medical_assistant import MedicalAssistant


class EstadoAtendimento(TypedDict, total=False):
    paciente_id: str
    pergunta: str
    risco: str
    exames_pendentes: List[str]
    conduta: str
    alerta: Optional[str]
    fontes: List[str]
    backend: str
    trilha: List[str]
    resumo_final: str


class _PythonGraphRunner:
    """Executa o mesmo fluxo quando LangGraph não está disponível no ambiente."""

    def __init__(self, owner: "FluxoAtendimento"):
        self.owner = owner

    def invoke(self, estado: EstadoAtendimento) -> EstadoAtendimento:
        estado = self.owner._no_triagem(estado)
        estado = self.owner._no_verificar_exames(estado)
        if self.owner._rota_por_risco(estado) == "emitir_alerta":
            estado = self.owner._no_emitir_alerta(estado)
        else:
            estado = self.owner._no_sugerir_conduta(estado)
        return self.owner._no_consolidar(estado)


class FluxoAtendimento:
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

    def _no_triagem(self, estado: EstadoAtendimento) -> EstadoAtendimento:
        reg = self.patient_db.get(estado["paciente_id"])
        risco = reg.get("classificacao_risco", "desconhecido") if reg else "desconhecido"
        return {**estado, "risco": risco, "trilha": estado.get("trilha", []) + ["triagem"]}

    def _no_verificar_exames(self, estado: EstadoAtendimento) -> EstadoAtendimento:
        pend = self.patient_db.exames_pendentes(estado["paciente_id"])
        return {
            **estado,
            "exames_pendentes": pend,
            "trilha": estado.get("trilha", []) + ["verificar_exames"],
        }

    def _no_sugerir_conduta(self, estado: EstadoAtendimento) -> EstadoAtendimento:
        pergunta = estado.get("pergunta") or "Quais critérios do protocolo devem ser conferidos neste caso?"
        r = self.assistant.responder(pergunta, paciente_id=estado["paciente_id"])
        trilha = estado.get("trilha", []) + ["sugerir_conduta"]
        self.auditor.registrar(AuditEvent(
            pergunta=pergunta,
            resposta=r.resposta,
            backend_llm=r.backend,
            paciente_id=estado["paciente_id"],
            fontes=r.fontes,
            guardrail_bloqueou=r.bloqueado_guardrail,
            guardrail_categorias=r.categorias_guardrail,
            fluxo_no="sugerir_conduta",
        ))
        return {
            **estado,
            "conduta": r.resposta,
            "fontes": r.fontes,
            "backend": r.backend,
            "trilha": trilha,
        }

    def _no_emitir_alerta(self, estado: EstadoAtendimento) -> EstadoAtendimento:
        pend = estado.get("exames_pendentes", [])
        alerta = (
            f"🚨 ALERTA — paciente {estado['paciente_id']} classificado como risco vermelho "
            "pelos indicadores disponíveis na base. Priorizar avaliação da equipe médica. "
            + (f"Exames pendentes registrados: {', '.join(pend)}." if pend else "Não há exame registrado como pendente.")
        )
        r = self.assistant.responder(
            estado.get("pergunta") or "Quais critérios do protocolo devem ser revisados em um caso de maior risco?",
            paciente_id=estado["paciente_id"],
        )
        trilha = estado.get("trilha", []) + ["emitir_alerta"]
        self.auditor.registrar(AuditEvent(
            pergunta="[fluxo] risco vermelho",
            resposta=alerta,
            backend_llm=r.backend,
            paciente_id=estado["paciente_id"],
            fontes=r.fontes,
            fluxo_no="emitir_alerta",
        ))
        return {
            **estado,
            "alerta": alerta,
            "conduta": r.resposta,
            "fontes": r.fontes,
            "backend": r.backend,
            "trilha": trilha,
        }

    def _no_consolidar(self, estado: EstadoAtendimento) -> EstadoAtendimento:
        partes = [f"Paciente {estado['paciente_id']} | risco: {estado.get('risco')}"]
        pend = estado.get("exames_pendentes", [])
        partes.append("Exames pendentes: " + (", ".join(pend) if pend else "nenhum registrado"))
        if estado.get("alerta"):
            partes.append(estado["alerta"])
        partes.append("Orientação baseada nas fontes:\n" + estado.get("conduta", "(sem resposta)"))
        if estado.get("fontes"):
            partes.append("Fontes: " + "; ".join(estado["fontes"]))
        trilha = estado.get("trilha", []) + ["consolidar"]
        partes.append("Trilha do fluxo: " + " → ".join(trilha))
        return {**estado, "resumo_final": "\n\n".join(partes), "trilha": trilha}

    @staticmethod
    def _rota_por_risco(estado: EstadoAtendimento) -> str:
        return "emitir_alerta" if estado.get("risco") == "vermelho" else "sugerir_conduta"

    def _compilar(self):
        if not LANGGRAPH_AVAILABLE:
            return _PythonGraphRunner(self)

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
        entrada: EstadoAtendimento = {"paciente_id": paciente_id, "trilha": []}
        if pergunta:
            entrada["pergunta"] = pergunta
        return self.app.invoke(entrada)
