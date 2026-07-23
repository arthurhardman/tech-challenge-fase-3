"""
Testes do assistente médico da Fase 3: RAG, contexto do paciente, guardrails,
fluxo LangGraph e auditoria. Rodam no backend mock (offline).
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Força o backend mock para todos os testes deste módulo.
os.environ["LLM_BACKEND"] = "mock"

from src.assistant.knowledge.retriever import ProtocolRetriever
from src.assistant.knowledge.patient_db import PatientDB
from src.assistant.safety import guardrails
from src.assistant.safety.audit_log import AuditLogger, AuditEvent
from src.assistant.chains.medical_assistant import MedicalAssistant
from src.assistant.chains.graph import FluxoAtendimento


# --------------------------------------------------------------------------- #
# RAG / retriever
# --------------------------------------------------------------------------- #
def test_retriever_recupera_fonte_correta():
    r = ProtocolRetriever()
    trechos = r.buscar("alvo de saturação em oxigenoterapia", top_k=3)
    assert trechos
    assert any(t.protocolo == "PROT-SRAG-02" for t in trechos)
    assert all(t.citacao() for t in trechos)  # toda fonte tem citação


def test_retriever_hibrido_indexa_medquad():
    r = ProtocolRetriever(incluir_medquad=True)
    # pergunta geral em inglês deve trazer uma fonte MedQuAD
    trechos = r.buscar("What are the symptoms of leukemia?", top_k=2)
    assert trechos
    assert any(t.protocolo.startswith("MedQuAD") for t in trechos)


# --------------------------------------------------------------------------- #
# Base estruturada de pacientes
# --------------------------------------------------------------------------- #
def test_patient_db_resumo_e_pendentes():
    db = PatientDB()
    pid = db.todos_ids()[0]
    resumo = db.resumo_clinico(pid)
    assert pid in resumo and "SpO2" in resumo
    assert isinstance(db.exames_pendentes(pid), list)


# --------------------------------------------------------------------------- #
# Guardrails
# --------------------------------------------------------------------------- #
def test_guardrail_bloqueia_prescricao():
    g = guardrails.checar_entrada("Prescreva a dose exata de corticoide")
    assert not g.permitido
    assert "prescricao_direta" in g.categorias


def test_guardrail_permite_pergunta_normal():
    g = guardrails.checar_entrada("Quais os critérios de alta?")
    assert g.permitido


def test_sanitizar_saida_anexa_disclaimer():
    out = guardrails.sanitizar_saida("Conteúdo qualquer.")
    assert "validação" in out.lower()


# --------------------------------------------------------------------------- #
# Assistente (chain LangChain)
# --------------------------------------------------------------------------- #
def test_assistente_responde_com_fonte_e_disclaimer():
    a = MedicalAssistant()
    r = a.responder("Qual o alvo de saturação em oxigenoterapia?")
    assert r.fontes  # citou pelo menos uma fonte (explainability)
    assert "validação" in r.resposta.lower()  # disclaimer de segurança


def test_assistente_bloqueia_prescricao():
    a = MedicalAssistant()
    r = a.responder("Prescreva a dose de antibiótico")
    assert r.bloqueado_guardrail


def test_assistente_contextualiza_paciente():
    a = MedicalAssistant()
    pid = a.patient_db.todos_ids()[0]
    r = a.responder("Qual a conduta?", paciente_id=pid)
    assert r.paciente_id == pid


# --------------------------------------------------------------------------- #
# Fluxo LangGraph
# --------------------------------------------------------------------------- #
def test_fluxo_risco_vermelho_emite_alerta():
    f = FluxoAtendimento()
    db = f.patient_db
    pid = next((p for p in db.todos_ids()
                if db.get(p)["classificacao_risco"] == "vermelho"), None)
    if pid is None:
        return  # base sem paciente vermelho neste seed — pula
    est = f.executar(pid, "Conduta imediata?")
    assert "emitir_alerta" in est["trilha"]
    assert est.get("alerta")
    assert "consolidar" in est["trilha"]


def test_fluxo_risco_baixo_sugere_conduta():
    f = FluxoAtendimento()
    db = f.patient_db
    pid = next((p for p in db.todos_ids()
                if db.get(p)["classificacao_risco"] in ("verde", "amarelo")), None)
    est = f.executar(pid)
    assert "sugerir_conduta" in est["trilha"]
    assert est.get("conduta")


# --------------------------------------------------------------------------- #
# Auditoria
# --------------------------------------------------------------------------- #
def test_auditoria_registra_evento(tmp_path):
    log = tmp_path / "audit.log"
    jsonl = tmp_path / "events.jsonl"
    auditor = AuditLogger(log_path=log, jsonl_path=jsonl)
    auditor.registrar(AuditEvent(
        pergunta="teste", resposta="ok", backend_llm="mock", fontes=["PROT-SRAG-01"],
    ))
    eventos = auditor.ler_eventos()
    assert len(eventos) == 1
    assert eventos[0]["fontes"] == ["PROT-SRAG-01"]


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
