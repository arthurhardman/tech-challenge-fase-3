"""
cli.py
------
Interface de linha de comando do assistente médico (Fase 3).

Exemplos:
    # pergunta clínica direta (RAG + guardrails)
    python -m src.assistant.cli perguntar "Qual o alvo de saturação em SRAG?"

    # pergunta contextualizada por paciente da base
    python -m src.assistant.cli perguntar "Qual a conduta?" --paciente PAC-0003

    # fluxo de decisão automatizado (LangGraph) para um paciente
    python -m src.assistant.cli fluxo --paciente PAC-0001

    # lista pacientes sintéticos disponíveis
    python -m src.assistant.cli pacientes
"""

from __future__ import annotations

import argparse

from .chains.graph import FluxoAtendimento
from .chains.medical_assistant import MedicalAssistant
from .knowledge.patient_db import PatientDB
from .safety.audit_log import AuditEvent, AuditLogger


def _cmd_perguntar(args) -> None:
    assistant = MedicalAssistant()
    auditor = AuditLogger()
    r = assistant.responder(args.texto, paciente_id=args.paciente)
    print(f"\n[backend: {r.backend}]")
    print(r.resposta)
    print(f"\nFontes: {', '.join(r.fontes) or '—'}")
    auditor.registrar(AuditEvent(
        pergunta=args.texto, resposta=r.resposta, backend_llm=r.backend,
        paciente_id=args.paciente, fontes=r.fontes,
        guardrail_bloqueou=r.bloqueado_guardrail,
        guardrail_categorias=r.categorias_guardrail, fluxo_no="cli_perguntar",
    ))


def _cmd_fluxo(args) -> None:
    fluxo = FluxoAtendimento()
    if not fluxo.patient_db.existe(args.paciente):
        print(f"Paciente {args.paciente} não encontrado. Use 'pacientes' para listar.")
        return
    estado = fluxo.executar(args.paciente, pergunta=args.pergunta)
    print(f"\n[backend: {estado.get('backend')}]")
    print(estado["resumo_final"])


def _cmd_pacientes(_args) -> None:
    db = PatientDB()
    ids = db.todos_ids(limit=15)
    if not ids:
        print("Base de pacientes vazia. Rode run_fase3.py ou informe --sivep.")
        return
    print(f"Base: {db.backend} | total: {db.count():,}")
    for pid in ids:
        reg = db.get(pid) or {}
        print(
            f"  {pid} | risco={reg.get('classificacao_risco', 'n/d'):8} "
            f"| ano={reg.get('ano_fonte', 'n/d')} "
            f"| pendentes={db.exames_pendentes(pid) or '—'}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Assistente médico SRAG (Fase 3).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("perguntar", help="Pergunta clínica direta.")
    p1.add_argument("texto")
    p1.add_argument("--paciente", default=None, help="ID do paciente (ex.: PAC-0003).")
    p1.set_defaults(func=_cmd_perguntar)

    p2 = sub.add_parser("fluxo", help="Fluxo de decisão automatizado (LangGraph).")
    p2.add_argument("--paciente", required=True, help="ID do paciente.")
    p2.add_argument("--pergunta", default=None, help="Pergunta opcional para o fluxo.")
    p2.set_defaults(func=_cmd_fluxo)

    p3 = sub.add_parser("pacientes", help="Lista pacientes disponíveis na base estruturada.")
    p3.set_defaults(func=_cmd_pacientes)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
