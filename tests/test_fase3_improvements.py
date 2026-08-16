"""Testes das melhorias propostas no MR da Fase 3."""

from pathlib import Path

from src.assistant import config
from src.assistant.chains.graph import FluxoAtendimento
from src.assistant.chains.medical_assistant import MedicalAssistant
from src.assistant.knowledge.patient_db import PatientDB
from src.assistant.knowledge.retriever import ProtocolRetriever
from src.assistant.knowledge.sivep_adapter import prepare_sqlite_from_prepared_csv
from src.assistant.safety.guardrails import sanitizar_saida
from src.assistant.finetuning.data_prep import anonimizar
from src.finetuning.dataset_prep import anonymize, build_dataset
from src.finetuning.local_validation import Config as LocalConfig, run as run_local_validation


def test_anonimizacao_remove_nome_rotulado():
    texto = "Paciente João da Silva, CPF 123.456.789-00, e-mail joao@x.com"
    assert "João da Silva" not in anonimizar(texto)
    assert "João da Silva" not in anonymize(texto)


def test_guardrail_saida_remove_instrucao_de_dose():
    out = sanitizar_saida("Tome 50 mg de medicamento agora.")
    assert "Tome 50 mg" not in out
    assert "bloqueada" in out
    assert "validação" in out


def test_retriever_carrega_protocolos_oficiais_e_pagina():
    r = ProtocolRetriever(incluir_medquad=False, incluir_pubmedqa=False)
    resultados = r.buscar("Quais sinais indicam evolução para SRAG?", top_k=5)
    assert resultados
    assert any(t.tipo == "protocolo_oficial" for t in resultados)
    assert any(t.page is not None for t in resultados if t.tipo == "protocolo_oficial")


def test_patient_db_sqlite_com_amostra_sivep(tmp_path):
    db_path = tmp_path / "patients.db"
    total = prepare_sqlite_from_prepared_csv(config.SIVEP_SAMPLE_PATH, db_path)
    db = PatientDB(db_path)
    assert total == 8000
    assert db.count() == 8000
    pid = db.todos_ids(limit=1)[0]
    resumo = db.resumo_clinico(pid)
    assert pid in resumo
    assert "risco derivado" in resumo


def test_fluxo_funciona_com_sivep_sqlite(tmp_path):
    db_path = tmp_path / "patients.db"
    prepare_sqlite_from_prepared_csv(config.SIVEP_SAMPLE_PATH, db_path)
    db = PatientDB(db_path)
    assistant = MedicalAssistant(patient_db=db)
    fluxo = FluxoAtendimento(assistant=assistant, patient_db=db)
    pid = db.primeiro_por_risco("vermelho") or db.todos_ids(limit=1)[0]
    estado = fluxo.executar(pid, "Quais critérios de gravidade devem ser revisados?")
    assert estado["trilha"][0:2] == ["triagem", "verificar_exames"]
    assert estado["trilha"][-1] == "consolidar"
    assert estado.get("fontes")


def test_dataset_principal_inclui_srag_oficial():
    stats = build_dataset(max_medquad=20, max_pubmedqa=20, synthetic_oversample=1)
    assert stats["por_fonte"]["Protocolos oficiais SRAG"] >= 80


def test_treino_local_real_gera_checkpoint(tmp_path):
    metrics = run_local_validation(
        output_dir=tmp_path,
        cfg=LocalConfig(
            pretrain_epochs=1,
            general_medical_epochs=1,
            finetune_epochs=1,
            batch_size=16,
            max_pretraining_pairs=20,
            max_general_medical_examples=20,
        ),
    )
    assert metrics["mode"] == "real-local-validation"
    assert metrics["training_examples"] > 0
    assert (tmp_path / "tiny_transformer.pt").exists()
    assert (tmp_path / "metrics.json").exists()


def test_fallback_local_detecta_saida_degenerada():
    from src.assistant.chains.llm_backend import _local_output_is_low_quality

    assert _local_output_is_low_quality("o do é a notificação a. a do o do o é.")
    assert not _local_output_is_low_quality(
        "Pacientes com sinais de agravamento devem ser reavaliados conforme o protocolo clínico."
    )


def test_fallback_extrativo_escolhe_frase_relacionada_a_pergunta():
    from src.assistant.chains.llm_backend import _extractive_grounded

    prompt = """CONTEXTO CLÍNICO RECUPERADO:
[Fonte: protocolo.pdf, p. 4]
A vacinação deve ser registrada no sistema. Pacientes com dispneia e saturação baixa devem ser avaliados quanto a sinais de gravidade.

PERGUNTA DO MÉDICO:
Quais sinais de gravidade respiratória devem ser revisados?
"""
    resposta = _extractive_grounded(prompt)
    assert "dispneia" in resposta.lower()
    assert "protocolo.pdf, p. 4" in resposta
