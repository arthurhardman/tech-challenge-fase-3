"""
Testes do pipeline de fine-tuning da Fase 3.

Cobrem anonimização de PII, curadoria, construção do dataset de instrução e o
laço de treinamento em modo demo (offline, sem GPU/rede).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.assistant.finetuning.data_prep import (
    anonimizar,
    curar_exemplos,
    preprocessar,
)
from src.assistant.finetuning.dataset_builder import build_dataset
from src.assistant.finetuning import train


# --------------------------------------------------------------------------- #
# Anonimização
# --------------------------------------------------------------------------- #
def test_anonimiza_cpf_email_telefone():
    txt = "Paciente João da Silva, CPF 123.456.789-00, tel (11) 91234-5678, joao@x.com"
    out = anonimizar(txt)
    assert "123.456.789-00" not in out
    assert "joao@x.com" not in out
    assert "91234-5678" not in out
    assert "[CPF]" in out and "[EMAIL]" in out


def test_anonimizacao_idempotente():
    txt = "email teste@hospital.com repetido teste@hospital.com"
    assert anonimizar(anonimizar(txt)) == anonimizar(txt)


def test_curadoria_remove_curtos_e_duplicados():
    exemplos = [
        {"prompt": "P1", "response": "resposta suficientemente longa para passar"},
        {"prompt": "P1", "response": "resposta suficientemente longa para passar"},  # dup
        {"prompt": "P2", "response": "curta"},  # curta demais
    ]
    curados = curar_exemplos(exemplos)
    assert len(curados) == 1


def test_preprocessar_normaliza_espacos():
    assert preprocessar("a   b\n\n\n\nc") == "a b\n\nc"


# --------------------------------------------------------------------------- #
# Dataset de instrução
# --------------------------------------------------------------------------- #
def test_build_dataset_gera_exemplos():
    ds = build_dataset(salvar=False)
    assert len(ds) > 0
    ex = ds[0]
    assert {"prompt", "response", "system", "origem"} <= set(ex.keys())
    # sem PII bruta
    assert "@" not in ex["response"] or "[EMAIL]" in ex["response"]


# --------------------------------------------------------------------------- #
# Treinamento (modo demo)
# --------------------------------------------------------------------------- #
def test_train_demo_produz_loss_decrescente():
    m = train.run(mode="demo", epochs=3)
    assert m["mode"] == "demo"
    losses = m["loss_por_passo"]
    assert len(losses) > 0
    # a perda final deve ser menor que a inicial (convergência)
    assert losses[-1] < losses[0]


def test_train_demo_salva_manifesto_adapter():
    train.run(mode="demo", epochs=2)
    from src.assistant import config
    assert (config.ADAPTER_DIR / "adapter_manifest.json").exists()


# --------------------------------------------------------------------------- #
# Datasets externos (PubMedQA + MedQuAD) — abordagem híbrida
# --------------------------------------------------------------------------- #
def test_fatias_externas_existem_e_tem_formato():
    from src.assistant.finetuning.external_datasets import carregar_externos
    ext = carregar_externos()
    assert len(ext) > 0
    origens = {e["origem"] for e in ext}
    assert "pubmedqa" in origens and "medquad" in origens
    # todo exemplo externo tem fonte (explainability) e idioma
    assert all(e.get("fonte") and e.get("idioma") == "en" for e in ext)


def test_dataset_hibrido_combina_hospital_e_externos():
    ds = build_dataset(salvar=False, incluir_externos=True)
    origens = {e["origem"] for e in ds}
    # hospital (PT) + datasets (EN)
    assert {"protocolo", "faq"} <= origens
    assert {"pubmedqa", "medquad"} <= origens


def test_dataset_sem_externos_fica_menor():
    com = build_dataset(salvar=False, incluir_externos=True)
    sem = build_dataset(salvar=False, incluir_externos=False)
    assert len(com) > len(sem)


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
