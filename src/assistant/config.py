"""
config.py
---------
Configuração central do assistente médico (Fase 3).

Centraliza caminhos, nomes de arquivos e flags de ambiente para que os demais
módulos não fixem paths no código. Segue a mesma filosofia das fases anteriores:
o fluxo padrão usa a amostra anonimizada do SIVEP incluída no projeto. Os dados
sintéticos antigos continuam disponíveis apenas como fallback/demonstração explícita.
"""

from __future__ import annotations

import os
from pathlib import Path

# Raiz do projeto (…/ml-tech). Três níveis acima deste arquivo.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# --- Base de conhecimento --------------------------------------------------- #
KB_DIR = PROJECT_ROOT / "data" / "knowledge_base"
PROTOCOLOS_DIR = KB_DIR / "protocolos"
FAQ_PATH = KB_DIR / "faq_medicos.jsonl"
LAUDOS_PATH = KB_DIR / "laudos_modelos.jsonl"
PRONTUARIOS_PATH = KB_DIR / "prontuarios.json"

# Base real anonimizada derivada do SIVEP/OpenDataSUS. O SQLite tem prioridade
# no PatientDB quando existir; o JSON acima continua disponível para a demo antiga.
SIVEP_DIR = KB_DIR / "sivep"
SIVEP_SAMPLE_PATH = SIVEP_DIR / "patients_sivep_sample.csv"
SIVEP_DB_PATH = SIVEP_DIR / "patients_sivep.db"

# Protocolos oficiais usados no RAG e perguntas curadas usadas no fine-tuning.
OFFICIAL_DIR = KB_DIR / "official"
OFFICIAL_CHUNKS_PATH = OFFICIAL_DIR / "protocol_chunks.jsonl"
OFFICIAL_FAQ_PATH = OFFICIAL_DIR / "protocol_faq.jsonl"
OFFICIAL_EVAL_PATH = OFFICIAL_DIR / "qa_eval.jsonl"

# --- Artefatos de fine-tuning ---------------------------------------------- #
FINETUNING_DIR = PROJECT_ROOT / "results" / "finetuning"
DATASET_PATH = FINETUNING_DIR / "dataset_instruct.jsonl"
ADAPTER_DIR = FINETUNING_DIR / "adapter"          # onde o LoRA seria salvo
TRAIN_METRICS_PATH = FINETUNING_DIR / "train_metrics.json"
EVAL_METRICS_PATH = FINETUNING_DIR / "eval_metrics.json"

# --- Logging / auditoria ---------------------------------------------------- #
AUDIT_LOG_PATH = PROJECT_ROOT / "results" / "finetuning" / "audit.log"

# --- Backend da LLM --------------------------------------------------------- #
# O modo auto prioriza o adapter LoRA quando existir e depois o checkpoint local.
# Ollama e mock continuam disponíveis para compatibilidade com a Fase 2.
LLM_BACKEND = os.environ.get("LLM_BACKEND", "auto")
LLM_MODEL = os.environ.get("LLM_MODEL", "llama3.1:8b")

# Modelo-base sugerido para o fine-tuning real (executável em GPU).
# Trocável por qualquer causal LM do Hugging Face (LLaMA, Falcon, Mistral…).
BASE_MODEL_ID = os.environ.get("BASE_MODEL_ID", "meta-llama/Llama-3.2-1B-Instruct")


def ensure_dirs() -> None:
    """Cria os diretórios de saída se não existirem."""
    for d in (KB_DIR, PROTOCOLOS_DIR, SIVEP_DIR, OFFICIAL_DIR, FINETUNING_DIR, ADAPTER_DIR):
        d.mkdir(parents=True, exist_ok=True)
