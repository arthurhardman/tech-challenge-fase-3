"""
config.py
---------
Configuração central do assistente médico (Fase 3).

Centraliza caminhos, nomes de arquivos e flags de ambiente para que os demais
módulos não fixem paths no código. Segue a mesma filosofia das fases anteriores:
tudo funciona "out of the box" com dados sintéticos locais, sem rede obrigatória.
"""

from __future__ import annotations

import os
from pathlib import Path

# Raiz do projeto (…/ml-tech). Três níveis acima deste arquivo.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# --- Base de conhecimento (dados sintéticos anonimizados) ------------------- #
KB_DIR = PROJECT_ROOT / "data" / "knowledge_base"
PROTOCOLOS_DIR = KB_DIR / "protocolos"
FAQ_PATH = KB_DIR / "faq_medicos.jsonl"
LAUDOS_PATH = KB_DIR / "laudos_modelos.jsonl"
PRONTUARIOS_PATH = KB_DIR / "prontuarios.json"

# --- Artefatos de fine-tuning ---------------------------------------------- #
FINETUNING_DIR = PROJECT_ROOT / "results" / "finetuning"
DATASET_PATH = FINETUNING_DIR / "dataset_instruct.jsonl"
ADAPTER_DIR = FINETUNING_DIR / "adapter"          # onde o LoRA seria salvo
TRAIN_METRICS_PATH = FINETUNING_DIR / "train_metrics.json"
EVAL_METRICS_PATH = FINETUNING_DIR / "eval_metrics.json"

# --- Logging / auditoria ---------------------------------------------------- #
AUDIT_LOG_PATH = PROJECT_ROOT / "results" / "finetuning" / "audit.log"

# --- Backend da LLM (reaproveita o cliente Ollama/mock da Fase 2) ----------- #
# LLM_BACKEND: "ollama" | "mock" | "auto"  (default: auto → cai em mock offline)
LLM_BACKEND = os.environ.get("LLM_BACKEND", "auto")
LLM_MODEL = os.environ.get("LLM_MODEL", "llama3.1:8b")

# Modelo-base sugerido para o fine-tuning real (executável em GPU).
# Trocável por qualquer causal LM do Hugging Face (LLaMA, Falcon, Mistral…).
BASE_MODEL_ID = os.environ.get("BASE_MODEL_ID", "meta-llama/Llama-3.2-1B-Instruct")


def ensure_dirs() -> None:
    """Cria os diretórios de saída se não existirem."""
    for d in (KB_DIR, PROTOCOLOS_DIR, FINETUNING_DIR, ADAPTER_DIR):
        d.mkdir(parents=True, exist_ok=True)
