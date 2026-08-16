"""Orquestra a Fase 3 sem sobrescrever os dados existentes.

Exemplos:
    # Execução reproduzível em CPU usando a amostra real do SIVEP incluída no repo
    python run_fase3.py --mode local

    # Preparar uma base maior com arquivos anuais do OpenDataSUS
    python run_fase3.py --mode local --sivep data/raw/INFLUD24.csv data/raw/sivep_multi_year/

    # Validar o dataset LoRA sem baixar modelo/GPU
    python run_fase3.py --mode lora-dry-run

    # Fine-tuning LoRA real (GPU + requirements-finetuning.txt)
    python run_fase3.py --mode lora-real

A base sintética antiga continua disponível, mas só é regenerada quando
``--generate-synthetic`` é passado explicitamente.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from src.assistant import config
from src.assistant.knowledge.patient_db import PatientDB
from src.assistant.knowledge.sivep_adapter import (
    prepare_sqlite_from_prepared_csv,
    prepare_sqlite_from_sivep,
)


def _run(desc: str, args: list[str], env: dict | None = None) -> None:
    print(f"\n=== {desc} ===")
    merged = os.environ.copy()
    if env:
        merged.update(env)
    subprocess.run([sys.executable, *args], check=True, env=merged)


def _prepare_patients(args) -> None:
    print("\n=== 1/6 Base estruturada de pacientes ===")
    if args.generate_synthetic:
        _run("Gerando a base sintética por solicitação explícita", ["scripts/gen_synthetic_data.py"])
        return

    if args.sivep:
        total = prepare_sqlite_from_sivep(
            args.sivep,
            config.SIVEP_DB_PATH,
            max_rows_per_file=args.max_rows_per_file,
        )
        print(f"[ok] {total:,} registros SIVEP preparados em {config.SIVEP_DB_PATH}")
        return

    if config.SIVEP_DB_PATH.exists():
        print(f"[ok] usando SQLite existente: {config.SIVEP_DB_PATH}")
        return

    if config.SIVEP_SAMPLE_PATH.exists():
        total = prepare_sqlite_from_prepared_csv(config.SIVEP_SAMPLE_PATH, config.SIVEP_DB_PATH)
        print(f"[ok] amostra real SIVEP importada: {total:,} registros")
        return

    if config.PRONTUARIOS_PATH.exists():
        print(f"[ok] usando JSON existente: {config.PRONTUARIOS_PATH}")
        return

    raise FileNotFoundError(
        "Nenhuma base de pacientes encontrada. Informe --sivep, mantenha a amostra "
        "versionada ou use --generate-synthetic para a demo antiga."
    )


def _choose_patient(requested: str | None) -> str:
    db = PatientDB()
    if requested and db.existe(requested):
        return requested
    return db.primeiro_por_risco("vermelho") or (db.todos_ids(limit=1)[0] if db.todos_ids(limit=1) else "")


def main() -> None:
    ap = argparse.ArgumentParser(description="Pipeline integrado da Fase 3.")
    ap.add_argument(
        "--mode",
        choices=["local", "lora-dry-run", "lora-real", "skip-train"],
        default="local",
        help="local executa treino real pequeno em CPU; lora-real exige GPU.",
    )
    ap.add_argument("--paciente", default=None)
    ap.add_argument("--sivep", nargs="*", default=None, help="CSV/Parquet ou diretório com INFLUD*.csv")
    ap.add_argument("--max-rows-per-file", type=int, default=None)
    ap.add_argument("--generate-synthetic", action="store_true")
    args = ap.parse_args()

    config.ensure_dirs()
    _prepare_patients(args)

    _run(
        "2/6 Dataset de fine-tuning (MedQuAD + PubMedQA + SRAG oficial)",
        ["-m", "src.finetuning.dataset_prep", "--build", "--synthetic-oversample", "5"],
    )

    backend_env = {"MEDICAL_LLM_BACKEND": "auto"}
    if args.mode == "local":
        _run("3/6 Fine-tuning real de validação em CPU", ["-m", "src.finetuning.local_validation"])
        backend_env = {"MEDICAL_LLM_BACKEND": "local"}
    elif args.mode == "lora-dry-run":
        _run("3/6 Validação do pipeline LoRA", ["-m", "src.finetuning.train_lora", "--dry-run"])
    elif args.mode == "lora-real":
        _run("3/6 Fine-tuning LoRA real", ["-m", "src.finetuning.train_lora"])
        backend_env = {"MEDICAL_LLM_BACKEND": "finetuned"}
    else:
        print("\n=== 3/6 Treino pulado por --mode skip-train ===")

    _run("4/6 Diagrama do fluxo", ["scripts/gen_diagrama_fluxo.py"])
    _run("5/6 Avaliação do assistente", ["-m", "src.assistant.finetuning.evaluate"], env=backend_env)

    paciente = _choose_patient(args.paciente)
    if not paciente:
        raise RuntimeError("A base estruturada não possui pacientes para executar o fluxo.")
    _run(
        f"6/6 Fluxo clínico para {paciente}",
        ["-m", "src.assistant.cli", "fluxo", "--paciente", paciente,
         "--pergunta", "Quais critérios de gravidade do protocolo devem ser revisados neste caso?"],
        env=backend_env,
    )

    print("\n[ok] Pipeline da Fase 3 concluído sem regenerar/sobrescrever a base de pacientes.")


if __name__ == "__main__":
    main()
