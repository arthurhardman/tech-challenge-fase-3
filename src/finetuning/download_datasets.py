"""
download_datasets.py
---------------------
Baixa os dados brutos usados pelo `dataset_prep.py`, para reprodutibilidade
(o repositório versiona apenas o dataset já processado — os brutos são
grandes e são recriados por este script).

  - MedQuAD (NIH/NIDDK/NINDS/CDC) — clonado via git (subpastas selecionadas,
    focadas em conteúdo de doenças respiratórias/infecciosas, coerente com o
    domínio SRAG da Fase 1/2, mas o parser em dataset_prep.py aceita
    qualquer subpasta do MedQuAD);
  - PubMedQA — baixa o split rotulado (`ori_pqal.json`) via git clone raso.

Uso:
    python -m src.finetuning.download_datasets
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_RAW_DIR = _PROJECT_ROOT / "data" / "finetuning" / "raw"

# Subpastas do MedQuAD que baixamos (mantém o download rápido; o parser
# funciona com qualquer subconjunto — edite esta lista para pegar mais).
MEDQUAD_SUBFOLDERS = ["6_NINDS_QA", "5_NIDDK_QA", "9_CDC_QA"]


def _run(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def download_medquad(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        _run(["git", "clone", "--depth", "1", "https://github.com/abachaa/MedQuAD.git", tmp])
        for folder in MEDQUAD_SUBFOLDERS:
            src = Path(tmp) / folder
            if src.exists():
                shutil.copytree(src, dest / folder, dirs_exist_ok=True)
    n = len(list(dest.rglob("*.xml")))
    print(f"[download_datasets] MedQuAD: {n} arquivos XML em {dest}")


def download_pubmedqa(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        _run(["git", "clone", "--depth", "1", "https://github.com/pubmedqa/pubmedqa.git", tmp])
        for fname in ["ori_pqal.json", "test_ground_truth.json"]:
            src = Path(tmp) / "data" / fname
            if src.exists():
                shutil.copy2(src, dest / fname)
    print(f"[download_datasets] PubMedQA salvo em {dest}")


def main() -> None:
    try:
        download_medquad(_RAW_DIR / "medquad")
        download_pubmedqa(_RAW_DIR / "pubmedqa")
    except subprocess.CalledProcessError as exc:
        print(
            f"Falha ao clonar via git ({exc}). Baixe manualmente:\n"
            "  MedQuAD:  https://github.com/abachaa/MedQuAD\n"
            "  PubMedQA: https://github.com/pubmedqa/pubmedqa (data/ori_pqal.json)\n"
            f"e coloque em {_RAW_DIR}/medquad/ e {_RAW_DIR}/pubmedqa/ respectivamente.",
            file=sys.stderr,
        )
        raise


if __name__ == "__main__":
    main()
