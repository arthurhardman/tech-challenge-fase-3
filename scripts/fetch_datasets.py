"""
fetch_datasets.py
-----------------
Baixa os datasets externos sugeridos no enunciado para `data/external/`:

  - PubMedQA  (github.com/pubmedqa/pubmedqa) — o subconjunto rotulado
    `ori_pqal.json` já vem no repositório.
  - MedQuAD   (github.com/abachaa/MedQuAD)   — XMLs de QAs do NIH.

Após baixar, gere as fatias curadas (offline) com:
    python -m src.assistant.finetuning.external_datasets

Requer `git` instalado. Os dados brutos NÃO são versionados; apenas as fatias
curadas em data/knowledge_base/external/ vão para o repositório.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "data" / "external"

REPOS = {
    "pubmedqa": "https://github.com/pubmedqa/pubmedqa.git",
    "medquad": "https://github.com/abachaa/MedQuAD.git",
}


def main() -> None:
    EXTERNAL.mkdir(parents=True, exist_ok=True)
    for nome, url in REPOS.items():
        destino = EXTERNAL / nome
        if destino.exists():
            print(f"[skip] {nome} já existe em {destino}")
            continue
        print(f"[baixando] {nome} ← {url}")
        subprocess.run(["git", "clone", "--depth", "1", url, str(destino)], check=True)
        print(f"[ok] {nome} em {destino}")
    print("\nPróximo passo: python -m src.assistant.finetuning.external_datasets")


if __name__ == "__main__":
    main()
