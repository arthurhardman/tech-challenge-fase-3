"""
run_fase3.py
------------
Orquestra o pipeline completo da Fase 3 com um único comando:

  1. gera a base sintética anonimizada;
  2. constrói o dataset de instruction tuning;
  3. roda o fine-tuning (modo demo por padrão) + curva de perda;
  4. gera o diagrama do fluxo LangChain/LangGraph;
  5. avalia o assistente;
  6. executa uma demonstração do fluxo LangGraph para um paciente.

Uso:
    python run_fase3.py                 # tudo em modo demo (offline)
    python run_fase3.py --mode real     # fine-tuning real (requer GPU + deps)
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def _run(desc: str, args: list[str]) -> None:
    print(f"\n=== {desc} ===")
    subprocess.run([sys.executable, *args], check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Pipeline completo da Fase 3.")
    ap.add_argument("--mode", choices=["demo", "real", "auto"], default="demo")
    ap.add_argument("--paciente", default="PAC-0001")
    args = ap.parse_args()

    _run("1/6 Base sintética", ["scripts/gen_synthetic_data.py"])
    _run("2/6 Dataset de instrução", ["-m", "src.assistant.finetuning.dataset_builder"])
    _run("3/6 Fine-tuning", ["-m", "src.assistant.finetuning.train", "--mode", args.mode])
    _run("4/6 Diagrama do fluxo", ["scripts/gen_diagrama_fluxo.py"])
    _run("5/6 Avaliação", ["-m", "src.assistant.finetuning.evaluate"])
    _run("6/6 Demonstração do fluxo LangGraph",
         ["-m", "src.assistant.cli", "fluxo", "--paciente", args.paciente])

    print("\n[ok] Pipeline da Fase 3 concluído.")


if __name__ == "__main__":
    main()
