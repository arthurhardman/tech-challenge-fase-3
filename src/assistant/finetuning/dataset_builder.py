"""
dataset_builder.py
------------------
Monta o dataset de *instruction tuning* a partir das três fontes internas
exigidas pelo enunciado:

  1. Protocolos médicos do hospital        (data/knowledge_base/protocolos/*.md)
  2. Perguntas frequentes de médicos        (faq_medicos.jsonl)
  3. Modelos de laudos/receitas/procedimentos (laudos_modelos.jsonl)

Cada exemplo vira um par {"prompt", "response"} no formato instrução→resposta,
já anonimizado e curado (via data_prep). O resultado é salvo em JSONL, pronto
para o `train.py`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from .. import config
from .data_prep import curar_exemplos, preprocessar

SYSTEM_INSTRUCTION = (
    "Você é um assistente clínico do hospital, especializado em SRAG. Responda com "
    "base nos protocolos internos, cite a fonte quando possível e nunca prescreva "
    "medicação sem validação médica."
)


def _exemplos_de_faq(path: Path) -> List[Dict]:
    exemplos = []
    if not path.exists():
        return exemplos
    for linha in path.read_text(encoding="utf-8").splitlines():
        if not linha.strip():
            continue
        item = json.loads(linha)
        fonte = item.get("fonte", "")
        resposta = item["resposta"]
        exemplos.append(
            {
                "prompt": item["pergunta"],
                "response": resposta,
                "origem": "faq",
                "fonte": fonte,
            }
        )
    return exemplos


def _exemplos_de_laudos(path: Path) -> List[Dict]:
    exemplos = []
    if not path.exists():
        return exemplos
    for linha in path.read_text(encoding="utf-8").splitlines():
        if not linha.strip():
            continue
        item = json.loads(linha)
        exemplos.append(
            {
                "prompt": item["instrucao"],
                "response": item["modelo"],
                "origem": f"modelo_{item.get('tipo', 'documento')}",
                "fonte": item.get("titulo", ""),
            }
        )
    return exemplos


def _exemplos_de_protocolos(diretorio: Path) -> List[Dict]:
    """
    Deriva pares Q&A dos protocolos: cada seção '## Título' vira uma pergunta
    'O que diz o protocolo sobre <título>?' com a resposta = corpo da seção.
    Isso ensina o modelo a recuperar conteúdo dos protocolos internos.
    """
    exemplos: List[Dict] = []
    if not diretorio.exists():
        return exemplos
    for md in sorted(diretorio.glob("*.md")):
        texto = md.read_text(encoding="utf-8")
        cod = md.stem.split("_")[0]  # ex.: PROT-SRAG-01
        secoes = texto.split("\n## ")
        for sec in secoes[1:]:
            linhas = sec.splitlines()
            titulo = linhas[0].split(". ", 1)[-1].strip()
            corpo = "\n".join(linhas[1:]).strip()
            if len(corpo) < 30:
                continue
            exemplos.append(
                {
                    "prompt": f"O que o protocolo {cod} orienta sobre {titulo.lower()}?",
                    "response": corpo,
                    "origem": "protocolo",
                    "fonte": cod,
                }
            )
    return exemplos


def _exemplos_oficiais(path: Path) -> List[Dict]:
    """Carrega as perguntas curadas dos protocolos oficiais com página de origem."""
    exemplos = []
    if not path.exists():
        return exemplos
    for linha in path.read_text(encoding="utf-8").splitlines():
        if not linha.strip():
            continue
        item = json.loads(linha)
        fonte = item.get("source", "protocolo_oficial")
        page = item.get("page")
        exemplos.append({
            "prompt": item.get("question", ""),
            "response": item.get("answer", ""),
            "origem": "protocolo_oficial",
            "fonte": f"{fonte}, p. {page}" if page is not None else fonte,
            "idioma": "pt",
        })
    return exemplos


def build_dataset(salvar: bool = True, incluir_externos: bool = True) -> List[Dict]:
    """
    Constrói o dataset completo (curado + anonimizado) e opcionalmente salva em
    JSONL em `config.DATASET_PATH`. Retorna a lista de exemplos.

    Abordagem híbrida: combina os dados internos do hospital (SRAG, PT-BR) com os
    datasets sugeridos PubMedQA + MedQuAD (EN), quando as fatias estão presentes.
    """
    config.ensure_dirs()
    brutos = (
        _exemplos_de_faq(config.FAQ_PATH)
        + _exemplos_de_laudos(config.LAUDOS_PATH)
        + _exemplos_de_protocolos(config.PROTOCOLOS_DIR)
        + _exemplos_oficiais(config.OFFICIAL_FAQ_PATH)
    )
    curados = curar_exemplos(brutos)

    # Injeta a instrução de sistema (formato chat) já pré-processada.
    for ex in curados:
        ex["system"] = preprocessar(SYSTEM_INSTRUCTION)
        ex.setdefault("idioma", "pt")

    # Datasets externos (PubMedQA + MedQuAD) — já vêm curados/anonimizados.
    if incluir_externos:
        from .external_datasets import carregar_externos
        curados += carregar_externos()

    if salvar:
        with open(config.DATASET_PATH, "w", encoding="utf-8") as f:
            for ex in curados:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    return curados


if __name__ == "__main__":
    ds = build_dataset(salvar=True)
    print(f"[ok] {len(ds)} exemplos de instruction tuning → {config.DATASET_PATH}")
    origens: Dict[str, int] = {}
    for e in ds:
        origens[e["origem"]] = origens.get(e["origem"], 0) + 1
    for k, v in sorted(origens.items()):
        print(f"     {k}: {v}")
