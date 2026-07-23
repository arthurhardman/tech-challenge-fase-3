"""
external_datasets.py
--------------------
Integra os datasets sugeridos no enunciado — **PubMedQA** e **MedQuAD** — ao
projeto (abordagem híbrida: hospital SRAG em PT-BR + estes datasets).

Converte ambos para o mesmo formato de *instruction tuning* usado no projeto
(`{system, prompt, response, origem, fonte, idioma}`), aplicando anonimização e
curadoria. Gera "fatias" curadas e enxutas em `data/knowledge_base/external/`,
que são versionadas para que o pipeline rode **offline** (sem baixar nada).

Fontes / licenças (atribuição obrigatória — ver CITATIONS.md):
  - PubMedQA (Jin et al., 2019) — licença MIT. Usamos o subconjunto rotulado
    `ori_pqal.json` (1.000 QAs de pesquisa com resposta longa + veredito).
  - MedQuAD (Ben Abacha & Demner-Fushman, 2019) — licença CC BY 4.0. Usamos os
    9 subconjuntos do NIH que mantêm as respostas (excluímos ADAM, MedlinePlus
    Drugs e Herbs, cujas respostas foram removidas por copyright do MedlinePlus).

Para regenerar as fatias a partir dos dados brutos:
    python scripts/fetch_datasets.py          # baixa os brutos
    python -m src.assistant.finetuning.external_datasets   # gera as fatias
"""

from __future__ import annotations

import glob
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List

from .. import config
from .data_prep import curar_exemplos, preprocessar

# --- Dados brutos (baixados por scripts/fetch_datasets.py; podem não existir) - #
EXTERNAL_RAW = config.PROJECT_ROOT / "data" / "external"
PUBMEDQA_RAW = EXTERNAL_RAW / "pubmedqa" / "data" / "ori_pqal.json"
MEDQUAD_RAW = EXTERNAL_RAW / "medquad"

# --- Fatias curadas (versionadas; rodam offline) ---------------------------- #
EXTERNAL_KB = config.KB_DIR / "external"
PUBMEDQA_SLICE = EXTERNAL_KB / "pubmedqa.jsonl"
MEDQUAD_SLICE = EXTERNAL_KB / "medquad.jsonl"

SYSTEM_EN = (
    "You are a clinical assistant. Answer based on medical evidence and cite the "
    "source. Never prescribe medication or a dose without human validation."
)

# Subconjuntos do MedQuAD que mantêm respostas (excluídos 10/11/12 por copyright).
_MEDQUAD_SUBSETS_OK = {
    "1_CancerGov_QA", "2_GARD_QA", "3_GHR_QA", "4_MPlus_Health_Topics_QA",
    "5_NIDDK_QA", "6_NINDS_QA", "7_SeniorHealth_QA", "8_NHLBI_QA_XML", "9_CDC_QA",
}
_MEDQUAD_FONTE = {
    "1_CancerGov_QA": "MedQuAD:CancerGov", "2_GARD_QA": "MedQuAD:GARD",
    "3_GHR_QA": "MedQuAD:GHR", "4_MPlus_Health_Topics_QA": "MedQuAD:MedlinePlus",
    "5_NIDDK_QA": "MedQuAD:NIDDK", "6_NINDS_QA": "MedQuAD:NINDS",
    "7_SeniorHealth_QA": "MedQuAD:NIHSeniorHealth", "8_NHLBI_QA_XML": "MedQuAD:NHLBI",
    "9_CDC_QA": "MedQuAD:CDC",
}


# --------------------------------------------------------------------------- #
# Loaders dos dados brutos
# --------------------------------------------------------------------------- #
def load_pubmedqa(max_itens: int = 250, max_chars: int = 800) -> List[Dict]:
    """Converte o PubMedQA rotulado (ori_pqal.json) em exemplos de instrução."""
    if not PUBMEDQA_RAW.exists():
        return []
    data = json.load(open(PUBMEDQA_RAW, encoding="utf-8"))
    exemplos: List[Dict] = []
    for pmid, item in list(data.items())[:max_itens]:
        pergunta = (item.get("QUESTION") or "").strip()
        longa = (item.get("LONG_ANSWER") or "").strip()
        decisao = (item.get("final_decision") or "").strip()
        if not pergunta or not longa:
            continue
        resposta = f"{decisao.capitalize()}. {longa}" if decisao else longa
        exemplos.append({
            "prompt": pergunta,
            "response": resposta[:max_chars],
            "system": SYSTEM_EN,
            "origem": "pubmedqa",
            "fonte": f"PubMedQA:PMID{pmid}",
            "idioma": "en",
        })
    return exemplos


def load_medquad(max_itens: int = 350, max_chars: int = 700) -> List[Dict]:
    """Converte os XMLs do MedQuAD (subsets com resposta) em exemplos de instrução."""
    if not MEDQUAD_RAW.exists():
        return []
    exemplos: List[Dict] = []
    for subset in sorted(_MEDQUAD_SUBSETS_OK):
        subdir = MEDQUAD_RAW / subset
        if not subdir.exists():
            continue
        fonte = _MEDQUAD_FONTE.get(subset, "MedQuAD")
        for xml_path in sorted(subdir.glob("*.xml")):
            try:
                root = ET.parse(xml_path).getroot()
            except ET.ParseError:
                continue
            for qa in root.iter("QAPair"):
                q = qa.find("Question")
                a = qa.find("Answer")
                pergunta = (q.text or "").strip() if q is not None else ""
                resposta = (a.text or "").strip() if a is not None else ""
                if not pergunta or len(resposta) < 20:
                    continue
                exemplos.append({
                    "prompt": pergunta,
                    "response": " ".join(resposta.split())[:max_chars],
                    "system": SYSTEM_EN,
                    "origem": "medquad",
                    "fonte": fonte,
                    "idioma": "en",
                })
                if len(exemplos) >= max_itens:
                    return exemplos
    return exemplos


# --------------------------------------------------------------------------- #
# Geração das fatias curadas (versionadas)
# --------------------------------------------------------------------------- #
def _salvar_jsonl(exemplos: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in exemplos:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def build_slices(n_pubmedqa: int = 250, n_medquad: int = 350) -> Dict[str, int]:
    """Gera as fatias curadas a partir dos dados brutos (requer download)."""
    pub = curar_exemplos(load_pubmedqa(n_pubmedqa))
    med = curar_exemplos(load_medquad(n_medquad))
    for ex in pub + med:
        ex["prompt"] = preprocessar(ex["prompt"])
        ex["response"] = preprocessar(ex["response"])
    _salvar_jsonl(pub, PUBMEDQA_SLICE)
    _salvar_jsonl(med, MEDQUAD_SLICE)
    return {"pubmedqa": len(pub), "medquad": len(med)}


# --------------------------------------------------------------------------- #
# Leitura das fatias (usada pelo pipeline; roda offline)
# --------------------------------------------------------------------------- #
def _ler_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def carregar_externos() -> List[Dict]:
    """Retorna os exemplos das fatias versionadas (PubMedQA + MedQuAD)."""
    return _ler_jsonl(PUBMEDQA_SLICE) + _ler_jsonl(MEDQUAD_SLICE)


if __name__ == "__main__":
    stats = build_slices()
    print(f"[ok] fatias geradas em {EXTERNAL_KB}/")
    print(f"     pubmedqa.jsonl: {stats['pubmedqa']} exemplos")
    print(f"     medquad.jsonl : {stats['medquad']} exemplos")
