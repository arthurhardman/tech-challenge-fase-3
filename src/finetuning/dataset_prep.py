"""
dataset_prep.py
----------------
Preparação do dataset de fine-tuning do assistente médico (Fase 3).

Combina três fontes, todas convertidas para um formato único de
instruction-tuning (estilo Alpaca: instruction / input / output):

  1. MedQuAD  — pares pergunta/resposta de saúde extraídos de fontes do
     NIH (NINDS, NIDDK, CDC), em XML. Fonte pública, sem dados de pacientes.
  2. PubMedQA — perguntas clínicas de pesquisa com contexto (abstract) e
     resposta longa, derivadas de publicações do PubMed. Fonte pública.
  3. Protocolos internos (sintéticos) — exemplos SINTÉTICOS no estilo dos
     documentos que um hospital real usaria (protocolos de conduta, FAQ de
     médicos, modelos de laudo/receita), no domínio de SRAG — o mesmo
     domínio clínico da Fase 1/2 deste projeto. Nenhum dado de paciente
     real é usado; tudo é gerado para fins de demonstração do pipeline.
     Em produção, este bloco seria substituído pelos documentos reais do
     hospital, passando pela mesma etapa de anonimização abaixo.

Etapas de curadoria aplicadas a TODAS as fontes:
  - normalização de texto (espaços, entidades HTML/XML residuais);
  - remoção de duplicatas (por hash do par pergunta+resposta);
  - filtro de tamanho (remove pares vazios ou longos demais para o contexto
    do modelo alvo);
  - anonimização: varredura por regex de padrões de PII/PHI (nomes de
    pacientes, CPF, prontuário, telefone, e-mail, datas de nascimento) —
    essencial para quando dados reais do hospital forem usados no lugar dos
    exemplos sintéticos;
  - split determinístico em train / val / test.

Uso:
    python -m src.finetuning.dataset_prep --build
    (ou `from src.finetuning.dataset_prep import build_dataset`)
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import random
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Optional
from xml.etree import ElementTree as ET

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_RAW_DIR = _PROJECT_ROOT / "data" / "finetuning" / "raw"
_PROCESSED_DIR = _PROJECT_ROOT / "data" / "finetuning" / "processed"

SYSTEM_PROMPT = (
    "Você é um assistente clínico do hospital, treinado com protocolos internos "
    "e literatura médica. Responda de forma objetiva, cite a fonte do protocolo "
    "quando disponível, e NUNCA prescreva um tratamento definitivo sem indicar "
    "que a decisão final exige validação de um médico responsável."
)


# ---------------------------------------------------------------------------
# Estrutura unificada de exemplo
# ---------------------------------------------------------------------------
@dataclass
class InstructionExample:
    instruction: str          # a pergunta / tarefa
    input: str                 # contexto adicional (pode ser vazio)
    output: str                 # resposta esperada
    source: str                 # proveniência, para explainability/auditoria
    category: str = "geral"     # protocolo | faq | laudo | receita | literatura

    def key(self) -> str:
        raw = f"{self.instruction.strip().lower()}|{self.output.strip().lower()}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def to_prompt_format(self) -> dict:
        """Formato de treino (chat-style), pronto para o `train_lora.py`."""
        return {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": self._user_turn()},
                {"role": "assistant", "content": self.output.strip()},
            ],
            "source": self.source,
            "category": self.category,
        }

    def _user_turn(self) -> str:
        if self.input.strip():
            return f"{self.instruction.strip()}\n\nContexto:\n{self.input.strip()}"
        return self.instruction.strip()


# ---------------------------------------------------------------------------
# Limpeza / anonimização
# ---------------------------------------------------------------------------
_WS_RE = re.compile(r"[ \t]+")
_NL_RE = re.compile(r"\n{3,}")

# Padrões de PII/PHI — usados para higienizar QUALQUER dado real que venha a
# substituir os exemplos sintéticos (CPF, telefone BR, e-mail, prontuário,
# nomes precedidos de "Sr./Sra./Dr.", datas no formato dd/mm/aaaa).
_PII_PATTERNS = [
    (re.compile(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b"), "[CPF_REMOVIDO]"),
    (re.compile(r"\b\(?\d{2}\)?[\s-]?9?\d{4}[\s-]?\d{4}\b"), "[TELEFONE_REMOVIDO]"),
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), "[EMAIL_REMOVIDO]"),
    (re.compile(r"\b(?:prontu[áa]rio|registro)\s*n?[ºo°]?\s*[:\-]?\s*\d{4,}\b", re.I),
     "[PRONTUARIO_REMOVIDO]"),
    (re.compile(r"\b(?:Sr\.?|Sra\.?|Dr\.?|Dra\.?)\s+[A-ZÀ-Ý][a-zà-ÿ]+(?:\s+[A-ZÀ-Ý][a-zà-ÿ]+)*"),
     "[NOME_REMOVIDO]"),
    (re.compile(r"\b\d{2}/\d{2}/\d{4}\b"), "[DATA_REMOVIDA]"),
]


def anonymize(text: str) -> str:
    """Aplica os padrões de PII/PHI acima. Idempotente e segura para texto sem PII."""
    for pattern, replacement in _PII_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def clean_text(text: str) -> str:
    """Normaliza espaços/quebras e decodifica entidades HTML/XML residuais."""
    text = html.unescape(text or "")
    text = _WS_RE.sub(" ", text)
    text = _NL_RE.sub("\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Fonte 1: MedQuAD (XML)
# ---------------------------------------------------------------------------
def load_medquad(raw_dir: Optional[Path] = None) -> List[InstructionExample]:
    raw_dir = Path(raw_dir or _RAW_DIR / "medquad")
    examples: List[InstructionExample] = []
    for xml_path in sorted(raw_dir.rglob("*.xml")):
        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError:
            continue
        focus = (root.findtext("Focus") or "").strip()
        source = root.attrib.get("source", "MedQuAD")
        for qa in root.iter("QAPair"):
            q = qa.findtext("Question")
            a = qa.findtext("Answer")
            if not q or not a:
                continue
            q, a = clean_text(q), clean_text(a)
            if len(a) < 20 or len(a) > 4000:
                continue
            examples.append(
                InstructionExample(
                    instruction=q,
                    input=f"Tópico: {focus}" if focus else "",
                    output=a,
                    source=f"MedQuAD/{source}/{xml_path.name}",
                    category="literatura",
                )
            )
    return examples


# ---------------------------------------------------------------------------
# Fonte 2: PubMedQA (JSON — split rotulado ori_pqal.json)
# ---------------------------------------------------------------------------
def load_pubmedqa(raw_dir: Optional[Path] = None) -> List[InstructionExample]:
    raw_dir = Path(raw_dir or _RAW_DIR / "pubmedqa")
    json_path = raw_dir / "ori_pqal.json"
    if not json_path.exists():
        return []
    data = json.loads(json_path.read_text(encoding="utf-8"))
    examples: List[InstructionExample] = []
    for pmid, rec in data.items():
        question = clean_text(rec.get("QUESTION", ""))
        contexts = rec.get("CONTEXTS") or []
        long_answer = clean_text(rec.get("LONG_ANSWER", ""))
        decision = rec.get("final_decision", "")
        if not question or not long_answer:
            continue
        contexto = clean_text(" ".join(contexts))[:2000]
        resposta = long_answer
        if decision:
            resposta = f"Resposta direta: {decision}. {resposta}"
        examples.append(
            InstructionExample(
                instruction=question,
                input=contexto,
                output=resposta,
                source=f"PubMedQA/PMID:{pmid}",
                category="literatura",
            )
        )
    return examples


# ---------------------------------------------------------------------------
# Fonte 3: Protocolos internos SINTÉTICOS (domínio SRAG, coerente com F1/F2)
# ---------------------------------------------------------------------------
def synthetic_hospital_examples() -> List[InstructionExample]:
    """
    Exemplos sintéticos representando os tipos de documento citados no
    desafio (protocolos internos, FAQ de médicos, modelos de laudo/receita).
    Todos fictícios — nenhum dado de paciente real. Usados aqui para que o
    pipeline de fine-tuning tenha, desde já, exemplos no formato/tom que o
    hospital usaria; substitua por documentos reais (após anonimização)
    quando disponíveis.
    """
    raw = [
        dict(
            instruction="Qual o protocolo interno para paciente com suspeita de SRAG à admissão?",
            input="",
            output=(
                "Protocolo Interno SRAG-01 (fictício): 1) Isolamento respiratório imediato; "
                "2) Coletar SpO2, RT-PCR para vírus respiratórios e radiografia de tórax nas "
                "primeiras 2h; 3) Classificar gravidade pelo escore interno (frequência "
                "respiratória, SpO2 e uso de musculatura acessória); 4) Se SpO2 < 92% em ar "
                "ambiente, iniciar suporte de O2 e acionar a equipe de terapia intensiva; "
                "5) Reavaliação clínica a cada 4h nas primeiras 24h. Toda conduta terapêutica "
                "definitiva deve ser validada pelo médico assistente responsável."
            ),
            source="Protocolo-Interno-SRAG-01 (sintético)",
            category="protocolo",
        ),
        dict(
            instruction="Quais exames são obrigatórios antes de liberar alta de um paciente SRAG internado?",
            input="",
            output=(
                "Protocolo Interno SRAG-04 (fictício): radiografia de tórax de controle, "
                "SpO2 em ar ambiente por 24h estável acima de 94%, hemograma sem sinais de "
                "infecção ativa e ausência de febre por 48h. A alta definitiva é sempre uma "
                "decisão médica; o assistente pode apenas listar as pendências identificadas."
            ),
            source="Protocolo-Interno-SRAG-04 (sintético)",
            category="protocolo",
        ),
        dict(
            instruction="Médico pergunta: em que situações devo suspeitar de coinfecção bacteriana em paciente com SRAG viral?",
            input="",
            output=(
                "Segundo o protocolo interno, suspeitar de coinfecção bacteriana quando houver: "
                "piora clínica após melhora inicial, novo pico febril após 5-7 dias de evolução, "
                "leucocitose com desvio à esquerda, ou infiltrado radiológico novo/em piora. "
                "Nesses casos, o protocolo recomenda coleta de culturas antes de eventual início "
                "de antibioticoterapia empírica, decisão que cabe ao médico assistente."
            ),
            source="FAQ-Medicos-Infeccao (sintético)",
            category="faq",
        ),
        dict(
            instruction="Modelo de laudo de radiografia de tórax para paciente com SRAG (padrão interno).",
            input="",
            output=(
                "Modelo de Laudo (fictício): 'Radiografia de tórax em PA e perfil. Achados: "
                "opacidades [descrever localização e lateralidade], [presença/ausência] de "
                "derrame pleural, área cardíaca [normal/aumentada]. Impressão diagnóstica: "
                "achados [compatíveis/não compatíveis] com acometimento pulmonar por SRAG, "
                "sugere-se correlação clínico-laboratorial.' — Este é um modelo de estrutura; "
                "o preenchimento e a assinatura são sempre de responsabilidade do radiologista."
            ),
            source="Modelo-Laudo-RX-Torax (sintético)",
            category="laudo",
        ),
        dict(
            instruction="Modelo de receita padrão interna para suporte sintomático em SRAG leve (ambulatorial).",
            input="",
            output=(
                "Modelo de Receita (fictício, sujeito à validação médica): "
                "'Paracetamol 750mg — 1 comprimido a cada 6h se febre ou dor, "
                "Hidratação oral abundante, Repouso relativo.' — Este modelo NÃO deve ser "
                "emitido automaticamente; serve apenas como referência de formatação. "
                "Qualquer prescrição real exige assinatura e CRM do médico responsável."
            ),
            source="Modelo-Receita-SRAG-Leve (sintético)",
            category="receita",
        ),
        dict(
            instruction="Médico pergunta: quando devo escalar um caso de SRAG para a UTI segundo o protocolo do hospital?",
            input="",
            output=(
                "Protocolo Interno SRAG-02 (fictício) indica escalonamento para UTI quando "
                "houver pelo menos um dos critérios: SpO2 < 90% mesmo com O2 suplementar, "
                "frequência respiratória > 30irpm, uso importante de musculatura acessória, "
                "instabilidade hemodinâmica ou rebaixamento do nível de consciência. O "
                "assistente pode sinalizar esses critérios como alerta, mas o acionamento da "
                "UTI é sempre confirmado pelo médico assistente."
            ),
            source="Protocolo-Interno-SRAG-02 (sintético)",
            category="protocolo",
        ),
    ]
    return [InstructionExample(**r) for r in raw]


# ---------------------------------------------------------------------------
# Curadoria: dedup + split + persistência
# ---------------------------------------------------------------------------
def deduplicate(examples: Iterable[InstructionExample]) -> List[InstructionExample]:
    seen = set()
    out = []
    for ex in examples:
        k = ex.key()
        if k in seen:
            continue
        seen.add(k)
        out.append(ex)
    return out


def anonymize_examples(examples: List[InstructionExample]) -> List[InstructionExample]:
    for ex in examples:
        ex.instruction = anonymize(ex.instruction)
        ex.input = anonymize(ex.input)
        ex.output = anonymize(ex.output)
    return examples


def split_dataset(
    examples: List[InstructionExample],
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
):
    rng = random.Random(seed)
    shuffled = examples[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_val = int(n * val_frac)
    n_test = int(n * test_frac)
    val = shuffled[:n_val]
    test = shuffled[n_val:n_val + n_test]
    train = shuffled[n_val + n_test:]
    return train, val, test


def _write_jsonl(examples: List[InstructionExample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex.to_prompt_format(), ensure_ascii=False) + "\n")


def build_dataset(
    raw_dir: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    max_medquad: Optional[int] = 1500,
    max_pubmedqa: Optional[int] = 1000,
    synthetic_oversample: int = 20,
    seed: int = 42,
) -> dict:
    """
    Executa o pipeline completo: carrega as 3 fontes, limpa, deduplica,
    anonimiza, faz split e grava train/val/test.jsonl + um dataset_card.

    `max_medquad` / `max_pubmedqa` limitam o tamanho (amostragem
    determinística) para manter o fine-tuning rápido em GPUs modestas;
    use `None` para usar tudo que estiver em `data/finetuning/raw/`.

    `synthetic_oversample`: os exemplos de protocolo interno são poucos
    (poucas dezenas) perto dos milhares de pares de literatura pública. Sem
    algum peso extra, o fine-tuning aprenderia basicamente "responder como o
    MedQuAD/PubMedQA" e ignoraria o tom/formato dos protocolos do hospital —
    que é o ponto central do desafio. Por isso os exemplos sintéticos são
    repetidos `synthetic_oversample` vezes SOMENTE no split de treino (val/test
    usam os exemplos originais, sem repetição, para não inflar a métrica).
    Em produção, com centenas/milhares de documentos internos reais, isso não
    seria necessário.
    """
    raw_dir = Path(raw_dir or _RAW_DIR)
    out_dir = Path(out_dir or _PROCESSED_DIR)
    rng = random.Random(seed)

    medquad = load_medquad(raw_dir / "medquad")
    pubmedqa = load_pubmedqa(raw_dir / "pubmedqa")
    synthetic = synthetic_hospital_examples()

    if max_medquad is not None and len(medquad) > max_medquad:
        medquad = rng.sample(medquad, max_medquad)
    if max_pubmedqa is not None and len(pubmedqa) > max_pubmedqa:
        pubmedqa = rng.sample(pubmedqa, max_pubmedqa)

    public_examples = deduplicate(medquad + pubmedqa)
    synthetic = deduplicate(synthetic)
    public_examples = anonymize_examples(public_examples)
    synthetic = anonymize_examples(synthetic)

    # Split feito só sobre os dados públicos + 1 cópia dos sintéticos, para
    # que val/test reflitam a distribuição real (sem oversampling).
    train, val, test = split_dataset(public_examples + synthetic, seed=seed)

    # Oversampling aplicado apenas ao treino, e apenas às linhas sintéticas
    # que caíram no split de treino.
    synthetic_keys = {ex.key() for ex in synthetic}
    train_synthetic = [ex for ex in train if ex.key() in synthetic_keys]
    train_public = [ex for ex in train if ex.key() not in synthetic_keys]
    train = train_public + train_synthetic * synthetic_oversample
    rng.shuffle(train)

    all_examples = train_public + train_synthetic + val + test  # para stats "reais"

    _write_jsonl(train, out_dir / "train.jsonl")
    _write_jsonl(val, out_dir / "val.jsonl")
    _write_jsonl(test, out_dir / "test.jsonl")

    stats = {
        "total_exemplos_unicos": len(all_examples),
        "train_linhas_gravadas": len(train),
        "train_exemplos_unicos": len(train_public) + len(train_synthetic),
        "val": len(val),
        "test": len(test),
        "synthetic_oversample_factor": synthetic_oversample,
        "por_fonte": {
            "MedQuAD": len(medquad),
            "PubMedQA": len(pubmedqa),
            "Protocolos internos (sintéticos)": len(synthetic),
        },
        "por_categoria": _count_by(all_examples, "category"),
        "seed": seed,
    }
    (out_dir / "dataset_card.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_dataset_card_md(stats, out_dir / "dataset_card.md")
    return stats


def _count_by(examples: List[InstructionExample], attr: str) -> dict:
    out: dict = {}
    for ex in examples:
        v = getattr(ex, attr)
        out[v] = out.get(v, 0) + 1
    return out


def _write_dataset_card_md(stats: dict, path: Path) -> None:
    lines = [
        "# Dataset Card — Assistente Médico (Fase 3)",
        "",
        "Dataset de instruction-tuning combinando fontes públicas e exemplos",
        "sintéticos no estilo de documentos internos hospitalares.",
        "",
        f"- **Total de exemplos únicos:** {stats['total_exemplos_unicos']}",
        f"- **Val / Test:** {stats['val']} / {stats['test']} (sem oversampling)",
        f"- **Train (linhas gravadas em train.jsonl):** {stats['train_linhas_gravadas']} "
        f"({stats['train_exemplos_unicos']} exemplos únicos; os "
        f"{stats['por_fonte']['Protocolos internos (sintéticos)']} exemplos de protocolo "
        f"interno foram repetidos {stats['synthetic_oversample_factor']}x no treino — ver "
        "nota abaixo)",
        "",
        "## Por fonte",
        "",
    ]
    for k, v in stats["por_fonte"].items():
        lines.append(f"- {k}: {v}")
    lines += ["", "## Por categoria", ""]
    for k, v in stats["por_categoria"].items():
        lines.append(f"- {k}: {v}")
    lines += [
        "",
        "## Observações sobre privacidade",
        "",
        "- MedQuAD e PubMedQA são datasets públicos, sem dados identificáveis de pacientes.",
        "- Os exemplos de \"protocolo interno\", \"faq\", \"laudo\" e \"receita\" são "
        "**sintéticos/fictícios**, criados apenas para dar ao pipeline o formato que os "
        "documentos reais do hospital teriam.",
        "- Todas as fontes passam pela mesma etapa de anonimização "
        "(`src/finetuning/dataset_prep.py::anonymize`) antes de entrar no dataset final — "
        "isso é o que garante que, ao trocar os exemplos sintéticos por documentos reais do "
        "hospital, CPF, telefone, e-mail, nome de paciente e número de prontuário sejam "
        "removidos automaticamente.",
        "",
        "## Nota sobre o oversampling dos protocolos internos",
        "",
        "Os exemplos de protocolo/FAQ/laudo/receita são propositalmente poucos neste "
        "repositório de demonstração (ver `synthetic_hospital_examples()`), então foram "
        "repetidos apenas no split de treino para o fine-tuning não ignorá-los diante do "
        "volume de dados públicos. Val/test usam os exemplos originais, sem repetição, "
        "para as métricas não ficarem infladas. Com documentos reais do hospital em maior "
        "volume, o oversampling deixa de ser necessário (ajuste `synthetic_oversample=1`).",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Prepara o dataset de fine-tuning (Fase 3).")
    parser.add_argument("--build", action="store_true", help="Executa o pipeline completo.")
    parser.add_argument("--max-medquad", type=int, default=1500)
    parser.add_argument("--max-pubmedqa", type=int, default=1000)
    parser.add_argument("--synthetic-oversample", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.build:
        stats = build_dataset(
            max_medquad=args.max_medquad,
            max_pubmedqa=args.max_pubmedqa,
            synthetic_oversample=args.synthetic_oversample,
            seed=args.seed,
        )
        print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
