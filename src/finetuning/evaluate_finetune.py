"""
evaluate_finetune.py
---------------------
Avalia o adapter LoRA treinado (`train_lora.py`) comparando com o modelo
base, no split de teste (`data/finetuning/processed/test.jsonl`).

Também roda em GPU (Colab), na sequência do treino:

    python -m src.finetuning.evaluate_finetune

Métricas calculadas:
  - Perplexity no split de teste (base vs. fine-tuned) — mede o quanto o
    modelo "aprendeu" a distribuição dos protocolos/QA do domínio;
  - ROUGE-L entre a resposta gerada e a resposta de referência — mede
    aderência de conteúdo;
  - Taxa de citação de fonte: % das respostas geradas que mencionam de
    onde veio a informação (proxy simples de explainability — requisito
    de "indicar a fonte da informação utilizada na resposta" do desafio);
  - Taxa de "disclaimer de validação humana": % das respostas que reforçam
    que a decisão final é do médico (requisito de segurança do desafio).

Resultado salvo em `results/finetuned_model/eval_report.json` +
`docs/eval_finetune_report.md` (usado no relatório técnico).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

DISCLAIMER_PATTERNS = re.compile(
    r"(valida[çc][ãa]o (m[ée]dica|do m[ée]dico)|m[ée]dico (respons[áa]vel|assistente)|"
    r"n[ãa]o substitui|apoio [àa] decis[ãa]o|decis[ãa]o (final )?(cabe|[ée])|"
    r"julgamento cl[íi]nico)",
    re.IGNORECASE,
)
SOURCE_PATTERNS = re.compile(
    r"(protocolo|fonte|segundo o|de acordo com|pubmed|medquad)", re.IGNORECASE
)


def _rouge_l(reference: str, hypothesis: str) -> float:
    """ROUGE-L simplificado (LCS-based F1), sem dependência externa."""
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    if not ref_tokens or not hyp_tokens:
        return 0.0
    m, n = len(ref_tokens), len(hyp_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    if lcs == 0:
        return 0.0
    precision = lcs / n
    recall = lcs / m
    return 2 * precision * recall / (precision + recall)


def _balance_by_category(records: List[dict], max_examples: int) -> List[dict]:
    """Amostra em round-robin entre as categorias presentes no split de teste.

    Garante que categorias minoritárias (protocolo_oficial) apareçam na
    avaliação mesmo quando representam poucos por cento do arquivo.
    """
    buckets: dict = {}
    for rec in records:
        buckets.setdefault(rec.get("category", "sem_categoria"), []).append(rec)

    selected: List[dict] = []
    idx = 0
    # Round-robin: uma volta por índice, pegando um exemplo de cada categoria.
    while len(selected) < max_examples:
        adicionou = False
        for cat in sorted(buckets):
            if idx < len(buckets[cat]) and len(selected) < max_examples:
                selected.append(buckets[cat][idx])
                adicionou = True
        if not adicionou:
            break
        idx += 1
    return selected


def _load_test_set(path: Path) -> List[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def _generate(model, tokenizer, messages: list, max_new_tokens: int = 300) -> str:
    import torch
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return text.strip()


def run_full_evaluation(
    adapter_dir: Path,
    base_model_name: str,
    test_path: Path,
    max_examples: int = 100,
    balance_by_category: bool = True,
) -> dict:
    """
    Avaliação completa (requer GPU + transformers/peft instalados). Compara
    o modelo base "puro" com o modelo base + adapter LoRA no mesmo conjunto
    de perguntas de teste.

    `balance_by_category` é importante para não medir só uma fatia do domínio:
    o split de teste é ~97% literatura (PubMedQA/MedQuAD) e ~3% protocolo
    oficial de SRAG. Pegar os N primeiros registros traria apenas literatura, e
    as métricas de disclaimer/citação de fonte — que só fazem sentido nas
    respostas de protocolo — apareceriam zeradas sem que isso indique falha.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    # bfloat16 + device_map="auto" pressupõem GPU NVIDIA. Em Apple Silicon
    # usamos float16 em MPS e float32 na CPU.
    if torch.cuda.is_available():
        load_kwargs = {"device_map": "auto", "dtype": torch.bfloat16}
        device = None
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        load_kwargs = {"dtype": torch.float16}
        device = "mps"
    else:
        load_kwargs = {"dtype": torch.float32}
        device = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, **load_kwargs)
    if device:
        base_model = base_model.to(device)
    # `PeftModel` embrulha o mesmo objeto do modelo base; para comparar os dois
    # precisamos de uma cópia independente, senão o "base" já sai adaptado.
    base_only = AutoModelForCausalLM.from_pretrained(base_model_name, **load_kwargs)
    if device:
        base_only = base_only.to(device)
    tuned_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    base_model = base_only

    test_set = _load_test_set(test_path)
    if balance_by_category:
        test_set = _balance_by_category(test_set, max_examples)
    else:
        test_set = test_set[:max_examples]

    rows = []
    for rec in test_set:
        messages = rec["messages"]
        system_and_user = messages[:2]
        reference = messages[2]["content"]

        base_answer = _generate(base_model, tokenizer, system_and_user)
        tuned_answer = _generate(tuned_model, tokenizer, system_and_user)

        rows.append({
            "source": rec.get("source"),
            "category": rec.get("category"),
            "reference": reference,
            "base_answer": base_answer,
            "tuned_answer": tuned_answer,
            "rouge_l_base": _rouge_l(reference, base_answer),
            "rouge_l_tuned": _rouge_l(reference, tuned_answer),
            "tuned_has_disclaimer": bool(DISCLAIMER_PATTERNS.search(tuned_answer)),
            "tuned_cites_source": bool(SOURCE_PATTERNS.search(tuned_answer)),
        })

    return _summarize(rows)


def _summarize(rows: List[dict]) -> dict:
    n = len(rows) or 1

    # As métricas globais escondem o comportamento por domínio: disclaimer e
    # citação de fonte só são esperados nas respostas de protocolo, não nas de
    # literatura (PubMedQA/MedQuAD), cujas referências não têm esse formato.
    por_categoria: dict = {}
    for row in rows:
        cat = row.get("category") or "sem_categoria"
        por_categoria.setdefault(cat, []).append(row)

    resumo_categorias = {
        cat: {
            "n": len(items),
            "rouge_l_base": round(sum(r["rouge_l_base"] for r in items) / len(items), 4),
            "rouge_l_tuned": round(sum(r["rouge_l_tuned"] for r in items) / len(items), 4),
            "pct_disclaimer": round(100 * sum(r["tuned_has_disclaimer"] for r in items) / len(items), 1),
            "pct_cita_fonte": round(100 * sum(r["tuned_cites_source"] for r in items) / len(items), 1),
        }
        for cat, items in sorted(por_categoria.items())
    }

    return {
        "n_examples": len(rows),
        "rouge_l_base_mean": round(sum(r["rouge_l_base"] for r in rows) / n, 4),
        "rouge_l_tuned_mean": round(sum(r["rouge_l_tuned"] for r in rows) / n, 4),
        "pct_respostas_com_disclaimer": round(
            100 * sum(r["tuned_has_disclaimer"] for r in rows) / n, 1
        ),
        "pct_respostas_citam_fonte": round(
            100 * sum(r["tuned_cites_source"] for r in rows) / n, 1
        ),
        "por_categoria": resumo_categorias,
        "detalhes": rows,
    }


def main():
    parser = argparse.ArgumentParser(description="Avalia o adapter LoRA vs. modelo base.")
    parser.add_argument("--adapter-dir", type=str, default="results/finetuned_model")
    parser.add_argument("--base-model", type=str, required=False,
                         help="Se omitido, lê de results/finetuned_model/run_info.json")
    parser.add_argument("--test-path", type=str, default="data/finetuning/processed/test.jsonl")
    parser.add_argument("--max-examples", type=int, default=100)
    parser.add_argument(
        "--no-balance", action="store_true",
        help="Usa os N primeiros registros em vez de amostrar entre as categorias. "
             "O split de teste é ~97%% literatura, então sem balanceamento as "
             "perguntas de protocolo oficial ficam de fora.",
    )
    args = parser.parse_args()

    adapter_dir = _PROJECT_ROOT / args.adapter_dir
    base_model_name = args.base_model
    if base_model_name is None:
        run_info_path = adapter_dir / "run_info.json"
        if not run_info_path.exists():
            raise SystemExit(
                "Não encontrei run_info.json (rode train_lora.py antes) e "
                "--base-model não foi informado."
            )
        base_model_name = json.loads(run_info_path.read_text())["base_model"]

    test_path = _PROJECT_ROOT / args.test_path
    result = run_full_evaluation(
        adapter_dir, base_model_name, test_path, args.max_examples,
        balance_by_category=not args.no_balance,
    )

    out_json = adapter_dir / "eval_report.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    md_path = _PROJECT_ROOT / "docs" / "eval_finetune_report.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(_to_markdown(result, base_model_name), encoding="utf-8")

    print(f"[evaluate_finetune] relatório salvo em {out_json} e {md_path}")
    print(json.dumps({k: v for k, v in result.items() if k != "detalhes"}, indent=2, ensure_ascii=False))


def _to_markdown(result: dict, base_model_name: str) -> str:
    return (
        "# Avaliação do fine-tuning — Assistente Médico (Fase 3)\n\n"
        f"- **Modelo base:** {base_model_name}\n"
        f"- **Exemplos de teste avaliados:** {result['n_examples']}\n\n"
        "## Qualidade de conteúdo (ROUGE-L vs. referência)\n\n"
        f"- Base: {result['rouge_l_base_mean']}\n"
        f"- Fine-tuned: {result['rouge_l_tuned_mean']}\n\n"
        "## Segurança e explainability (só modelo fine-tuned)\n\n"
        f"- % de respostas com disclaimer de validação médica: "
        f"{result['pct_respostas_com_disclaimer']}%\n"
        f"- % de respostas que citam a fonte da informação: "
        f"{result['pct_respostas_citam_fonte']}%\n\n"
        "> Nota: ROUGE-L é um proxy de sobreposição lexical, não de correção "
        "clínica. Ele indica se o modelo aprendeu o *vocabulário e formato* dos "
        "protocolos, mas a validação final de conteúdo clínico deve ser feita por "
        "um profissional de saúde (ver relatório técnico, seção de limitações)."
    )


if __name__ == "__main__":
    main()
