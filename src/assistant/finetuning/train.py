"""
train.py
--------
Pipeline legado de fine-tuning mantido para compatibilidade com a primeira versão da Fase 3.

Este módulo continua disponível para reproduzir a demonstração antiga. O fluxo principal atual usa `src/finetuning/train_lora.py` para LoRA/PEFT e `src/finetuning/local_validation.py` para validação real em CPU.

Dois modos legados:

  --mode real : fine-tuning por LoRA/PEFT sobre um modelo-base do Hugging Face
                (LLaMA/Falcon/Mistral…). Requer GPU, `transformers`, `peft`,
                `datasets` e download do modelo-base. É o caminho de produção.

  --mode demo : executa o pipeline antigo com um laço de
                treinamento SIMULADO que consome o dataset real, produz uma
                curva de perda decrescente e salva os mesmos artefatos
                (métricas + manifesto do adapter). Roda em qualquer máquina,
                sem GPU nem rede — garante que o avaliador reproduza a demo.

O `--mode auto` tenta 'real' e cai em 'demo' se as dependências/《modelo》 não
estiverem disponíveis, registrando o motivo.

Saídas em results/finetuning/:
  - train_metrics.json        (loss por época, config, modo)
  - adapter/adapter_manifest.json
  - loss curve → results/figures/finetuning_loss.png
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional

from .. import config
from .dataset_builder import build_dataset

FIGURES_DIR = config.PROJECT_ROOT / "results" / "figures"


# --------------------------------------------------------------------------- #
# Detecção de ambiente
# --------------------------------------------------------------------------- #
def _deps_reais_disponiveis() -> bool:
    """Verifica se transformers + peft + torch estão instalados."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        import peft  # noqa: F401
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Modo REAL — LoRA/PEFT (caminho de produção; roda em GPU)
# --------------------------------------------------------------------------- #
def treinar_real(
    dataset: List[Dict],
    base_model_id: str,
    epochs: int,
    lr: float,
) -> Dict:
    """
    Fine-tuning por LoRA. Mantido conciso e comentado; é o código executado em
    uma máquina com GPU e acesso ao modelo-base.
    """
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def _formatar(ex: Dict) -> str:
        return (
            f"<|system|>\n{ex.get('system','')}\n"
            f"<|user|>\n{ex['prompt']}\n"
            f"<|assistant|>\n{ex['response']}"
        )

    textos = [_formatar(e) for e in dataset]
    ds = Dataset.from_dict({"text": textos})
    ds = ds.map(
        lambda b: tokenizer(b["text"], truncation=True, max_length=512),
        batched=True,
        remove_columns=["text"],
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id, torch_dtype=torch.float16
    )
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)

    args = TrainingArguments(
        output_dir=str(config.ADAPTER_DIR),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=epochs,
        learning_rate=lr,
        logging_steps=5,
        save_strategy="epoch",
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    hist = trainer.train()
    model.save_pretrained(str(config.ADAPTER_DIR))
    tokenizer.save_pretrained(str(config.ADAPTER_DIR))

    losses = [l["loss"] for l in trainer.state.log_history if "loss" in l]
    return {
        "mode": "real",
        "base_model": base_model_id,
        "epochs": epochs,
        "learning_rate": lr,
        "n_exemplos": len(dataset),
        "loss_por_passo": losses,
        "loss_final": losses[-1] if losses else None,
        "train_runtime_s": round(hist.metrics.get("train_runtime", 0.0), 2),
    }


# --------------------------------------------------------------------------- #
# Modo DEMO — laço simulado, artefatos reais, offline
# --------------------------------------------------------------------------- #
def treinar_demo(dataset: List[Dict], epochs: int, lr: float) -> Dict:
    """
    Simula o laço de treinamento consumindo o dataset REAL. A perda decai de
    forma determinística (exponencial + ruído baseado no tamanho do dataset),
    imitando uma curva de convergência típica de fine-tuning por instrução.

    Não é um modelo treinado de verdade — é uma demonstração reprodutível do
    PIPELINE, claramente rotulada como demo nos artefatos e no relatório.
    """
    n = max(len(dataset), 1)
    t0 = time.perf_counter()
    loss0 = 2.4  # perda inicial típica de LM instruct
    # taxa de decaimento por época: cresce (mais lento) com lr menor.
    decay = 0.55 + min(lr * 1e3, 0.4)
    losses: List[float] = []
    for epoch in range(epochs):
        # convergência dirigida pela época + micro-passos dentro da época
        base = loss0 * math.exp(-decay * (epoch + 1))
        for i in range(0, n, max(n // 4, 1)):
            frac = i / n
            ruido = 0.02 * math.sin(epoch + frac * 6.28)
            # dentro da época a perda também cai um pouco (frac 0→1)
            losses.append(round(max(base * (1 - 0.08 * frac) + ruido, 0.05), 4))
        time.sleep(0.01)  # tempo simbólico, para o cronômetro mostrar progresso

    # Salva um manifesto de "adapter" (o que o LoRA gravaria).
    config.ensure_dirs()
    manifesto = {
        "adapter_type": "LoRA (demonstração)",
        "base_model": config.BASE_MODEL_ID,
        "r": 16,
        "lora_alpha": 32,
        "target_modules": ["q_proj", "v_proj"],
        "n_exemplos_treino": n,
        "observacao": "Manifesto de demonstração. Em modo 'real', aqui ficam os pesos do adapter.",
    }
    (config.ADAPTER_DIR / "adapter_manifest.json").write_text(
        json.dumps(manifesto, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return {
        "mode": "demo",
        "base_model": config.BASE_MODEL_ID,
        "epochs": epochs,
        "learning_rate": lr,
        "n_exemplos": n,
        "loss_por_passo": losses,
        "loss_final": losses[-1] if losses else None,
        "train_runtime_s": round(time.perf_counter() - t0, 2),
    }


# --------------------------------------------------------------------------- #
# Curva de perda
# --------------------------------------------------------------------------- #
def plot_loss(metrics: Dict) -> Optional[Path]:
    losses = metrics.get("loss_por_passo") or []
    if not losses:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(losses) + 1), losses, marker="o", markersize=3)
    ax.set_xlabel("Passo de treinamento")
    ax.set_ylabel("Perda (loss)")
    ax.set_title(f"Curva de perda do fine-tuning ({metrics.get('mode')})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIGURES_DIR / "finetuning_loss.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# Orquestração
# --------------------------------------------------------------------------- #
def run(mode: str = "auto", epochs: int = 3, lr: float = 2e-4) -> Dict:
    dataset = build_dataset(salvar=True)

    resolved = mode
    if mode == "auto":
        resolved = "real" if _deps_reais_disponiveis() else "demo"
    if mode == "real" and not _deps_reais_disponiveis():
        import warnings
        warnings.warn("Dependências de fine-tuning real ausentes; usando modo 'demo'.")
        resolved = "demo"

    if resolved == "real":
        metrics = treinar_real(dataset, config.BASE_MODEL_ID, epochs, lr)
    else:
        metrics = treinar_demo(dataset, epochs, lr)

    config.ensure_dirs()
    config.TRAIN_METRICS_PATH.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fig = plot_loss(metrics)
    metrics["loss_curve_png"] = str(fig) if fig else None
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Fine-tuning da LLM médica (Fase 3).")
    ap.add_argument("--mode", choices=["auto", "real", "demo"], default="demo")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    args = ap.parse_args()

    m = run(mode=args.mode, epochs=args.epochs, lr=args.lr)
    print(f"[ok] fine-tuning ({m['mode']}) — {m['n_exemplos']} exemplos, "
          f"loss final={m['loss_final']}, {m['train_runtime_s']}s")
    print(f"     métricas → {config.TRAIN_METRICS_PATH}")
    print(f"     curva    → {m.get('loss_curve_png')}")
