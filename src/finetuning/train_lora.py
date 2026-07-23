"""
train_lora.py
-------------
Script de fine-tuning LoRA/PEFT do assistente médico (Fase 3).

PROJETADO PARA RODAR EM GPU (Google Colab, ou GPU própria com >= 16GB VRAM).
Este container de desenvolvimento não tem GPU nem acesso ao Hugging Face
Hub, então este script não é executado aqui — mas o pipeline de dados
(`dataset_prep.py`) já foi validado com dados reais, e este script consome
exatamente os arquivos que ele gera (`data/finetuning/processed/*.jsonl`).

Como rodar no Google Colab (GPU T4/L4, runtime gratuito ou Pro):
    !git clone <seu-repositorio>
    %cd <repo>/
    !pip install -r requirements-finetuning.txt
    !python -m src.finetuning.dataset_prep --build   # gera os .jsonl
    !huggingface-cli login                            # se o modelo exigir aceite de licença
    !python -m src.finetuning.train_lora

Como rodar com GPU própria:
    pip install -r requirements-finetuning.txt
    python -m src.finetuning.dataset_prep --build
    python -m src.finetuning.train_lora --base-model tiiuae/falcon-7b-instruct

O adapter LoRA resultante é salvo em `results/finetuned_model/` (apenas os
pesos do adapter, poucos MB — o modelo base NÃO é re-salvo).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


def load_config(path: Optional[Path] = None) -> dict:
    path = Path(path or _CONFIG_PATH)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Aplica overrides do tipo `training.num_train_epochs=1` vindos da CLI."""
    for item in overrides:
        key_path, _, value = item.partition("=")
        keys = key_path.split(".")
        node = cfg
        for k in keys[:-1]:
            node = node[k]
        # tenta converter para int/float/bool; senão mantém string
        parsed: object
        if value.lower() in ("true", "false"):
            parsed = value.lower() == "true"
        else:
            try:
                parsed = int(value)
            except ValueError:
                try:
                    parsed = float(value)
                except ValueError:
                    parsed = value
        node[keys[-1]] = parsed
    return cfg


def format_chat_example(example: dict, tokenizer) -> str:
    """Converte um registro `{"messages": [...]}` em texto de treino usando o
    chat template do tokenizer do próprio modelo (garante o formato correto
    de tags especiais para cada arquitetura)."""
    return tokenizer.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=False
    )


def main():
    parser = argparse.ArgumentParser(description="Fine-tuning LoRA do assistente médico (Fase 3).")
    parser.add_argument("--config", type=str, default=str(_CONFIG_PATH))
    parser.add_argument("--base-model", type=str, default=None,
                         help="Sobrescreve o modelo ativo definido no config.yaml (nome do HF Hub).")
    parser.add_argument("--override", action="append", default=[],
                         help="Override pontual, ex: --override training.num_train_epochs=1")
    parser.add_argument("--dry-run", action="store_true",
                         help="Só valida config + dataset, sem carregar o modelo (útil para CI/CPU).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.override)
    base_model_name = args.base_model or cfg["base_models"][cfg["active"]]

    train_path = _PROJECT_ROOT / cfg["data"]["train_path"]
    val_path = _PROJECT_ROOT / cfg["data"]["val_path"]
    if not train_path.exists():
        raise FileNotFoundError(
            f"{train_path} não encontrado. Rode antes: "
            "python -m src.finetuning.dataset_prep --build"
        )

    print(f"[train_lora] modelo base: {base_model_name}")
    print(f"[train_lora] treino: {train_path} | validação: {val_path}")

    if args.dry_run:
        # Validação leve, sem GPU/rede: garante que o dataset está no formato
        # esperado (chat-style com system/user/assistant) antes de ir pro Colab.
        n = 0
        with open(train_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                assert "messages" in rec and len(rec["messages"]) == 3
                n += 1
        print(f"[train_lora] dry-run OK — {n} exemplos de treino no formato esperado.")
        return

    # Imports pesados (transformers/peft/bitsandbytes) só acontecem se formos
    # treinar de verdade — assim `--dry-run` roda em qualquer máquina.
    import torch
    from datasets import load_dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig

    quant_cfg = cfg["quantization"]
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_cfg["load_in_4bit"],
        bnb_4bit_quant_type=quant_cfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, quant_cfg["bnb_4bit_compute_dtype"]),
        bnb_4bit_use_double_quant=quant_cfg["bnb_4bit_use_double_quant"],
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    lora_cfg = cfg["lora"]
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        task_type=lora_cfg["task_type"],
        target_modules=lora_cfg["target_modules"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    dataset = load_dataset(
        "json",
        data_files={"train": str(train_path), "validation": str(val_path)},
    )

    def _to_text(batch):
        return {"text": [format_chat_example({"messages": m}, tokenizer) for m in batch["messages"]]}

    dataset = dataset.map(_to_text, batched=True, remove_columns=dataset["train"].column_names)

    t = cfg["training"]
    output_dir = _PROJECT_ROOT / t["output_dir"]
    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        per_device_eval_batch_size=t["per_device_eval_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        lr_scheduler_type=t["lr_scheduler_type"],
        warmup_ratio=t["warmup_ratio"],
        weight_decay=t["weight_decay"],
        max_seq_length=t["max_seq_length"],
        logging_steps=t["logging_steps"],
        eval_strategy=t["eval_strategy"],
        eval_steps=t["eval_steps"],
        save_strategy=t["save_strategy"],
        save_steps=t["save_steps"],
        save_total_limit=t["save_total_limit"],
        bf16=t["bf16"],
        gradient_checkpointing=t["gradient_checkpointing"],
        seed=t["seed"],
        report_to=t["report_to"],
        dataset_text_field="text",
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
    )

    trainer.train()

    # Salva SÓ o adapter LoRA (leve) + tokenizer, para uso em inference.py /
    # integração LangChain — não re-salva os pesos do modelo base.
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    run_info = {
        "base_model": base_model_name,
        "lora_config": lora_cfg,
        "training_config": t,
        "train_examples": len(dataset["train"]),
        "val_examples": len(dataset["validation"]),
    }
    (output_dir / "run_info.json").write_text(
        json.dumps(run_info, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[train_lora] adapter salvo em: {output_dir}")


if __name__ == "__main__":
    main()
