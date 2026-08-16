"""Inferência com o modelo ajustado da Fase 3.

Há dois caminhos reais:
- ``hf``: modelo-base do Hugging Face + adapter LoRA salvo por ``train_lora.py``;
- ``local``: Transformer pequeno treinado por ``local_validation.py`` para validar
  o pipeline em CPU.

O backend local não substitui o LLM da entrega, mas evita que a validação de
engenharia dependa de um mock quando não há GPU/Hugging Face disponível.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_ADAPTER_DIR = _PROJECT_ROOT / "results" / "finetuned_model"
_DEFAULT_LOCAL_DIR = _PROJECT_ROOT / "results" / "finetuning" / "local_validation"


class FineTunedAssistant:
    def __init__(
        self,
        adapter_dir: Optional[Path] = None,
        local_dir: Optional[Path] = None,
        backend: str = "auto",
    ):
        self.adapter_dir = Path(adapter_dir or _DEFAULT_ADAPTER_DIR)
        self.local_dir = Path(local_dir or _DEFAULT_LOCAL_DIR)
        self.backend = self._resolve_backend(backend)
        self._model = None
        self._tokenizer = None
        self._local_cfg = None
        if self.backend == "hf":
            self._load_hf_model()
        elif self.backend == "local":
            self._load_local_model()

    def _resolve_backend(self, backend: str) -> str:
        backend = backend.lower()
        if backend not in {"auto", "hf", "local", "mock"}:
            raise ValueError("backend deve ser auto, hf, local ou mock")
        if backend == "mock":
            return "mock"

        if backend in {"auto", "hf"} and (self.adapter_dir / "run_info.json").exists():
            try:
                import torch  # noqa: F401
                import transformers  # noqa: F401
                import peft  # noqa: F401
                return "hf"
            except ImportError:
                if backend == "hf":
                    raise

        if backend in {"auto", "local"} and (self.local_dir / "tiny_transformer.pt").exists():
            return "local"
        if backend == "local":
            raise FileNotFoundError(
                f"Checkpoint local não encontrado em {self.local_dir}. "
                "Rode: python -m src.finetuning.local_validation"
            )
        return "mock"

    def _load_hf_model(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        run_info = json.loads((self.adapter_dir / "run_info.json").read_text())
        base_model_name = run_info["base_model"]
        self._tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name, device_map="auto", torch_dtype=torch.bfloat16
        )
        self._model = PeftModel.from_pretrained(base_model, str(self.adapter_dir))

    def _load_local_model(self) -> None:
        import torch
        from .local_validation import Config, TinyTransformer, Tokenizer

        checkpoint = torch.load(
            self.local_dir / "tiny_transformer.pt", map_location="cpu", weights_only=True
        )
        vocab = json.loads((self.local_dir / "tokenizer.json").read_text(encoding="utf-8"))
        self._tokenizer = Tokenizer(vocab)
        self._local_cfg = Config(**checkpoint["config"])
        self._model = TinyTransformer(
            checkpoint["vocab_size"], checkpoint["pad_id"],
            max_len=max(self._local_cfg.max_src, self._local_cfg.max_tgt) + 4,
        )
        self._model.load_state_dict(checkpoint["model_state"])
        self._model.eval()

    def ask(
        self,
        question: str,
        system: Optional[str] = None,
        context: Optional[str] = None,
        max_new_tokens: int = 300,
    ) -> str:
        if self.backend == "hf":
            messages: List[dict] = []
            if system:
                messages.append({"role": "system", "content": system})
            user_content = f"{question}\n\nContexto:\n{context}" if context else question
            messages.append({"role": "user", "content": user_content})
            return self._ask_hf(messages, max_new_tokens)
        if self.backend == "local":
            return self._ask_local(question)
        return self._ask_mock(question, context)

    def _ask_hf(self, messages: List[dict], max_new_tokens: int) -> str:
        import torch
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        return self._tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

    def _ask_local(self, question: str) -> str:
        import torch
        ids = torch.tensor(
            [self._tokenizer.encode(question, self._local_cfg.max_src)], dtype=torch.long
        )
        generated = self._model.generate(
            ids, self._tokenizer.bos_id, self._tokenizer.eos_id,
            max_new_tokens=self._local_cfg.max_tgt - 1,
        )
        return self._tokenizer.decode(generated[0].tolist())

    @staticmethod
    def _ask_mock(question: str, context: Optional[str]) -> str:
        return (
            "[backend de demonstração sem modelo ajustado]\n"
            "O adapter LoRA e o checkpoint local não estão disponíveis. "
            "Rode o treinamento antes de usar este backend para avaliação clínica."
        )


def get_assistant(backend: Optional[str] = None, **kwargs) -> FineTunedAssistant:
    backend = backend or os.environ.get("FINETUNE_BACKEND", "auto")
    return FineTunedAssistant(backend=backend, **kwargs)
