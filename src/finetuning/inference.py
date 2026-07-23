"""
inference.py
------------
Wrapper fino para gerar respostas com o modelo fine-tuned (base + adapter
LoRA). Ponto único de acesso ao modelo — é isso que o pipeline LangChain
(próxima etapa da Fase 3) vai importar, em vez de lidar com
transformers/peft diretamente.

Segue o mesmo padrão de fallback do `src/llm/client.py` da Fase 2: se o
adapter treinado ou a stack de GPU não estiverem disponíveis (como neste
container de desenvolvimento), cai para um backend "mock" que responde de
forma determinística — assim notebooks, testes e a integração LangChain
podem ser desenvolvidos e testados sem GPU, e passam a usar o modelo real
assim que `results/finetuned_model/` existir (gerado pelo `train_lora.py`
no Colab).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_ADAPTER_DIR = _PROJECT_ROOT / "results" / "finetuned_model"


class FineTunedAssistant:
    """
    Uso:
        assistant = FineTunedAssistant()   # auto-detecta backend disponível
        resposta = assistant.ask(
            "Qual o protocolo para paciente com suspeita de SRAG?",
            system="...",
        )
    """

    def __init__(self, adapter_dir: Optional[Path] = None, backend: str = "auto"):
        self.adapter_dir = Path(adapter_dir or _DEFAULT_ADAPTER_DIR)
        self.backend = self._resolve_backend(backend)
        self._model = None
        self._tokenizer = None
        if self.backend == "hf":
            self._load_hf_model()

    def _resolve_backend(self, backend: str) -> str:
        if backend == "mock":
            return "mock"
        run_info = self.adapter_dir / "run_info.json"
        if backend in ("hf", "auto") and run_info.exists():
            try:
                import torch  # noqa: F401
                import transformers  # noqa: F401
                import peft  # noqa: F401
                return "hf"
            except ImportError:
                if backend == "hf":
                    raise
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

    def ask(self, question: str, system: Optional[str] = None,
             context: Optional[str] = None, max_new_tokens: int = 300) -> str:
        messages: List[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        user_content = f"{question}\n\nContexto:\n{context}" if context else question
        messages.append({"role": "user", "content": user_content})

        if self.backend == "hf":
            return self._ask_hf(messages, max_new_tokens)
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
        text = self._tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        return text.strip()

    @staticmethod
    def _ask_mock(question: str, context: Optional[str]) -> str:
        """
        Resposta determinística por template — mantém a interface idêntica
        à do modelo real para que a integração LangChain/LangGraph seja
        desenvolvida e testada de ponta a ponta antes do fine-tuning real
        rodar em GPU.
        """
        base = (
            "[modo demonstração — adapter fine-tuned não encontrado em "
            "results/finetuned_model/, rode src/finetuning/train_lora.py em uma "
            "GPU para habilitar o modelo real]\n"
        )
        resposta = (
            f"Com base nos protocolos internos disponíveis, sobre '{question.strip()}': "
            "recomenda-se seguir a conduta padrão documentada e, diante de qualquer "
            "dúvida ou caso fora do escopo do protocolo, encaminhar para validação do "
            "médico responsável antes de qualquer prescrição ou conduta definitiva."
        )
        if context:
            resposta += f" Contexto considerado: {context[:200]}"
        return base + resposta


def get_assistant(backend: Optional[str] = None, **kwargs) -> FineTunedAssistant:
    backend = backend or os.environ.get("FINETUNE_BACKEND", "auto")
    return FineTunedAssistant(backend=backend, **kwargs)
