"""Backend da LLM usado pelo assistente.

Quando LangChain está instalado, ``CustomMedicalLLM`` implementa a interface
``LLM`` normalmente. Em ambientes de teste sem a dependência, mantemos uma
interface mínima com ``invoke`` para que RAG, segurança, banco e LangGraph
possam ser validados sem mascarar a ausência da biblioteca.

O backend também consegue usar o adapter LoRA de ``src/finetuning`` quando ele
já foi treinado e salvo em ``results/finetuned_model``.
"""

from __future__ import annotations

import os
import re
from typing import Any, List, Optional

try:  # caminho usado na entrega com LangChain instalado
    from langchain_core.callbacks.manager import CallbackManagerForLLMRun
    from langchain_core.language_models.llms import LLM as _LangChainLLM
    LANGCHAIN_AVAILABLE = True
except ImportError:  # permite testar o restante do projeto em CPU/offline
    CallbackManagerForLLMRun = Any
    LANGCHAIN_AVAILABLE = False

    class _LangChainLLM:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

        def invoke(self, prompt: str, **kwargs):
            return self._call(prompt, **kwargs)

from src.llm.client import LLMClient, get_client
from src.finetuning.inference import FineTunedAssistant


def _extractive_grounded(prompt: str) -> str:
    """Fallback extrativo ancorado no contexto, usado somente sem modelo disponível."""
    fontes = re.findall(r"\[Fonte:\s*([^\]]+)\]", prompt)
    blocos = re.split(r"\[Fonte:[^\]]+\]\s*", prompt)
    trecho = ""
    if len(blocos) > 1:
        bruto = blocos[1].split("DADOS DO PACIENTE")[0].split("PERGUNTA DO MÉDICO")[0]
        frases = re.split(r"(?<=[.!?])\s+", bruto.strip())
        trecho = " ".join(frases[:2]).strip()
    fonte_txt = ", ".join(fontes[:3]) or "fonte clínica recuperada"
    if trecho:
        return f"Com base no contexto recuperado: {trecho}\n\nFontes: {fonte_txt}."
    return f"Não encontrei conteúdo suficiente para responder com segurança. Fonte consultada: {fonte_txt}."


class CustomMedicalLLM(_LangChainLLM):
    """Wrapper único para Ollama, adapter LoRA ou fallback extrativo."""

    client: Any = None
    finetuned: Any = None
    system_prompt: Optional[str] = None
    selected_backend: str = "auto"

    def __init__(
        self,
        client: Optional[LLMClient] = None,
        system_prompt: Optional[str] = None,
        backend: Optional[str] = None,
        **kwargs,
    ):
        selected = (backend or os.environ.get("MEDICAL_LLM_BACKEND") or "auto").lower()
        super().__init__(**kwargs)
        self.system_prompt = system_prompt
        self.selected_backend = selected
        self.finetuned = None
        self.client = client

        if selected in {"finetuned", "hf", "local", "auto"}:
            requested_ft = "auto" if selected == "auto" else ("hf" if selected in {"finetuned", "hf"} else "local")
            candidate = FineTunedAssistant(backend=requested_ft)
            if candidate.backend in {"hf", "local"}:
                self.finetuned = candidate
                return
            if selected in {"finetuned", "hf", "local"}:
                raise RuntimeError(
                    "Backend fine-tuned solicitado, mas o adapter/dependências não estão disponíveis."
                )

        if selected in {"ollama", "mock", "auto"}:
            self.client = client or get_client(backend=None if selected == "auto" else selected)

    @property
    def _llm_type(self) -> str:
        return "custom_medical_llm"

    @property
    def backend(self) -> str:
        if self.finetuned is not None:
            return f"finetuned-{self.finetuned.backend}"
        if self.client is not None:
            return getattr(self.client, "active_backend", "ollama")
        return "extractive"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        system = kwargs.get("system", self.system_prompt)
        if self.finetuned is not None:
            return self.finetuned.ask(prompt, system=system)
        if self.client is not None and self.backend != "mock":
            return self.client.generate(prompt, system=system)
        # O mock antigo não é usado para validar conteúdo clínico. Sem um modelo
        # real, a resposta fica extrativa e visivelmente ancorada no RAG.
        return _extractive_grounded(prompt)
