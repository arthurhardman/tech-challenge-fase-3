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


def _content_tokens(text: str) -> set[str]:
    stop = {
        "a", "o", "as", "os", "de", "da", "do", "das", "dos", "e", "em",
        "um", "uma", "para", "por", "com", "que", "qual", "quais", "como",
        "este", "esta", "esse", "essa", "ser", "são", "no", "na", "nos", "nas",
    }
    return {
        token
        for token in re.findall(r"[a-zà-ÿ0-9]+", text.lower())
        if len(token) >= 3 and token not in stop
    }


def _extractive_grounded(prompt: str) -> str:
    """Resposta extrativa curta, escolhida entre os trechos já recuperados pelo RAG.

    O fallback anterior pegava simplesmente as primeiras frases do primeiro chunk.
    Agora usamos a própria pergunta para escolher a frase com maior sobreposição de
    termos. É simples, determinístico e mantém a resposta ancorada no protocolo.
    """
    question_match = re.search(
        r"PERGUNTA DO MÉDICO:\s*(.+?)(?:\n\n|$)", prompt, re.S
    )
    question = question_match.group(1).strip() if question_match else prompt
    q_tokens = _content_tokens(question)

    context_match = re.search(
        r"CONTEXTO CLÍNICO RECUPERADO:\s*(.+?)(?:\n\nDADOS DO PACIENTE|\n\nPERGUNTA DO MÉDICO:)",
        prompt,
        re.S,
    )
    context = context_match.group(1).strip() if context_match else prompt

    blocks = re.findall(
        r"\[Fonte:\s*([^\]]+)\]\s*(.*?)(?=\n\n\[Fonte:|$)",
        context,
        re.S,
    )
    candidates: list[tuple[float, str, str]] = []
    for source, text in blocks:
        sentences = re.split(r"(?<=[.!?])\s+|\s+[•●]\s+", text.strip())
        for sentence in sentences:
            sentence = " ".join(sentence.split())
            words = sentence.split()
            if len(words) < 6 or len(words) > 90:
                continue
            s_tokens = _content_tokens(sentence)
            overlap = len(q_tokens & s_tokens)
            # Jaccard ajuda a não escolher uma frase enorme só por ter muitos termos.
            union = len(q_tokens | s_tokens) or 1
            score = overlap + (overlap / union)
            candidates.append((score, sentence, source.strip()))

    if candidates:
        _, sentence, source = max(candidates, key=lambda item: item[0])
        return f"Com base no contexto recuperado: {sentence}\n\nFonte: {source}."

    fontes = re.findall(r"\[Fonte:\s*([^\]]+)\]", context)
    fonte_txt = ", ".join(fontes[:3]) or "fonte clínica recuperada"
    return (
        "Não encontrei conteúdo suficiente para responder com segurança. "
        f"Fonte consultada: {fonte_txt}."
    )


def _local_output_is_low_quality(text: str) -> bool:
    """Detecta respostas degeneradas do Transformer pequeno usado na validação CPU."""
    words = re.findall(r"[a-zà-ÿ0-9]+", text.lower())
    if len(words) < 8:
        return True
    content = [w for w in words if len(w) >= 4]
    if len(set(content)) < 3:
        return True
    if len(set(words)) / max(1, len(words)) < 0.35:
        return True
    return False


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
            answer = self.finetuned.ask(prompt, system=system)
            # O Transformer local prova o treino real, mas ainda é pequeno. Se ele
            # produzir uma frase degenerada, preferimos devolver um trecho do RAG
            # a mostrar texto sem qualidade clínica. O adapter LoRA não usa esse fallback.
            if self.finetuned.backend == "local" and _local_output_is_low_quality(answer):
                return _extractive_grounded(prompt)
            return answer
        if self.client is not None and self.backend != "mock":
            return self.client.generate(prompt, system=system)
        # O mock antigo não é usado para validar conteúdo clínico. Sem um modelo
        # real, a resposta fica extrativa e visivelmente ancorada no RAG.
        return _extractive_grounded(prompt)
