"""
llm_backend.py
--------------
Integra a LLM customizada ao LangChain (requisito 2).

Reaproveita o `LLMClient` da Fase 2 (backend Ollama local + fallback mock) e o
expõe como um `LLM` do LangChain, para ser usado nas chains e no LangGraph. Em
produção, o mesmo wrapper serve o modelo *fine-tuned* (basta apontar o Ollama
para o modelo com o adapter LoRA aplicado, ou trocar por um endpoint HF).

Assim, o pipeline LangChain fica desacoplado do backend: hoje demonstramos com
Ollama/mock; amanhã, com a LLM fine-tuned, sem mudar as chains.
"""

from __future__ import annotations

import re
from typing import Any, List, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM

# Cliente da Fase 2 (Ollama/mock). Import defensivo p/ mensagem clara.
from src.llm.client import LLMClient, get_client


def _mock_grounded(prompt: str) -> str:
    """
    Mock extrativo para a demo OFFLINE do assistente.

    O mock genérico da Fase 2 foi feito para explicar predições de risco; aqui,
    no assistente clínico, geramos uma resposta ANCORADA no bloco CONTEXTO do
    prompt — extraindo o primeiro trecho de protocolo e sua fonte. Assim a demo
    roda sem Ollama e ainda produz respostas fundamentadas e citando a fonte.
    Em produção, este caminho é substituído pela LLM fine-tuned (via Ollama/HF).
    """
    # Fontes marcadas como "[Fonte: PROT-... — Título]"
    fontes = re.findall(r"\[Fonte:\s*([^\]]+)\]", prompt)
    # Blocos de contexto: texto após cada marcador de fonte.
    blocos = re.split(r"\[Fonte:[^\]]+\]\s*", prompt)
    trecho = ""
    if len(blocos) > 1:
        # primeiro bloco de conteúdo após uma fonte; corta no próximo cabeçalho
        bruto = blocos[1].split("DADOS DO PACIENTE")[0].split("PERGUNTA DO MÉDICO")[0]
        # pega as 2 primeiras frases
        frases = re.split(r"(?<=[.!?])\s+", bruto.strip())
        trecho = " ".join(frases[:2]).strip()

    codigos = ", ".join(sorted({f.split(" — ")[0].strip() for f in fontes})) or "protocolo interno"
    if not trecho:
        return (
            "Com base nos protocolos internos, recomenda-se seguir a conduta "
            f"prevista em {codigos}. "
            "Trata-se de apoio à decisão; a conduta final é do médico responsável."
        )
    return (
        f"[demonstração — resposta extrativa dos protocolos]\n"
        f"Com base nos protocolos internos: {trecho} "
        f"\n\nFontes: {codigos}."
    )


class CustomMedicalLLM(LLM):
    """
    LLM do LangChain que delega a geração ao `LLMClient` (Ollama/mock/fine-tuned).

    O `system` é passado via kwargs em cada chamada; se ausente, usa o system
    padrão configurado na chain.
    """

    client: Any = None
    system_prompt: Optional[str] = None

    def __init__(self, client: Optional[LLMClient] = None, system_prompt: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        # `client` e `system_prompt` são campos pydantic declarados acima.
        self.client = client or get_client()
        self.system_prompt = system_prompt

    @property
    def _llm_type(self) -> str:
        return "custom_medical_llm"

    @property
    def backend(self) -> str:
        return getattr(self.client, "active_backend", "desconhecido")

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        system = kwargs.get("system", self.system_prompt)
        # Na demo offline (backend mock), usa resposta extrativa ancorada no
        # contexto; com Ollama/fine-tuned, delega a geração ao cliente real.
        if self.backend == "mock":
            return _mock_grounded(prompt)
        return self.client.generate(prompt, system=system)
