"""
client.py
---------
Cliente de LLM com dois backends intercambiáveis:

  - "ollama" : chama um modelo local servido pelo Ollama (http://localhost:11434).
               Backend principal, demonstrado no vídeo. Sem custo de API.
  - "mock"   : gera respostas por template a partir do próprio prompt/contexto,
               sem nenhuma dependência externa. Garante que notebooks/testes/CI
               rodem em qualquer máquina (inclusive a do avaliador).

Seleção:
  - parâmetro `backend` ou variável de ambiente `LLM_BACKEND` (ollama|mock);
  - se `backend="ollama"` (ou "auto") e o servidor não responder, faz **fallback
    automático para `mock`**, registrando um aviso.
"""

from __future__ import annotations

import os
import re
from typing import Optional


DEFAULT_MODEL = os.environ.get("LLM_MODEL", "llama3.1:8b")
DEFAULT_OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")


class LLMClient:
    """Wrapper fino que expõe `generate(prompt, system) -> str`."""

    def __init__(
        self,
        backend: str = "auto",
        model: str = DEFAULT_MODEL,
        ollama_url: str = DEFAULT_OLLAMA_URL,
        timeout: int = 60,
    ):
        self.requested_backend = backend
        self.model = model
        self.ollama_url = ollama_url.rstrip("/")
        self.timeout = timeout
        self.active_backend = self._resolve_backend(backend)

    # ---- resolução de backend -------------------------------------------------
    def _resolve_backend(self, backend: str) -> str:
        if backend == "mock":
            return "mock"
        if backend in ("ollama", "auto"):
            if self._ollama_available():
                return "ollama"
            if backend == "ollama":
                import warnings
                warnings.warn(
                    "Ollama indisponível em "
                    f"{self.ollama_url}; usando backend 'mock' como fallback."
                )
            return "mock"
        raise ValueError(f"Backend '{backend}' inválido. Use 'ollama', 'mock' ou 'auto'.")

    def _ollama_available(self) -> bool:
        try:
            import requests
            r = requests.get(f"{self.ollama_url}/api/tags", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    # ---- geração --------------------------------------------------------------
    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        if self.active_backend == "ollama":
            try:
                return self._generate_ollama(prompt, system)
            except Exception as exc:  # fallback resiliente em runtime
                import warnings
                warnings.warn(f"Falha no Ollama ({exc}); usando 'mock'.")
                self.active_backend = "mock"
        return self._generate_mock(prompt, system)

    def _generate_ollama(self, prompt: str, system: Optional[str]) -> str:
        import requests
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system or "",
            "stream": False,
            "options": {"temperature": 0.2},
        }
        r = requests.post(
            f"{self.ollama_url}/api/generate", json=payload, timeout=self.timeout
        )
        r.raise_for_status()
        return r.json().get("response", "").strip()

    # ---- backend mock (template determinístico) -------------------------------
    @staticmethod
    def _generate_mock(prompt: str, system: Optional[str]) -> str:
        """
        Constrói uma explicação por template a partir do conteúdo do prompt.
        Extrai a predição/probabilidade e os fatores listados, sem chamar LLM.
        """
        pred = _search(r"Predição do modelo:\s*\**([^\*\(\n]+)", prompt) or "indefinida"
        prob = _search(r"probabilidade estimada:\s*([0-9.,]+%)", prompt) or "n/d"
        pred = pred.strip()

        fatores = re.findall(r"^- (.+)$", prompt, flags=re.MULTILINE)
        # Remove eventuais bullets do bloco de exemplo few-shot.
        fatores = [f for f in fatores if "→" in f][:3]
        if fatores:
            lista = "; ".join(_humanize_factor(f) for f in fatores)
            corpo = (
                f"O modelo prevê o desfecho '{pred}' com probabilidade de {prob}. "
                f"Os fatores que mais pesaram nessa estimativa foram: {lista}. "
                "Recomenda-se atenção a esses pontos e reavaliação clínica conforme evolução."
            )
        else:
            corpo = (
                f"O modelo prevê o desfecho '{pred}' com probabilidade de {prob}, "
                "sem fatores individuais destacados no contexto fornecido."
            )
        return (
            f"[modo demonstração — resposta gerada por template]\n{corpo} "
            "Esta é uma estimativa de apoio à decisão e não substitui a avaliação médica."
        )


def _search(pattern: str, text: str) -> Optional[str]:
    m = re.search(pattern, text)
    return m.group(1).strip() if m else None


def _humanize_factor(raw: str) -> str:
    """'IDADE = 72 → ↑ aumenta o risco (peso +0.310)' → frase curta."""
    nome = raw.split("=")[0].strip()
    if "aumenta o risco" in raw:
        return f"{nome} (elevando o risco)"
    if "reduz o risco" in raw:
        return f"{nome} (reduzindo o risco)"
    return nome


def get_client(backend: Optional[str] = None, **kwargs) -> LLMClient:
    """
    Fábrica de cliente. Precedência do backend:
      argumento `backend` > variável de ambiente LLM_BACKEND > 'auto'.
    """
    backend = backend or os.environ.get("LLM_BACKEND", "auto")
    return LLMClient(backend=backend, **kwargs)
