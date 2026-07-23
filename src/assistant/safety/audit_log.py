"""
audit_log.py
------------
Logging detalhado para rastreamento e auditoria (requisito 3).

Cada interação do assistente gera um registro estruturado (JSON-lines) contendo:
timestamp, pergunta, paciente consultado, fontes citadas, backend da LLM,
decisões de guardrail e a resposta final. Isso permite reconstruir depois
"por que o assistente respondeu isso" — a espinha dorsal da auditabilidade.

Grava em dois canais:
  - arquivo texto legível (results/finetuning/audit.log) via `logging`;
  - arquivo JSONL estruturado (audit_events.jsonl) para consulta programática.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from .. import config

_JSONL_PATH = config.PROJECT_ROOT / "results" / "finetuning" / "audit_events.jsonl"


@dataclass
class AuditEvent:
    """Registro auditável de uma interação com o assistente."""
    pergunta: str
    resposta: str
    backend_llm: str
    paciente_id: Optional[str] = None
    fontes: List[str] = field(default_factory=list)
    guardrail_bloqueou: bool = False
    guardrail_categorias: List[str] = field(default_factory=list)
    fluxo_no: Optional[str] = None            # nó do LangGraph que respondeu
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class AuditLogger:
    """Registra `AuditEvent` em log de texto + JSONL."""

    def __init__(self, log_path: Optional[Path] = None, jsonl_path: Optional[Path] = None):
        config.ensure_dirs()
        self.log_path = Path(log_path or config.AUDIT_LOG_PATH)
        self.jsonl_path = Path(jsonl_path or _JSONL_PATH)

        self.logger = logging.getLogger("assistant.audit")
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            handler = logging.FileHandler(self.log_path, encoding="utf-8")
            handler.setFormatter(
                logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
            )
            self.logger.addHandler(handler)

    def registrar(self, evento: AuditEvent) -> None:
        """Persiste o evento nos dois canais."""
        self.logger.info(
            "pergunta=%r paciente=%s fontes=%s backend=%s bloqueio=%s no=%s",
            evento.pergunta[:120],
            evento.paciente_id,
            evento.fontes,
            evento.backend_llm,
            evento.guardrail_bloqueou,
            evento.fluxo_no,
        )
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(evento), ensure_ascii=False) + "\n")

    def ler_eventos(self) -> List[dict]:
        """Lê todos os eventos JSONL (para inspeção/relatório)."""
        if not self.jsonl_path.exists():
            return []
        linhas = self.jsonl_path.read_text(encoding="utf-8").splitlines()
        return [json.loads(l) for l in linhas if l.strip()]
