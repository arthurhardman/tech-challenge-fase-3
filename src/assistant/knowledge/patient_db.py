"""Consulta à base estruturada de pacientes usada pelo assistente.

Mantém compatibilidade com o JSON sintético original, mas usa SQLite quando uma
base SIVEP preparada estiver disponível. Assim o mesmo código continua simples
para a demo e também escala para centenas de milhares de registros reais.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

from .. import config


class PatientDB:
    """Consulta pacientes em SQLite (preferencial) ou JSON (compatibilidade)."""

    def __init__(self, path: Optional[Path] = None):
        chosen = Path(path) if path else self._default_path()
        self.path = chosen
        self.backend = "sqlite" if chosen.suffix.lower() in {".db", ".sqlite", ".sqlite3"} else "json"
        self._por_id: Dict[str, Dict] = {}

        if self.backend == "json" and self.path.exists():
            registros = json.loads(self.path.read_text(encoding="utf-8"))
            self._por_id = {r["paciente_id"]: r for r in registros}

    @staticmethod
    def _default_path() -> Path:
        # Se já existe uma base real preparada, ela tem prioridade. O JSON
        # sintético continua disponível para quem quiser reproduzir a demo antiga.
        if config.SIVEP_DB_PATH.exists():
            return config.SIVEP_DB_PATH
        return config.PRONTUARIOS_PATH

    def existe(self, paciente_id: str) -> bool:
        return self.get(paciente_id) is not None

    def get(self, paciente_id: str) -> Optional[Dict]:
        if self.backend == "json":
            return self._por_id.get(paciente_id)
        if not self.path.exists():
            return None
        with sqlite3.connect(self.path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM patients WHERE patient_id = ? LIMIT 1",
                (paciente_id,),
            ).fetchone()
        return dict(row) if row else None

    def exames_pendentes(self, paciente_id: str) -> List[str]:
        reg = self.get(paciente_id)
        if not reg:
            return []
        if isinstance(reg.get("exames"), dict):
            return [nome for nome, status in reg["exames"].items() if status == "pendente"]
        raw = reg.get("exames_pendentes", "")
        return [x.strip() for x in str(raw).split(",") if x.strip()]

    def resumo_clinico(self, paciente_id: str) -> str:
        """Monta o contexto usando somente os fatos existentes no registro."""
        r = self.get(paciente_id)
        if not r:
            return f"(paciente {paciente_id} não encontrado na base)"

        # Formato sintético antigo.
        if "spo2" in r:
            comorb = ", ".join(r.get("comorbidades", [])) or "nenhuma"
            pend = self.exames_pendentes(paciente_id)
            return (
                f"Paciente {paciente_id}: {r.get('idade')} anos, sexo {r.get('sexo')}, "
                f"SpO2 {r.get('spo2')}%, FR {r.get('freq_respiratoria')} irpm, "
                f"febre={'sim' if r.get('febre') else 'não'}, "
                f"dispneia={'sim' if r.get('dispneia') else 'não'}, "
                f"comorbidades: {comorb}, risco: {r.get('classificacao_risco')}, "
                f"exames pendentes: {', '.join(pend) if pend else 'nenhum'}."
            )

        # Formato SIVEP anonimizado. SATURACAO é um indicador de saturação <95%,
        # não um valor numérico de SpO2, por isso não transformamos esse campo em número.
        fields = [
            ("idade", "idade"), ("sexo", "sexo"), ("febre", "febre"),
            ("tosse", "tosse"), ("dispneia", "dispneia"),
            ("desc_resp", "desconforto respiratório"),
            ("saturacao", "SpO2/saturação <95%"), ("uti", "UTI"),
            ("suporte_ventilatorio", "suporte ventilatório"),
            ("resultado_pcr", "PCR"), ("classificacao_final", "classificação final"),
            ("desfecho_registrado", "desfecho registrado"),
            ("ano_fonte", "ano do registro"),
        ]
        parts = [f"Paciente {paciente_id}"]
        for key, label in fields:
            value = r.get(key)
            if value is not None and str(value).strip() not in {"", "nan", "não informado"}:
                parts.append(f"{label}: {value}")
        parts.append(f"risco derivado para o fluxo: {r.get('classificacao_risco', 'desconhecido')}")
        pend = self.exames_pendentes(paciente_id)
        parts.append("exames pendentes: " + (", ".join(pend) if pend else "nenhum registrado como pendente"))
        return "; ".join(parts) + "."

    def todos_ids(self, limit: int | None = None) -> List[str]:
        if self.backend == "json":
            ids = list(self._por_id.keys())
            return ids[:limit] if limit else ids
        if not self.path.exists():
            return []
        sql = "SELECT patient_id FROM patients ORDER BY patient_id"
        params = ()
        if limit:
            sql += " LIMIT ?"
            params = (int(limit),)
        with sqlite3.connect(self.path) as conn:
            return [row[0] for row in conn.execute(sql, params).fetchall()]

    def count(self) -> int:
        if self.backend == "json":
            return len(self._por_id)
        if not self.path.exists():
            return 0
        with sqlite3.connect(self.path) as conn:
            return int(conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0])

    def primeiro_por_risco(self, risco: str) -> Optional[str]:
        if self.backend == "json":
            for pid, row in self._por_id.items():
                if row.get("classificacao_risco") == risco:
                    return pid
            return None
        if not self.path.exists():
            return None
        with sqlite3.connect(self.path) as conn:
            row = conn.execute(
                "SELECT patient_id FROM patients WHERE classificacao_risco = ? LIMIT 1",
                (risco,),
            ).fetchone()
        return row[0] if row else None
