"""
patient_db.py
-------------
Acesso à base ESTRUTURADA de pacientes (prontuários sintéticos). Cumpre o
requisito 2 do enunciado: "realizar consultas em base de dados estruturadas
(como prontuários e registros)" e permite contextualizar a resposta da LLM com
os dados atualizados do paciente.

A base é um JSON de prontuários anonimizados (IDs fictícios PAC-XXXX). As
consultas aqui são determinísticas e auditáveis — nada é inventado pela LLM.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from .. import config


class PatientDB:
    """Consulta simples sobre a base de prontuários sintéticos."""

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path or config.PRONTUARIOS_PATH)
        self._por_id: Dict[str, Dict] = {}
        if self.path.exists():
            registros = json.loads(self.path.read_text(encoding="utf-8"))
            self._por_id = {r["paciente_id"]: r for r in registros}

    def existe(self, paciente_id: str) -> bool:
        return paciente_id in self._por_id

    def get(self, paciente_id: str) -> Optional[Dict]:
        """Retorna o prontuário do paciente ou None."""
        return self._por_id.get(paciente_id)

    def exames_pendentes(self, paciente_id: str) -> List[str]:
        """Lista os exames com status 'pendente' — usado pelo fluxo de decisão."""
        reg = self.get(paciente_id)
        if not reg:
            return []
        return [nome for nome, status in reg.get("exames", {}).items()
                if status == "pendente"]

    def resumo_clinico(self, paciente_id: str) -> str:
        """
        Monta um resumo textual do paciente para injetar no contexto do prompt.
        Somente fatos presentes no prontuário — base para contextualização.
        """
        r = self.get(paciente_id)
        if not r:
            return f"(paciente {paciente_id} não encontrado na base)"
        comorb = ", ".join(r.get("comorbidades", [])) or "nenhuma"
        pend = self.exames_pendentes(paciente_id)
        pend_txt = ", ".join(pend) if pend else "nenhum"
        return (
            f"Paciente {r['paciente_id']}: {r['idade']} anos, sexo {r['sexo']}, "
            f"SpO2 {r['spo2']}%, FR {r['freq_respiratoria']} irpm, "
            f"febre={'sim' if r['febre'] else 'não'}, "
            f"dispneia={'sim' if r['dispneia'] else 'não'}, "
            f"comorbidades: {comorb}, "
            f"classificação de risco: {r['classificacao_risco']}, "
            f"exames pendentes: {pend_txt}."
        )

    def todos_ids(self) -> List[str]:
        return list(self._por_id.keys())
