"""Recuperação de contexto clínico com rastreio de fonte.

O índice junta três tipos de material:
- protocolos sintéticos que já existiam no projeto;
- protocolos oficiais versionados como chunks com arquivo e página;
- PubMedQA/MedQuAD como referência médica complementar.

A busca continua em TF-IDF porque o corpus é pequeno o bastante para isso e o
resultado é rápido, determinístico e simples de explicar no vídeo.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .. import config


@dataclass
class Trecho:
    protocolo: str
    titulo_secao: str
    texto: str
    score: float = 0.0
    page: Optional[int] = None
    chunk_id: Optional[str] = None
    tipo: str = "protocolo"

    def citacao(self) -> str:
        if self.page is not None:
            return f"{self.protocolo}, p. {self.page}"
        return f"{self.protocolo} — {self.titulo_secao}"


class ProtocolRetriever:
    """Índice TF-IDF híbrido com proveniência por documento/página quando disponível."""

    def __init__(
        self,
        protocolos_dir: Optional[Path] = None,
        incluir_medquad: bool = True,
        incluir_pubmedqa: bool = True,
        incluir_oficiais: bool = True,
    ):
        self.dir = Path(protocolos_dir or config.PROTOCOLOS_DIR)
        self.incluir_medquad = incluir_medquad
        self.incluir_pubmedqa = incluir_pubmedqa
        self.incluir_oficiais = incluir_oficiais
        self._trechos: List[Trecho] = []
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._matrix = None
        self._carregar()

    def _carregar(self) -> None:
        self._carregar_protocolos_markdown()
        if self.incluir_oficiais:
            self._carregar_protocolos_oficiais()
        if self.incluir_medquad:
            self._carregar_jsonl_externo(config.KB_DIR / "external" / "medquad.jsonl", "MedQuAD")
        if self.incluir_pubmedqa:
            self._carregar_jsonl_externo(config.KB_DIR / "external" / "pubmedqa.jsonl", "PubMedQA")

        if self._trechos:
            corpus = [
                f"{t.protocolo} {t.titulo_secao} {t.texto}"
                for t in self._trechos
            ]
            self._vectorizer = TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                ngram_range=(1, 2),
                min_df=1,
                sublinear_tf=True,
            )
            self._matrix = self._vectorizer.fit_transform(corpus)

    def _carregar_protocolos_markdown(self) -> None:
        for md in sorted(self.dir.glob("*.md")):
            cod = md.stem.split("_")[0]
            texto = md.read_text(encoding="utf-8")
            secoes = texto.split("\n## ")
            for sec in secoes[1:]:
                linhas = sec.splitlines()
                titulo = linhas[0].split(". ", 1)[-1].strip()
                corpo = "\n".join(linhas[1:]).strip()
                if corpo:
                    self._trechos.append(
                        Trecho(cod, titulo, corpo, tipo="protocolo_sintetico")
                    )

    def _carregar_protocolos_oficiais(self) -> None:
        path = config.OFFICIAL_CHUNKS_PATH
        if not path.exists():
            return
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            self._trechos.append(
                Trecho(
                    protocolo=item.get("source", "protocolo_oficial"),
                    titulo_secao=f"página {item.get('page', '?')}",
                    texto=item.get("text", ""),
                    page=int(item["page"]) if item.get("page") is not None else None,
                    chunk_id=item.get("chunk_id"),
                    tipo="protocolo_oficial",
                )
            )

    def _carregar_jsonl_externo(self, path: Path, fallback: str) -> None:
        if not path.exists():
            return
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            fonte = item.get("fonte") or item.get("source") or fallback
            pergunta = item.get("prompt") or item.get("question") or ""
            resposta = item.get("response") or item.get("answer") or ""
            if not resposta:
                continue
            self._trechos.append(
                Trecho(
                    protocolo=fonte,
                    titulo_secao=pergunta[:100],
                    texto=resposta,
                    tipo=fallback.lower(),
                )
            )

    def buscar(self, consulta: str, top_k: int = 3) -> List[Trecho]:
        if not self._trechos or self._vectorizer is None:
            return []
        q = self._vectorizer.transform([consulta])
        sims = cosine_similarity(q, self._matrix)[0]
        ordenados = sims.argsort()[::-1][: max(1, top_k)]
        resultados: List[Trecho] = []
        for i in ordenados:
            if sims[i] <= 0:
                continue
            t = self._trechos[int(i)]
            resultados.append(
                Trecho(
                    protocolo=t.protocolo,
                    titulo_secao=t.titulo_secao,
                    texto=t.texto,
                    score=float(sims[i]),
                    page=t.page,
                    chunk_id=t.chunk_id,
                    tipo=t.tipo,
                )
            )
        return resultados

    def contexto_formatado(self, trechos: List[Trecho]) -> str:
        if not trechos:
            return "(nenhuma fonte relevante encontrada)"
        return "\n\n".join(
            f"[Fonte: {t.citacao()}]\n{t.texto}" for t in trechos
        )

    def total_trechos(self) -> int:
        return len(self._trechos)
