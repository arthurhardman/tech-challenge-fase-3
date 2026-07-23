"""
retriever.py
------------
Recuperação de contexto (RAG) sobre os protocolos internos, com rastreio de
FONTE — peça central da *explainability* exigida pelo enunciado (requisito 3).

Implementação leve e sem rede: indexa os protocolos em chunks e usa TF-IDF +
similaridade de cosseno (scikit-learn, já dependência do projeto). Cada trecho
recuperado carrega o código do protocolo de origem (ex.: PROT-SRAG-02), o que
permite ao assistente citar explicitamente de onde veio a informação.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .. import config


@dataclass
class Trecho:
    """Um chunk recuperável de protocolo, com metadados de fonte."""
    protocolo: str          # ex.: "PROT-SRAG-02"
    titulo_secao: str       # ex.: "Alvo de saturação"
    texto: str
    score: float = 0.0

    def citacao(self) -> str:
        return f"{self.protocolo} — {self.titulo_secao}"


class ProtocolRetriever:
    """
    Índice TF-IDF do conhecimento clínico, com recuperação top-k e fontes.

    Base primária: protocolos internos SRAG (PT-BR). Opcionalmente, inclui o
    MedQuAD (EN) como base de referência complementar — abordagem híbrida. Toda
    fonte é rastreada (explainability), seja um protocolo (PROT-SRAG-0x) ou um
    subconjunto do MedQuAD (MedQuAD:CDC, etc.).
    """

    def __init__(self, protocolos_dir: Optional[Path] = None, incluir_medquad: bool = True):
        self.dir = Path(protocolos_dir or config.PROTOCOLOS_DIR)
        self.incluir_medquad = incluir_medquad
        self._trechos: List[Trecho] = []
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._matrix = None
        self._carregar()

    def _carregar(self) -> None:
        """Lê os protocolos (.md) e, opcionalmente, o MedQuAD; monta o TF-IDF."""
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
                        Trecho(protocolo=cod, titulo_secao=titulo, texto=corpo)
                    )

        if self.incluir_medquad:
            self._carregar_medquad()

        if self._trechos:
            corpus = [f"{t.titulo_secao} {t.texto}" for t in self._trechos]
            self._vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
            self._matrix = self._vectorizer.fit_transform(corpus)

    def _carregar_medquad(self) -> None:
        """Indexa as entradas do MedQuAD (fatia versionada) como trechos citáveis."""
        import json
        slice_path = config.KB_DIR / "external" / "medquad.jsonl"
        if not slice_path.exists():
            return
        for linha in slice_path.read_text(encoding="utf-8").splitlines():
            if not linha.strip():
                continue
            item = json.loads(linha)
            self._trechos.append(Trecho(
                protocolo=item.get("fonte", "MedQuAD"),
                titulo_secao=item.get("prompt", "")[:80],
                texto=item.get("response", ""),
            ))

    def buscar(self, consulta: str, top_k: int = 3) -> List[Trecho]:
        """Retorna os `top_k` trechos mais relevantes para a consulta."""
        if not self._trechos or self._vectorizer is None:
            return []
        q = self._vectorizer.transform([consulta])
        sims = cosine_similarity(q, self._matrix)[0]
        ordenados = sims.argsort()[::-1][:top_k]
        resultados: List[Trecho] = []
        for i in ordenados:
            if sims[i] <= 0:
                continue
            t = self._trechos[i]
            resultados.append(
                Trecho(t.protocolo, t.titulo_secao, t.texto, score=float(sims[i]))
            )
        return resultados

    def contexto_formatado(self, trechos: List[Trecho]) -> str:
        """Monta o bloco de contexto para o prompt, com marcação de fonte."""
        if not trechos:
            return "(nenhum protocolo relevante encontrado)"
        blocos = []
        for t in trechos:
            blocos.append(f"[Fonte: {t.citacao()}]\n{t.texto}")
        return "\n\n".join(blocos)
