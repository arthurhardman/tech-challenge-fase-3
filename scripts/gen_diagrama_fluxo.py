"""
gen_diagrama_fluxo.py
---------------------
Gera o diagrama do fluxo LangChain/LangGraph do assistente médico (Fase 3),
salvo em results/figures/fluxo_langchain.png — usado no relatório técnico.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures" / "fluxo_langchain.png"

AZUL = "#2c6fbb"
VERDE = "#2e8b57"
VERMELHO = "#c0392b"
CINZA = "#555555"
LARANJA = "#d98324"


def caixa(ax, xy, w, h, texto, cor, fs=9, fc=None):
    x, y = xy
    box = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.6, edgecolor=cor, facecolor=fc or "#f4f8fc",
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, texto, ha="center", va="center",
            fontsize=fs, color="#1a1a1a", wrap=True)
    return (x + w / 2, y)  # ponto inferior central


def seta(ax, p1, p2, cor=CINZA, rótulo=None, estilo="-|>"):
    arr = FancyArrowPatch(p1, p2, arrowstyle=estilo, mutation_scale=14,
                          linewidth=1.5, color=cor, shrinkA=2, shrinkB=2)
    ax.add_patch(arr)
    if rótulo:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.text(mx + 0.15, my, rótulo, fontsize=8, color=cor, style="italic")


def main() -> Path:
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 11)
    ax.axis("off")
    ax.set_title("Assistente Médico SRAG — Fluxo LangChain + LangGraph (Fase 3)",
                 fontsize=13, weight="bold", pad=14)

    # Coluna esquerda: chain LangChain (RAG)
    ax.text(2.4, 10.4, "Chain LangChain (RAG + contexto)", fontsize=10,
            weight="bold", color=AZUL, ha="center")
    e = caixa(ax, (1.1, 9.2), 2.6, 0.8, "Pergunta do médico\n(+ paciente_id)", AZUL)
    g_in = caixa(ax, (1.1, 8.0), 2.6, 0.8, "Guardrail de entrada\n(bloqueia prescrição)", LARANJA)
    rag = caixa(ax, (0.3, 6.6), 1.7, 0.9, "Retriever\nTF-IDF\n(protocolos)", VERDE)
    pdb = caixa(ax, (2.2, 6.6), 1.5, 0.9, "PatientDB\n(prontuários)", VERDE)
    llm = caixa(ax, (1.1, 5.2), 2.6, 0.8, "LLM customizada\n(Ollama / fine-tuned / mock)", AZUL)
    g_out = caixa(ax, (1.1, 4.0), 2.6, 0.8, "Guardrail de saída\n(disclaimer + fontes)", LARANJA)
    resp = caixa(ax, (1.1, 2.8), 2.6, 0.8, "Resposta + FONTES\n(explainability)", AZUL, fc="#eaf3ea")

    seta(ax, (2.4, 9.2), (2.4, 8.8))
    seta(ax, (2.4, 8.0), (1.15, 7.5), rótulo="")
    seta(ax, (2.4, 8.0), (2.95, 7.5))
    seta(ax, (1.15, 6.6), (2.0, 6.0))
    seta(ax, (2.95, 6.6), (2.6, 6.0))
    seta(ax, (2.4, 5.2), (2.4, 4.8))
    seta(ax, (2.4, 4.0), (2.4, 3.6))

    # Coluna direita: fluxo LangGraph
    ax.text(8.6, 10.4, "Fluxo LangGraph (decisão automatizada)", fontsize=10,
            weight="bold", color=VERMELHO, ha="center")
    t = caixa(ax, (7.3, 9.2), 2.6, 0.8, "triagem\n(risco do paciente)", CINZA)
    ve = caixa(ax, (7.3, 8.0), 2.6, 0.8, "verificar_exames\n(pendentes)", CINZA)
    # roteamento
    ax.text(8.6, 7.4, "risco?", fontsize=9, weight="bold", ha="center", color="#1a1a1a")
    alerta = caixa(ax, (5.6, 6.0), 2.5, 0.9, "emitir_alerta\n(equipe médica)", VERMELHO, fc="#fbeceb")
    conduta = caixa(ax, (9.0, 6.0), 2.6, 0.9, "sugerir_conduta\n(chain LangChain →)", VERDE, fc="#eaf3ea")
    cons = caixa(ax, (7.3, 4.4), 2.6, 0.9, "consolidar\n(resumo + trilha)", CINZA)
    fim = caixa(ax, (7.9, 3.0), 1.4, 0.7, "END", "#1a1a1a", fc="#efefef")

    seta(ax, (8.6, 9.2), (8.6, 8.8))
    seta(ax, (8.6, 8.0), (8.6, 7.6))
    seta(ax, (8.4, 7.3), (6.85, 6.9), cor=VERMELHO, rótulo="vermelho")
    seta(ax, (8.8, 7.3), (10.3, 6.9), cor=VERDE, rótulo="verde/amarelo")
    seta(ax, (6.85, 6.0), (8.2, 5.3), cor=VERMELHO)
    seta(ax, (10.3, 6.0), (9.0, 5.3), cor=VERDE)
    seta(ax, (8.6, 4.4), (8.6, 3.7))

    # Ponte: sugerir_conduta usa a chain LangChain (esquerda) — conector curvo
    ponte = FancyArrowPatch(
        (9.0, 6.45), (3.7, 3.2), arrowstyle="-|>", mutation_scale=14,
        linewidth=1.4, color=AZUL, linestyle=(0, (4, 3)),
        connectionstyle="arc3,rad=-0.35", shrinkA=4, shrinkB=4,
    )
    ax.add_patch(ponte)
    ax.text(6.35, 5.7, "reutiliza a chain\n(RAG + guardrails)", fontsize=8,
            color=AZUL, style="italic", ha="center")

    # Auditoria (transversal)
    aud = caixa(ax, (4.6, 1.2), 3.0, 0.8, "AuditLogger — log + JSONL\n(rastreamento/auditoria)", LARANJA, fc="#fdf1e3")
    seta(ax, (2.4, 2.8), (4.9, 2.0), cor=LARANJA)
    seta(ax, (8.6, 3.0), (7.3, 2.0), cor=LARANJA)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("[ok]", OUT)
    return OUT


if __name__ == "__main__":
    main()
