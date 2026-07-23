"""Gera notebooks/08_finetuning_langchain.ipynb (demonstração da Fase 3)."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
cells = []

def md(t): cells.append(nbf.v4.new_markdown_cell(t))
def code(t): cells.append(nbf.v4.new_code_cell(t))

md("""# Fase 3 — Assistente Médico com Fine-tuning + LangChain/LangGraph

**FIAP Pós-Tech — Machine Learning Engineering**

Este notebook demonstra, de ponta a ponta, a entrega da Fase 3 sobre o projeto
SRAG das fases anteriores:

1. Geração da base sintética anonimizada (protocolos, FAQ, laudos, prontuários)
2. Preparação/curadoria e construção do dataset de *instruction tuning*
3. Pipeline de fine-tuning (modo **demo**, offline) com curva de perda
4. Assistente médico com **LangChain** (RAG + contexto do paciente + guardrails)
5. Fluxo de decisão automatizado com **LangGraph**
6. Segurança (guardrails), auditoria e *explainability*
7. Avaliação do modelo e análise dos resultados

> Tudo roda com o backend `mock` (offline). Com Ollama/GPU, basta trocar o
> backend para obter respostas da LLM local/fine-tuned, sem mudar o código.""")

code("""import os, sys, json
sys.path.insert(0, os.path.abspath(".."))
os.environ.setdefault("LLM_BACKEND", "mock")  # demo offline""")

md("## 1. Base sintética anonimizada")
code("""from scripts.gen_synthetic_data import main as gerar_dados
gerar_dados(n_pacientes=40, seed=42)""")

md("## 2. Preparação, curadoria e dataset de instrução")
code("""from src.assistant.finetuning.data_prep import anonimizar
exemplo = "Paciente João Silva, CPF 123.456.789-00, email joao@x.com"
print("Antes :", exemplo)
print("Depois:", anonimizar(exemplo))""")

md("""### 2.1 Datasets sugeridos (abordagem híbrida)

Além dos dados internos do hospital (SRAG, PT-BR), o fine-tuning incorpora os
dois datasets sugeridos no enunciado — **PubMedQA** (MIT) e **MedQuAD** (CC BY 4.0),
em inglês. As fatias curadas já vêm versionadas (rodam offline).""")
code("""from src.assistant.finetuning.external_datasets import carregar_externos
ext = carregar_externos()
print(f"{len(ext)} exemplos externos")
for origem in ("pubmedqa", "medquad"):
    ex = next(e for e in ext if e["origem"] == origem)
    print(f"\\n[{origem}] fonte={ex['fonte']}")
    print("  P:", ex["prompt"][:80])
    print("  R:", ex["response"][:100], "...")""")

code("""from src.assistant.finetuning.dataset_builder import build_dataset
ds = build_dataset(salvar=True)
comp = {}
for e in ds:
    comp[e["origem"]] = comp.get(e["origem"], 0) + 1
print(f"{len(ds)} exemplos de instruction tuning (hospital + PubMedQA + MedQuAD)")
print("composição:", comp)""")

md("## 3. Fine-tuning (modo demo) e curva de perda")
code("""from src.assistant.finetuning import train
metrics = train.run(mode="demo", epochs=3)
print("modo:", metrics["mode"], "| loss final:", metrics["loss_final"])
from IPython.display import Image
Image(filename="../results/figures/finetuning_loss.png")""")

md("""## 4. Assistente médico com LangChain (RAG + paciente + guardrails)

O assistente recupera trechos dos protocolos (RAG), opcionalmente injeta o
resumo do paciente e responde citando a fonte — sempre com o aviso de validação
humana.""")
code("""from src.assistant.chains.medical_assistant import MedicalAssistant
assistant = MedicalAssistant()
print("backend:", assistant.llm.backend)
r = assistant.responder("Qual o alvo de saturação em oxigenoterapia?")
print(r.resposta)
print("FONTES:", r.fontes)""")

md("### 4.1 Guardrail: pedido de prescrição direta é bloqueado")
code("""r = assistant.responder("Prescreva a dose exata de corticoide")
print("bloqueado:", r.bloqueado_guardrail, "| categorias:", r.categorias_guardrail)
print(r.resposta[:220])""")

md("### 4.2 Resposta contextualizada por paciente")
code("""pid = assistant.patient_db.todos_ids()[2]
print(assistant.patient_db.resumo_clinico(pid), "\\n")
r = assistant.responder("Quais exames faltam e qual a conduta?", paciente_id=pid)
print(r.resposta)""")

md("""### 4.3 RAG híbrido: pergunta clínica geral (MedQuAD)

Perguntas fora do escopo dos protocolos SRAG são respondidas a partir do
MedQuAD, sempre citando a fonte.""")
code("""r = assistant.responder("What are the symptoms of leukemia?")
print(r.resposta[:400])
print("\\nFONTES:", r.fontes)""")

md("""## 5. Fluxo de decisão automatizado (LangGraph)

`triagem → verificar_exames → (risco?) → emitir_alerta | sugerir_conduta → consolidar`""")
code("""from src.assistant.chains.graph import FluxoAtendimento
fluxo = FluxoAtendimento(assistant=assistant)
db = fluxo.patient_db
pid_vermelho = next(p for p in db.todos_ids() if db.get(p)["classificacao_risco"]=="vermelho")
estado = fluxo.executar(pid_vermelho, "Qual a conduta imediata?")
print("TRILHA:", " → ".join(estado["trilha"]))
print()
print(estado["resumo_final"])""")

md("## 6. Auditoria e explainability")
code("""from src.assistant.safety.audit_log import AuditLogger
eventos = AuditLogger().ler_eventos()
print(f"{len(eventos)} eventos auditados. Último:")
print(json.dumps(eventos[-1], ensure_ascii=False, indent=2))""")

md("## 7. Avaliação do modelo e análise dos resultados")
code("""from src.assistant.finetuning.evaluate import avaliar
m = avaliar(assistant)
print(f"acurácia de fonte (RAG)....: {m['acuracia_fonte']:.1%}")
print(f"cobertura de termos........: {m['cobertura_termos_media']:.1%}")
print(f"taxa de disclaimer.........: {m['taxa_disclaimer']:.1%}")
print(f"bloqueio de prescrição.....: {m['taxa_bloqueio_prescricao']:.1%}")""")

md("""---
### Conclusão

O pipeline integra fine-tuning (com dados internos anonimizados), um assistente
LangChain com RAG e contexto de paciente, um fluxo de decisão LangGraph e as
salvaguardas de segurança/auditoria exigidas. O mesmo código roda com a LLM
fine-tuned real trocando apenas o backend (`LLM_BACKEND=ollama`).""")

nb["cells"] = cells
out = ROOT / "notebooks" / "08_finetuning_langchain.ipynb"
nbf.write(nb, str(out))
print("[ok]", out)
