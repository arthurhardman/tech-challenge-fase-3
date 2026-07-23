# Tech Challenge — SRAG (Fase 3)
## Fase 3 — Assistente Médico com Fine-tuning + LangChain/LangGraph

A Fase 3 adiciona um **assistente virtual médico** sobre o mesmo domínio SRAG:
ele responde dúvidas clínicas com base nos **protocolos internos**, consulta
**prontuários** e coordena um **fluxo de decisão automatizado e seguro**. Todos
os dados são **sintéticos e anonimizados**.

Relatório completo: [`docs/relatorio_fase3.md`](docs/relatorio_fase3.md) ·
Plano: [`docs/plano_fase3.md`](docs/plano_fase3.md) ·
Diagrama: `results/figures/fluxo_langchain.png`

### Datasets (abordagem híbrida)

O fine-tuning combina os **dados internos do hospital** (SRAG, PT-BR) com os dois
datasets sugeridos no enunciado:

- **PubMedQA** (MIT) — 250 exemplos (subconjunto rotulado). Fine-tuning.
- **MedQuAD** (CC BY 4.0) — 350 exemplos (9 subconjuntos do NIH com resposta).
  Fine-tuning **e** RAG complementar.

Total do dataset de instrução: **627 exemplos** (27 hospital + 250 + 350).
Citações/licenças: [`data/knowledge_base/external/CITATIONS.md`](data/knowledge_base/external/CITATIONS.md).

```bash
python scripts/fetch_datasets.py                       # baixa PubMedQA + MedQuAD
python -m src.assistant.finetuning.external_datasets   # gera as fatias curadas (offline depois disso)
```
> As fatias curadas já vêm versionadas, então o pipeline roda offline sem baixar nada.

### O que foi entregue

| Requisito | Implementação |
|-----------|---------------|
| Fine-tuning com dados internos | `src/assistant/finetuning/` — dataset de instrução a partir de protocolos, FAQ e modelos de laudo; LoRA/PEFT (modo real) + laço demo offline |
| Preprocessing, anonimização, curadoria | `finetuning/data_prep.py` (regex de PII, normalização, dedupe) |
| Assistente com LangChain | `chains/medical_assistant.py` (RAG + contexto do paciente + guardrails) |
| Consulta a base estruturada | `knowledge/patient_db.py` (prontuários sintéticos) |
| Fluxos do LangGraph | `chains/graph.py` (triagem → exames → alerta/conduta → consolidar) |
| Segurança (nunca prescrever) | `safety/guardrails.py` |
| Logging/auditoria | `safety/audit_log.py` (texto + JSONL) |
| Explainability (fonte) | `knowledge/retriever.py` (RAG TF-IDF que retorna o protocolo) |

### Estrutura (Fase 3)

```
data/knowledge_base/          # base sintética anonimizada
  ├── protocolos/*.md
  ├── faq_medicos.jsonl
  ├── laudos_modelos.jsonl
  └── prontuarios.json
src/assistant/
  ├── config.py
  ├── finetuning/  (data_prep, dataset_builder, train, evaluate)
  ├── knowledge/   (retriever, patient_db)
  ├── safety/      (guardrails, audit_log)
  ├── chains/      (llm_backend, medical_assistant, graph)
  └── cli.py
notebooks/08_finetuning_langchain.ipynb
```

### Como executar (offline, sem GPU)

```bash
# instala langchain-core, langgraph, nbformat (já no requirements.txt)
pip install -r requirements.txt

# pipeline completo da Fase 3 (base → dataset → fine-tuning demo → diagrama → avaliação → fluxo)
python run_fase3.py            # ou: make fase3

# comandos individuais
python scripts/gen_synthetic_data.py                       # base sintética
python -m src.assistant.finetuning.train --mode demo       # fine-tuning (demo)
python -m src.assistant.finetuning.evaluate                # métricas
python -m src.assistant.cli pacientes                      # lista pacientes
python -m src.assistant.cli perguntar "Qual o alvo de SpO2?" --paciente PAC-0003
python -m src.assistant.cli fluxo --paciente PAC-0001      # fluxo LangGraph

# testes da Fase 3 (18)
pytest tests/test_finetuning.py tests/test_assistant.py -v
```

> **Backend plugável.** Por padrão roda com o backend `mock` (offline). Com
> Ollama disponível, use `LLM_BACKEND=ollama`. O mesmo código passa a servir a
> LLM *fine-tuned* real trocando apenas o backend — as chains não mudam.

### Fine-tuning real (GPU)

O modo `demo` (padrão) simula o laço de treino e produz artefatos reais (curva
de perda, métricas), garantindo reprodutibilidade sem GPU. Para o fine-tuning
real por **LoRA/PEFT**:

```bash
pip install torch transformers peft datasets accelerate
python -m src.assistant.finetuning.train --mode real --epochs 3
```

### Resultados da avaliação (assistente, backend mock)

| Métrica | Valor |
|---------|-------|
| Acurácia de fonte (RAG, gold SRAG) | 100% |
| Recuperação MedQuAD (RAG híbrido) | 100% |
| Taxa de disclaimer de validação | 100% |
| Bloqueio de prescrição direta | 100% |
| Cobertura de termos (proxy) | ~59% |
