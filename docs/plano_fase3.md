# Plano — Tech Challenge Fase 3

**FIAP Pós-Tech — Machine Learning Engineering**
**Tema:** Assistente virtual médico com fine-tuning de LLM + fluxos de decisão com LangChain/LangGraph
**Repositório:** `ml-tech` (continuação das Fases 1 e 2)

---

## 1. Objetivo

Construir um **assistente virtual médico** treinado com dados internos do
"hospital" e organizar **fluxos de decisão automatizados e seguros** com
LangChain/LangGraph. Mantemos o domínio das fases anteriores — **SRAG**
(Síndrome Respiratória Aguda Grave) — de modo que o projeto seja coeso: o mesmo
hospital que classificou desfechos (Fase 1) e otimizou modelos com AG + LLM
explicativa (Fase 2) agora ganha um assistente clínico sobre seus protocolos.

## 2. Requisitos do enunciado × como atendemos

| # | Requisito | Como é atendido | Onde |
|---|-----------|-----------------|------|
| 1 | Fine-tuning com protocolos, FAQ e modelos de laudos/receitas | Dataset de *instruction tuning* montado das 3 fontes; LoRA/PEFT (modo real) + laço demo offline | `src/assistant/finetuning/` |
| 1 | Preprocessing, anonimização e curadoria | `data_prep.py`: regex de PII (CPF, CNS, telefone, e-mail…), normalização, dedupe | `finetuning/data_prep.py` |
| 2 | Pipeline LangChain integrando a LLM customizada | `CustomMedicalLLM` (wrapper do cliente Ollama/mock/fine-tuned) + chain | `chains/llm_backend.py`, `chains/medical_assistant.py` |
| 2 | Consulta a base estruturada (prontuários) | `PatientDB` sobre `prontuarios.json` (sintético/anonimizado) | `knowledge/patient_db.py` |
| 2 | Contextualização com dados do paciente | Resumo clínico injetado no prompt | `medical_assistant.py` |
| 3 | Limites (nunca prescrever sem validação) | Guardrails de entrada/saída | `safety/guardrails.py` |
| 3 | Logging detalhado para auditoria | `AuditLogger` (texto + JSONL) | `safety/audit_log.py` |
| 3 | Explainability (indicar a fonte) | RAG TF-IDF que retorna o protocolo de origem; fontes na resposta | `knowledge/retriever.py` |
| — | Fluxos do LangGraph | `FluxoAtendimento` (triagem → exames → alerta/conduta → consolidar) | `chains/graph.py` |
| 4 | Código modular + README | Pacote `src/assistant/` + seção no README | — |

## 3. Decisões de escopo (explícitas para avaliação)

1. **Domínio mantido (SRAG).** Coerência com as Fases 1–2. Os protocolos, FAQ e
   prontuários são sobre SRAG.
2. **Dados híbridos: hospital sintético (PT) + PubMedQA/MedQuAD (EN).** Os
   protocolos, FAQ e prontuários do hospital são sintéticos e anonimizados
   (nenhum dado real). Os datasets sugeridos entram como dados complementares.
   As funções de anonimização são reais e aplicáveis a produção.
3. **Fine-tuning: código real (LoRA/PEFT) + modo demo offline.** Espelha a
   filosofia da Fase 2 (Ollama ↔ mock): o modo `real` roda em GPU com um
   modelo-base do Hugging Face; o modo `demo` (padrão) executa todo o pipeline
   de dados e um laço de treino simulado que produz artefatos reais (curva de
   perda, métricas), garantindo reprodutibilidade na máquina do avaliador.
4. **RAG leve com TF-IDF** (scikit-learn), em vez de vetor store pesado —
   controla a explainability (retorna a fonte) e roda sem rede/GPU.
5. **Backend da LLM plugável.** As chains não dependem do backend: hoje
   Ollama/mock; amanhã a LLM fine-tuned, sem alterar o código do assistente.

## 4. Datasets sugeridos (PubMedQA / MedQuAD) — integrados

Adotamos a **abordagem híbrida**: os dados internos do hospital (SRAG, PT-BR)
são combinados com os dois datasets sugeridos, no fine-tuning e no RAG.

- **PubMedQA** (Jin et al., 2019, licença MIT) — 250 exemplos da fatia curada do
  subconjunto rotulado; entram no fine-tuning.
- **MedQuAD** (Ben Abacha & Demner-Fushman, 2019, licença CC BY 4.0) — 350
  exemplos dos 9 subconjuntos do NIH com resposta; entram no fine-tuning **e** no
  RAG como base de referência complementar (com citação de fonte).

Loaders em `finetuning/external_datasets.py`; download em `scripts/fetch_datasets.py`;
citações em `data/knowledge_base/external/CITATIONS.md`. Os dados do hospital
atendem ao requisito literal ("dados próprios do hospital") e os datasets à
sugestão — cobrindo os dois.

## 5. Entregáveis

- Código (fine-tuning + LangChain + LangGraph) em `src/assistant/`
- Base sintética anonimizada em `data/knowledge_base/`
- Notebook `notebooks/08_finetuning_langchain.ipynb`
- Relatório técnico `docs/relatorio_fase3.md` + diagrama `results/figures/fluxo_langchain.png`
- Testes `tests/test_finetuning.py`, `tests/test_assistant.py`
- Roteiro do vídeo `docs/roteiro_video_fase3.md`
