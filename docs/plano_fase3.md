# Plano — Tech Challenge Fase 3

**FIAP Pós-Tech — Machine Learning Engineering**  
**Tema:** Assistente médico com fine-tuning, LangChain e LangGraph  
**Domínio:** SRAG / SIVEP-Gripe

---

## 1. Objetivo

A Fase 3 continua o projeto das fases anteriores no mesmo domínio de SRAG. A ideia é ter um assistente que consiga consultar protocolos, usar dados estruturados do paciente, explicar de onde veio a informação e executar um fluxo de triagem com validação humana.

O projeto não tenta substituir o médico. Pedidos de prescrição/dose são bloqueados e as respostas sempre deixam explícita a necessidade de validação profissional.

## 2. Como os requisitos são atendidos

| Requisito | Implementação |
|---|---|
| Fine-tuning de LLM | `src/finetuning/train_lora.py` com LoRA/PEFT sobre Falcon/LLaMA/Mistral |
| Validação de treino sem GPU | `src/finetuning/local_validation.py`, Transformer pequeno treinado de verdade em CPU |
| Protocolos/FAQ/documentos | protocolos oficiais SRAG + 100 Q&As curados + poucos exemplos sintéticos de formatos internos |
| PubMedQA e MedQuAD | entram no dataset de fine-tuning e também no RAG |
| Preprocessing/anonimização | `src/finetuning/dataset_prep.py` + seleção de campos no adaptador SIVEP |
| Dados estruturados | `PatientDB` sobre SQLite com amostra real anonimizada do SIVEP |
| Base completa | adaptador aceita os CSVs anuais do OpenDataSUS sem carregar todos os anos juntos na memória |
| LangChain | `MedicalAssistant` + backend compatível com LangChain |
| LangGraph | fluxo triagem → exames → alerta/conduta → consolidação |
| Segurança | guardrails de entrada e saída, inclusive remoção de instruções de dose |
| Logging/auditoria | `AuditLogger` com pergunta, paciente, backend, fontes e decisão dos guardrails |
| Explainability | RAG retorna arquivo, página, `chunk_id` e score quando disponíveis |

## 3. Dados usados

### Pacientes

O repositório acompanha uma amostra de **8.000 registros reais anonimizados do SIVEP**, distribuída entre 2023, 2024, 2025 e 2026. Ela é convertida para SQLite durante a execução.

O adaptador também foi validado com os quatro arquivos anuais completos usados pelo grupo, totalizando **1.071.699 registros**.

### Conhecimento textual

O RAG combina:

- protocolos originais do projeto;
- 5 documentos oficiais relacionados a SRAG/vigilância;
- 857 chunks com arquivo e página;
- PubMedQA;
- MedQuAD.

Para o domínio SRAG existem ainda **100 perguntas/respostas curadas**, com 20 perguntas separadas para avaliação.

## 4. Fine-tuning

O dataset principal fica em `src/finetuning/` e combina:

- 1.500 exemplos MedQuAD;
- 1.000 PubMedQA;
- 100 exemplos oficiais de SRAG;
- 6 exemplos sintéticos de formatos internos.

Após deduplicação são **2.587 exemplos únicos**. O caminho de entrega usa LoRA/PEFT. Como esse treino depende de GPU, existe também um Transformer pequeno para validar em CPU, sem simular loss, o fluxo completo de treino/checkpoint/inferência.

## 5. Fluxo do assistente

```text
pergunta
   ↓
guardrail de entrada
   ↓
RAG + dados do paciente
   ↓
backend ajustado / Ollama / checkpoint local
   ↓
guardrail de saída
   ↓
resposta + fontes + auditoria
```

O LangGraph organiza a etapa operacional:

```text
triagem → verificar_exames → emitir_alerta | sugerir_conduta → consolidar
```

## 6. Avaliação esperada

A avaliação usa as 20 perguntas SRAG que não entram no treino e mede:

- recuperação da fonte correta;
- recuperação de fonte + página;
- presença do aviso de validação médica;
- bloqueio de prescrição;
- recuperação de MedQuAD e PubMedQA em testes de sanidade.

## 7. Entregáveis

- código modular em `src/assistant/` e `src/finetuning/`;
- amostra SIVEP anonimizada em `data/knowledge_base/sivep/`;
- protocolos/fontes em `data/knowledge_base/official/`;
- notebook `notebooks/08_finetuning_langchain.ipynb`;
- relatório `docs/relatorio_fase3.md`;
- diagrama em `results/figures/fluxo_langchain.png`;
- testes automatizados em `tests/`.
