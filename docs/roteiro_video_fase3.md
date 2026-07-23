# Roteiro do Vídeo — Fase 3 (até 15 min)

**Objetivo:** demonstrar o assistente médico (fine-tuning + LangChain/LangGraph),
um fluxo automatizado, respostas clínicas contextualizadas e os logs/validação.

Dica: rode tudo com `LLM_BACKEND=mock` para garantir que funcione ao vivo,
offline. Se tiver Ollama, mostre também com `LLM_BACKEND=ollama`.

---

## 0. Abertura (0:00–1:00)
- Apresentar o grupo e recapitular o projeto: SRAG, Fases 1–2 (classificação e
  otimização + LLM explicativa) e o salto da Fase 3 (assistente + fluxos).
- Mostrar a estrutura `src/assistant/` e o diagrama `results/figures/fluxo_langchain.png`.

## 1. Dados e fine-tuning (1:00–4:30)
- Rodar `python scripts/gen_synthetic_data.py` — mostrar protocolos, FAQ,
  laudos e prontuários sintéticos/anonimizados.
- Mostrar `data_prep.anonimizar()` mascarando CPF/e-mail (ao vivo, no notebook).
- **Datasets sugeridos (abordagem híbrida):** explicar que, além dos dados do
  hospital (SRAG, PT-BR), o projeto incorpora **PubMedQA** (licença MIT) e
  **MedQuAD** (CC BY 4.0). Mostrar a célula `carregar_externos()` no notebook com
  uma amostra de cada. Citar `data/knowledge_base/external/CITATIONS.md`.
- Rodar `python -m src.assistant.finetuning.dataset_builder` — **627 exemplos**
  (27 hospital + 250 PubMedQA + 350 MedQuAD); apontar a composição impressa.
- Rodar `python -m src.assistant.finetuning.train --mode demo` — abrir a curva
  de perda `results/figures/finetuning_loss.png`.
- Explicar a decisão modo `real` (LoRA/PEFT, GPU) × modo `demo` (offline), e o
  porquê do híbrido: atende ao requisito literal ("dados do hospital") **e** à
  sugestão de datasets.

## 2. Assistente com LangChain (4:30–8:00)
- No notebook `08_finetuning_langchain.ipynb`:
  - Pergunta clínica: *"Qual o alvo de saturação em oxigenoterapia?"* → mostrar
    resposta **com fonte** (PROT-SRAG-02) e o disclaimer.
  - **Guardrail:** *"Prescreva a dose exata de corticoide"* → mostrar bloqueio.
  - **Contexto do paciente:** escolher um `PAC-XXXX`, mostrar o resumo clínico e
    uma resposta contextualizada (exames pendentes).
  - **RAG híbrido:** perguntar *"What are the symptoms of leukemia?"* → mostrar
    que o assistente responde a partir do **MedQuAD**, citando a fonte
    (`MedQuAD:CancerGov`). Explicar: perguntas de SRAG puxam os protocolos PT-BR;
    perguntas gerais puxam o MedQuAD — sempre com a fonte.

## 3. Fluxo automatizado com LangGraph (8:00–11:00)
- `python -m src.assistant.cli fluxo --paciente PAC-0001` (paciente vermelho):
  - mostrar a **trilha** `triagem → verificar_exames → emitir_alerta → consolidar`
    e o **alerta** para a equipe.
- Rodar o fluxo para um paciente verde/amarelo → trilha com `sugerir_conduta`.
- Explicar o roteamento condicional por risco no `graph.py`.

## 4. Segurança, auditoria e explainability (11:00–13:00)
- Abrir `results/finetuning/audit_events.jsonl` — mostrar um evento com fontes,
  backend, paciente e decisão de guardrail.
- Reforçar: a resposta sempre cita o protocolo (explainability) e nunca
  prescreve sem validação humana.

## 5. Avaliação e encerramento (13:00–15:00)
- `python -m src.assistant.finetuning.evaluate` — mostrar as métricas (fonte
  SRAG 100%, recuperação MedQuAD 100%, disclaimer 100%, bloqueio 100%, cobertura
  ~59%) e a composição do dataset (hospital + PubMedQA + MedQuAD); analisá-las.
- Rodar `pytest tests/test_finetuning.py tests/test_assistant.py -v` (18 testes).
- Fechar: arquitetura plugável — trocar o backend ativa a LLM fine-tuned real
  sem mudar as chains; e o híbrido cobre requisito literal + datasets sugeridos.

---

### Checklist de gravação
- [ ] Terminal com fonte legível e `LLM_BACKEND=mock` exportado.
- [ ] Notebook já com o kernel selecionado.
- [ ] Figuras abertas: `finetuning_loss.png`, `fluxo_langchain.png`.
- [ ] `audit_events.jsonl` à mão.
