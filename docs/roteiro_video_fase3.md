# Roteiro do Vídeo — Fase 3 (até 15 min)

**Objetivo:** mostrar o treino, o assistente contextualizado, o fluxo LangGraph, segurança, fontes e auditoria sem depender de dados fictícios para a demonstração principal.

---

## 0. Abertura (0:00–1:00)

- Recapitular rapidamente Fases 1 e 2.
- Explicar que a Fase 3 adiciona fine-tuning, RAG, LangChain/LangGraph e consulta estruturada de pacientes.
- Mostrar a estrutura `src/assistant/`, `src/finetuning/` e o diagrama.

## 1. Dados e preparação (1:00–4:00)

- Mostrar `patients_sivep_sample.csv` e explicar que são 8 mil registros reais anonimizados do SIVEP, de 2023 a 2026.
- Mostrar que o adaptador também aceita os CSVs completos do OpenDataSUS.
- Abrir `data/knowledge_base/official/` e mostrar os protocolos oficiais/chunks com página.
- Mostrar a composição do dataset: MedQuAD + PubMedQA + 100 Q&As SRAG + poucos formatos internos sintéticos.
- Demonstrar rapidamente a anonimização com um texto contendo nome/CPF/e-mail.

## 2. Fine-tuning (4:00–6:30)

- Rodar `python -m src.finetuning.train_lora --dry-run` para validar o dataset do LoRA.
- Explicar que o treino LoRA completo requer GPU e usa Falcon/LLaMA/Mistral.
- Mostrar a validação real em CPU (`src/finetuning/local_validation.py`) e as métricas salvas, deixando claro que esse Transformer pequeno valida o pipeline, mas não substitui o LLM pré-treinado.

## 3. Assistente contextualizado (6:30–9:30)

- No notebook `08_finetuning_langchain.ipynb`, selecionar um paciente real anonimizado do SQLite.
- Fazer uma pergunta sobre sinais de atenção no caso.
- Mostrar a resposta e as fontes com arquivo/página.
- Fazer uma pergunta geral de saúde para mostrar a recuperação complementar de MedQuAD/PubMedQA.

## 4. Segurança e LangGraph (9:30–12:30)

- Perguntar uma dose/prescrição direta e mostrar que o guardrail bloqueia.
- Mostrar o fluxo `triagem → verificar_exames → emitir_alerta/sugerir_conduta → consolidar`.
- Explicar que risco e exames vêm somente dos campos existentes no PatientDB; o fluxo não inventa medidas ausentes.

## 5. Avaliação, logs e fechamento (12:30–15:00)

- Mostrar `results/finetuning/eval_metrics.json`:
  - fonte correta top-k: 80%;
  - fonte + página: 70%;
  - aviso médico: 100%;
  - bloqueio de prescrição: 100%;
  - sanity checks MedQuAD/PubMedQA: 100%.
- Abrir o log de auditoria e mostrar pergunta, paciente, fontes e decisão de guardrail.
- Rodar `pytest tests -q` e mostrar o total de testes passando.
- Encerrar lembrando que a solução é apoio à decisão e exige validação humana.

---

### Antes de gravar

- [ ] Rodar `python run_fase3.py --mode local` pelo menos uma vez.
- [ ] Conferir que `patients_sivep.db` foi criado.
- [ ] Deixar o notebook 08 aberto nos pontos principais.
- [ ] Deixar `eval_metrics.json`, diagrama e audit log fáceis de abrir.
- [ ] Se houver GPU disponível, mostrar também os artefatos reais do LoRA.
