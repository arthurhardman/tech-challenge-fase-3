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

- Mostrar `src/finetuning/config.yaml`: o pipeline aceita Falcon, LLaMA, Mistral
  ou Qwen; basta trocar `active`.
- **Dizer explicitamente qual modelo foi treinado e por quê:** o fine-tuning desta
  entrega usou **Qwen2.5-1.5B-Instruct**, porque Falcon-7B em 4-bit depende de
  `bitsandbytes`, que só tem kernels para GPU NVIDIA. O PDF pede "LLaMA, Falcon
  ou um outro", então a troca está dentro do enunciado — melhor explicar antes de
  a banca perguntar.
- Mostrar que o treino foi **real**, não simulado:
  - `results/finetuned_model/adapter_model.safetensors` (os pesos do adapter);
  - `run_info.json` com device, `train_loss` e `eval_loss` medidos;
  - a curva de loss descendo ao longo dos steps.
- Rodar `python -m src.finetuning.train_lora --dry-run` para mostrar a validação
  do dataset (2.095 exemplos no formato chat) sem precisar treinar de novo no vídeo.
- Mostrar a comparação **modelo base vs. modelo ajustado**
  (`results/finetuned_model/eval_report.json`): ROUGE-L, taxa de citação de fonte
  e taxa de disclaimer. Esse é o argumento mais forte de "o fine-tuning funcionou"
  e atende o entregável "Avaliação do modelo e análise dos resultados".
- Citar `src/finetuning/local_validation.py` como o caminho alternativo em CPU
  pura, deixando claro que ele valida o pipeline mas não é o modelo da entrega.

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

- [ ] Criar o ambiente com **Python 3.12** (`python3.12 -m venv venv`); 3.13/3.14
      não têm wheels compatíveis com os pins de numpy/tensorflow.
- [ ] Rodar `python run_fase3.py --mode lora-real` pelo menos uma vez (usa o
      adapter LoRA treinado). O `--mode local` continua valendo como alternativa.
- [ ] Conferir que `patients_sivep.db` foi criado.
- [ ] Conferir que `results/finetuned_model/` tem `adapter_model.safetensors` e
      `run_info.json` — são a prova do fine-tuning real.
- [ ] Deixar o notebook 08 aberto nos pontos principais.
- [ ] Deixar abertos: `eval_metrics.json`, `eval_report.json`, diagrama e audit log.

### Cobertura dos 4 itens exigidos no PDF

| Item do PDF | Onde aparece no roteiro |
|---|---|
| Treinamento e funcionamento da LLM personalizada | seção 2 |
| Execução de um fluxo automatizado | seção 4 (LangGraph) |
| Resposta a perguntas clínicas contextualizadas | seção 3 |
| Logs e validação das respostas | seções 4 e 5 |

### O que NÃO prometer no vídeo

O modelo ajustado é um 1.5B treinado por 1 época: ele responde em português
coerente e ancorado nas fontes, mas não tem qualidade de um modelo comercial
grande. Apresente a entrega pelo que ela é — um **pipeline completo e auditável**
de fine-tuning + RAG + fluxo + segurança — e não como um produto clínico pronto.
Essa limitação já está declarada na seção 10 do relatório técnico.
