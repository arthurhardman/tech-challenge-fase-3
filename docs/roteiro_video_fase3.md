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

**Gravar pela interface visual** — é mais legível que o terminal:

```bash
streamlit run app/assistente_app.py
```

- Abrir a aba **🧑‍⚕️ Paciente** e mostrar um caso real anonimizado do SIVEP:
  risco, idade, comorbidades e o resumo clínico que é enviado à LLM.
- Ir para **💬 Consulta clínica**, escolher uma pergunta de exemplo e consultar.
- Mostrar, lado a lado, a resposta e o painel **📚 Fontes utilizadas** com
  arquivo e página — é a explainability pedida no desafio.
- Apontar o rodapé `backend: finetuned-hf`: prova de que quem respondeu foi o
  modelo com o adapter LoRA, não um mock.
- Fazer uma pergunta geral de saúde para mostrar a recuperação complementar de
  MedQuAD/PubMedQA.

> A primeira resposta demora ~40 s (carregamento do adapter) e as seguintes
> ~20-30 s. Deixe o app aberto e faça uma pergunta antes de começar a gravar,
> para o modelo já estar em memória.

## 4. Segurança e LangGraph (9:30–12:30)

- Na aba **🔀 Fluxo automatizado**, executar o grafo para o paciente de risco
  vermelho. A trilha `triagem → verificar_exames → emitir_alerta → consolidar`
  acende passo a passo e o alerta aparece destacado.
- Explicar que risco e exames vêm somente dos campos existentes no PatientDB; o
  fluxo não inventa medidas ausentes.
- Na aba **🔒 Segurança e auditoria**, clicar em “Testar pedido de prescrição” e
  mostrar o guardrail recusando a dose — é o requisito destacado em amarelo no
  enunciado.

## 5. Avaliação, logs e fechamento (12:30–15:00)

- Mostrar `results/finetuning/eval_metrics.json`:
  - fonte correta top-k: 80%;
  - fonte + página: 70%;
  - aviso médico: 100%;
  - bloqueio de prescrição: 100%;
  - sanity checks MedQuAD/PubMedQA: 100%.
- Ainda na aba **🔒 Segurança e auditoria**, mostrar a trilha de auditoria
  preenchida em tempo real pelas consultas feitas durante o vídeo: pergunta,
  paciente, fontes, backend e decisão do guardrail.
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
- [ ] Subir o app (`streamlit run app/assistente_app.py`) e fazer **uma pergunta
      antes de gravar**, para o adapter já estar carregado em memória.
- [ ] Deixar o notebook 08 aberto como material de apoio.
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
