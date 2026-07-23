# Dataset Card — Assistente Médico (Fase 3)

Dataset de instruction-tuning combinando fontes públicas e exemplos
sintéticos no estilo de documentos internos hospitalares.

- **Total de exemplos únicos:** 2487
- **Val / Test:** 248 / 248 (sem oversampling)
- **Train (linhas gravadas em train.jsonl):** 2105 (1991 exemplos únicos; os 6 exemplos de protocolo interno foram repetidos 20x no treino — ver nota abaixo)

## Por fonte

- MedQuAD: 1500
- PubMedQA: 1000
- Protocolos internos (sintéticos): 6

## Por categoria

- literatura: 2481
- protocolo: 3
- laudo: 1
- faq: 1
- receita: 1

## Observações sobre privacidade

- MedQuAD e PubMedQA são datasets públicos, sem dados identificáveis de pacientes.
- Os exemplos de "protocolo interno", "faq", "laudo" e "receita" são **sintéticos/fictícios**, criados apenas para dar ao pipeline o formato que os documentos reais do hospital teriam.
- Todas as fontes passam pela mesma etapa de anonimização (`src/finetuning/dataset_prep.py::anonymize`) antes de entrar no dataset final — isso é o que garante que, ao trocar os exemplos sintéticos por documentos reais do hospital, CPF, telefone, e-mail, nome de paciente e número de prontuário sejam removidos automaticamente.

## Nota sobre o oversampling dos protocolos internos

Os exemplos de protocolo/FAQ/laudo/receita são propositalmente poucos neste repositório de demonstração (ver `synthetic_hospital_examples()`), então foram repetidos apenas no split de treino para o fine-tuning não ignorá-los diante do volume de dados públicos. Val/test usam os exemplos originais, sem repetição, para as métricas não ficarem infladas. Com documentos reais do hospital em maior volume, o oversampling deixa de ser necessário (ajuste `synthetic_oversample=1`).