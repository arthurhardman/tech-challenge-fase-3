# Dataset Card — Assistente Médico (Fase 3)

Dataset de instruction-tuning combinando literatura pública, protocolos oficiais
de SRAG e poucos exemplos sintéticos de formatos internos hospitalares.

- **Total de exemplos únicos:** 2587
- **Val / Test:** 258 / 258 (sem oversampling)
- **Train (linhas gravadas em train.jsonl):** 2095 (2071 exemplos únicos; os 6 exemplos de protocolo interno foram repetidos 5x no treino — ver nota abaixo)

## Por fonte

- MedQuAD: 1500
- PubMedQA: 1000
- Protocolos oficiais SRAG: 100
- Protocolos internos (sintéticos): 6

## Por categoria

- literatura: 2481
- protocolo_oficial: 100
- protocolo: 3
- laudo: 1
- faq: 1
- receita: 1

## Observações sobre privacidade

- MedQuAD e PubMedQA são datasets públicos, sem dados identificáveis de pacientes.
- Os 100 exemplos de protocolo SRAG foram curados a partir dos documentos oficiais e mantêm referência de fonte/página.
- Apenas os poucos exemplos de formato interno (laudo/receita/procedimento e FAQ interna) são sintéticos; eles representam os tipos de documentos pedidos no challenge.
- Todas as fontes passam pela mesma etapa de anonimização (`src/finetuning/dataset_prep.py::anonymize`) antes de entrar no dataset final.

## Nota sobre o oversampling dos exemplos internos

Os exemplos sintéticos de formato interno são poucos quando comparados à literatura pública. Por isso, eles podem ser repetidos somente no split de treino para não desaparecerem no volume total.
Val/test continuam sem repetição. Como o dataset agora também possui 100 exemplos oficiais de SRAG, o oversampling padrão foi reduzido para 5x.