# Dataset Card — Assistente Médico (Fase 3)

Dataset de instruction-tuning combinando literatura pública, protocolos oficiais
de SRAG e poucos exemplos sintéticos de formatos internos hospitalares.

- **Total de exemplos únicos:** 146
- **Val / Test:** 14 / 14 (sem oversampling)
- **Train (linhas gravadas em train.jsonl):** 118 (118 exemplos únicos; os 6 exemplos de protocolo interno foram repetidos 1x no treino — ver nota abaixo)

## Por fonte

- MedQuAD: 20
- PubMedQA: 20
- Protocolos oficiais SRAG: 100
- Protocolos internos (sintéticos): 6

## Por categoria

- protocolo_oficial: 100
- literatura: 40
- laudo: 1
- protocolo: 3
- receita: 1
- faq: 1

## Observações sobre privacidade

- MedQuAD e PubMedQA são datasets públicos, sem dados identificáveis de pacientes.
- Os 100 exemplos de protocolo SRAG foram curados a partir dos documentos oficiais e mantêm referência de fonte/página.
- Apenas os poucos exemplos de formato interno (laudo/receita/procedimento e FAQ interna) são sintéticos; eles representam os tipos de documentos pedidos no challenge.
- Todas as fontes passam pela mesma etapa de anonimização (`src/finetuning/dataset_prep.py::anonymize`) antes de entrar no dataset final.

## Nota sobre o oversampling dos exemplos internos

Os exemplos sintéticos de formato interno são poucos quando comparados à literatura pública. Por isso, eles podem ser repetidos somente no split de treino para não desaparecerem no volume total.
Val/test continuam sem repetição. Como o dataset agora também possui 100 exemplos oficiais de SRAG, o oversampling padrão foi reduzido para 5x.