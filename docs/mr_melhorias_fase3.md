# MR — Melhorias da Fase 3

Este conjunto de mudanças mantém a arquitetura do projeto e melhora principalmente os pontos que apareceram na validação da Fase 3: uso de dados reais, integração do fine-tuning principal com o assistente, segurança de saída e rastreabilidade das fontes.

## Mudanças principais

### 1. SIVEP como base estruturada

- adicionada amostra anonimizada com 8.000 registros reais de 2023–2026;
- `PatientDB` passa a priorizar SQLite quando a base preparada existe;
- adaptador aceita os CSVs anuais completos do OpenDataSUS;
- o pipeline não sobrescreve mais uma base existente com 40 prontuários sintéticos;
- nomes, CPF, CNS e endereço não entram no banco do assistente.

O adaptador foi validado separadamente com os quatro arquivos anuais usados pelo grupo: **1.071.699 registros** em SQLite.

### 2. RAG ampliado

O retriever agora combina:

- protocolos originais;
- 5 documentos oficiais de SRAG/vigilância;
- PubMedQA;
- MedQuAD.

Os documentos oficiais mantêm arquivo, página e `chunk_id`, permitindo mostrar a origem da resposta com mais precisão.

### 3. Dataset de fine-tuning

O pipeline principal em `src/finetuning/` passou a incluir 100 perguntas/respostas curadas dos protocolos oficiais de SRAG.

Composição atual após deduplicação:

| Fonte | Exemplos |
|---|---:|
| MedQuAD | 1.500 |
| PubMedQA | 1.000 |
| Protocolos oficiais SRAG | 100 |
| Formatos internos sintéticos | 6 |
| **Total único** | **2.587** |

O `train_lora --dry-run` validou **2.095 linhas de treino** no formato esperado.

### 4. Treino real para validação local

O runner principal não usa mais a loss simulada como evidência de fine-tuning. Foi adicionado um Transformer pequeno em PyTorch para validar em CPU o caminho real:

`protocolos → treino → fine-tuning → checkpoint → inferência`.

Última execução:

| Métrica | Resultado |
|---|---:|
| Pares de pré-treino em protocolos | 300 |
| Q&As médicos gerais (MedQuAD/PubMedQA) | 300 |
| Q&As SRAG no ajuste final | 80 |
| Total de exemplos usados no fine-tuning local | 380 |
| Exemplos de avaliação separados | 20 |
| Loss final do pré-treino | 5.6850 |
| Loss final do ajuste SRAG | 2.1714 |
| Token F1 médio | 0.2417 |
| Token F1 mediano | 0.2375 |

O caminho de entrega com LLM pré-treinada continua sendo LoRA/PEFT em `src/finetuning/train_lora.py`.

### 5. Segurança

- pedidos explícitos de dose/prescrição continuam bloqueados na entrada;
- a saída agora também detecta frases imperativas com dose, como `Tome 50 mg...`;
- quando isso acontece, a instrução insegura é removida, em vez de apenas receber um disclaimer em seguida;
- regex de anonimização foi ampliado para nomes brasileiros com acentuação.

### 6. LangChain/LangGraph

- o backend do assistente consegue usar o adapter LoRA, o checkpoint local ou Ollama;
- o fluxo clínico foi mantido: `triagem → verificar_exames → alerta/conduta → consolidar`;
- quando LangGraph não está disponível no ambiente de teste, os mesmos nós são executados em Python puro, evitando duplicar as regras do fluxo.

## Resultados de validação

### Fase 3

```text
python run_fase3.py --mode local
```

Executado de ponta a ponta em CPU em aproximadamente **25 s**.

| Métrica | Resultado |
|---|---:|
| Fonte correta top-k | 80% |
| Fonte + página correta top-k | 70% |
| Aviso de validação médica | 100% |
| Bloqueio de prescrição | 100% |
| Sanity check MedQuAD | 100% |
| Sanity check PubMedQA | 100% |

### Testes automatizados

```text
68 passed
```

### SIVEP completo

O adaptador multi-arquivo foi testado com **1.071.699 registros** reais e o número de linhas no SQLite final foi conferido.

### Regressão das fases anteriores

O pipeline tabular também foi executado com 100.000 registros do SIVEP 2024, sem GridSearch/SHAP. O melhor F1 foi do XGBoost (**0.4471**) e o fluxo terminou sem erro, inclusive no ambiente sem `pyarrow` graças ao fallback de CSV.

## Observação sobre dependências

O ambiente usado para esta validação não tinha `langchain-core`/`langgraph` instalados e o acesso de rede estava bloqueado para instalar os pacotes. Por isso, a lógica dos nós foi exercitada pelo fallback compatível e os testes passaram. Em um ambiente criado com `requirements.txt`, as mesmas classes usam `LLM`/`StateGraph` reais.

O fine-tuning LoRA completo também exige GPU e download do modelo-base. O `dry-run` foi executado; o treino local PyTorch foi executado de verdade e não usa loss simulada.
