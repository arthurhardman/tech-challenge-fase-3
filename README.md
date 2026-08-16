# Tech Challenge — SRAG (Fases 1, 2 e 3)

Projeto desenvolvido para acompanhar a evolução do Tech Challenge usando o mesmo
domínio clínico de **Síndrome Respiratória Aguda Grave (SRAG)**.

- **Fase 1:** classificação de desfecho clínico e validação com dados/imagens;
- **Fase 2:** otimização do XGBoost com Algoritmo Genético e explicações com LLM;
- **Fase 3:** assistente médico com fine-tuning, RAG, LangChain/LangGraph, dados estruturados,
  segurança, fontes e auditoria.

A Fase 3 continua sendo uma ferramenta de apoio. Ela não prescreve medicamentos e
não substitui a avaliação do médico responsável.

---

## O que mudou na Fase 3

O projeto original da fase já tinha LangChain/LangGraph, PubMedQA, MedQuAD e uma
base sintética. Nesta versão o fluxo foi ampliado para trabalhar também com **dados
reais anonimizados do SIVEP-Gripe/OpenDataSUS** e protocolos oficiais de SRAG.

### Dados estruturados

O repositório acompanha uma amostra anonimizada com **8.000 registros reais** do
SIVEP, distribuídos entre 2023, 2024, 2025 e 2026:

```text
data/knowledge_base/sivep/patients_sivep_sample.csv
```

Durante a execução ela é convertida para SQLite:

```text
data/knowledge_base/sivep/patients_sivep.db
```

O banco `.db` é gerado localmente e fica fora do Git. O `PatientDB` usa SQLite
quando esse arquivo existe e mantém compatibilidade com o antigo
`prontuarios.json`.

Também é possível preparar os CSVs anuais completos do OpenDataSUS:

```bash
python run_fase3.py --mode local --sivep \
  data/raw/INFLUD24-26-06-2025.csv \
  data/raw/sivep_multi_year/
```

O adaptador foi testado com os quatro arquivos anuais usados no projeto, somando
**1.071.699 registros**. Os arquivos brutos não entram no repositório por causa do
tamanho.

Nenhum nome, CPF, CNS ou endereço do SIVEP é levado para a base do assistente.
Cada linha recebe um identificador novo e só os campos clínicos necessários são
mantidos.

### Protocolos e literatura

O RAG combina:

- protocolos SRAG originais do projeto;
- **5 documentos oficiais** versionados em `data/knowledge_base/official/protocols/`;
- chunks com arquivo, página e `chunk_id` em `protocol_chunks.jsonl`;
- **PubMedQA**;
- **MedQuAD**.

As perguntas curadas de SRAG ficam em:

```text
data/knowledge_base/official/protocol_faq.jsonl
```

São **100 pares pergunta/resposta** com referência do arquivo e da página usada na
curadoria.

---

## Fine-tuning

O caminho principal de fine-tuning está em:

```text
src/finetuning/
```

Ele combina quatro grupos de dados:

1. MedQuAD;
2. PubMedQA;
3. perguntas curadas de protocolos oficiais de SRAG;
4. poucos exemplos sintéticos de documentos internos, mantidos para demonstrar o
   formato de laudo/receita/procedimento pedido no challenge.

A versão atual gera:

| Fonte | Exemplos únicos usados na preparação |
|---|---:|
| MedQuAD | 1.500 |
| PubMedQA | 1.000 |
| Protocolos oficiais SRAG | 100 |
| Exemplos internos sintéticos | 6 |
| **Total após deduplicação** | **2.587** |

O split atual gera 2.095 linhas de treino, 258 de validação e 258 de teste. O
oversampling dos exemplos sintéticos foi reduzido porque agora existem exemplos
reais do domínio SRAG.

### LoRA/PEFT — caminho de entrega

O fine-tuning de um modelo pré-treinado continua em:

```bash
python -m src.finetuning.dataset_prep --build
python -m src.finetuning.train_lora --dry-run
```

O `--dry-run` valida todo o dataset sem baixar o modelo. Para o treinamento real:

```bash
pip install -r requirements-finetuning.txt
python -m src.finetuning.train_lora
```

O adapter é salvo em:

```text
results/finetuned_model/
```

O `CustomMedicalLLM` passa a usar esse adapter automaticamente quando
`run_info.json` e as dependências do Hugging Face/PEFT estão disponíveis.

### Validação real em CPU

Também existe um Transformer pequeno em:

```text
src/finetuning/local_validation.py
```

Ele não substitui Falcon/LLaMA/Mistral. Serve para validar de verdade, em CPU e
sem simular loss, o caminho:

```text
protocolos -> treino -> fine-tuning -> checkpoint -> inferência
```

Para rodar:

```bash
pip install -r requirements-fase3-local.txt
python -m src.finetuning.local_validation
```

A validação local agora usa uma base maior sem misturar o conjunto de teste: primeiro o modelo vê trechos dos protocolos, depois uma amostra de MedQuAD/PubMedQA e, por último, os exemplos curados de SRAG. As 20 perguntas de avaliação continuam fora do treinamento. Durante o teste, o contexto vem dos três melhores resultados do retriever; a fonte/página esperada não é usada para escolher o texto de entrada.

Última execução validada neste projeto:

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

---

## Assistente médico

O assistente fica em:

```text
src/assistant/
```

Fluxo principal:

```text
pergunta
   ↓
guardrail de entrada
   ↓
RAG (protocolos oficiais + PubMedQA/MedQuAD)
   ↓
PatientDB (SQLite ou JSON)
   ↓
LLM ajustada / Ollama / backend local
   ↓
guardrail de saída
   ↓
resposta + fontes + auditoria
```

### LangChain

`src/assistant/chains/medical_assistant.py` monta a resposta contextualizada e
`llm_backend.py` expõe o backend como LLM do LangChain quando a dependência está
instalada.

### LangGraph

`src/assistant/chains/graph.py` organiza o fluxo:

```text
triagem
  ↓
verificar_exames
  ↓
risco vermelho? ── sim ──> emitir_alerta
       │
       não
       ↓
sugerir_conduta
       ↓
consolidar
```

Com `langgraph` instalado, esse fluxo é compilado como `StateGraph`. Em ambientes
de teste sem a biblioteca, os mesmos nós podem ser executados em Python puro para
validar as regras e o estado do fluxo.

---

## Segurança e anonimização

Foram mantidas duas verificações de segurança:

- **entrada:** bloqueia pedidos diretos de prescrição, dose e posologia;
- **saída:** detecta frases imperativas como `Tome 50 mg...` ou
  `Administre 20 mg...` e remove a instrução antes de apresentar a resposta.

Exemplo de saída bloqueada:

```text
A resposta gerada continha uma instrução de dose/prescrição e foi bloqueada pelo guardrail.
```

A anonimização textual também cobre CPF, CNS, telefone, e-mail, prontuário, datas
e nomes quando aparecem identificados como `Paciente`, `Nome`, `Sr.`, `Dra.` etc.

Para a base SIVEP, a proteção principal é mais simples: campos de identificação
nem chegam ao banco usado pelo assistente.

---

## Explainability e fontes

O retriever agora guarda, quando disponível:

```text
arquivo
página
chunk_id
score
```

Assim uma resposta pode citar, por exemplo:

```text
guia_srag_ministerio_saude_2025.pdf, p. 4
```

Na avaliação atual com 20 perguntas SRAG separadas do treino:

| Métrica | Resultado |
|---|---:|
| Fonte correta no top-k | 80% |
| Fonte + página correta no top-k | 70% |
| Prescrição direta bloqueada | 100% |
| Respostas com aviso de validação | 100% |
| Recuperação MedQuAD no teste de sanidade | 100% |
| Recuperação PubMedQA no teste de sanidade | 100% |

---

## Estrutura principal da Fase 3

```text
data/knowledge_base/
├── protocolos/                    # protocolos sintéticos originais
├── external/                      # fatias PubMedQA e MedQuAD
├── official/
│   ├── protocols/                 # PDFs oficiais
│   ├── protocol_chunks.jsonl      # chunks com arquivo/página
│   ├── protocol_faq.jsonl         # 100 Q&As SRAG
│   └── qa_eval.jsonl              # 20 perguntas de avaliação
└── sivep/
    └── patients_sivep_sample.csv  # 8 mil registros reais anonimizados

src/assistant/
├── knowledge/
│   ├── patient_db.py
│   ├── retriever.py
│   └── sivep_adapter.py
├── chains/
│   ├── medical_assistant.py
│   ├── llm_backend.py
│   └── graph.py
├── safety/
│   ├── guardrails.py
│   └── audit_log.py
└── cli.py

src/finetuning/
├── dataset_prep.py                # dataset principal
├── train_lora.py                  # LoRA/PEFT em GPU
├── inference.py                   # adapter LoRA ou checkpoint local
├── local_validation.py            # treino real pequeno em CPU
└── evaluate_finetune.py
```

> `src/assistant/finetuning/` continua no projeto por compatibilidade com o
> notebook/estrutura anterior. Para novos treinamentos, use `src/finetuning/`.

---

## Instalação

```bash
python -m venv venv

# Linux/Mac
source venv/bin/activate

# Windows
# venv\Scripts\activate

pip install -r requirements.txt
```

Para a validação real em CPU:

```bash
pip install -r requirements-fase3-local.txt
```

Para LoRA em GPU:

```bash
pip install -r requirements-finetuning.txt
```

---

## Execução da Fase 3

### Pipeline reproduzível em CPU

```bash
python run_fase3.py --mode local
```

Esse comando:

1. cria o SQLite a partir da amostra real do SIVEP sem sobrescrever a origem;
2. prepara o dataset de fine-tuning;
3. executa um fine-tuning real pequeno em CPU;
4. gera o diagrama;
5. avalia RAG, segurança e fontes;
6. roda um fluxo completo com um paciente da base.

### Validar o LoRA sem GPU

```bash
python run_fase3.py --mode lora-dry-run
```

### LoRA real

```bash
python run_fase3.py --mode lora-real
```

### Usar CSVs completos do OpenDataSUS

```bash
python run_fase3.py --mode local --sivep \
  data/raw/INFLUD24-26-06-2025.csv \
  data/raw/sivep_multi_year/
```

### Base sintética antiga

Ela não é mais regenerada automaticamente. Se quiser reproduzir a demo original:

```bash
python run_fase3.py --mode skip-train --generate-synthetic
```

---

## CLI

```bash
python -m src.assistant.cli pacientes
python -m src.assistant.cli perguntar "Quais sinais indicam evolução para SRAG?"
python -m src.assistant.cli fluxo --paciente <ID>
```

---

## Testes

```bash
python -m pytest tests -q
```

Estado validado desta versão:

```text
68 passed
```

O pipeline completo `python run_fase3.py --mode local` também foi executado de
ponta a ponta com a amostra SIVEP versionada.

Para conferir sintaxe sem executar treino:

```bash
make lint
```

---

## Resultados e documentação

- relatório da Fase 3: [`docs/relatorio_fase3.md`](docs/relatorio_fase3.md)
- plano da Fase 3: [`docs/plano_fase3.md`](docs/plano_fase3.md)
- roteiro do vídeo: [`docs/roteiro_video_fase3.md`](docs/roteiro_video_fase3.md)
- diagrama: `results/figures/fluxo_langchain.png`
- métricas do assistente: `results/finetuning/eval_metrics.json`
- métricas do treino local: `results/finetuning/local_validation/metrics.json`

---

## Observação sobre os datasets grandes

Os CSVs completos do OpenDataSUS não devem ser enviados no MR. O projeto versiona
somente a amostra anonimizada e os manifestos/fontes necessários para reproduzir
a preparação. Para uma execução com os anos completos, coloque os arquivos em
`data/raw/` e passe os caminhos usando `--sivep`.
