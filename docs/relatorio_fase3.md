# Relatório Técnico — Tech Challenge Fase 3
## Assistente Médico com Fine-tuning, LangChain e LangGraph

## 1. Objetivo

A Fase 3 amplia o projeto de SRAG das fases anteriores com um assistente clínico
capaz de consultar protocolos, usar dados estruturados do paciente, responder com
fontes e executar um fluxo automatizado de triagem/alerta com LangGraph.

A implementação continua sendo uma ferramenta de apoio: não prescreve, não define
dose e sempre exige validação do médico responsável.

---

## 2. Dados usados

### 2.1 SIVEP-Gripe / OpenDataSUS

O assistente passou a aceitar registros reais do SIVEP. Para o repositório é
versionada uma amostra anonimizada de 8.000 casos, com registros de 2023 a 2026.

O adaptador também foi testado com os quatro arquivos completos usados no projeto,
somando 1.071.699 registros. Os CSVs brutos não são versionados por causa do
tamanho.

Durante a preparação:

- campos de identificação não entram na base do assistente;
- um novo `patient_id` é criado;
- somente variáveis clínicas úteis são mantidas;
- o resultado é salvo em SQLite para não carregar todos os pacientes na memória.

### 2.2 Protocolos e literatura

O RAG usa:

- protocolos originais do projeto;
- 5 documentos oficiais de SRAG/vigilância;
- 857 chunks com arquivo, página e `chunk_id`;
- PubMedQA;
- MedQuAD.

Também foram incluídos 100 pares pergunta/resposta curados a partir dos protocolos
oficiais. Vinte perguntas ficam separadas para avaliação.

---

## 3. Fine-tuning

O pipeline oficial está em `src/finetuning/`.

### 3.1 Dataset

A preparação combina:

| Fonte | Exemplos |
|---|---:|
| MedQuAD | 1.500 |
| PubMedQA | 1.000 |
| Protocolos oficiais SRAG | 100 |
| Exemplos internos sintéticos | 6 |
| **Total único** | **2.587** |

O split atual gera 2.095 linhas de treino, 258 de validação e 258 de teste.

Os exemplos sintéticos foram mantidos apenas para representar formatos como laudo,
receita e procedimento interno. O domínio SRAG agora é reforçado principalmente
pelos 100 exemplos oficiais, então o oversampling sintético pôde ser reduzido.

### 3.2 LoRA/PEFT

`src/finetuning/train_lora.py` implementa o caminho de fine-tuning de um modelo
pré-treinado com LoRA/PEFT. O pipeline aceita Falcon, LLaMA ou Mistral conforme a
configuração.

O comando abaixo foi validado no ambiente atual:

```bash
python -m src.finetuning.train_lora --dry-run
```

Resultado: 2.095 exemplos de treino no formato chat esperado.

O treinamento LoRA completo exige GPU e as dependências de
`requirements-finetuning.txt`.

### 3.3 Validação real em CPU

Para não depender de loss simulada, foi adicionado um Transformer pequeno em
`src/finetuning/local_validation.py`. Ele serve para testar de ponta a ponta o
processo de treino em CPU.

Última execução:

| Métrica | Resultado |
|---|---:|
| Pares de pré-treino | 95 |
| Exemplos de fine-tuning | 80 |
| Exemplos de avaliação | 20 |
| Loss final do pré-treino | 5.7822 |
| Loss final do fine-tuning | 3.4328 |
| Token F1 médio | 0.2111 |

Esse modelo pequeno não substitui o LLM da entrega. Ele comprova que preparação,
treino, checkpoint e inferência estão funcionando sem simulação.

---

## 4. Assistente com LangChain

`MedicalAssistant` executa as seguintes etapas:

1. guardrail de entrada;
2. busca de contexto no RAG;
3. consulta ao paciente no `PatientDB`;
4. geração com adapter LoRA, Ollama ou checkpoint local;
5. guardrail de saída;
6. resposta com fontes.

O `PatientDB` usa SQLite quando existe uma base SIVEP preparada e mantém
compatibilidade com o JSON sintético anterior.

O backend da LLM continua compatível com LangChain. Quando o adapter LoRA está
presente, ele é carregado por `src/finetuning/inference.py`.

---

## 5. Fluxo LangGraph

O fluxo implementa:

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

Os mesmos nós podem ser executados em Python puro quando LangGraph não está
instalado, o que permite testar a lógica do fluxo em CI. Com LangGraph instalado,
a implementação usa `StateGraph` normalmente.

---

## 6. Segurança e anonimização

### 6.1 Entrada

Pedidos como "prescreva", "qual a dose" ou "qual a posologia" são bloqueados
antes de chegar à LLM.

### 6.2 Saída

A resposta também é verificada. Frases imperativas com dose, por exemplo:

```text
Tome 50 mg de medicamento agora.
```

são removidas e substituídas por uma mensagem segura.

### 6.3 Anonimização

A anonimização textual cobre CPF, CNS, telefone, e-mail, prontuário, datas e nomes
rotulados. O padrão foi ajustado para nomes brasileiros com acentuação.

No caminho SIVEP, a proteção principal é não carregar identificadores pessoais na
base usada pelo assistente.

---

## 7. Explainability e auditoria

Cada trecho recuperado pode carregar:

- arquivo;
- página;
- `chunk_id`;
- score de similaridade.

As respostas exibem essas fontes e o `AuditLogger` registra pergunta, paciente,
backend, fontes e decisões de guardrail em log/JSONL.

---

## 8. Resultados da avaliação

A avaliação usa 20 perguntas SRAG separadas da curadoria de treino.

| Métrica | Resultado |
|---|---:|
| Fonte correta no top-k | 80% |
| Fonte + página correta no top-k | 70% |
| Aviso de validação médica | 100% |
| Bloqueio de prescrição direta | 100% |
| Recuperação MedQuAD no teste de sanidade | 100% |
| Recuperação PubMedQA no teste de sanidade | 100% |

Em comparação com a versão anterior, o projeto agora consegue apontar arquivo e
página, usa dados reais estruturados e não depende do treino simulado para validar
o pipeline.

---

## 9. Testes e validação

Foram executados:

```bash
python -m compileall -q src tests scripts run_pipeline.py run_fase3.py
python -m pytest tests -q
python run_fase3.py --mode local
python -m src.finetuning.train_lora --dry-run
```

Resultado atual dos testes automatizados:

```text
66 passed
```

O adaptador SIVEP também foi testado separadamente com os quatro CSVs completos,
criando um SQLite com 1.071.699 registros.

---

## 10. Limitações

- O fine-tuning LoRA completo ainda precisa ser executado em uma máquina com GPU.
- O Transformer local é propositalmente pequeno e não deve ser usado como modelo
  clínico real.
- Os dados SIVEP servem para contextualização epidemiológica/estruturada e não
  substituem um prontuário eletrônico completo do hospital.
- A solução continua exigindo revisão humana para qualquer decisão clínica.

---

## 11. Conclusão

A Fase 3 passa a reunir dados reais anonimizados do SIVEP, protocolos oficiais,
PubMedQA, MedQuAD, fine-tuning LoRA, validação real em CPU, RAG com arquivo/página,
PatientDB em SQLite, LangChain, LangGraph, guardrails e auditoria.

A arquitetura mantém os componentes anteriores do projeto, mas deixa o caminho de
dados e de treinamento mais próximo do que seria usado em uma implantação real.
