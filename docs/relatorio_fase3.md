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

### 3.2 LoRA/PEFT — executado

`src/finetuning/train_lora.py` implementa o fine-tuning de um modelo
pré-treinado com LoRA/PEFT. O `config.yaml` aceita Falcon, LLaMA, Mistral ou
Qwen; basta trocar a chave `active`.

**O fine-tuning foi executado de verdade nesta entrega**, e não apenas validado.

Modelo base escolhido: **Qwen2.5-1.5B-Instruct**. A escolha é uma consequência do
hardware: a configuração original (Falcon-7B em 4-bit) depende de `bitsandbytes`,
que só possui kernels para GPU NVIDIA. O treino foi feito em Apple Silicon (M4,
16 GB de memória unificada), onde o caminho viável é carregar um modelo menor sem
quantização. O enunciado pede "LLaMA, Falcon ou um outro", então a substituição
está dentro do escopo.

O `train_lora.py` detecta o device e ajusta a estratégia sozinho:

| Device | Precisão | Quantização 4-bit |
|---|---|---|
| CUDA (NVIDIA) | bfloat16 | sim (QLoRA via bitsandbytes) |
| MPS (Apple Silicon) | float16 | não |
| CPU | float32 | não |

Resultado da execução:

| Item | Valor |
|---|---|
| Modelo base | `Qwen/Qwen2.5-1.5B-Instruct` |
| Device | MPS (Apple M4) |
| Parâmetros treináveis | 18.464.768 (**1,18%** de 1,56 B) |
| Exemplos de treino | 2.095 |
| Épocas / steps | 1 / 131 |
| `max_seq_length` | 512 |
| Duração | ~1 h 06 min |
| **Loss de treino** | **1,3352** |
| **Loss de validação** | **1,1879** |
| Curva de loss | 1,946 → 1,219 |

A loss de validação ficar abaixo da de treino indica que não houve overfitting no
regime de 1 época.

O adapter treinado (74 MB) fica em `results/finetuned_model/`:

```text
adapter_model.safetensors   # pesos LoRA
adapter_config.json
run_info.json               # device, losses e log_history medidos
tokenizer.json / vocab.json / chat_template.jinja
```

Para reproduzir:

```bash
pip install -r requirements-finetuning.txt
python -m src.finetuning.dataset_prep --build
python -m src.finetuning.train_lora            # treino real
python -m src.finetuning.train_lora --dry-run  # só valida o dataset
```

### 3.3 Validação real em CPU

Para não depender de loss simulada, foi adicionado um Transformer pequeno em
`src/finetuning/local_validation.py`. Ele serve para testar de ponta a ponta o
processo de treino em CPU.

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

Esse modelo pequeno não substitui o LLM da entrega. Ele comprova que preparação,
treino, checkpoint e inferência estão funcionando sem simulação. No conjunto de
avaliação, o contexto é recuperado pelo retriever top-3 a partir da pergunta; a
fonte/página de referência não é usada para montar a entrada do modelo.

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

São duas avaliações complementares, com objetivos diferentes.

### 8.1 Modelo ajustado vs. modelo base

`src/finetuning/evaluate_finetune.py` gera a mesma pergunta nos dois modelos e
compara com a resposta de referência do split de teste. Resultado com 14 exemplos
amostrados de forma balanceada entre as categorias:

| Categoria | n | ROUGE-L base | ROUGE-L ajustado | Ganho |
|---|---:|---:|---:|---:|
| Literatura (PubMedQA/MedQuAD) | 7 | 0,0467 | **0,1453** | 3,1× |
| Protocolo oficial SRAG | 7 | 0,0683 | **0,1242** | 1,8× |
| **Geral** | **14** | **0,0575** | **0,1347** | **2,3×** |

Em uma amostragem anterior só com literatura (20 exemplos), o ganho foi de
0,0491 → 0,1697 (3,5×). O fine-tuning melhora a aderência à resposta esperada em
todas as fatias medidas.

O efeito também é visível no formato da resposta:

```text
Pergunta do PubMedQA
  base     : "Sim, os endometriomas (polposes endometrais) em mulheres..."
  ajustado : "Resposta direta: yes. The results suggest that..."
```

O modelo ajustado adota a convenção do dataset ("Resposta direta: ..."), que o
modelo base desconhece.

> **Sobre as taxas de disclaimer e citação de fonte nesse relatório:** elas
> aparecem como 0% e isso é esperado. As respostas de *referência* do dataset são
> textos técnicos curtos, sem aviso de validação e sem citar arquivo/página — o
> modelo acerta ao não inventá-los. No produto final, disclaimer e fontes não são
> responsabilidade da LLM: são aplicados determinísticamente pelo guardrail de
> saída e pelo retriever, e medidos na avaliação 8.2 (onde dão 100%).

A amostragem é balanceada por categoria porque o split de teste é ~97% literatura
e ~3% protocolo oficial; pegar os N primeiros registros mediria só uma fatia.
Para reproduzir:

```bash
python -m src.finetuning.evaluate_finetune --max-examples 14
```

### 8.2 Assistente completo (RAG + segurança)

`src/assistant/finetuning/evaluate.py` usa 20 perguntas SRAG separadas da
curadoria de treino e mede o sistema inteiro, não a LLM isolada:

| Métrica | Resultado |
|---|---:|
| Fonte correta no top-k | 80% |
| Fonte + página correta no top-k | 70% |
| Aviso de validação médica | 100% |
| Bloqueio de prescrição direta | 100% |
| Recuperação MedQuAD no teste de sanidade | 100% |
| Recuperação PubMedQA no teste de sanidade | 100% |

Em comparação com a versão anterior, o projeto agora consegue apontar arquivo e
página, usa dados reais estruturados, treina um adapter LoRA de verdade e não
depende de treino simulado para validar o pipeline.

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
68 passed
```

O adaptador SIVEP também foi testado separadamente com os quatro CSVs completos,
criando um SQLite com 1.071.699 registros.

---

## 10. Limitações

- O fine-tuning LoRA foi executado, mas em um modelo de **1,5 B por 1 época**, por
  restrição de hardware (16 GB de memória unificada). O modelo ajustado responde
  em português coerente e ancorado nas fontes, porém não tem a qualidade de um
  modelo de 7 B+ treinado por mais épocas. Com GPU NVIDIA, basta trocar `active`
  no `config.yaml` para Falcon/LLaMA/Mistral e reexecutar o mesmo script.
- O Transformer de `local_validation.py` é propositalmente pequeno e serve apenas
  para validar o pipeline em CPU pura; não deve ser usado como modelo clínico.
- Os dados SIVEP servem para contextualização epidemiológica/estruturada e não
  substituem um prontuário eletrônico completo do hospital.
- O retriever usa TF-IDF. É rápido, determinístico e fácil de explicar, mas perde
  sinônimos e paráfrases que um índice vetorial denso capturaria — o que explica
  parte dos 20% de perguntas cuja fonte correta não entra no top-k.
- A solução continua exigindo revisão humana para qualquer decisão clínica.

---

## 11. Conclusão

A Fase 3 passa a reunir dados reais anonimizados do SIVEP, protocolos oficiais,
PubMedQA, MedQuAD, fine-tuning LoRA, validação real em CPU, RAG com arquivo/página,
PatientDB em SQLite, LangChain, LangGraph, guardrails e auditoria.

A arquitetura mantém os componentes anteriores do projeto, mas deixa o caminho de
dados e de treinamento mais próximo do que seria usado em uma implantação real.
