# Plano — Tech Challenge Fase 2 (Projeto 1)

**FIAP Pós-Tech — Machine Learning Engineering**
**Projeto escolhido:** Projeto 1 — Otimização de Modelos de Diagnóstico
**LLM:** modelo open-source local (Ollama — `llama3.1:8b` ou `mistral`)
**Repositório:** `ml-tech` (continuação da Fase 1)

---

## 1. Objetivo

Otimizar os modelos de classificação de desfecho clínico em SRAG construídos na Fase 1
usando **Algoritmos Genéticos (AG)** para busca de hiperparâmetros, e integrar uma **LLM
local** para gerar explicações em linguagem natural dos diagnósticos, voltadas a
profissionais de saúde.

Reaproveitamos integralmente a Fase 1:
- Modelos treinados em `results/models/` (LogReg, Decision Tree, Random Forest, XGBoost, CNN).
- Pipeline modular em `src/tabular/` (load, preprocessing, modeling, evaluation).
- Splits e artefatos já gerados em `data/processed/`.

> **Escopo da otimização (decisão do grupo):** a Fase 2 otimiza o **modelo tabular
> principal (XGBoost)**. A CNN de imagem da Fase 1 permanece como **amostra/extra da Fase 1**
> e **não** é alvo do AG — otimizar o XGBoost já atende plenamente o requisito e é a entrega
> mais segura. Isso será deixado **explícito no relatório técnico** para evitar ambiguidade
> na avaliação.

---

## 2. Escopo e requisitos obrigatórios

| # | Requisito do enunciado | Como será atendido |
|---|------------------------|--------------------|
| 1 | AG para otimização de hiperparâmetros | Módulo `src/genetic/` próprio (codificação, seleção, crossover, mutação, fitness) |
| 2 | Codificação de genes adequada | Cromossomo = vetor dos hiperparâmetros do XGBoost (ver §4.1) |
| 3 | Operadores de seleção, cruzamento e mutação | Torneio + crossover uniforme + mutação gaussiana/uniforme |
| 4 | Função fitness baseada em métricas | F1-score (e recall de óbito) via StratifiedKFold |
| 5 | Comparar otimizado vs. original | Tabela comparativa vs. modelos do GridSearchCV da Fase 1 |
| 6 | ≥ 3 experimentos com configs diferentes de AG | Variar população, taxa de mutação, gerações (ver §5) |
| 7 | Escalabilidade automática + monitoramento/logging | Avaliação paralela (`joblib`) + `tracker.py` com métricas por geração |
| 8 | Documentar arquitetura e decisões | Este documento + `docs/relatorio_fase2.md` |
| 9 | Integração com LLM pré-treinada | Ollama local → explicações dos diagnósticos |
| 10 | Prompt engineering | Templates em `src/llm/prompts.py`, documentados e versionados |
| 11 | Avaliar qualidade das interpretações | Rubrica qualitativa + exemplos comentados |
| 12 | Projeto bem estruturado + venv | venv já existente; deps novas em `requirements.txt` |
| 13 | Testes automatizados | `pytest` para operadores do AG e parser da LLM |
| 14 | Nuvem (IaC) | **Opcional** — pontuação extra; fora do escopo mínimo |

---

## 3. Arquitetura (adições ao repositório)

```
src/
├── genetic/                  # NOVO — algoritmo genético
│   ├── __init__.py
│   ├── chromosome.py         # codificação dos hiperparâmetros (genes) + decode
│   ├── operators.py          # seleção (torneio), crossover, mutação
│   ├── fitness.py            # fitness = F1/recall via cross-validation
│   └── ga_optimizer.py       # loop evolutivo + elitismo + logging por geração
├── llm/                      # NOVO — interpretação via LLM local
│   ├── __init__.py
│   ├── client.py             # wrapper Ollama (chamada local, sem custo de API)
│   ├── prompts.py            # templates de prompt engineering médico
│   └── explainer.py          # predição + features/SHAP → explicação PT-BR
└── monitoring/               # NOVO — tracking de experimentos
    ├── __init__.py
    └── tracker.py            # métricas por geração → CSV/JSON + gráficos de convergência

notebooks/
├── 06_genetic_optimization.ipynb   # NOVO — roda os 3+ experimentos
└── 07_llm_interpretation.ipynb     # NOVO — demonstra explicações

experiments/                         # NOVO — configs (YAML/JSON) + resultados dos experimentos
results/
├── figures/                         # + gráficos de convergência do AG
└── models/                          # + xgboost_ga_optimized.pkl

docs/
├── plano_fase2.md                   # este arquivo
└── relatorio_fase2.md               # NOVO — relatório técnico final

tests/
├── test_genetic.py                  # NOVO — operadores do AG
└── test_llm.py                      # NOVO — parser/contrato da saída da LLM
```

---

## 4. Detalhamento do Algoritmo Genético

### 4.1 Codificação (cromossomo)

Modelo-alvo: **XGBoost** (melhor desempenho na Fase 1). Cada indivíduo é um vetor de genes,
um por hiperparâmetro, com domínio definido:

| Gene | Hiperparâmetro | Tipo | Domínio |
|------|----------------|------|---------|
| g1 | `n_estimators` | int | 50–600 |
| g2 | `max_depth` | int | 2–12 |
| g3 | `learning_rate` | float | 0.01–0.3 |
| g4 | `subsample` | float | 0.5–1.0 |
| g5 | `colsample_bytree` | float | 0.5–1.0 |
| g6 | `min_child_weight` | int | 1–10 |
| g7 | `gamma` | float | 0.0–5.0 |
| g8 | `scale_pos_weight` | float | 1.0–ratio (desbalanceamento) |

`decode()` converte o vetor em `dict` de kwargs do `XGBClassifier`. Normalização opcional
dos genes para [0,1] internamente (facilita mutação) com mapeamento para o domínio real.

### 4.2 Operadores

- **Seleção:** torneio (k=3) — bom equilíbrio entre pressão seletiva e diversidade.
- **Crossover:** uniforme (cada gene vem de um dos pais com p=0.5); taxa configurável.
- **Mutação:** gaussiana para genes contínuos, perturbação ±1 com clamp para inteiros;
  taxa configurável por gene.
- **Elitismo:** preserva os N melhores indivíduos por geração (evita regressão).

### 4.3 Função fitness

```
fitness(indivíduo) = média( F1-score do XGBoost(decode(indivíduo)) )
                     em StratifiedKFold(n_splits=3) sobre X_train/y_train
```

- Métrica primária: **F1** (alternativa: recall de óbito, dado o custo clínico de falso-negativo).
- Penalização opcional por complexidade (n_estimators × max_depth) para favorecer modelos
  mais enxutos/rápidos — alinhado ao requisito de "eficiência".
- Avaliação da população **paralelizada com `joblib`** (atende escalabilidade).

### 4.4 Loop evolutivo (`ga_optimizer.py`)

```
inicializa população aleatória
para cada geração:
    avalia fitness (paralelo)
    registra no tracker (melhor, média, desvio)
    seleciona pais (torneio)
    aplica crossover
    aplica mutação
    aplica elitismo
    monta nova população
retorna melhor indivíduo + histórico
```

Critério de parada: nº fixo de gerações **ou** estagnação (sem melhora por K gerações).

---

## 5. Experimentos (≥ 3)

Todos sobre o XGBoost, fitness = F1 em CV. Configs em `experiments/*.yaml`.

Configuração **leve** (proposital): mantém custo de execução baixo — importante porque cada
indivíduo treina um XGBoost com CV — e ainda atende os 3 experimentos exigidos. Dá para
escalar depois se houver tempo/máquina.

| Exp | População | Gerações | Taxa crossover | Taxa mutação | Objetivo |
|-----|-----------|----------|----------------|--------------|----------|
| A | 10 | 8 | 0.8 | 0.10 | baseline leve/rápido |
| B | 20 | 10 | 0.9 | 0.05 | mais exploração, mutação baixa |
| C | 30 | 12 | 0.8 | 0.20 | população maior, mutação alta |

**Saídas por experimento:** melhor cromossomo, fitness final, curva de convergência
(melhor/média por geração), tempo de execução, hiperparâmetros encontrados.

**Comparativo final:** tabela XGBoost-original (GridSearch da Fase 1) vs. XGBoost-AG em
accuracy, precision, recall, F1 e AUC no conjunto de teste.

---

## 6. Integração com LLM (local)

### 6.1 Stack
- **Ollama** rodando localmente; modelo `llama3.1:8b` (fallback `mistral:7b`) — **backend principal**.
- `src/llm/client.py`: wrapper fino sobre a API local do Ollama (`http://localhost:11434`),
  sem dependência de chave/custo.
- **Modo `mock`/`demo` (obrigatório no nosso setup):** backend alternativo que devolve
  respostas por **template/cache** (sem precisar de LLM local). Garante que **notebooks e
  testes rodem em qualquer máquina** — inclusive na do professor, cujo setup é desconhecido.
  Seleção via parâmetro/variável de ambiente (`LLM_BACKEND=ollama|mock`); `mock` é o
  **fallback automático** quando o Ollama não está acessível. Respostas de demonstração
  ficam versionadas em `src/llm/demo_responses/` (ou geradas por template a partir das
  features/SHAP do caso).

### 6.2 Fluxo
```
predição do modelo (classe + probabilidade)
      + principais features do paciente
      + contribuições SHAP (já geradas na Fase 1)
           → prompt estruturado → LLM → explicação clínica em PT-BR
```

### 6.3 Prompt engineering (`prompts.py`)
- Papel: "assistente clínico que explica predições de risco para a equipe médica".
- Entrada estruturada (predição, probabilidade, top features SHAP com sinal/magnitude).
- Restrições: linguagem acessível, não inventar dados, deixar claro que é apoio à decisão
  (não substitui julgamento clínico), citar as features que mais pesaram.
- Versões de prompt comparadas (v1 simples vs. v2 com few-shot) para avaliação.

### 6.4 Avaliação de qualidade
Rubrica qualitativa em 4 critérios (1–5): **fidelidade** às features, **clareza**,
**acionabilidade**, **segurança** (sem alucinação). Aplicada a ~5–10 casos de exemplo,
documentada no relatório.

---

## 7. Monitoramento e escalabilidade

- `monitoring/tracker.py`: registra por geração (melhor/média/desvio do fitness, tempo) em
  CSV/JSON; gera gráfico de convergência em `results/figures/`.
- Escalabilidade: avaliação paralela da população (`joblib`, `n_jobs` configurável via env,
  reaproveitando o padrão `SRAG_GRID_N_JOBS` já usado na Fase 1).
- **Nuvem (opcional / extra):** se houver tempo, conteinerizar e descrever IaC; não bloqueia
  a entrega mínima.

---

## 8. Testes automatizados

- `test_genetic.py`: crossover e mutação produzem cromossomos válidos (domínios respeitados);
  torneio seleciona dentro da população; elitismo preserva o melhor; `decode()` gera kwargs
  válidos para o XGBoost.
- `test_llm.py`: contrato da saída do `explainer` (campos esperados, parser robusto a
  resposta vazia/malformada); `client` é mockável (sem exigir Ollama no CI).

---

## 9. Entregáveis

- **Repositório Git** (`ml-tech`, branch `fase-2` → merge na `main`): código-fonte completo,
  notebooks de demonstração (`06`, `07`), `experiments/`, testes.
- **Relatório técnico** (`docs/relatorio_fase2.md`): implementação do AG e resultados da
  otimização; integração com LLM (abordagem, prompts, avaliação); comparativo original vs.
  otimizado; desafios e soluções.
- **Vídeo** (YouTube/Vimeo, ≤ 15 min): sistema em execução, componentes da solução,
  resultados do AG, demonstração da LLM.

---

## 10. Sequência de execução

1. **Setup** — branch `fase-2`; adicionar deps (`pyyaml`, `requests`/`ollama`); instalar
   Ollama + baixar modelo; criar esqueleto de pastas.
2. **AG core** — `chromosome` → `operators` → `fitness` → `ga_optimizer`, com testes a cada passo.
3. **Experimentos** — notebook `06`; rodar Exp A/B/C; salvar em `experiments/` e
   `results/models/xgboost_ga_optimized.pkl`.
4. **LLM** — `client` + `prompts` + `explainer`; notebook `07` ligando predição → explicação.
5. **Monitoramento** — curvas de convergência + logging estruturado.
6. **Relatório + roteiro do vídeo** — preencher `docs/relatorio_fase2.md`; gravar.

---

## 11. Decisões técnicas

- **AG escrito do zero** (sem DEAP): o enunciado pontua explicitamente codificação de genes
  e os três operadores; implementação própria (~200 linhas) é mais demonstrável no vídeo.
- **Modelo-alvo XGBoost**: melhor desempenho na Fase 1 e maior espaço de hiperparâmetros para
  o AG explorar. Extensível a Random Forest se sobrar tempo.
- **Fitness por F1/recall de óbito**: custo clínico de falso-negativo justifica priorizar recall.
- **LLM local (Ollama)**: sem custo de API, roda offline, atende o requisito de LLM
  pré-treinada; SHAP da Fase 1 alimenta o prompt e dá fidelidade às explicações.
- **Backend `mock` desde o início**: garante reprodutibilidade da entrega independente do
  setup de quem avalia; o Ollama é o caminho "real" demonstrado no vídeo, o `mock` é a rede
  de segurança para notebooks/testes/CI.
- **Foco no XGBoost (tabular)**: a CNN de imagem fica como extra da Fase 1, fora do alvo do AG.

---

## 12. Divisão sugerida do grupo (4 integrantes)

- **AG (core + experimentos):** 1–2 pessoas.
- **LLM (client, prompts, avaliação):** 1 pessoa.
- **Monitoramento, testes e integração:** 1 pessoa.
- **Relatório e vídeo:** responsabilidade compartilhada, montagem final por 1 pessoa.
</content>
</invoke>
