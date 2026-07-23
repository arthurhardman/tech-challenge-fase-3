# Relatório Técnico — Tech Challenge Fase 2 (Projeto 1)

**FIAP Pós-Tech — Machine Learning Engineering**
**Projeto:** Otimização de Modelos de Diagnóstico com Algoritmos Genéticos e LLM
**Repositório:** `ml-tech` (continuação da Fase 1)

**Integrantes:** Arthur Augusto Paula Hardman (rm372971) · Gabriel Franco Garcia (rm373872) ·
Icaro Oliveira Costa (rm371223) · Daniel Ruiz De Campos Pascoalato (rm372385)

---

## 1. Introdução e objetivo

Na Fase 1 construímos modelos de classificação do desfecho clínico (Cura × Óbito) de
pacientes hospitalizados com SRAG, a partir dos dados do SIVEP-Gripe. A Fase 2 evolui esse
trabalho em duas frentes:

1. **Otimização por Algoritmo Genético (AG)** dos hiperparâmetros do modelo tabular principal
   (XGBoost), buscando ganho de desempenho sobre o modelo original (ajustado por GridSearchCV).
2. **Integração com LLM** para traduzir as predições numéricas em **explicações clínicas em
   linguagem natural**, apoiando a interpretação pela equipe de saúde.

> **Escopo da otimização.** O AG otimiza o **XGBoost** (melhor modelo tabular da Fase 1). A
> CNN de classificação de imagens de raio-X permanece como **entrega extra da Fase 1** e não é
> alvo da otimização desta fase — o XGBoost concentra o maior espaço de hiperparâmetros e o
> impacto direto sobre a tarefa de prognóstico.

---

## 2. Reaproveitamento da Fase 1

| Artefato da Fase 1 | Uso na Fase 2 |
|--------------------|---------------|
| Pipeline `src/tabular/` (load, preprocessing) | Carregamento e geração dos splits (treino/val/teste) |
| `results/models/xgboost.pkl` (GridSearchCV) | **Baseline** para o comparativo original × otimizado |
| Cálculo de SHAP (`src/tabular/evaluation.py`) | Fatores por paciente que alimentam o prompt da LLM |

O dataset bruto (`INFLUD24-26-06-2025`, 267.984 registros × 194 variáveis) é pré-processado
exatamente como na Fase 1: filtragem de `EVOLUCAO` válida, imputação ajustada só no treino,
codificação de categóricas e normalização. O alvo binário resultante é fortemente
desbalanceado (~13% de óbitos), o que orienta a escolha da métrica de fitness.

---

## 3. Algoritmo Genético

Implementação **própria** (sem bibliotecas de AG), em `src/genetic/`, para tornar explícitos
os componentes exigidos: codificação, seleção, cruzamento, mutação e fitness.

### 3.1 Codificação (representação dos genes)

Cada indivíduo é um vetor de 8 genes, um por hiperparâmetro do XGBoost, com domínio fechado:

| Gene | Tipo | Domínio |
|------|------|---------|
| `n_estimators` | int | 50–600 |
| `max_depth` | int | 2–12 |
| `learning_rate` | float | 0.01–0.3 |
| `subsample` | float | 0.5–1.0 |
| `colsample_bytree` | float | 0.5–1.0 |
| `min_child_weight` | int | 1–10 |
| `gamma` | float | 0.0–5.0 |
| `scale_pos_weight` | float | 1.0–12.0 |

`decode()` converte o indivíduo em kwargs do `XGBClassifier`. Todo operador devolve indivíduos
recortados ao domínio/tipo (`clip_individual`), garantindo validade.

### 3.2 Operadores

- **Seleção por torneio** (k configurável): sorteia k indivíduos e promove o de maior fitness —
  equilíbrio simples entre pressão seletiva e diversidade.
- **Crossover uniforme**: com probabilidade `crossover_rate`, cada gene é trocado de forma
  independente entre os pais (p = 0,5).
- **Mutação**: gaussiana para genes contínuos (desvio proporcional à amplitude do domínio) e
  perturbação inteira para genes discretos, com probabilidade `mutation_rate` por gene.
- **Elitismo**: preserva os N melhores indivíduos por geração, impedindo regressão do melhor
  fitness.

### 3.3 Função fitness

```
fitness(indivíduo) = média( F1-score do XGBoost(decode(indivíduo)) )
                     em StratifiedKFold(n_splits=3) sobre a amostra de treino
```

Escolhemos **F1** como métrica primária por equilibrar precisão e recall na classe minoritária
(Óbito). Há suporte opcional a penalização por complexidade (favorecendo modelos mais enxutos)
e a outras métricas (recall, ROC-AUC). A avaliação da população é **paralelizada com `joblib`**.

### 3.4 Loop evolutivo

`GeneticOptimizer` inicializa a população, avalia (em paralelo), registra o histórico por
geração (melhor/média/desvio), aplica torneio → crossover → mutação → elitismo, e itera até o
número de gerações ou um critério opcional de *early stopping*.

### 3.5 Estratégia de desempenho

Cada avaliação de fitness treina um XGBoost com validação cruzada; sobre as 187 mil linhas de
treino, avaliar centenas de indivíduos seria proibitivo. Adotamos uma estratégia padrão:

1. o **AG busca** sobre uma **amostra estratificada** de 30 mil registros (mantém a proporção
   de óbitos);
2. o **melhor indivíduo** de cada experimento é **retreinado no conjunto de treino completo**
   antes da avaliação final no teste.

---

## 4. Experimentos e resultados

Três experimentos com configurações distintas de AG (atendendo ao requisito de ≥ 3), todos
otimizando F1 por validação cruzada. Configurações em `experiments/experimentos.yaml`.

| Exp | População | Gerações | Crossover | Mutação | F1 (busca, CV) | Tempo |
|-----|-----------|----------|-----------|---------|----------------|-------|
| A | 10 | 8 | 0.8 | 0.10 | 0.5057 | ~14 s |
| B | 20 | 10 | 0.9 | 0.05 | 0.5091 | ~29 s |
| C | 30 | 12 | 0.8 | 0.20 | 0.5126 | ~74 s |

As curvas de convergência (melhor e média por geração) estão em
`results/figures/ga_convergence_exp_*.png` e o comparativo em
`results/figures/ga_convergence_comparativo.png`. Observa-se aumento do fitness médio e
redução do desvio ao longo das gerações — sinal de convergência da população.

### 4.1 Comparativo: modelo original × otimizado (conjunto de teste)

| Modelo | Accuracy | Precision | Recall | **F1** | ROC-AUC |
|--------|----------|-----------|--------|--------|---------|
| XGBoost original (Fase 1, GridSearchCV) | 0.8523 | 0.3301 | 0.6928 | **0.4471** | 0.8797 |
| XGBoost-AG (exp_A) — **melhor F1** | 0.9113 | 0.4869 | 0.5429 | **0.5134** | 0.9055 |
| XGBoost-AG (exp_B) | 0.8996 | 0.4407 | 0.6124 | 0.5126 | 0.9055 |
| XGBoost-AG (exp_C) | 0.9140 | 0.5012 | 0.5243 | 0.5125 | 0.9059 |

**Leitura dos resultados.** O modelo otimizado pelo AG melhora o **F1 de 0,447 para 0,513
(+14,8%)**, com ganho expressivo de **precisão** (0,330 → 0,487) e de **ROC-AUC** (0,880 →
0,906). Há um *trade-off* esperado: o recall recua (0,693 → 0,543), pois o modelo original
priorizava sensibilidade às custas de muitos falsos positivos. O AG, guiado pelo F1, encontra
um equilíbrio melhor entre identificar óbitos e não superalarmar. O melhor modelo foi
serializado em `results/models/xgboost_ga_optimized.pkl`.

> **Nota.** Se a prioridade clínica for **maximizar recall de óbito** (minimizar falso-negativo),
> basta trocar a métrica de fitness para `recall` em `experiments/experimentos.yaml` — a
> arquitetura já suporta. O exp_B, com maior recall (0,612) mantendo F1 alto, é uma alternativa
> nesse sentido.

---

## 5. Integração com LLM

### 5.1 Abordagem

A camada `src/llm/` transforma a saída do modelo em explicação clínica:

```
predição (classe + probabilidade) + fatores SHAP do paciente
        → prompt estruturado → LLM → explicação em PT-BR
```

Os **fatores SHAP** (já calculados na Fase 1, via `TreeExplainer`) indicam, por paciente, quais
variáveis empurraram a predição para Óbito (contribuição positiva) ou Cura (negativa). As
`top-k` features por magnitude alimentam o prompt, dando **fidelidade** à explicação.

### 5.2 Backends e robustez

| Backend | Descrição |
|---------|-----------|
| `ollama` | LLM open-source local (`llama3.1:8b`), via API em `localhost:11434`. Sem custo, offline. Backend **principal**, demonstrado no vídeo. |
| `mock` | Respostas por **template** a partir do próprio contexto (predição + fatores). Sem dependência externa. |

A seleção é por `LLM_BACKEND=ollama|mock|auto`. Em `auto`/`ollama`, se o servidor ou o modelo
não estiverem disponíveis, há **fallback automático para `mock`**. Isso garante que **notebooks
e testes rodem em qualquer máquina** — decisão importante, já que o ambiente de avaliação é
desconhecido. (Para a LLM real: instalar o Ollama e `ollama pull llama3.1:8b`.)

### 5.3 Prompt engineering

O `system prompt` fixa o papel ("assistente clínico de apoio à decisão"), restrições de
segurança (não inventar dados, citar os fatores, reforçar que não substitui o julgamento
médico) e o idioma. Foram avaliadas **duas versões** de prompt de usuário:

- **v1** — instrução direta;
- **v2** — instrução + **exemplo few-shot** que fixa tom e formato.

Ambas são parametrizáveis no `DiagnosisExplainer`, permitindo comparação objetiva (notebook
`07`, seção 5).

### 5.4 Avaliação da qualidade

Rubrica qualitativa (1–5) em quatro critérios — **fidelidade**, **clareza**, **acionabilidade**
e **segurança** — aplicada a casos representativos (óbito de alta confiança, cura de alta
confiança e caso de fronteira). No backend `mock`, fidelidade e segurança são altas por
construção (template ancorado nos fatores); com `ollama`, a clareza e a acionabilidade tendem a
melhorar, e as notas devem ser reavaliadas e exemplificadas no vídeo/relatório final.

---

## 6. Monitoramento, logging e escalabilidade

- **Tracking** (`src/monitoring/tracker.py`): a cada geração registra melhor/média/desvio do
  fitness e tempo, persistindo `history.csv`, `summary.json` e `run.log` em
  `experiments/<exp>/`, além do gráfico de convergência.
- **Escalabilidade**: a avaliação da população é paralelizada (`joblib`, `n_jobs` configurável),
  reaproveitando o padrão de paralelismo já adotado na Fase 1. A implementação em nuvem (IaC)
  era **opcional** (pontuação extra) e não foi incluída no escopo mínimo.

---

## 7. Testes automatizados

Suíte em `tests/` (executável com `pytest`, sem depender de Ollama):

- `test_genetic.py` — validade de codificação/`decode`, recorte de domínios, seleção por
  torneio, crossover, mutação, elitismo e convergência do loop (com avaliador *fake* rápido);
- `test_llm.py` — seleção de backend e fallback para `mock`, extração de predição/fatores,
  contrato de saída do `explainer` e ordenação de fatores a partir de SHAP.

Resultado: **37 testes** no total (Fase 1 + Fase 2), todos passando, sem regressão.

---

## 8. Desafios e soluções

| Desafio | Solução |
|---------|---------|
| Custo de avaliar o fitness sobre 187k linhas | Busca em amostra estratificada (30k) + retreino do melhor no treino completo |
| Combinações inválidas de hiperparâmetros derrubando a evolução | `fitness` captura exceções e atribui fitness 0; operadores sempre recortam ao domínio |
| Paralelismo aninhado (XGBoost × joblib) estourando memória | `n_jobs=1` no modelo durante a busca; paralelismo apenas no nível da população |
| Ambiente de avaliação desconhecido (LLM local pode não existir) | Backend `mock` + fallback automático, garantindo reprodutibilidade |
| Desbalanceamento de classes (~13% óbitos) | Fitness por F1 e `scale_pos_weight` como gene evoluído |

---

## 9. Arquitetura da solução

```
src/
├── tabular/        # pipeline da Fase 1 (load, preprocessing, modeling, evaluation)
├── genetic/        # AG: chromosome, operators, fitness, ga_optimizer
├── llm/            # client (ollama|mock), prompts, explainer
└── monitoring/     # tracker (logging + convergência)
notebooks/
├── 06_genetic_optimization.ipynb   # experimentos de AG + comparativo
└── 07_llm_interpretation.ipynb     # explicações via LLM
experiments/        # configs + artefatos por experimento
results/            # figures (convergência, SHAP) + models (.pkl)
tests/              # test_genetic, test_llm (+ testes da Fase 1)
```

---

## 10. Conclusões e próximos passos

- O Algoritmo Genético **superou o ajuste original** do XGBoost em F1 (+14,8%) e ROC-AUC,
  demonstrando o valor da busca evolutiva de hiperparâmetros sobre o GridSearchCV.
- A camada de LLM entrega **interpretabilidade acionável**, convertendo predições + SHAP em
  texto clínico, com prompt engineering versionado e avaliação de qualidade.
- A solução é **reprodutível** (backend `mock`, testes, configs versionadas) e **monitorável**
  (tracking por geração).
- **Módulo 3:** esta base prepara a integração com dados textuais (prontuários/laudos),
  evoluindo o explicador para um assistente médico mais completo.

---

## 11. Como reproduzir

```bash
# 1. Ambiente (Python 3.11/3.12) e dependências
pip install -r requirements.txt

# 2. Dataset bruto em data/raw/INFLUD24-26-06-2025.csv (separador ';', latin-1)

# 3. Testes
pytest tests/ -v

# 4. Experimentos de AG e comparativo
jupyter nbconvert --to notebook --execute notebooks/06_genetic_optimization.ipynb

# 5. Interpretação via LLM (mock por padrão; para LLM real:)
#    ollama pull llama3.1:8b && export LLM_BACKEND=ollama
jupyter nbconvert --to notebook --execute notebooks/07_llm_interpretation.ipynb
```
</content>
