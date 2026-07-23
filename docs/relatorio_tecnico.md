# Relatório Técnico — Tech Challenge Fase 1
## Classificação de Desfecho Clínico em SRAG (SIVEP-Gripe)
### FIAP Pos-Tech — Machine Learning Engineering

---

## 1. Introdução e Formulação do Problema

Este trabalho desenvolve um modelo de classificação binária supervisionada para prever o desfecho clínico de pacientes hospitalizados com Síndrome Respiratória Aguda Grave (SRAG), usando dados do SIVEP-Gripe (Sistema de Informação de Vigilância Epidemiológica da Gripe) do Ministério da Saúde do Brasil.

**Pergunta central:** Com base nos dados clínicos disponíveis na admissão, é possível prever se um paciente com SRAG hospitalizado irá se curar ou morrer?

**Variável alvo:** `EVOLUCAO`, binarizada como `OBITO`:

| Classe | `EVOLUCAO` original | `OBITO` codificado |
|--------|--------------------|--------------------|
| Cura | 1 | 0 |
| Óbito | 2 | 1 |

Registros com `EVOLUCAO = 3` (óbito por outras causas) e `EVOLUCAO = 9` (ignorado) são removidos — o modelo aprende apenas com desfechos clínicos definitivos e clinicamente interpretáveis.

Como validação complementar, implementamos um segundo pipeline usando deep learning para classificação de radiografias de tórax (COVID-19 Radiography Database, Kaggle), demonstrando a aplicação de CNN no mesmo domínio de saúde respiratória.

---

## 2. Estratégias de Pré-processamento

### 2.1 Pipeline Tabular

O pré-processamento é implementado em `src/tabular/preprocessing.py` e segue este fluxo:

```
[1] Filtrar EVOLUCAO válida → apenas 1=Cura e 2=Óbito
[2] Criar target binário OBITO (0=Cura, 1=Óbito)
[3] Selecionar 35 features clínicas
[4] Split estratificado 70/15/15  ← ANTES de qualquer transformação
[5] Imputação de valores ausentes
    ├── Numéricas → mediana (fit exclusivamente no X_train)
    └── Categóricas → moda (fit exclusivamente no X_train)
[6] Codificação categórica → LabelEncoder (fit exclusivamente no X_train)
[7] Normalização → StandardScaler (fit exclusivamente no X_train)
```

#### Decisão crítica: ausência de data leakage

O split estratificado acontece **antes** de qualquer transformação. Todos os transformadores (imputadores, encoders, scaler) são fitados exclusivamente no `X_train` e aplicados via `transform` nos conjuntos de validação e teste. Isso garante que as métricas finais reflitam desempenho real em dados nunca vistos pelo pipeline.

#### Seleção de features

35 features clínicas foram selecionadas com base em três critérios:

1. **Disponibilidade na admissão** — apenas variáveis coletadas na entrada do paciente
2. **Relevância clínica** — alinhamento com a literatura de fatores de risco para mortalidade por SRAG
3. **Qualidade de preenchimento** — exclusão de colunas com missingness excessivo ou preenchimento tardio

| Grupo | Qtd | Features |
|-------|-----|---------|
| Demográficas | 7 | `CS_SEXO`, `NU_IDADE_N`, `TP_IDADE`, `CS_GESTANT`, `CS_RACA`, `CS_ESCOL_N`, `CS_ZONA` |
| Sintomas na admissão | 12 | `FEBRE`, `TOSSE`, `GARGANTA`, `DISPNEIA`, `DESC_RESP`, `SATURACAO`, `DIARREIA`, `VOMITO`, `DOR_ABD`, `FADIGA`, `PERD_OLFT`, `PERD_PALA` |
| Comorbidades | 12 | `PUERPERA`, `CARDIOPATI`, `HEMATOLOGI`, `SIND_DOWN`, `HEPATICA`, `ASMA`, `DIABETES`, `NEUROLOGIC`, `PNEUMOPATI`, `IMUNODEPRE`, `RENAL`, `OBESIDADE` |
| Internação | 4 | `HOSPITAL`, `UTI`, `SUPORT_VEN`, `NOSOCOMIAL` |
| Vacinação | 2 | `VACINA`, `VACINA_COV` |

**Excluídas deliberadamente (data leakage):** `DT_EVOLUCA`, `DT_ENCERRA`, `NU_DO`, `CRITERIO`, resultados laboratoriais com preenchimento tardio.

#### Tratamento de desbalanceamento

O dataset é desbalanceado (maioria dos pacientes sobrevive). Estratégia adotada:

- Modelos sklearn: `class_weight='balanced'` — penaliza erros na classe minoritária proporcionalmente ao desbalanceamento
- XGBoost: `scale_pos_weight = n_cura / n_obito` — mecanismo equivalente, calculado automaticamente a partir do `y_train`

#### Divisão treino/validação/teste

| Split | Proporção | Uso |
|-------|-----------|-----|
| Treino | 70% | Fit dos modelos e transformadores |
| Validação | 15% | Tuning de hiperparâmetros (GridSearchCV interno) |
| Teste | 15% | Avaliação final — utilizado uma única vez |

Estratificação por `OBITO` garante a mesma proporção de óbitos nos três splits.

### 2.2 Pipeline de Imagens

Implementado em `src/image/image_preprocessing.py`:

- **Carregamento:** `cv2` (BGR → RGB), redimensionamento para 224×224×3
- **Normalização:** pixels divididos por 255.0 (range [0, 1])
- **Split:** estratificado 80/20 (treino/teste)
- **Data augmentation** (apenas no treino): rotação ≤10°, zoom ≤10%, deslocamentos ≤5%, flip horizontal

As transformações de augmentation foram escolhidas como conservadoras intencionalmente: radiografias de tórax têm orientação e escala anatômica clinicamente relevantes — transformações agressivas introduziriam artefatos não representativos da realidade clínica.

---

## 3. Modelos Utilizados e Justificativas

### 3.1 Modelos Tabulares

Implementados em `src/tabular/modeling.py`. Todos passam por `GridSearchCV` com `StratifiedKFold(n_splits=5)`, otimizando **F1-score da classe positiva (Óbito)**.

#### Regressão Logística

**Papel:** baseline linear e interpretável.

**Justificativa:** estabelece o piso de desempenho e permite verificar se a separabilidade linear do problema é suficiente. Alta interpretabilidade facilita a explicação do modelo para audiências não técnicas.

**Hiperparâmetros tuned:** `C` ∈ {0.01, 0.1, 1.0, 10.0}, `solver` ∈ {lbfgs, liblinear}

#### Árvore de Decisão

**Papel:** modelo não-linear com regras explícitas e completamente interpretável.

**Justificativa:** gera regras de decisão legíveis ("se saturação < X e UTI = 1, então Óbito") que podem ser validadas clinicamente. Serve também para identificar os splits mais informativos.

**Hiperparâmetros tuned:** `max_depth` ∈ {5, 10, 20, None}, `min_samples_split` ∈ {2, 10, 50}, `criterion` ∈ {gini, entropy}

#### Random Forest

**Papel:** ensemble robusto com boa generalização.

**Justificativa:** reduz a variância da Árvore de Decisão via bagging. É menos suscetível a overfitting em datasets grandes e fornece `feature_importances_` estável.

**Hiperparâmetros tuned:** `n_estimators` ∈ {100, 300}, `max_depth` ∈ {10, 20, None}, `min_samples_split` ∈ {2, 10}

#### XGBoost

**Papel:** modelo de alta performance baseado em boosting.

**Justificativa:** historicamente dominante em competições de dados tabulares. O boosting foca iterativamente nos exemplos mais difíceis, o que é especialmente útil com desbalanceamento de classes.

**Nota técnica:** `XGBClassifier` não aceita `class_weight='balanced'`. A correção aplicada é `scale_pos_weight = n_cura / n_obito` — mecanismo oficial do XGBoost para desbalanceamento.

**Hiperparâmetros tuned:** `n_estimators` ∈ {100, 300}, `max_depth` ∈ {4, 6, 8}, `learning_rate` ∈ {0.05, 0.1, 0.2}, `subsample` ∈ {0.8, 1.0}

### 3.2 Modelo de Imagem — CNN Própria

Implementado em `src/image/image_model.py` usando Keras Sequential.

**Justificativa da CNN própria (sem transfer learning):** o objetivo do trabalho é praticar o pipeline de deep learning de ponta a ponta. Usar modelos pré-treinados (VGG16, ResNet) eliminaria a maior parte do aprendizado técnico e tornaria a explicação em banca mais difícil.

**Arquitetura:**

```
Input: (224, 224, 3)
Conv2D(32, 3×3, relu)  → 222×222×32
MaxPooling2D(2×2)      → 111×111×32
Conv2D(64, 3×3, relu)  → 109×109×64
MaxPooling2D(2×2)      → 54×54×64
GlobalAveragePooling2D → 64
Dense(64, relu)        → 64
Dense(4, softmax)      → 4 classes

Total: 23.812 parâmetros treináveis
```

**Decisão arquitetural:** `GlobalAveragePooling2D` em vez de `Flatten`. Com `Flatten`, o vetor resultante teria 54×54×64 = 186.624 valores, gerando >23 milhões de parâmetros — desproporcional para 960 imagens de treino e garantidamente causaria overfitting severo. `GlobalAveragePooling2D` reduz o modelo em ~1.000×.

**Compilação:** Adam, `sparse_categorical_crossentropy`, 15 épocas, batch 16.

---

## 4. Resultados e Interpretação

### 4.1 Métricas de Avaliação — Pipeline Tabular

As métricas abaixo são calculadas no conjunto de **teste** (15% dos dados, nunca visto durante treino ou tuning):

| Métrica | Descrição | Relevância no projeto |
|---------|-----------|----------------------|
| **F1-score (Óbito)** | Média harmônica de Precision e Recall para a classe positiva | **Métrica principal** — equilíbrio entre identificar óbitos e não gerar alarmes excessivos |
| **Recall (Óbito)** | Dos pacientes que morreram, quantos o modelo identificou | Clinicamente crítico — falso negativo tem custo alto |
| **Precision (Óbito)** | Dos que o modelo previu como Óbito, quantos realmente eram | Relevante para gestão de recursos (evitar alocação desnecessária) |
| **ROC-AUC** | Área sob a curva ROC, independente de threshold | Comparação robusta entre modelos em dados desbalanceados |
| **Accuracy** | Proporção de acertos total | Referência apenas — enganosa com classes desbalanceadas |

> **Por que não usar Accuracy como métrica principal?** Um modelo que prevê sempre "Cura" já teria ~75% de accuracy sem nenhum valor clínico. F1 e ROC-AUC são robustos ao desbalanceamento.

#### Visualizações geradas

Todos os gráficos são salvos em `results/figures/`:

| Arquivo | Descrição |
|---------|-----------|
| `confusion_matrix_{modelo}.png` | Matriz de confusão por modelo — evidencia falsos negativos |
| `roc_curves.png` | Curvas ROC de todos os modelos num único gráfico |
| `precision_recall_curves.png` | Curvas PR — mais informativas com classes desbalanceadas |
| `feature_importance_{modelo}.png` | Top 20 features por importância (modelos tree-based) |
| `shap_summary_{modelo}.png` | SHAP summary plot — impacto e direção de cada feature |

### 4.2 Interpretabilidade — Feature Importance e SHAP

**Feature Importance** (Random Forest e XGBoost) indica quais variáveis o modelo utiliza mais nas suas divisões. As features de maior importância esperada são: `NU_IDADE_N` (idade), `SATURACAO`, `UTI` e `SUPORT_VEN`.

**SHAP (SHapley Additive exPlanations)** vai além: mostra não apenas a magnitude da contribuição de cada feature, mas também a **direção** do efeito sobre a predição. Exemplos esperados:

- `SATURACAO = 1` (saturação baixa) → empurra predição para Óbito (SHAP positivo)
- `UTI = 1` (internado em UTI) → empurra predição para Óbito
- `NU_IDADE_N` alto → empurra predição para Óbito
- `VACINA_COV = 1` (vacinado) → empurra predição para Cura (SHAP negativo)

Essa interpretação é clinicamente plausível e indica que o modelo aprendeu padrões reais, não artefatos dos dados.

Para modelos tree-based, usa-se `TreeExplainer` (eficiente em tempo). Para Regressão Logística, usa-se `KernelExplainer` como fallback.

### 4.3 Discussão Crítica

**O modelo pode ser usado na prática?**

Como ferramenta de **apoio à decisão clínica**, sim. Como substituto do julgamento médico, não. O modelo produz um score de risco que pode auxiliar na triagem de pronto-socorro e priorização de leitos de UTI. O médico sempre tem a palavra final.

**Limitações identificadas:**

| Limitação | Impacto | Mitigação |
|-----------|---------|-----------|
| `9=Ignorado` tratado como categoria, não como ausente | Viés na imputação | Substituir por NaN antes da imputação |
| Dados de um único ano (2024) | Pode não generalizar para outros anos/ondas | Análise temporal cross-year |
| Features de internação (`UTI`, `SUPORT_VEN`) podem ser atualizadas após piora | Risco de leakage parcial | Auditoria temporal no dicionário de dados |
| Modelo não calibrado | Probabilidades podem não refletir frequências reais | `CalibratedClassifierCV` em produção |
| Threshold fixo em 0.5 | Não otimizado para o custo assimétrico dos erros clínicos | Threshold tuning favorecendo recall de Óbito |

### 4.4 Métricas de Avaliação — Pipeline de Imagem

A avaliação gera `classification_report` do sklearn (precision, recall, F1 por classe) e matriz de confusão visualizada. A CNN própria com dataset reduzido (300 imagens/classe) serve como prova de conceito — resultados melhorariam com mais dados e regularização adicional (Dropout, early stopping).

---

## 5. Reprodutibilidade

`RANDOM_STATE = 42` é aplicado em todos os pontos de aleatoriedade: `train_test_split`, `StratifiedKFold`, todos os estimadores sklearn e XGBoost, amostragem SHAP, e o split do pipeline de imagem.

Todos os transformadores do pipeline tabular são serializados com `joblib` em `data/processed/`, garantindo que inferência em novos dados use exatamente os mesmos parâmetros do treino.

**18 testes automatizados** cobrem as invariantes críticas do pipeline:
- Filtro correto de `EVOLUCAO`
- Target binário sem valores fora de {0, 1}
- Proporções do split 70/15/15 (±2%)
- Ausência de overlap entre splits
- Imputação sem leakage de mediana global
- LabelEncoder mapeando categorias desconhecidas para -1
- StandardScaler resultando em média ≈ 0 no treino
- Interface correta dos 4 modelos (fit/predict/predict\_proba)
