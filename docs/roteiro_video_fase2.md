# Roteiro do Vídeo — Tech Challenge Fase 2 (Projeto 1)

**Duração-alvo:** ≤ 15 min · **Upload:** YouTube/Vimeo (público ou não listado)
**Link já registrado:** https://www.youtube.com/watch?v=Bc4YDASEAGM

> **Antes de gravar:** para demonstrar a LLM real (não o template), rode:
> ```bash
> ollama pull llama3.1:8b
> export LLM_BACKEND=ollama
> jupyter nbconvert --to notebook --execute --inplace notebooks/07_llm_interpretation.ipynb
> ```
> Tenha aberto: o repositório no editor, os dois notebooks já executados, as figuras de
> convergência e o `docs/relatorio_fase2.md`.

---

## Distribuição do tempo (visão geral)

| Bloco | Tempo | Quem mostra |
|-------|-------|-------------|
| 1. Abertura e contexto | 0:00–1:30 | slides / fala |
| 2. Visão geral da solução e arquitetura | 1:30–3:00 | diagrama / editor |
| 3. Algoritmo Genético — código | 3:00–6:00 | `src/genetic/` |
| 4. Experimentos e resultados | 6:00–9:00 | notebook 06 + figuras |
| 5. Comparativo original × otimizado | 9:00–10:30 | tabela do notebook 06 |
| 6. Integração com LLM | 10:30–13:00 | `src/llm/` + notebook 07 |
| 7. Testes, monitoramento e fechamento | 13:00–15:00 | terminal + conclusões |

---

## Bloco 1 — Abertura e contexto (0:00–1:30)

- Apresentar o grupo e o projeto escolhido: **Projeto 1 — Otimização de Modelos de Diagnóstico**.
- Recapitular a Fase 1 em uma frase: classificação de desfecho (Cura × Óbito) em SRAG com dados
  do SIVEP-Gripe; melhor modelo tabular = **XGBoost**.
- Enunciar o objetivo da Fase 2: **otimizar o XGBoost com Algoritmo Genético** e **explicar os
  diagnósticos com uma LLM local**.
- Deixar claro o escopo: o AG otimiza o **modelo tabular**; a CNN de imagem fica como extra da
  Fase 1.

**Fala-chave:** "Vamos mostrar que um algoritmo genético supera o ajuste por GridSearch e que
uma LLM transforma a saída do modelo em explicação clínica útil."

---

## Bloco 2 — Visão geral e arquitetura (1:30–3:00)

- Mostrar a estrutura do repositório (`src/genetic`, `src/llm`, `src/monitoring`, `notebooks`,
  `experiments`, `tests`).
- Explicar o fluxo em uma frase: **dados da Fase 1 → AG otimiza hiperparâmetros → modelo
  otimizado → SHAP → LLM gera explicação**.
- Citar as decisões de projeto: AG escrito do zero (para evidenciar os operadores), fitness por
  F1, LLM local via Ollama com fallback `mock`.

**Mostrar na tela:** seção 9 (Arquitetura) do `relatorio_fase2.md`.

---

## Bloco 3 — Algoritmo Genético no código (3:00–6:00)

Percorrer `src/genetic/` destacando cada requisito do enunciado:

1. **Codificação** (`chromosome.py`): os 8 genes (hiperparâmetros) e seus domínios; `decode()`
   para kwargs do XGBoost. *Mostrar a tabela `GENES`.*
2. **Operadores** (`operators.py`):
   - **seleção por torneio** — sorteia k, promove o melhor;
   - **crossover uniforme** — troca gene a gene;
   - **mutação** — gaussiana (contínuos) / inteira (discretos);
   - **elitismo** — preserva os melhores.
3. **Fitness** (`fitness.py`): F1 via `StratifiedKFold`; tratamento de combinações inválidas;
   menção à penalização opcional de complexidade.
4. **Loop evolutivo** (`ga_optimizer.py`): inicialização → avaliação (paralela) → nova geração;
   histórico por geração e early stopping opcional.

**Fala-chave:** "Cada peça que o desafio pede — codificação, seleção, cruzamento, mutação e
fitness — está implementada explicitamente."

---

## Bloco 4 — Experimentos e resultados (6:00–9:00)

- Abrir `06_genetic_optimization.ipynb`.
- Explicar a **estratégia de desempenho**: busca em amostra estratificada de 30k + retreino do
  melhor no treino completo.
- Mostrar os **3 experimentos** (A/B/C) variando população, gerações e taxa de mutação.
- Exibir as **curvas de convergência** (`results/figures/ga_convergence_*.png`): fitness médio
  sobe e desvio cai ao longo das gerações.

**Mostrar na tela:** o comparativo de convergência `ga_convergence_comparativo.png`.

---

## Bloco 5 — Comparativo original × otimizado (9:00–10:30)

- Mostrar a tabela final do notebook 06:

| Modelo | F1 | ROC-AUC | Accuracy |
|--------|----|---------|----------|
| Original (Fase 1) | 0.447 | 0.880 | 0.852 |
| **XGBoost-AG (melhor)** | **0.513** | **0.906** | **0.914** |

- Destacar o ganho: **F1 +14,8%**, ROC-AUC +2,6 pts, precisão quase dobrou.
- Comentar o **trade-off** consciente: recall recua, mas o equilíbrio (F1/AUC) melhora; se a
  prioridade fosse recall de óbito, bastaria trocar a métrica de fitness.

**Fala-chave:** "O ganho é real e medido no conjunto de teste, não na busca."

---

## Bloco 6 — Integração com LLM (10:30–13:00)

- Abrir `src/llm/` e explicar a abordagem: **predição + fatores SHAP → prompt → LLM →
  explicação em PT-BR**.
- Mostrar `prompts.py`: o *system prompt* (papel, restrições de segurança) e as versões **v1 e
  v2 (few-shot)**.
- Citar os **dois backends**: `ollama` (local, principal) e `mock` (fallback reprodutível).
- Rodar a célula de explicações do `07_llm_interpretation.ipynb` **com o Ollama ativo** e ler
  uma explicação gerada para um caso de óbito de alta confiança.
- Mostrar a **comparação v1 × v2** e a **rubrica de avaliação**.

**Fala-chave:** "A LLM cita exatamente os fatores que pesaram na predição e reforça que é apoio
à decisão, não diagnóstico."

---

## Bloco 7 — Testes, monitoramento e fechamento (13:00–15:00)

- Rodar `pytest tests/ -v` ao vivo: **37 testes passando** (Fase 1 + Fase 2).
- Mostrar os artefatos de monitoramento em `experiments/<exp>/` (history.csv, summary.json,
  run.log).
- Fechar com as conclusões: AG superou o GridSearch; LLM entrega interpretabilidade; solução
  reprodutível e monitorável; base pronta para o assistente médico do Módulo 3.
- Agradecer e citar o link do repositório: https://github.com/arthurhardman/ml-tech

---

## Checklist de gravação

- [ ] `ollama pull llama3.1:8b` feito e `LLM_BACKEND=ollama` exportado
- [ ] Notebooks 06 e 07 já executados (saídas visíveis)
- [ ] Figuras de convergência abertas
- [ ] `pytest tests/ -v` testado antes (para rodar liso no vídeo)
- [ ] Áudio e tela em boa resolução; vídeo ≤ 15 min
- [ ] Upload público/não listado e link atualizado no PDF de entrega
</content>
