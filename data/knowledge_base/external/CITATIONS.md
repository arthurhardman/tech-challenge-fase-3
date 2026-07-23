# Datasets externos — Citações e Licenças

Este projeto usa, na abordagem híbrida da Fase 3, dois datasets públicos
sugeridos no enunciado, além dos dados internos sintéticos do hospital (SRAG).
As fatias curadas ficam em `data/knowledge_base/external/` e são derivadas das
fontes abaixo. A atribuição é obrigatória pelas licenças.

## PubMedQA

- **Subconjunto usado:** rotulado (`ori_pqal.json`, 1.000 QAs); usamos uma fatia curada de 250.
- **Licença:** MIT.
- **Site:** https://pubmedqa.github.io/ · **Repositório:** https://github.com/pubmedqa/pubmedqa
- **Citação:**

> Jin, Q., Dhingra, B., Liu, Z., Cohen, W., & Lu, X. (2019). *PubMedQA: A Dataset
> for Biomedical Research Question Answering.* Proceedings of the 2019 Conference
> on Empirical Methods in Natural Language Processing (EMNLP). arXiv:1909.06146.

## MedQuAD

- **Subconjuntos usados:** os 9 conjuntos do NIH que mantêm as respostas
  (CancerGov, GARD, GHR, MedlinePlus Health Topics, NIDDK, NINDS, NIHSeniorHealth,
  NHLBI, CDC). **Excluímos** os 3 subconjuntos cujas respostas foram removidas por
  copyright do MedlinePlus (ADAM, MedlinePlus Drugs, MedlinePlus Herbs). Fatia
  curada de 350.
- **Licença:** Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Repositório:** https://github.com/abachaa/MedQuAD
- **Citação:**

> Ben Abacha, A., & Demner-Fushman, D. (2019). *A Question-Entailment Approach to
> Question Answering.* BMC Bioinformatics, 20(1), 511:1–511:23.

## Como regenerar as fatias a partir das fontes

```bash
python scripts/fetch_datasets.py                      # baixa os brutos em data/external/
python -m src.assistant.finetuning.external_datasets  # gera as fatias curadas
```

> Observação: as fatias versionadas permitem rodar o pipeline **offline**. Os
> dados brutos completos (`data/external/`) não são versionados.
