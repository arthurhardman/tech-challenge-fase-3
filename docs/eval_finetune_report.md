# Avaliação do fine-tuning — Assistente Médico (Fase 3)

- **Modelo base:** Qwen/Qwen2.5-1.5B-Instruct
- **Exemplos de teste avaliados:** 14

## Qualidade de conteúdo (ROUGE-L vs. referência)

- Base: 0.0575
- Fine-tuned: 0.1347

## Segurança e explainability (só modelo fine-tuned)

- % de respostas com disclaimer de validação médica: 0.0%
- % de respostas que citam a fonte da informação: 0.0%

> Nota: ROUGE-L é um proxy de sobreposição lexical, não de correção clínica. Ele indica se o modelo aprendeu o *vocabulário e formato* dos protocolos, mas a validação final de conteúdo clínico deve ser feita por um profissional de saúde (ver relatório técnico, seção de limitações).