"""
gen_synthetic_data.py
---------------------
Gera a base de conhecimento SINTÉTICA e ANONIMIZADA da Fase 3.

Nada aqui contém dados reais de pacientes: protocolos, perguntas frequentes,
modelos de laudo/receita e prontuários são todos gerados proceduralmente, com
identificadores fictícios. Isso atende ao requisito de "dataset anonimizado ou
exemplo de dados sintéticos" sem qualquer risco de exposição de PII.

Domínio: SRAG (Síndrome Respiratória Aguda Grave) — o mesmo das Fases 1 e 2,
mantendo o projeto coeso (mesmo "hospital", agora com um assistente clínico).

Saídas (em data/knowledge_base/):
  - protocolos/*.md          → protocolos clínicos internos (fonte do RAG)
  - faq_medicos.jsonl        → perguntas frequentes de médicos + respostas
  - laudos_modelos.jsonl     → modelos de laudo, receita e procedimento
  - prontuarios.json         → base estruturada de pacientes sintéticos

Uso:
    python scripts/gen_synthetic_data.py [--n-pacientes 40] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KB_DIR = PROJECT_ROOT / "data" / "knowledge_base"
PROTO_DIR = KB_DIR / "protocolos"


# --------------------------------------------------------------------------- #
# 1. Protocolos clínicos internos (fonte primária de verdade do assistente)
# --------------------------------------------------------------------------- #
PROTOCOLOS = {
    "PROT-SRAG-01_triagem.md": """# PROT-SRAG-01 — Triagem e Classificação de Risco em SRAG

**Versão:** 3.1 | **Revisão:** anual | **Aprovação:** Comissão de Protocolos Clínicos

## 1. Definição de caso
Considera-se SRAG o paciente com síndrome gripal que apresente dispneia,
saturação de O2 < 95% em ar ambiente, desconforto respiratório ou piora de
condição de base.

## 2. Classificação de risco na admissão
- **Verde (leve):** SpO2 >= 95%, sem dispneia, sem comorbidade descompensada.
- **Amarelo (moderado):** SpO2 90-94% ou dispneia leve; reavaliar em 2h.
- **Vermelho (grave):** SpO2 < 90%, uso de musculatura acessória, confusão
  mental ou comorbidade descompensada. Acionar equipe de resposta rápida.

## 3. Exames mínimos na admissão
Hemograma, PCR, gasometria arterial, RX de tórax e, quando indicado,
tomografia. Registrar SpO2 em ar ambiente antes de qualquer suplementação.

## 4. Sinais de alerta para reavaliação imediata
Queda de SpO2 > 3 pontos, frequência respiratória > 28 irpm, hipotensão.
""",
    "PROT-SRAG-02_oxigenoterapia.md": """# PROT-SRAG-02 — Oxigenoterapia e Suporte Ventilatório

**Versão:** 2.4 | **Aprovação:** Comissão de Protocolos Clínicos

## 1. Alvo de saturação
Manter SpO2 entre 92% e 96% (94-98% em gestantes). Evitar hiperóxia.

## 2. Escalonamento
1. Cateter nasal (1-5 L/min).
2. Máscara com reservatório se SpO2 < 92% com cateter.
3. Cânula nasal de alto fluxo (CNAF) em falha da máscara.
4. Considerar ventilação mecânica se FR > 35, SpO2 < 90% com CNAF, ou
   rebaixamento do nível de consciência.

## 3. Critérios para UTI
SpO2 < 90% refratária, necessidade de ventilação mecânica, instabilidade
hemodinâmica. A decisão de intubação é sempre de médico plantonista.
""",
    "PROT-SRAG-03_farmacologico.md": """# PROT-SRAG-03 — Manejo Farmacológico de Suporte

**Versão:** 1.9 | **Aprovação:** Comissão de Farmácia e Terapêutica

## 1. Princípios
Toda prescrição é ato médico. O assistente virtual NÃO prescreve: apenas
sugere condutas previstas em protocolo, que exigem validação humana.

## 2. Corticoterapia
Indicada em pacientes com necessidade de oxigenoterapia suplementar, conforme
avaliação médica. Dose e via a critério do prescritor.

## 3. Profilaxia de tromboembolismo
Avaliar profilaxia em todo paciente internado por SRAG sem contraindicação.

## 4. Antibioticoterapia
Somente com suspeita de coinfecção bacteriana documentada. Evitar uso empírico
prolongado. Reavaliar em 48-72h com culturas.
""",
    "PROT-SRAG-04_alta_isolamento.md": """# PROT-SRAG-04 — Critérios de Alta e Isolamento

**Versão:** 2.0 | **Aprovação:** CCIH

## 1. Critérios de alta hospitalar
SpO2 >= 94% em ar ambiente por >= 24h, ausência de febre por 48h sem
antitérmico, estabilidade clínica e tolerância à dieta.

## 2. Isolamento
Precaução para gotículas e contato até definição etiológica. Suspender
conforme resultado laboratorial e diretriz da CCIH.

## 3. Orientações de alta
Retorno em caso de dispneia, febre persistente ou queda de saturação medida
em oxímetro domiciliar.
""",
}


# --------------------------------------------------------------------------- #
# 2. FAQ de médicos (perguntas frequentes → base para fine-tuning e avaliação)
# --------------------------------------------------------------------------- #
FAQ_MEDICOS = [
    {
        "pergunta": "Qual o alvo de saturação para paciente com SRAG em oxigenoterapia?",
        "resposta": "Manter SpO2 entre 92% e 96% (94-98% em gestantes), evitando hiperóxia, conforme PROT-SRAG-02.",
        "fonte": "PROT-SRAG-02",
    },
    {
        "pergunta": "Quando devo considerar transferência para UTI?",
        "resposta": "Considere UTI em SpO2 < 90% refratária, necessidade de ventilação mecânica ou instabilidade hemodinâmica. A decisão de intubação é do médico plantonista (PROT-SRAG-02).",
        "fonte": "PROT-SRAG-02",
    },
    {
        "pergunta": "Como classifico o risco de um paciente na admissão?",
        "resposta": "Use a classificação verde/amarelo/vermelho do PROT-SRAG-01, baseada em SpO2, dispneia e comorbidades descompensadas. Registre a SpO2 em ar ambiente antes de suplementar O2.",
        "fonte": "PROT-SRAG-01",
    },
    {
        "pergunta": "Quais exames pedir na admissão de um caso de SRAG?",
        "resposta": "Hemograma, PCR, gasometria arterial, RX de tórax e tomografia quando indicada, conforme PROT-SRAG-01.",
        "fonte": "PROT-SRAG-01",
    },
    {
        "pergunta": "Posso iniciar antibiótico empírico em todo paciente com SRAG?",
        "resposta": "Não. Antibioticoterapia só com suspeita de coinfecção bacteriana documentada; evite uso empírico prolongado e reavalie em 48-72h com culturas (PROT-SRAG-03).",
        "fonte": "PROT-SRAG-03",
    },
    {
        "pergunta": "Quando o paciente pode receber alta?",
        "resposta": "SpO2 >= 94% em ar ambiente por >= 24h, sem febre por 48h sem antitérmico, estável e tolerando dieta (PROT-SRAG-04).",
        "fonte": "PROT-SRAG-04",
    },
    {
        "pergunta": "Quais os sinais de alerta para reavaliação imediata?",
        "resposta": "Queda de SpO2 maior que 3 pontos, frequência respiratória acima de 28 irpm e hipotensão (PROT-SRAG-01).",
        "fonte": "PROT-SRAG-01",
    },
    {
        "pergunta": "Como escalonar o suporte de oxigênio?",
        "resposta": "Cateter nasal (1-5 L/min) → máscara com reservatório → cânula nasal de alto fluxo → ventilação mecânica conforme critérios do PROT-SRAG-02.",
        "fonte": "PROT-SRAG-02",
    },
    {
        "pergunta": "O assistente virtual pode prescrever medicação?",
        "resposta": "Não. Toda prescrição é ato médico. O assistente apenas sugere condutas previstas em protocolo, sempre sujeitas a validação humana (PROT-SRAG-03).",
        "fonte": "PROT-SRAG-03",
    },
    {
        "pergunta": "Preciso avaliar profilaxia de trombose em quem interna por SRAG?",
        "resposta": "Sim. Avalie profilaxia de tromboembolismo em todo paciente internado por SRAG sem contraindicação (PROT-SRAG-03).",
        "fonte": "PROT-SRAG-03",
    },
]


# --------------------------------------------------------------------------- #
# 3. Modelos de laudo, receita e procedimento (para fine-tuning de formato)
# --------------------------------------------------------------------------- #
LAUDOS_MODELOS = [
    {
        "tipo": "laudo",
        "titulo": "Laudo de RX de tórax — SRAG",
        "instrucao": "Redija um laudo estruturado de RX de tórax para caso de SRAG com infiltrado bilateral.",
        "modelo": (
            "LAUDO DE RADIOGRAFIA DE TÓRAX\n"
            "Técnica: incidência PA em inspiração.\n"
            "Achados: opacidades em vidro fosco de distribuição bilateral e periférica, "
            "predominando em bases. Ausência de derrame pleural volumoso.\n"
            "Impressão: achados compatíveis com acometimento pulmonar por SRAG. "
            "Correlacionar com quadro clínico e laboratorial.\n"
            "Observação: laudo de apoio; conclusão diagnóstica é do médico assistente."
        ),
    },
    {
        "tipo": "receita",
        "titulo": "Modelo de receita de suporte (a ser validada)",
        "instrucao": "Gere um modelo de orientação de suporte para alta de SRAG leve.",
        "modelo": (
            "ORIENTAÇÕES DE ALTA (modelo — requer validação e assinatura médica)\n"
            "1. Repouso relativo e hidratação oral.\n"
            "2. Antitérmico se temperatura > 37,8 C, conforme prescrição médica.\n"
            "3. Oximetria domiciliar: retornar se SpO2 < 94%.\n"
            "4. Sinais de alarme: dispneia, febre persistente, dor torácica.\n"
            "Este é um modelo de apoio; a prescrição efetiva é ato médico."
        ),
    },
    {
        "tipo": "procedimento",
        "titulo": "Checklist de admissão de SRAG",
        "instrucao": "Liste o procedimento de admissão de um paciente com suspeita de SRAG.",
        "modelo": (
            "PROCEDIMENTO DE ADMISSÃO — SRAG\n"
            "[ ] Medir SpO2 em ar ambiente e registrar.\n"
            "[ ] Classificar risco (verde/amarelo/vermelho) — PROT-SRAG-01.\n"
            "[ ] Solicitar exames mínimos (hemograma, PCR, gasometria, RX).\n"
            "[ ] Instituir precaução para gotículas e contato — PROT-SRAG-04.\n"
            "[ ] Registrar comorbidades e medicações de uso contínuo.\n"
            "[ ] Acionar resposta rápida se classificação vermelha."
        ),
    },
]


# --------------------------------------------------------------------------- #
# 4. Prontuários sintéticos (base ESTRUTURADA consultada pelo assistente)
# --------------------------------------------------------------------------- #
_COMORBIDADES = ["diabetes", "cardiopatia", "obesidade", "asma", "renal", "nenhuma"]
_SEXOS = ["M", "F"]


def _gera_prontuario(idx: int, rng: random.Random) -> dict:
    """Cria um prontuário sintético com ID fictício (sem qualquer PII real)."""
    idade = rng.randint(20, 92)
    spo2 = rng.choice([88, 90, 91, 93, 94, 95, 96, 97])
    fr = rng.randint(16, 34)
    febre = rng.random() < 0.6
    dispneia = spo2 < 94 or rng.random() < 0.4
    comorb = rng.sample(_COMORBIDADES, k=rng.randint(1, 2))
    comorb = [c for c in comorb if c != "nenhuma"] or ["nenhuma"]

    if spo2 < 90:
        risco = "vermelho"
    elif spo2 < 95 or dispneia:
        risco = "amarelo"
    else:
        risco = "verde"

    # Exames: alguns pendentes de propósito, para o fluxo de decisão testar isso.
    exames = {
        "hemograma": rng.choice(["concluido", "pendente"]),
        "pcr": rng.choice(["concluido", "pendente"]),
        "gasometria": rng.choice(["concluido", "pendente"]),
        "rx_torax": rng.choice(["concluido", "pendente"]),
    }

    return {
        "paciente_id": f"PAC-{idx:04d}",  # identificador fictício, anonimizado
        "idade": idade,
        "sexo": rng.choice(_SEXOS),
        "spo2": spo2,
        "freq_respiratoria": fr,
        "febre": febre,
        "dispneia": dispneia,
        "comorbidades": comorb,
        "classificacao_risco": risco,
        "exames": exames,
        "internado_em": "leito_enfermaria" if risco != "vermelho" else "aguardando_uti",
    }


# --------------------------------------------------------------------------- #
# Escrita
# --------------------------------------------------------------------------- #
def main(n_pacientes: int = 40, seed: int = 42) -> None:
    rng = random.Random(seed)
    PROTO_DIR.mkdir(parents=True, exist_ok=True)

    for nome, conteudo in PROTOCOLOS.items():
        (PROTO_DIR / nome).write_text(conteudo, encoding="utf-8")

    with open(KB_DIR / "faq_medicos.jsonl", "w", encoding="utf-8") as f:
        for item in FAQ_MEDICOS:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(KB_DIR / "laudos_modelos.jsonl", "w", encoding="utf-8") as f:
        for item in LAUDOS_MODELOS:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    prontuarios = [_gera_prontuario(i + 1, rng) for i in range(n_pacientes)]
    with open(KB_DIR / "prontuarios.json", "w", encoding="utf-8") as f:
        json.dump(prontuarios, f, ensure_ascii=False, indent=2)

    print(f"[ok] {len(PROTOCOLOS)} protocolos em {PROTO_DIR}")
    print(f"[ok] {len(FAQ_MEDICOS)} perguntas em faq_medicos.jsonl")
    print(f"[ok] {len(LAUDOS_MODELOS)} modelos em laudos_modelos.jsonl")
    print(f"[ok] {len(prontuarios)} prontuários sintéticos em prontuarios.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Gera base sintética anonimizada (Fase 3).")
    ap.add_argument("--n-pacientes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(n_pacientes=args.n_pacientes, seed=args.seed)
