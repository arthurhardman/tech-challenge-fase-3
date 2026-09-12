"""Interface visual do assistente médico da Fase 3.

Demonstra, em uma única tela, o que o desafio pede: um assistente treinado com
dados do hospital que auxilia em condutas clínicas, responde dúvidas de médicos
e sugere procedimentos a partir dos protocolos internos — sempre com fonte
rastreável, guardrail de prescrição e trilha de auditoria.

Rodar com:

    streamlit run app/assistente_app.py

O modelo ajustado (adapter LoRA) é carregado uma única vez por sessão via
``@st.cache_resource``. O primeiro carregamento leva ~40 s; as respostas
seguintes levam ~20-30 s em CPU/MPS, porque a geração é feita localmente.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# O Streamlit executa o arquivo diretamente, então o pacote `src` só fica
# importável se a raiz do projeto estiver no sys.path.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Precisa vir antes de importar transformers (conflito Keras 3 / TensorFlow).
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import streamlit as st

from src.assistant.chains.graph import FluxoAtendimento
from src.assistant.chains.medical_assistant import MedicalAssistant
from src.assistant.knowledge.patient_db import PatientDB
from src.assistant.safety.audit_log import AuditLogger

PERGUNTAS_EXEMPLO = [
    "Quais sinais indicam evolução para SRAG?",
    "Quando as amostras respiratórias devem ser coletadas para detecção viral?",
    "Quais critérios de gravidade devem ser observados numa síndrome gripal?",
    "Em que momento aplicar medidas de prevenção e controle?",
]

PERGUNTA_BLOQUEADA = "Qual a dose de dexametasona que devo prescrever?"


st.set_page_config(
    page_title="Assistente Médico SRAG",
    page_icon="🩺",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def carregar_assistente(backend: str) -> MedicalAssistant:
    """Carrega LLM + RAG + PatientDB uma única vez por sessão."""
    os.environ["MEDICAL_LLM_BACKEND"] = backend
    return MedicalAssistant()


@st.cache_resource(show_spinner=False)
def carregar_db() -> PatientDB:
    return PatientDB()


def _badge_risco(risco: str) -> str:
    cores = {"vermelho": "🔴", "amarelo": "🟡", "verde": "🟢"}
    return f"{cores.get(risco, '⚪')} {risco or 'desconhecido'}"


def _mostrar_fontes(fontes: list[str]) -> None:
    """Explainability: de onde veio cada informação da resposta."""
    if not fontes:
        st.caption("Nenhuma fonte recuperada para esta pergunta.")
        return
    st.markdown("**📚 Fontes utilizadas**")
    for fonte in fontes:
        st.markdown(f"- `{fonte}`")


# --------------------------------------------------------------------------- #
# Barra lateral: backend, paciente e estado do modelo
# --------------------------------------------------------------------------- #
st.sidebar.title("🩺 Assistente SRAG")
st.sidebar.caption("Tech Challenge — Fase 3")

adapter_ok = (_PROJECT_ROOT / "results" / "finetuned_model" / "run_info.json").exists()
backend_escolhido = st.sidebar.selectbox(
    "Backend da LLM",
    options=["finetuned", "auto", "local"],
    index=0 if adapter_ok else 1,
    help=(
        "finetuned = modelo base + adapter LoRA treinado nesta entrega. "
        "local = Transformer pequeno de validação. auto = escolhe o disponível."
    ),
)

if adapter_ok:
    st.sidebar.success("Adapter LoRA encontrado")
else:
    st.sidebar.warning(
        "Adapter LoRA ausente — rode `python -m src.finetuning.train_lora`."
    )

db = carregar_db()
st.sidebar.metric("Pacientes na base", f"{db.count():,}".replace(",", "."))

st.sidebar.divider()
st.sidebar.subheader("Paciente em contexto")

usar_paciente = st.sidebar.checkbox("Contextualizar pelo paciente", value=True)
paciente_id = None
if usar_paciente:
    vermelho = db.primeiro_por_risco("vermelho")
    ids = db.todos_ids(limit=40)
    if vermelho and vermelho in ids:
        ids.remove(vermelho)
    if vermelho:
        ids.insert(0, vermelho)
    paciente_id = st.sidebar.selectbox("ID do paciente", options=ids)

st.sidebar.divider()
st.sidebar.caption(
    "Ferramenta de apoio à decisão. Não prescreve e não substitui a avaliação "
    "do médico responsável."
)


# --------------------------------------------------------------------------- #
# Cabeçalho
# --------------------------------------------------------------------------- #
st.title("Assistente Médico SRAG")
st.caption(
    "LLM ajustada por LoRA + RAG sobre protocolos oficiais + base estruturada do "
    "SIVEP, com guardrails e auditoria."
)

aba_chat, aba_paciente, aba_fluxo, aba_auditoria = st.tabs(
    ["💬 Consulta clínica", "🧑‍⚕️ Paciente", "🔀 Fluxo automatizado", "🔒 Segurança e auditoria"]
)


# --------------------------------------------------------------------------- #
# Aba 1 — Consulta clínica (pergunta -> resposta + fontes)
# --------------------------------------------------------------------------- #
with aba_chat:
    st.subheader("Dúvida clínica")

    exemplo = st.selectbox(
        "Perguntas de exemplo", options=["(escrever a minha)"] + PERGUNTAS_EXEMPLO
    )
    valor_inicial = "" if exemplo.startswith("(") else exemplo
    pergunta = st.text_area("Pergunta do médico", value=valor_inicial, height=90)

    if st.button("Consultar protocolos", type="primary", disabled=not pergunta.strip()):
        with st.spinner("Recuperando protocolos e gerando a resposta…"):
            assistente = carregar_assistente(backend_escolhido)
            resposta = assistente.responder(pergunta, paciente_id=paciente_id)

        col_resp, col_fontes = st.columns([3, 2])
        with col_resp:
            if resposta.bloqueado_guardrail:
                st.error("⛔ Bloqueado pelo guardrail de entrada")
                st.caption(
                    "Categorias: " + ", ".join(resposta.categorias_guardrail)
                )
            st.markdown("**Resposta**")
            st.info(resposta.resposta)
            st.caption(f"backend: `{resposta.backend}`")
            if paciente_id:
                st.caption(f"contextualizado pelo paciente `{paciente_id}`")
        with col_fontes:
            _mostrar_fontes(resposta.fontes)


# --------------------------------------------------------------------------- #
# Aba 2 — Paciente (consulta à base estruturada)
# --------------------------------------------------------------------------- #
with aba_paciente:
    if not paciente_id:
        st.info("Ative “Contextualizar pelo paciente” na barra lateral.")
    else:
        registro = db.get(paciente_id) or {}
        risco = registro.get("classificacao_risco", "desconhecido")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Paciente", paciente_id)
        c2.metric("Idade", registro.get("idade", "—"))
        c3.metric("Sexo", registro.get("sexo", "—"))
        c4.metric("Risco", _badge_risco(risco))

        st.markdown("**Resumo clínico enviado à LLM**")
        st.code(db.resumo_clinico(paciente_id), language="text")

        pendentes = db.exames_pendentes(paciente_id)
        st.markdown("**Exames pendentes**")
        if pendentes:
            for exame in pendentes:
                st.warning(f"⏳ {exame}")
        else:
            st.success("Nenhum exame registrado como pendente.")

        with st.expander("Ver todos os campos do registro (anonimizado)"):
            st.json(registro)

        st.caption(
            "Dados reais do SIVEP-Gripe/OpenDataSUS. Nome, CPF, CNS e endereço "
            "não são importados para a base do assistente."
        )


# --------------------------------------------------------------------------- #
# Aba 3 — Fluxo automatizado (LangGraph)
# --------------------------------------------------------------------------- #
with aba_fluxo:
    st.subheader("Fluxo de decisão com LangGraph")
    st.caption("triagem → verificar_exames → (emitir_alerta | sugerir_conduta) → consolidar")

    if not paciente_id:
        st.info("Selecione um paciente na barra lateral para executar o fluxo.")
    else:
        pergunta_fluxo = st.text_input(
            "Pergunta enviada ao fluxo",
            value="Quais critérios de gravidade do protocolo devem ser revisados neste caso?",
        )
        if st.button("Executar fluxo", type="primary"):
            with st.spinner("Executando o grafo de decisão…"):
                assistente = carregar_assistente(backend_escolhido)
                fluxo = FluxoAtendimento(assistant=assistente, patient_db=db)
                estado = fluxo.executar(paciente_id, pergunta_fluxo)

            trilha = estado.get("trilha", [])
            st.markdown("**Trilha percorrida**")
            cols = st.columns(len(trilha) or 1)
            for col, no in zip(cols, trilha):
                destaque = "🚨" if no == "emitir_alerta" else "✅"
                col.success(f"{destaque} {no}")

            if estado.get("alerta"):
                st.error(estado["alerta"])

            st.markdown("**Orientação consolidada**")
            st.info(estado.get("conduta", "(sem resposta)"))
            _mostrar_fontes(estado.get("fontes", []))

            with st.expander("Estado completo do grafo"):
                st.json({k: v for k, v in estado.items() if k != "conduta"})


# --------------------------------------------------------------------------- #
# Aba 4 — Segurança e auditoria
# --------------------------------------------------------------------------- #
with aba_auditoria:
    st.subheader("Guardrail de prescrição")
    st.caption(
        "O assistente localiza critérios nos protocolos, mas nunca devolve dose "
        "ou posologia — requisito destacado no enunciado do desafio."
    )

    st.code(PERGUNTA_BLOQUEADA, language="text")
    if st.button("Testar pedido de prescrição"):
        with st.spinner("Executando…"):
            assistente = carregar_assistente(backend_escolhido)
            r = assistente.responder(PERGUNTA_BLOQUEADA)
        if r.bloqueado_guardrail:
            st.error("⛔ Guardrail de entrada acionado — prescrição direta recusada")
            st.caption("Categorias: " + ", ".join(r.categorias_guardrail))
        st.info(r.resposta)

    st.divider()
    st.subheader("Trilha de auditoria")

    eventos = AuditLogger().ler_eventos()
    if not eventos:
        st.info("Nenhum evento registrado ainda. Faça uma consulta nas outras abas.")
    else:
        bloqueados = sum(1 for e in eventos if e.get("guardrail_bloqueou"))
        c1, c2, c3 = st.columns(3)
        c1.metric("Eventos registrados", len(eventos))
        c2.metric("Bloqueios de guardrail", bloqueados)
        c3.metric("Com fonte citada", sum(1 for e in eventos if e.get("fontes")))

        st.markdown("**Últimos eventos**")
        for evento in reversed(eventos[-8:]):
            marca = "⛔" if evento.get("guardrail_bloqueou") else "✅"
            titulo = f"{marca} {evento.get('timestamp', '')[:19]} — {evento.get('pergunta', '')[:70]}"
            with st.expander(titulo):
                st.json(evento)

        st.caption(
            "Registrado em results/finetuning/audit_events.jsonl e audit.log — "
            "permite reconstruir por que o assistente respondeu o que respondeu."
        )
