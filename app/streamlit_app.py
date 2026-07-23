from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model


# =========================
# Configurações principais
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "results" / "models" / "image_cnn_model.keras"

CLASS_NAMES = [
    "COVID",
    "Normal",
    "Lung_Opacity",
    "Viral Pneumonia"
]


# =========================
# Carregamento do modelo
# =========================

@st.cache_resource
def carregar_modelo():
    """
    Carrega o modelo treinado apenas uma vez.
    Isso evita recarregar o modelo a cada interação.
    """
    return load_model(MODEL_PATH)


# =========================
# Pré-processamento da imagem
# =========================

def preparar_imagem(uploaded_file, image_size=(224, 224)):
    """
    Recebe a imagem enviada pelo usuário, converte para RGB,
    redimensiona para o tamanho usado no treino e normaliza os pixels.
    """

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)

    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise ValueError("Não foi possível ler a imagem enviada.")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, image_size)

    img_normalized = img_resized / 255.0
    img_input = np.expand_dims(img_normalized, axis=0)

    return img_rgb, img_input


# =========================
# Interface
# =========================

st.set_page_config(
    page_title="Assistente de Validação por Imagem",
    layout="centered"
)

st.title("Assistente de Validação por Imagem")
st.write(
    """
    Esta interface permite enviar uma radiografia torácica e receber
    uma previsão feita pelo modelo CNN treinado no projeto.

    O resultado é apenas acadêmico e não deve ser usado como diagnóstico médico.
    """
)

if not MODEL_PATH.exists():
    st.error(
        f"Modelo não encontrado em: {MODEL_PATH}. "
        "Treine e salve o modelo antes de usar a interface."
    )
    st.stop()


model = carregar_modelo()

uploaded_file = st.file_uploader(
    "Envie uma imagem de raio-x",
    type=["png", "jpg", "jpeg"]
)


if uploaded_file is not None:
    with st.chat_message("user"):
        st.write("Imagem enviada para análise.")

    try:
        image, image_input = preparar_imagem(uploaded_file)

        st.image(
    image,
    caption="Imagem recebida",
    use_column_width=True
)

        prediction = model.predict(image_input)
        predicted_index = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        predicted_class = CLASS_NAMES[predicted_index]

        with st.chat_message("assistant"):
            st.write("Análise concluída.")
            st.write(f"Classe prevista: **{predicted_class}**")
            st.write(f"Confiança do modelo: **{confidence:.2%}**")

            st.write("Distribuição das probabilidades:")

            for class_name, prob in zip(CLASS_NAMES, prediction[0]):
                st.write(f"- {class_name}: {prob:.2%}")

            st.warning(
                "Atenção: este modelo foi desenvolvido para fins acadêmicos. "
                "Ele não substitui avaliação médica profissional."
            )

    except Exception as erro:
        st.error(f"Erro ao processar a imagem: {erro}")