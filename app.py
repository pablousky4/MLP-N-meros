
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------
# Cargar modelo entrenado
# -----------------------------
# Asegúrate de haber guardado tu modelo con:
# model.save("mlp_mnist.h5")
model = tf.keras.models.load_model("model.h5")

st.title("🔢 Clasificador de Números MNIST")
st.write("Sube una imagen de un número escrito a mano (0-9) y la IA intentará reconocerlo.")

# -----------------------------
# Subida de imagen
# -----------------------------
uploaded_file = st.file_uploader("📤 Sube una imagen", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Mostrar la imagen original
    image = Image.open(uploaded_file).convert("L")  # Escala de grises
    st.image(image, caption="Imagen subida", width=150)

    # -----------------------------
    # Preprocesamiento
    # -----------------------------
    img_resized = image.resize((28, 28))  # Redimensionar a 28x28
    img_array = np.array(img_resized)

    # Invertir colores si está en blanco sobre negro
    if np.mean(img_array) > 127:
        img_array = 255 - img_array

    img_array = img_array / 255.0          # Normalizar a [0,1]
    img_array = img_array.reshape(1, 28, 28)

    # -----------------------------
    # Predicción
    # -----------------------------
    pred = model.predict(img_array)
    predicted_digit = np.argmax(pred)

    st.subheader(f"✅ Predicción: **{predicted_digit}**")
