
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import datetime
import base64
from pymongo import MongoClient

# -----------------------------
# Cargar variables de entorno
# -----------------------------
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

# -----------------------------
# Conectar con MongoDB Atlas
# -----------------------------
client = MongoClient(MONGO_URI)
db = client["mlpdb"]
collection = db["predicciones"]

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
    # Abrir imagen en escala de grises
    image = Image.open(uploaded_file).convert("L")

    # Ajustar al tamaño esperado por MNIST (28x28)
    image_resized = image.resize((28, 28))

    # Convertir a array numpy
    img_array = np.array(image_resized)

    # Invertir colores (MNIST son dígitos blancos sobre fondo negro)
    img_array = 255 - img_array  

    # Normalizar a [0,1]
    img_array = img_array.astype("float32") / 255.0

    # Aplanar 28x28 → 784 (porque tu modelo es MLP)
    img_array = img_array.reshape(1, 28 * 28)

    # Predicción
    prediction = model.predict(img_array)
    predicted_label = np.argmax(prediction, axis=1)[0]

    # Mostrar resultados
    st.image(image_resized, caption=f"Predicción: {predicted_label}", width=150)
    st.success(f"✅ La IA predice que es un **{predicted_label}**")

    buffered = io.BytesIO()
    image_resized.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    doc = {
        "fecha": datetime.datetime.now(),
        "prediccion": int(predicted_label),
        "imagen": img_base64
    }
    collection.insert_one(doc)

# -----------------------------
# Mostrar historial
# -----------------------------
st.subheader("📊 Historial de predicciones")

docs = collection.find().sort("fecha", -1).limit(10)

for doc in docs:
    img_data = base64.b64decode(doc["imagen"])
    img = Image.open(io.BytesIO(img_data))
    st.image(img, width=60)
    st.write(f"🕒 {doc['fecha']} → Predicción: **{doc['prediccion']}**")