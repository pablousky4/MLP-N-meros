# MLP-N-meros
https://github.com/pablousky4/MLP-N-meros
https://mlp-n-meros-deuhxb3ynbiwnm8rqhcgqc.streamlit.app/

# 🔢 Clasificador de Dígitos con IA + Firebase

Este proyecto entrena un modelo de red neuronal (MLP) con **MNIST** para reconocer números escritos a mano.  
Incluye una **app en Streamlit** que permite subir imágenes, hacer predicciones y guardar los resultados en **Firebase Firestore**.

## 🚀 Estructura
- `train_model.ipynb` → Entrena el modelo y lo guarda como `model.h5`.
- `app.ipynb` → App en Streamlit para predecir y guardar registros.
- `requirements.txt` → Librerías necesarias.
- `model.h5` → Modelo entrenado.
- `firebase_key.json` → Clave privada de Firebase (no subir al repo público).

## ▶️ Cómo ejecutar
1. Instala dependencias:
   ```bash
   pip install -r requirements.txt
