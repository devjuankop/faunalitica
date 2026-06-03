import io
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict/image")


st.set_page_config(
    page_title="Predicción de Especies (API)",
    page_icon="🧠",
    layout="wide",
)

# --- Header and Introduction ---
st.markdown(
    """
    # 🌿 Predicción de Especies Silvestres 🐾
    ¡Bienvenido al clasificador de imágenes de especies!
    Sube una fotografía de un animal o planta y nuestra inteligencia artificial te dirá
    a qué especie podría pertenecer, utilizando un modelo de inferencia servido por una API.
    """
)

# --- Sidebar for Configuration ---
with st.sidebar:
    st.header("⚙️ Configuración del Servicio")
    st.markdown(
        """
        Ajusta los parámetros para la comunicación con la API de inferencia.
        """
    )
    api_url_input = st.text_input(
        "🔗 URL del Servicio de Predicción (API)", value=API_URL
    )
    top_k = st.slider(
        "🔢 Número de Top-K Predicciones",
        min_value=1,
        max_value=10,
        value=3,
        help="Muestra las N predicciones con mayor probabilidad."
    )

# --- Main Content Layout ---
col_upload, col_results = st.columns([1, 1])

with col_upload:
    st.subheader("📤 Sube tu Imagen Aquí")
    uploaded_file = st.file_uploader(
        "Arrastra y suelta tu imagen o haz click para buscar",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=False
    )
    if uploaded_file is None:
        st.info("Esperando que subas una imagen para clasificar...")
    else:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Imagen Cargada", use_container_width=True)

with col_results:
    st.subheader("📊 Resultados de la Predicción")
    if uploaded_file is not None:
        if st.button("🚀 Realizar Predicción", type="primary"):
            with st.spinner("✨ Enviando imagen a la API y obteniendo predicción..."):
                try:
                    # Convert image to bytes
                    img_byte_arr = io.BytesIO()
                    image.save(img_byte_arr, format=image.format or "PNG")
                    img_byte_arr = img_byte_arr.getvalue()

                    files = {"file": (uploaded_file.name, img_byte_arr, uploaded_file.type)}
                    response = requests.post(
                        api_url_input, files=files, params={"top_k": top_k}
                    )
                    response.raise_for_status()  # Raise an exception for HTTP errors
                    result = response.json()

                    st.success("✅ Predicción Completada")
                    
                    # Display top prediction prominently
                    st.metric(
                        label="Especie Predicha (más probable)",
                        value=result["predicted_class"],
                        delta=f"{result['predictions'][0]['probability']:.2%}" # Display probability as delta for visual emphasis
                    )

                    # Detailed predictions in an expander
                    with st.expander("🔍 Ver Todas las Predicciones"):
                        predictions_df = pd.DataFrame(result["predictions"])
                        predictions_df["probability"] = predictions_df["probability"].round(6)
                        predictions_df.rename(columns={
                            "class_name": "Nombre de Clase",
                            "probability": "Probabilidad",
                            "class_index": "Índice de Clase"
                        }, inplace=True)
                        st.dataframe(predictions_df, use_container_width=True)
                        st.bar_chart(predictions_df.set_index("Nombre de Clase")["Probabilidad"])

                except requests.exceptions.RequestException as e:
                    st.error(f"❌ Error al conectar con la API de predicción: {e}")
                    st.exception(e) # Show full traceback for debugging
                except json.JSONDecodeError:
                    st.error("⚠️ Error: La API devolvió una respuesta JSON inválida.")
                    st.text(f"Respuesta de la API: {response.text}")
                except Exception as exc:
                    st.error(f"🚫 No se pudo realizar la predicción debido a un error interno: {exc}")
                    st.exception(exc) # Show full traceback for debugging
        else:
            st.info("Presiona 'Realizar Predicción' para obtener los resultados.")
    else:
        st.info("Primero sube una imagen en la columna izquierda.")

