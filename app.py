import cv2
import streamlit as st
import numpy as np
import pandas as pd
import torch
import os
import sys

# Configuración de página Streamlit
st.set_page_config(
    page_title="Detección de Objetos en Tiempo Real",
    page_icon="🔍",
    layout="wide"
)

# ========================
# 1. Cambiar fondo y color de texto
# ========================
page_style = """
<style>
    .stApp {
        background-color: #582f0e;
        color: #c9ada7;
    }
    /* Color para títulos y subtítulos */
    h1, h2, h3, h4, h5, h6, p, div, span, label {
        color: #c9ada7 !important;
    }
    /* Centrar cámara */
    .camera-container {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    /* Reducir tamaño del input de cámara */
    div[data-testid="stCameraInput"] video, 
    div[data-testid="stCameraInput"] button {
        max-width: 300px !important;
    }
</style>
"""
st.markdown(page_style, unsafe_allow_html=True)


# Función para cargar el modelo YOLOv5
@st.cache_resource
def load_yolov5_model(model_path='yolov5s.pt'):
    try:
        import yolov5
        try:
            model = yolov5.load(model_path, weights_only=False)
            return model
        except TypeError:
            try:
                model = yolov5.load(model_path)
                return model
            except Exception:
                st.warning("Intentando método alternativo de carga...")
                current_dir = os.path.dirname(os.path.abspath(__file__))
                if current_dir not in sys.path:
                    sys.path.append(current_dir)
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
                return model
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {str(e)}")
        return None

# Título y descripción
st.title("🔍 Detección de Objetos en Imágenes")
st.markdown("Esta aplicación utiliza YOLOv5 para detectar objetos en imágenes capturadas con tu cámara.")

# Cargar modelo
with st.spinner("Cargando modelo YOLOv5..."):
    model = load_yolov5_model()

if model:
    # Sidebar
    st.sidebar.title("Parámetros")
    with st.sidebar:
        st.subheader('Configuración de detección')
        model.conf = st.slider('Confianza mínima', 0.0, 1.0, 0.25, 0.01)
        model.iou = st.slider('Umbral IoU', 0.0, 1.0, 0.45, 0.01)

    # ========================
    # 3. Cámara centrada y más pequeña
    # ========================
    st.markdown('<div class="camera-container">', unsafe_allow_html=True)
    picture = st.camera_input("Capturar imagen", key="camera")
    st.markdown('</div>', unsafe_allow_html=True)

    if picture:
        # Procesar imagen
        bytes_data = picture.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        with st.spinner("Detectando objetos..."):
            results = model(cv2_img)

        predictions = results.pred[0]
        categories = predictions[:, 5] if len(predictions) > 0 else []

        if len(categories) > 0:
            # ========================
            # 2. Mostrar Whoa.jpeg si hay detecciones
            # ========================
            st.image("Whoa.jpeg", caption="¡Whoa! Objeto detectado 🎉", use_container_width=False)

        # Mostrar resultados
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Imagen con detecciones")
            results.render()
            st.image(cv2_img, channels='BGR', use_container_width=True)
        with col2:
            st.subheader("Objetos detectados")
            if len(categories) > 0:
                label_names = model.names
                category_count = {}
                for category in categories:
                    idx = int(category.item()) if hasattr(category, 'item') else int(category)
                    category_count[idx] = category_count.get(idx, 0) + 1

                data = []
                for category, count in category_count.items():
                    label = label_names[category]
                    data.append({"Categoría": label, "Cantidad": count})
                df = pd.DataFrame(data)
                st.dataframe(df, use_container_width=True)
                st.bar_chart(df.set_index('Categoría')['Cantidad'])
            else:
                st.info("No se detectaron objetos con los parámetros actuales.")

else:
    st.error("No se pudo cargar el modelo. Verifica dependencias e inténtalo nuevamente.")
