#!/usr/bin/env python3
"""
PNEUMOVISION AI - FIXED VERSION
Perbaikan:
- Tambah deteksi versi TensorFlow & Keras
- Helper load_model_safely mencoba tf.keras dan keras (Keras 3)
- Logging error model loading lebih jelas (tidak dipotong)
- Fallback ke demo model kalau benar-benar tidak bisa load
"""

import os
import sys

# ==================== CRITICAL: SET ENV VARS FIRST ====================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# MacOS specific fixes
if sys.platform == 'darwin':
    os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    os.environ['NO_MACOS_FORK_SAFETY'] = '1'

# ==================== IMPORT STREAMLIT ====================
import streamlit as st

# Set page config
st.set_page_config(
    page_title="PneumoVision AI - Pneumonia Detection",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    .main {
        background-color: #0f172a;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
        color: white;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: white !important;
    }
    
    p, span, div {
        color: #e2e8f0 !important;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%) !important;
        color: white !important;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.6);
    }
    
    .card {
        background: rgba(30, 41, 59, 0.8) !important;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 1rem;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ==================== CHECK IF TENSORFLOW / KERAS IS AVAILABLE ====================
TENSORFLOW_AVAILABLE = False
KERAS_AVAILABLE = False

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
    TF_VERSION = tf.__version__ if hasattr(tf, '__version__') else "Unknown"
except Exception as e:
    TF_VERSION = f"Error: {repr(e)}"

try:
    import keras
    KERAS_AVAILABLE = True
    KERAS_VERSION = keras.__version__ if hasattr(keras, '__version__') else "Unknown"
except Exception as e:
    KERAS_VERSION = f"Error: {repr(e)}"

# ==================== IMPORT OTHER LIBRARIES ====================
import numpy as np
from PIL import Image
import time
import random

# ==================== DEMO MODE FUNCTIONS ====================
def create_demo_model(model_name="efficientnet"):
    """Create a simple demo model (dummy probabilistic model)."""
    class DemoModel:
        def __init__(self, name):
            self.name = name
            self.input_shape = (224, 224, 3)
            
        def predict(self, X, verbose=0):
            if len(X.shape) == 4:
                brightness = np.mean(X)
                # Heuristik demo: gambar lebih gelap -> lebih "pneumonia"
                pneumonia_prob = max(0.1, min(0.9, (0.7 - brightness) * 2))
                pneumonia_prob += random.uniform(-0.2, 0.2)
                pneumonia_prob = np.clip(pneumonia_prob, 0.1, 0.9)
                normal_prob = 1 - pneumonia_prob
                return np.array([[normal_prob, pneumonia_prob]])
            else:
                return np.array([[0.5, 0.5]])
    
    return DemoModel(model_name)

def load_demo_models():
    """Load or create demo models."""
    models = {
        'efficientnet': create_demo_model('efficientnet'),
        'mobilenet': create_demo_model('mobilenet'),
        'meta_learner': create_demo_model('meta_learner')
    }
    return models

def preprocess_image_demo(image: Image.Image):
    """Preprocess image for demo / generic CNN: resize 224x224, scale 0-1."""
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

# ==================== SAFE MODEL LOADER ====================
def load_model_safely(model_path: str, model_name: str):
    """
    Coba load model dari file .keras / .h5 dengan beberapa strategi:
    1) tf.keras.models.load_model
    2) keras.models.load_model (kalau Keras 3 tersedia)
    
    Kalau gagal semua -> return None (nanti digantikan demo model).
    """
    errors = []

    if not os.path.exists(model_path):
        st.warning(f"⚠️ File model untuk {model_name} tidak ditemukan: {model_path}")
        return None

    # 1. Coba via tf.keras
    if TENSORFLOW_AVAILABLE:
        try:
            from tensorflow.keras.models import load_model as tf_load_model
            model = tf_load_model(model_path, compile=False)
            return model
        except Exception as e:
            errors.append(f"tf.keras load_model error: {repr(e)}")

    # 2. Coba via keras (Keras 3)
    if KERAS_AVAILABLE:
        try:
            model = keras.models.load_model(model_path, compile=False)
            return model
        except Exception as e:
            errors.append(f"keras.load_model error: {repr(e)}")

    # Kalau dua-duanya gagal, tampilkan error lengkap supaya kamu bisa diagnosa
    st.error(
        f"⛔️ Tidak bisa memuat model '{model_name}' dari '{model_path}'. "
        f"Detail error:\n\n" + "\n\n".join(errors)
    )
    return None

# ==================== MAIN APPLICATION ====================
def main():
    # Header
    st.markdown("""
    <div style="background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%); 
                padding: 2rem; border-radius: 15px; margin-bottom: 2rem;">
        <h1 style="color: white; text-align: center; margin: 0;">
            🫁 PneumoVision AI
        </h1>
        <p style="color: #e2e8f0; text-align: center; margin: 0.5rem 0 0 0;">
            Pneumonia Detection from Chest X-Ray Images
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <h3 style="color: #e2e8f0;">⚙️ System Info</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Show TensorFlow / Keras status
        if TENSORFLOW_AVAILABLE:
            st.success(f"✅ TensorFlow: {TF_VERSION}")
        else:
            st.warning("⚠️ TensorFlow tidak tersedia (Demo Mode)")
        
        if KERAS_AVAILABLE:
            st.info(f"ℹ️ Keras: {KERAS_VERSION}")
        else:
            st.info("ℹ️ Keras (standalone) tidak terpasang")
        
        st.markdown("---")
        
        # Model directory
        model_dir = st.text_input(
            "Direktori Model",
            value="saved_models",
            help="Path ke file model (.keras / .h5)"
        )
        
        if os.path.exists(model_dir):
            st.success("✅ Direktori ditemukan")
            files = os.listdir(model_dir)
            if files:
                st.info(f"📁 {len(files)} file di direktori model")
            else:
                st.warning("📂 Direktori kosong")
        else:
            st.warning("⚠️ Direktori tidak ditemukan")
        
        st.markdown("---")
        
        # Instructions
        st.markdown("""
        <div class="card">
            <h4 style="margin: 0; color: white;">📋 Cara Menggunakan</h4>
            <ol style="margin: 1rem 0 0 0; padding-left: 1.2rem;">
                <li>Upload citra X-ray dada</li>
                <li>Pilih model</li>
                <li>Klik <strong>Analyze Image</strong></li>
                <li>Lihat hasil prediksi</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="card">
            <h3 style="margin: 0; color: white;">📤 Upload X-Ray Image</h3>
            <p style="margin: 0.5rem 0 0 0;">
                Upload citra X-ray dada (JPG, JPEG, atau PNG)
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose file",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )
    
    with col2:
        st.markdown("""
        <div class="card" style="background: linear-gradient(135deg, #10b981 0%, #059669 100%);">
            <h4 style="margin: 0; color: white;">⚡ Quick Start</h4>
            <p style="margin: 1rem 0 0 0;">
                Analisis AI instan untuk citra X-ray dada (riset & edukasi).
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Process uploaded file
    if uploaded_file is not None:
        try:
            # Load image
            image = Image.open(uploaded_file).convert("RGB")
            
            st.markdown("---")
            
            # Two columns layout
            img_col, analysis_col = st.columns([1, 1])
            
            with img_col:
                st.markdown("""
                <div class="card">
                    <h4 style="margin: 0 0 1rem 0; color: white;">🖼️ Uploaded Image</h4>
                </div>
                """, unsafe_allow_html=True)
                
                st.image(image, use_column_width=True)
                
                # Image info
                st.markdown(f"""
                <div class="card">
                    <strong>Image Information</strong><br>
                    Size: {image.size[0]} × {image.size[1]} pixels<br>
                    Format: {uploaded_file.type}
                </div>
                """, unsafe_allow_html=True)
            
            with analysis_col:
                st.markdown("""
                <div class="card">
                    <h4 style="margin: 0 0 1rem 0; color: white;">🤖 AI Analysis</h4>
                </div>
                """, unsafe_allow_html=True)
                
                # Load models
                with st.spinner("Menyiapkan model AI..."):
                    if TENSORFLOW_AVAILABLE:
                        try:
                            models = {}
                            model_files = [
                                ('efficientnet', 'efficientnet_model.keras'),
                                ('mobilenet', 'mobilenet_model.keras'),
                                ('mobilenetv3', 'mobilenetv3_model.keras'),
                                ('meta_learner', 'meta_learner_model.keras')
                            ]
                            
                            for model_name, filename in model_files:
                                model_path = os.path.join(model_dir, filename)
                                model = load_model_safely(model_path, model_name)
                                
                                if model is not None:
                                    st.success(f"✅ {model_name} dimuat dari {filename}")
                                    models[model_name] = model
                                else:
                                    st.warning(f"⚠️ {model_name} tidak tersedia, gunakan demo model")
                                    models[model_name] = create_demo_model(model_name)
                            
                        except Exception as e:
                            st.error(f"❌ Error saat load model: {repr(e)}")
                            st.info("Berpindah ke demo mode.")
                            models = load_demo_models()
                    else:
                        st.info("TensorFlow tidak tersedia. Menggunakan demo mode.")
                        models = load_demo_models()
                
                # Model selection
                available_models = list(models.keys())
                selected_model = st.selectbox(
                    "Pilih Model",
                    available_models,
                    format_func=lambda x: x.capitalize()
                )
                
                # Analysis button
                if st.button("🚀 Analyze Image", type="primary", use_container_width=True):
                    with st.spinner("Menganalisis citra..."):
                        # Preprocess image
                        img_array = preprocess_image_demo(image)
                        
                        # Get model and predict
                        model = models[selected_model]
                        
                        try:
                            if hasattr(model, 'predict'):
                                prediction = model.predict(img_array, verbose=0)[0]
                            else:
                                # Fallback demo
                                brightness = np.mean(img_array)
                                pneumonia_prob = max(0.1, min(0.9, (0.7 - brightness) * 2))
                                pneumonia_prob += random.uniform(-0.2, 0.2)
                                pneumonia_prob = np.clip(pneumonia_prob, 0.1, 0.9)
                                normal_prob = 1 - pneumonia_prob
                                prediction = np.array([normal_prob, pneumonia_prob])
                            
                            # Extract probabilities
                            if len(prediction) == 2:
                                normal_prob = float(prediction[0])
                                pneumonia_prob = float(prediction[1])
                            else:
                                pneumonia_prob = float(prediction[-1]) if len(prediction) > 1 else 0.5
                                normal_prob = 1 - pneumonia_prob
                            
                            # Clamp & normalize
                            normal_prob = max(0.0, min(1.0, normal_prob))
                            pneumonia_prob = max(0.0, min(1.0, pneumonia_prob))
                            
                            total = normal_prob + pneumonia_prob
                            if total > 0:
                                normal_prob /= total
                                pneumonia_prob /= total
                            
                            # Determine result
                            is_pneumonia = pneumonia_prob > 0.5
                            confidence = pneumonia_prob if is_pneumonia else normal_prob
                            
                            # Display results
                            st.markdown("---")
                            st.markdown("""
                            <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); 
                                        padding: 1.5rem; border-radius: 12px; margin-bottom: 1rem;">
                                <h3 style="margin: 0; color: white;">📊 Analysis Results</h3>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Metrics in columns
                            col_a, col_b, col_c = st.columns(3)
                            
                            with col_a:
                                st.metric("Model Used", selected_model.capitalize())
                            
                            with col_b:
                                st.metric("Prediction", "PNEUMONIA" if is_pneumonia else "NORMAL")
                            
                            with col_c:
                                st.metric("Confidence", f"{confidence*100:.1f}%")
                            
                            # Visualization
                            st.markdown("---")
                            st.markdown("#### Probability Distribution")
                            
                            import matplotlib.pyplot as plt
                            
                            fig, ax = plt.subplots(figsize=(8, 4))
                            
                            categories = ['Normal', 'Pneumonia']
                            probabilities = [normal_prob, pneumonia_prob]
                            colors = ['#10b981', '#ef4444']
                            
                            bars = ax.bar(categories, probabilities, color=colors)
                            ax.set_ylim(0, 1.1)
                            ax.set_ylabel('Probability')
                            
                            for bar, prob in zip(bars, probabilities):
                                height = bar.get_height()
                                ax.text(
                                    bar.get_x() + bar.get_width() / 2.0,
                                    height + 0.02,
                                    f'{prob:.1%}',
                                    ha='center',
                                    fontweight='bold'
                                )
                            
                            st.pyplot(fig)
                            
                            # Result message
                            st.markdown("---")
                            
                            if is_pneumonia:
                                st.error("""
                                ⚠️ **PNEUMONIA DETECTED**
                                
                                **Disclaimer:** Ini hanya demonstrasi.
                                Silakan konsultasi dengan tenaga medis profesional untuk diagnosis yang akurat.
                                """)
                            else:
                                st.success("""
                                ✅ **NORMAL CHEST X-RAY**
                                
                                Tidak terdeteksi indikasi pneumonia pada citra ini (berdasarkan model).
                                """)
                            
                        except Exception as e:
                            st.error(f"❌ Error during analysis: {repr(e)}")
                            
                            # Fallback ke random demo
                            pneumonia_prob = random.uniform(0.1, 0.9)
                            is_pneumonia = pneumonia_prob > 0.5
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Pneumonia Probability", f"{pneumonia_prob*100:.1f}%")
                            with col2:
                                st.metric("Prediction", "PNEUMONIA" if is_pneumonia else "NORMAL")
                
        except Exception as e:
            st.error(f"❌ Error processing image: {repr(e)}")
    
    else:
        # Welcome screen
        st.markdown("---")
        
        st.markdown("""
        <div style="text-align: center; padding: 3rem;">
            <h2 style="color: #e2e8f0;">👆 Upload an X-Ray Image to Begin</h2>
            <p style="color: #94a3b8; font-size: 1.1rem;">
                AI akan menganalisis citra X-ray dada untuk mendeteksi indikasi pneumonia
            </p>
            
            <div style="display: flex; justify-content: center; gap: 3rem; margin-top: 2rem; flex-wrap: wrap;">
                <div style="text-align: center;">
                    <div style="font-size: 3rem;">🫁</div>
                    <div style="font-weight: bold; margin-top: 0.5rem;">Pneumonia Detection</div>
                </div>
                
                <div style="text-align: center;">
                    <div style="font-size: 3rem;">⚡</div>
                    <div style="font-weight: bold; margin-top: 0.5rem;">Fast Analysis</div>
                </div>
                
                <div style="text-align: center;">
                    <div style="font-size: 3rem;">🔬</div>
                    <div style="font-weight: bold; margin-top: 0.5rem;">AI-Powered</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #64748b; padding: 2rem;">
        <p>
            <strong>🫁 PneumoVision AI</strong> | Medical AI Research | For Educational Purposes
        </p>
        <p style="font-size: 0.9rem; margin-top: 0.5rem;">
            Selalu konsultasikan dengan tenaga kesehatan profesional untuk keputusan klinis
        </p>
    </div>
    """, unsafe_allow_html=True)

# ==================== RUN APPLICATION ====================
if __name__ == "__main__":
    main()
