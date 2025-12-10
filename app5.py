import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import time
import logging
import os
import sys

# Set environment variables BEFORE importing TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Import TensorFlow/Keras
import tensorflow as tf
import keras

# Suppress TensorFlow logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('tensorflow').disabled = True

try:
    tf.get_logger().setLevel('ERROR')
except:
    pass

# Disable GPU and configure threading
try:
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
except:
    pass

import streamlit as st


# Set page config FIRST
st.set_page_config(
    page_title="PneumoVision AI - Deteksi Pneumonia",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Dark theme
st.markdown("""
<style>
    .main {
        background-color: #1a1a2e;
        padding: 0rem 1rem;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    
    .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span {
        color: #e8e8ff !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3a5f 0%, #0f1624 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: #e8e8ff !important;
    }
</style>
""", unsafe_allow_html=True)


# ==================== SAFE MODEL LOADER ====================
def load_model_safely(filepath: str, model_name: str):
    """
    Coba load model dengan urutan:
    1) tf.keras.models.load_model
    2) keras.models.load_model

    Return:
        model atau None kalau semua gagal.
    """
    errors = []

    if not os.path.exists(filepath):
        st.warning(f"⚠️ File untuk {model_name} tidak ditemukan: {filepath}")
        return None

    # 1. Coba lewat tf.keras (lebih cocok untuk model yang disimpan dari tf.keras)
    try:
        from tensorflow.keras.models import load_model as tf_load_model
        model = tf_load_model(filepath, compile=False)
        return model
    except Exception as e:
        errors.append(f"tf.keras load_model error: {repr(e)}")

    # 2. Coba lewat keras (standalone Keras 3, kalau file memang dibuat dengan itu)
    try:
        model = keras.models.load_model(filepath, compile=False)
        return model
    except Exception as e:
        errors.append(f"keras.load_model error: {repr(e)}")

    # Kalau dua-duanya gagal, tampilkan detail lengkap supaya bisa dianalisis
    st.error(
        f"⛔️ Tidak bisa memuat model '{model_name}' dari '{filepath}'.\n\n"
        + "\n\n".join(errors)
    )
    return None


# ==================== MODEL LOADING ====================
@st.cache_resource(show_spinner=False)
def load_models_simple(model_dir):
    """Load models with proper error handling"""
    
    if not os.path.exists(model_dir):
        return None, f"Directory '{model_dir}' tidak ditemukan!"
    
    models = {}
    loaded_count = 0
    
    model_files_config = {
        'efficientnet': [
            'efficientnet_model.keras',
            'efficientnet_model.h5',
            'efficientnet.keras',
            'efficientnet.h5'
        ],
        'mobilenetv3': [
            'mobilenetv3_model.keras',
            'mobilenetv3_model.h5',
            'mobilenetv3.keras',
            'mobilenetv3.h5'
        ],
        'mobilenet': [
            'mobilenet_model.keras',
            'mobilenet_model.h5',
            'mobilenetv2_model.keras',
            'mobilenetv2_model.h5'
        ],
        'meta_learner': [
            'meta_learner_model.keras',
            'meta_learner_model.h5',
            'meta_learner.keras',
            'meta_learner.h5'
        ]
    }
    
    for model_name, file_list in model_files_config.items():
        model_loaded = False
        
        for filename in file_list:
            filepath = os.path.join(model_dir, filename)
            
            if os.path.exists(filepath):
                with st.spinner(f"🔄 Loading {model_name} dari {filename}..."):
                    model = load_model_safely(filepath, model_name)
                    if model is not None:
                        models[model_name] = model
                        loaded_count += 1
                        model_loaded = True
                        st.success(f"✅ {model_name} berhasil dimuat dari {filename}")
                        break
                    else:
                        # Gagal di filepath ini, coba nama file berikutnya
                        continue
        
        if not model_loaded:
            st.warning(f"⚠️ {model_name} tidak tersedia (tidak ada file yang berhasil dimuat)")
    
    if loaded_count == 0:
        st.error("❌ Tidak ada model yang bisa dimuat!")
        st.warning("🚨 Creating emergency fallback model...")
        emergency_model = create_emergency_model()
        models['emergency'] = emergency_model
        return models, "Menggunakan model emergency"
    elif loaded_count < 4:
        st.warning(f"⚠️ Hanya {loaded_count}/4 model yang berhasil dimuat")
        return models, f"Hanya {loaded_count} model dimuat"
    else:
        st.success(f"🎉 Semua {loaded_count} model berhasil dimuat!")
        return models, None


def create_emergency_model():
    """Create emergency fallback model"""
    model = keras.Sequential([
        keras.layers.Input(shape=(224, 224, 3)),
        keras.layers.Rescaling(1./255),
        keras.layers.Conv2D(32, 3, activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(64, 3, activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(2, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    return model


# ==================== ENSEMBLE CLASS ====================
class PneumoniaStackingEnsemble:
    def __init__(self, input_shape=(224, 224, 3), load_from_dir=None):
        self.input_shape = input_shape
        self.efficientnet_model = None
        self.mobilenetv3_model = None
        self.mobilenet_model = None
        self.meta_learner = None
        self.last_prediction_details = None
        self.available_models = []
        self.model_info = {}
        
        if load_from_dir:
            self.load_models_with_fallback(load_from_dir)
    
    def load_models_with_fallback(self, load_dir):
        """Load models with fallback"""
        st.info("🔄 Memulai proses loading model...")
        
        models, error_message = load_models_simple(load_dir)
        
        if not models:
            st.error("❌ Gagal total: Tidak bisa memuat atau membuat model")
            return False
        
        if 'efficientnet' in models:
            self.efficientnet_model = models['efficientnet']
            self.available_models.append('EfficientNetB0')
            self.model_info['EfficientNetB0'] = {
                'params': f"{self.efficientnet_model.count_params():,}",
                'type': 'EfficientNetB0',
                'status': '✅ Loaded'
            }
        
        if 'mobilenetv3' in models:
            self.mobilenetv3_model = models['mobilenetv3']
            self.available_models.append('MobileNetV3Small')
            self.model_info['MobileNetV3Small'] = {
                'params': f"{self.mobilenetv3_model.count_params():,}",
                'type': 'MobileNetV3Small',
                'status': '✅ Loaded'
            }
        
        if 'mobilenet' in models:
            self.mobilenet_model = models['mobilenet']
            self.available_models.append('MobileNetV2')
            self.model_info['MobileNetV2'] = {
                'params': f"{self.mobilenet_model.count_params():,}",
                'type': 'MobileNetV2',
                'status': '✅ Loaded'
            }
        
        if 'meta_learner' in models:
            self.meta_learner = models['meta_learner']
            self.model_info['MetaLearner'] = {
                'params': f"{self.meta_learner.count_params():,}",
                'type': 'Meta Learner',
                'status': '✅ Loaded'
            }
        
        if 'emergency' in models:
            self.efficientnet_model = models['emergency']
            self.available_models.append('Emergency CNN')
            self.model_info['Emergency CNN'] = {
                'params': f"{models['emergency'].count_params():,}",
                'type': 'Emergency Model',
                'status': '⚠️ Fallback'
            }
            st.warning("⚠️ Menggunakan model emergency (akurasi terbatas)")
        
        if all([self.efficientnet_model, self.mobilenetv3_model, 
               self.mobilenet_model, self.meta_learner]):
            self.available_models.append('Stacking Ensemble')
            self.model_info['Stacking Ensemble'] = {
                'params': 'Combined',
                'type': 'Stacking Ensemble',
                'status': '✅ Available'
            }
        
        st.success(f"✅ {len(self.available_models)} model siap digunakan")
        return True
    
    def preprocess_input(self, X):
        """Preprocess input for each model"""
        X = X.astype('float32')
        
        X_efficientnet = X.copy()
        X_mobilenetv3 = X.copy()
        X_mobilenet = X.copy()
        
        if self.efficientnet_model is not None:
            try:
                X_efficientnet = keras.applications.efficientnet.preprocess_input(X_efficientnet)
            except:
                X_efficientnet = X_efficientnet / 255.0
        
        if self.mobilenetv3_model is not None:
            try:
                X_mobilenetv3 = keras.applications.mobilenet_v3.preprocess_input(X_mobilenetv3)
            except:
                X_mobilenetv3 = X_mobilenetv3 / 255.0
        
        if self.mobilenet_model is not None:
            try:
                X_mobilenet = keras.applications.mobilenet_v2.preprocess_input(X_mobilenet)
            except:
                X_mobilenet = X_mobilenet / 255.0
        
        return [X_efficientnet, X_mobilenetv3, X_mobilenet]
    
    def predict_single_image(self, image, selected_model):
        """Predict with selected model"""
        try:
            image = image.resize((224, 224), Image.Resampling.LANCZOS)
            X = np.array(image, dtype=np.float32)[np.newaxis, ...]
            
            X_processed = self.preprocess_input(X)
            
            prediction_details = {
                'model_used': selected_model,
                'base_model_predictions': {},
                'final_prediction': None,
                'preprocessed_image': image,
                'preprocessed_array': X
            }
            
            if selected_model == "Stacking Ensemble":
                if not all([self.efficientnet_model, self.mobilenetv3_model, 
                           self.mobilenet_model, self.meta_learner]):
                    st.error("❌ Stacking Ensemble requires all models!")
                    return None
                
                efficientnet_pred = self.efficientnet_model.predict(X_processed[0], verbose=0)[0]
                mobilenetv3_pred = self.mobilenetv3_model.predict(X_processed[1], verbose=0)[0]
                mobilenet_pred = self.mobilenet_model.predict(X_processed[2], verbose=0)[0]
                
                prediction_details['base_model_predictions'] = {
                    'EfficientNetB0': efficientnet_pred.tolist(),
                    'MobileNetV3Small': mobilenetv3_pred.tolist(),
                    'MobileNetV2': mobilenet_pred.tolist()
                }
                
                meta_features = np.hstack([
                    efficientnet_pred.reshape(1, -1),
                    mobilenetv3_pred.reshape(1, -1),
                    mobilenet_pred.reshape(1, -1)
                ])
                
                probs = self.meta_learner.predict(meta_features, verbose=0)[0]
                result = {
                    'class': 'Pneumonia' if np.argmax(probs) == 1 else 'Normal',
                    'probability': float(probs[1])
                }
                
                prediction_details['final_prediction'] = {
                    'probabilities': probs.tolist(),
                    'class': result['class'],
                    'confidence': float(result['probability'])
                }
                
            else:
                models_map = {
                    'EfficientNetB0': (self.efficientnet_model, 0),
                    'MobileNetV3Small': (self.mobilenetv3_model, 1),
                    'MobileNetV2': (self.mobilenet_model, 2),
                    'Emergency CNN': (self.efficientnet_model, 0)
                }
                
                if selected_model not in models_map:
                    st.error(f"❌ Model {selected_model} tidak tersedia!")
                    return None
                
                model, idx = models_map[selected_model]
                if model is None:
                    st.error(f"❌ Model {selected_model} belum dimuat!")
                    return None
                
                probs = model.predict(X_processed[idx], verbose=0)[0]
                result = {
                    'class': 'Pneumonia' if np.argmax(probs) == 1 else 'Normal',
                    'probability': float(probs[1])
                }
                
                prediction_details['final_prediction'] = {
                    'probabilities': probs.tolist(),
                    'class': result['class'],
                    'confidence': float(result['probability'])
                }
            
            self.last_prediction_details = prediction_details
            return result
            
        except Exception as e:
            st.error(f"❌ Error selama prediksi: {str(e)}")
            return None


# ==================== VISUALIZATION ====================
def draw_cnn_architecture(layer_configs, current_step=-1, title="CNN Architecture"):
    """Draw CNN architecture"""
    fig, ax = plt.subplots(figsize=(18, 9))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    ax.set_xlim(-0.5, 21)
    ax.set_ylim(-1, 11)
    ax.axis('off')
    
    x_start = 1
    y_center = 5
    
    colors = {
        'input': '#4facfe',
        'conv': '#667eea',
        'pool': '#764ba2',
        'dense': '#f093fb',
        'output': '#11998e',
        'active': '#ff6b6b',
        'inactive': '#4a4a6a'
    }
    
    ax.text(10, 10, title, ha='center', va='center', fontsize=16, 
           fontweight='bold', color='#ffffff')
    
    for idx, config in enumerate(layer_configs):
        layer_type = config['type']
        width = config.get('width', 0.8)
        height = config.get('height', 3)
        label = config.get('label', '')
        
        if current_step >= 0 and idx <= current_step:
            color = colors['active']
            alpha = 0.95
            edge_width = 3
        else:
            color = colors.get(layer_type, '#4a4a6a')
            alpha = 0.7 if current_step < 0 else 0.4
            edge_width = 2
        
        front = FancyBboxPatch(
            (x_start, y_center - height/2), 
            width, height,
            boxstyle="round,pad=0.05",
            linewidth=edge_width,
            edgecolor='#ffffff' if current_step >= 0 and idx <= current_step else '#888',
            facecolor=color,
            alpha=alpha
        )
        ax.add_patch(front)
        
        ax.text(x_start + width/2, y_center, label, 
               ha='center', va='center', fontsize=9, fontweight='bold',
               color='white')
        
        if 'shape' in config:
            ax.text(x_start + width/2, y_center - height/2 - 0.4, 
                   config['shape'], ha='center', va='top', 
                   fontsize=7, color='#b8c5ff')
        
        if idx < len(layer_configs) - 1:
            arrow_color = '#ff6b6b' if current_step >= 0 and idx < current_step else '#888'
            arrow = FancyArrowPatch(
                (x_start + width + 0.15, y_center),
                (x_start + width + 0.65, y_center),
                arrowstyle='-|>', mutation_scale=18, 
                linewidth=2.5, color=arrow_color
            )
            ax.add_patch(arrow)
        
        x_start += width + 0.8
    
    plt.tight_layout()
    return fig


def animate_prediction_process(model, image, selected_model):
    """Animate prediction process"""
    
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 1.5rem; border-radius: 12px; margin-bottom: 1.5rem;'>
        <h2 style='color: white; margin: 0;'>🔬 Neural Network Visualization</h2>
    </div>
    """, unsafe_allow_html=True)
    
    progress_bar = st.progress(0)
    arch_placeholder = st.empty()
    
    architecture = [
        {'type': 'input', 'width': 1.3, 'height': 3.8, 'label': 'Input\nImage', 'shape': '224×224×3'},
        {'type': 'conv', 'width': 1.1, 'height': 3.5, 'label': 'Conv1\n64', 'shape': '112×112×64'},
        {'type': 'pool', 'width': 0.9, 'height': 3.0, 'label': 'Pool', 'shape': '56×56×64'},
        {'type': 'conv', 'width': 1.1, 'height': 2.7, 'label': 'Conv2\n128', 'shape': '28×28×128'},
        {'type': 'pool', 'width': 0.9, 'height': 2.4, 'label': 'Pool', 'shape': '14×14×128'},
        {'type': 'conv', 'width': 1.1, 'height': 2.2, 'label': 'Conv3\n256', 'shape': '7×7×256'},
        {'type': 'dense', 'width': 0.7, 'height': 3.2, 'label': 'Dense\n512', 'shape': '512'},
        {'type': 'dense', 'width': 0.7, 'height': 2.7, 'label': 'Dense\n256', 'shape': '256'},
        {'type': 'output', 'width': 0.6, 'height': 2.2, 'label': 'Output', 'shape': '2'},
    ]
    
    for i in range(len(architecture)):
        progress = int((i + 1) / len(architecture) * 100)
        progress_bar.progress(progress)
        
        fig = draw_cnn_architecture(architecture, current_step=i, title=selected_model)
        arch_placeholder.pyplot(fig)
        plt.close(fig)
        
        time.sleep(0.5)
    
    progress_bar.progress(100)
    visualize_prediction_process(model.last_prediction_details)


def visualize_prediction_process(prediction_details):
    """Visualize prediction results"""
    
    st.markdown("---")
    st.markdown("""
    <div style='background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
                padding: 1.5rem; border-radius: 12px; margin-bottom: 1.5rem;'>
        <h2 style='color: white; margin: 0;'>📊 Prediction Results</h2>
    </div>
    """, unsafe_allow_html=True)
    
    final_pred = prediction_details['final_prediction']
    is_pneumonia = final_pred['class'] == 'Pneumonia'
    
    if prediction_details['model_used'] == "Stacking Ensemble":
        st.markdown("#### 🔀 Stacking Ensemble Results")
        
        col1, col2, col3 = st.columns(3)
        base_preds = prediction_details['base_model_predictions']
        
        for col, (model_name, preds) in zip([col1, col2, col3], base_preds.items()):
            with col:
                st.markdown(f"**{model_name}**")
                st.write(f"Normal: {preds[0]:.2%}")
                st.write(f"Pneumonia: {preds[1]:.2%}")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    
    colors = ['#11998e', '#ff6b6b']
    bars = ax.bar(['Normal', 'Pneumonia'], final_pred['probabilities'], color=colors)
    ax.set_ylim(0, 1.1)
    ax.set_title("Final Prediction", fontsize=14, color='white', fontweight='bold')
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    for bar, v in zip(bars, final_pred['probabilities']):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
               f'{v:.2%}', ha='center', fontsize=11, color='white', fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    
    st.markdown("---")
    
    if is_pneumonia:
        st.error(f"""
        ### ⚠️ PNEUMONIA DETECTED
        **Confidence:** {final_pred['confidence']:.1%}
        
        Segera konsultasikan dengan dokter!
        """)
    else:
        st.success(f"""
        ### ✅ NORMAL
        **Confidence:** {final_pred['confidence']:.1%}
        
        Tidak terdeteksi tanda pneumonia.
        """)


# ==================== MAIN ====================
def main():
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 2.5rem; border-radius: 15px; margin-bottom: 2rem;'>
        <h1 style='color: white; text-align: center; margin: 0;'>
            🫁 PneumoVision AI
        </h1>
        <p style='color: #e8e8ff; text-align: center; margin: 0.8rem 0 0 0;'>
            Aplikasi Deteksi Pneumonia
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        # Info versi buat ngecek kecocokan environment
        st.info(f"TensorFlow: {tf.__version__}")
        try:
            st.info(f"Keras: {keras.__version__}")
        except:
            pass
        
        model_dir = st.text_input(
            "📁 Model Directory",
            "saved_models"
        )
        
        if st.button("🔍 Check Models"):
            if os.path.exists(model_dir):
                files = os.listdir(model_dir)
                st.success(f"✅ Found {len(files)} files")
                for f in files:
                    st.info(f"📄 {f}")
            else:
                st.error(f"❌ Directory not found: {model_dir}")
    
    uploaded_file = st.file_uploader(
        "Upload X-Ray Image",
        type=["jpg", "jpeg", "png"]
    )
    
    if uploaded_file:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption="Uploaded Image", width=350)
            
            with col2:
                with st.spinner("🤖 Loading models..."):
                    model = PneumoniaStackingEnsemble(
                        input_shape=(224, 224, 3),
                        load_from_dir=model_dir
                    )
                    
                    if model.available_models:
                        selected_model = st.selectbox(
                            "Select Model:",
                            model.available_models
                        )
                        
                        if st.button("🚀 Analyze", type="primary"):
                            result = model.predict_single_image(image, selected_model)
                            
                            if result:
                                st.success(f"**Prediction:** {result['class']}")
                                st.info(f"**Confidence:** {result['probability']:.1%}")
                                
                                animate_prediction_process(model, image, selected_model)
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
    
    else:
        st.info("👆 Upload an X-Ray image to start")
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #888;'>
        <p>⚠️ For educational purposes only</p>
        <p>🫁 PneumoVision AI © 2025</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
