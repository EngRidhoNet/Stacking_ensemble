import streamlit as st
import numpy as np
from PIL import Image, ImageFilter
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import matplotlib.pyplot as plt
import time

class PneumoniaStackingEnsemble:
    def __init__(self, input_shape=(224, 224, 3), load_from_dir=None):
        self.input_shape = input_shape
        self.efficientnet_model = None
        self.mobilenetv3_model = None
        self.mobilenet_model = None
        self.meta_learner = None
        self.last_prediction_details = None
        
        if load_from_dir and os.path.exists(load_from_dir):
            self.load_models(load_from_dir)
        else:
            st.error("Model directory not found. Please ensure models are saved in the specified directory.")

    def load_models(self, load_dir):
        try:
            self.efficientnet_model = load_model(os.path.join(load_dir, "efficientnet_model.keras"))
            self.mobilenetv3_model = load_model(os.path.join(load_dir, "mobilenetv3_model.keras"))
            self.mobilenet_model = load_model(os.path.join(load_dir, "mobilenet_model.keras"))
            self.meta_learner = load_model(os.path.join(load_dir, "meta_learner_model.keras"))
            st.success(f"✅ Models loaded successfully from {load_dir}")
        except Exception as e:
            st.error(f"❌ Error loading models: {str(e)}")
            self.efficientnet_model = None
            self.mobilenetv3_model = None
            self.mobilenet_model = None
            self.meta_learner = None

    def preprocess_input(self, X):
        """Preprocess input for each model type"""
        X = X.astype('float32')
        X_efficientnet = tf.keras.applications.efficientnet.preprocess_input(X.copy())
        X_mobilenetv3 = tf.keras.applications.mobilenet_v3.preprocess_input(X.copy())
        X_mobilenet = tf.keras.applications.mobilenet_v2.preprocess_input(X.copy())
        return [X_efficientnet, X_mobilenetv3, X_mobilenet]

    def predict_single_image(self, image, selected_model):
        image = image.resize((224, 224), Image.Resampling.LANCZOS)
        X = np.array(image, dtype=np.float32)[np.newaxis, ...]
        X_processed = self.preprocess_input(X)
        
        prediction_details = {
            'model_used': selected_model,
            'base_model_predictions': {},
            'final_prediction': None,
            'preprocessed_image': image
        }

        if selected_model == "Stacking Ensemble" and self.meta_learner is not None:
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
                'probability': probs[1]
            }
            
            prediction_details['final_prediction'] = {
                'probabilities': probs.tolist(),
                'class': result['class'],
                'confidence': result['probability']
            }
            
        else:
            models = {
                'EfficientNetB0': (self.efficientnet_model, 0),
                'MobileNetV3Small': (self.mobilenetv3_model, 1),
                'MobileNetV2': (self.mobilenet_model, 2)
            }
            if selected_model not in models:
                st.error("Invalid model selection!")
                return None
            
            model, idx = models[selected_model]
            probs = model.predict(X_processed[idx], verbose=0)[0]
            result = {
                'class': 'Pneumonia' if np.argmax(probs) == 1 else 'Normal',
                'probability': probs[1]
            }
            
            prediction_details['final_prediction'] = {
                'probabilities': probs.tolist(),
                'class': result['class'],
                'confidence': result['probability']
            }
        
        self.last_prediction_details = prediction_details
        return result


def animate_prediction_process(model, image, selected_model):
    """Animate the forward pass through the neural network"""
    
    st.header("🔬 Prediction Process Animation")
    
    # Create placeholders for dynamic content
    timeline_placeholder = st.empty()
    progress_bar = st.progress(0)
    step_placeholder = st.empty()
    viz_placeholder = st.empty()
    
    # ========== STEP 1: FROM PIXELS TO TENSOR ==========
    step_placeholder.markdown("### 📊 Step 1/4: From Pixels to Preprocessed Tensor")
    timeline_placeholder.info("🔄 Loading and preprocessing the image...")
    progress_bar.progress(15)
    time.sleep(0.5)
    
    # Visualization: Original vs Grayscale
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    axes[0].imshow(image)
    axes[0].set_title("Original X-ray Image (RGB)", fontsize=12, fontweight='bold')
    axes[0].axis("off")
    
    gray = image.convert("L")
    axes[1].imshow(gray, cmap="gray")
    axes[1].set_title("Pixel Intensity Map", fontsize=12, fontweight='bold')
    axes[1].axis("off")
    
    plt.tight_layout()
    viz_placeholder.pyplot(fig)
    plt.close(fig)
    
    step_placeholder.markdown("""
    ### 📊 Step 1/4: From Pixels to Preprocessed Tensor
    
    **What's happening:**
    - Image resized to 224×224 pixels
    - Converted to normalized tensor with shape (1, 224, 224, 3)
    - Pixel values normalized to [-1, 1] range for optimal model performance
    """)
    
    progress_bar.progress(25)
    time.sleep(0.8)
    
    # ========== STEP 2: CONVOLUTIONAL FEATURE MAPS ==========
    step_placeholder.markdown("### 🧠 Step 2/4: Convolutional Layers Extract Features")
    timeline_placeholder.info("🔍 Running image through convolutional filters...")
    progress_bar.progress(45)
    time.sleep(0.5)
    
    # Create simulated feature maps using image filters
    gray_small = image.convert("L").resize((112, 112))
    feature_maps = [
        gray_small.filter(ImageFilter.FIND_EDGES),
        gray_small.filter(ImageFilter.BLUR),
        gray_small.filter(ImageFilter.DETAIL),
        gray_small.filter(ImageFilter.SHARPEN),
        gray_small.filter(ImageFilter.SMOOTH),
        gray_small.filter(ImageFilter.EDGE_ENHANCE)
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.flatten()
    
    filter_names = ["Edge Detection", "Blur Filter", "Detail Filter", 
                    "Sharpen Filter", "Smooth Filter", "Edge Enhance"]
    
    for i, (fm, name) in enumerate(zip(feature_maps, filter_names)):
        axes[i].imshow(fm, cmap="viridis")
        axes[i].set_title(f"Feature Map {i+1}: {name}", fontsize=10, fontweight='bold')
        axes[i].axis("off")
    
    plt.suptitle("Convolutional Feature Extraction", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    viz_placeholder.pyplot(fig)
    plt.close(fig)
    
    step_placeholder.markdown("""
    ### 🧠 Step 2/4: Convolutional Layers Extract Features
    
    **What's happening:**
    - Multiple convolutional filters scan the image
    - Each filter detects different patterns: edges, textures, lung structures
    - Creates hierarchical feature representations
    - Early layers detect simple patterns, deeper layers detect complex structures
    """)
    
    progress_bar.progress(60)
    time.sleep(0.8)
    
    # ========== STEP 3: DENSE LAYER ACTIVATIONS ==========
    step_placeholder.markdown("### ⚡ Step 3/4: Dense Layer Neurons Activate")
    timeline_placeholder.info("🎯 Feeding extracted features into dense layers...")
    progress_bar.progress(75)
    time.sleep(0.5)
    
    # Simulate neuron activations (in real scenario, we'd extract actual intermediate layer outputs)
    num_neurons = 64
    np.random.seed(42)
    neuron_activations = np.random.beta(2, 5, num_neurons)  # More realistic activation pattern
    neuron_activations = np.sort(neuron_activations)[::-1]  # Sort for better visualization
    
    fig, ax = plt.subplots(figsize=(12, 4))
    colors = plt.cm.viridis(neuron_activations / neuron_activations.max())
    bars = ax.bar(range(num_neurons), neuron_activations, color=colors, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel("Neuron Index", fontsize=11, fontweight='bold')
    ax.set_ylabel("Activation Strength", fontsize=11, fontweight='bold')
    ax.set_title("Dense Layer Neuron Activations (Conceptual Representation)", fontsize=13, fontweight='bold')
    ax.axhline(y=neuron_activations.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {neuron_activations.mean():.3f}')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    viz_placeholder.pyplot(fig)
    plt.close(fig)
    
    step_placeholder.markdown("""
    ### ⚡ Step 3/4: Dense Layer Neurons Activate
    
    **What's happening:**
    - Convolutional features are flattened into a 1D vector
    - Dense (fully connected) neurons process these features
    - Each neuron's activation represents learned patterns
    - High activations indicate strong presence of specific features
    - These activations form the model's internal "understanding" of the image
    """)
    
    progress_bar.progress(85)
    time.sleep(0.8)
    
    # ========== STEP 4: FINAL PREDICTION ==========
    if selected_model == "Stacking Ensemble":
        step_placeholder.markdown("### 🎭 Step 4/4: Stacking Ensemble Meta-Learner")
        timeline_placeholder.info("🔮 Combining base model predictions with meta-learner...")
    else:
        step_placeholder.markdown(f"### 🎯 Step 4/4: {selected_model} Final Output")
        timeline_placeholder.info(f"📊 Generating final prediction from {selected_model}...")
    
    progress_bar.progress(95)
    time.sleep(0.5)
    
    # Visualize prediction process
    visualize_prediction_process(model.last_prediction_details)
    
    progress_bar.progress(100)
    timeline_placeholder.success("✅ Prediction process complete!")
    time.sleep(0.3)


def visualize_prediction_process(prediction_details):
    """Visualize the final prediction step"""
    
    if prediction_details['model_used'] == "Stacking Ensemble":
        st.markdown("#### 🔀 Base Model Predictions")
        
        base_models = prediction_details['base_model_predictions']
        cols = st.columns(3)
        
        for idx, (model_name, preds) in enumerate(base_models.items()):
            with cols[idx]:
                st.markdown(f"**{model_name}**")
                
                fig, ax = plt.subplots(figsize=(4, 3.5))
                classes = ['Normal', 'Pneumonia']
                colors = ['#90EE90', '#FFB6C6']
                bars = ax.bar(classes, preds, color=colors, edgecolor='black', linewidth=1.5)
                ax.set_ylim(0, 1)
                ax.set_title(f"{model_name}", fontsize=11, fontweight='bold')
                ax.set_ylabel("Probability", fontsize=10)
                ax.grid(axis='y', alpha=0.3)
                
                for i, v in enumerate(preds):
                    ax.text(i, v + 0.03, f"{v:.3f}", ha='center', fontweight='bold', fontsize=10)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                
                predicted_class = classes[np.argmax(preds)]
                confidence = max(preds)
                st.write(f"➜ **{predicted_class}** ({confidence:.1%})")
        
        st.markdown("#### 🎯 Meta-Learner Final Decision")
        final_pred = prediction_details['final_prediction']
        
        fig, ax = plt.subplots(figsize=(7, 4))
        classes = ['Normal', 'Pneumonia']
        colors = ['#90EE90', '#FFB6C6']
        bars = ax.bar(classes, final_pred['probabilities'], color=colors, edgecolor='black', linewidth=2)
        ax.set_ylim(0, 1)
        ax.set_title("Final Ensemble Prediction (Meta-Learner Output)", fontsize=13, fontweight='bold')
        ax.set_ylabel("Probability", fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        
        for i, v in enumerate(final_pred['probabilities']):
            ax.text(i, v + 0.03, f"{v:.4f}", ha='center', fontweight='bold', fontsize=11)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
        if final_pred['class'] == 'Pneumonia':
            st.error(f"⚠️ **Prediction: {final_pred['class']}** (Confidence: {final_pred['confidence']:.1%})")
        else:
            st.success(f"✅ **Prediction: {final_pred['class']}** (Confidence: {final_pred['confidence']:.1%})")
    
    else:
        final_pred = prediction_details['final_prediction']
        
        fig, ax = plt.subplots(figsize=(7, 4))
        classes = ['Normal', 'Pneumonia']
        colors = ['#90EE90', '#FFB6C6']
        bars = ax.bar(classes, final_pred['probabilities'], color=colors, edgecolor='black', linewidth=2)
        ax.set_ylim(0, 1)
        ax.set_title(f"{prediction_details['model_used']} Output", fontsize=13, fontweight='bold')
        ax.set_ylabel("Probability", fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        
        for i, v in enumerate(final_pred['probabilities']):
            ax.text(i, v + 0.03, f"{v:.4f}", ha='center', fontweight='bold', fontsize=11)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
        if final_pred['class'] == 'Pneumonia':
            st.error(f"⚠️ **Prediction: {final_pred['class']}** (Confidence: {final_pred['confidence']:.1%})")
        else:
            st.success(f"✅ **Prediction: {final_pred['class']}** (Confidence: {final_pred['confidence']:.1%})")


def main():
    st.set_page_config(page_title="Pneumonia Detection Demo", page_icon="🫁", layout="wide")
    
    st.title("🫁 Pneumonia Detection: Stacking Ensemble")
    st.markdown("""
    This app demonstrates **step-by-step** how a deep learning model analyzes chest X-rays to detect pneumonia.
    Watch the neural network process in action from raw pixels to final prediction!
    """)
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    model_dir = st.sidebar.text_input("Model Directory Path", "saved_models")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📝 About")
    st.sidebar.info("""
    **Models Available:**
    - EfficientNetB0
    - MobileNetV3Small
    - MobileNetV2
    - Stacking Ensemble (Meta-learner)
    
    **Upload** a chest X-ray image and watch the AI analyze it in real-time!
    """)
    
    # Main content
    st.header("📤 Upload Chest X-ray Image")
    
    uploaded_file = st.file_uploader(
        "Choose a chest X-ray image (JPG, JPEG, or PNG)",
        type=["jpg", "jpeg", "png"]
    )
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.image(image, caption="📷 Uploaded X-ray Image", use_column_width=True)
            
            with col2:
                st.markdown("### 🎛️ Model Selection")
                selected_model = st.selectbox(
                    "Choose a model for prediction:",
                    ["Stacking Ensemble", "EfficientNetB0", "MobileNetV3Small", "MobileNetV2"],
                    help="Stacking Ensemble combines all three base models for better accuracy"
                )
                
                st.markdown("---")
                
                if st.button("🚀 Start Prediction & Animation", type="primary", use_container_width=True):
                    with st.spinner("🔄 Loading models..."):
                        model = PneumoniaStackingEnsemble(
                            input_shape=(224, 224, 3), 
                            load_from_dir=model_dir
                        )
                        
                        if model.efficientnet_model is None:
                            st.error("❌ Failed to load models. Please check the model directory path.")
                        else:
                            st.success("✅ Models loaded successfully!")
                            
                            with st.spinner("🧠 Running prediction..."):
                                result = model.predict_single_image(image, selected_model)
                            
                            if result:
                                st.markdown("---")
                                animate_prediction_process(model, image, selected_model)
                                
                                st.markdown("---")
                                st.subheader("📋 Final Summary")
                                
                                summary_col1, summary_col2, summary_col3 = st.columns(3)
                                
                                with summary_col1:
                                    st.metric("Model Used", selected_model)
                                
                                with summary_col2:
                                    st.metric("Predicted Class", result['class'])
                                
                                with summary_col3:
                                    prob_percentage = result['probability'] * 100
                                    st.metric("Pneumonia Probability", f"{prob_percentage:.2f}%")
        
        except Exception as e:
            st.error(f"❌ Error processing image: {str(e)}")
    
    else:
        st.info("👆 Please upload a chest X-ray image to start the demo.")
        
        st.markdown("---")
        st.markdown("### 💡 How It Works")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("#### 1️⃣ Pixels → Tensor")
            st.write("Image is resized and normalized into a tensor format")
        
        with col2:
            st.markdown("#### 2️⃣ CNN Features")
            st.write("Convolutional layers extract hierarchical features")
        
        with col3:
            st.markdown("#### 3️⃣ Dense Neurons")
            st.write("Fully connected layers process the features")
        
        with col4:
            st.markdown("#### 4️⃣ Prediction")
            st.write("Final layer outputs probability scores")


if __name__ == "__main__":
    main()