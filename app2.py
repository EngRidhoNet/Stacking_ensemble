import streamlit as st
import numpy as np
from pathlib import Path
import kagglehub
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
from keras.layers import TFSMLayer

class PneumoniaStackingEnsemble:
    def __init__(self, input_shape=(224, 224, 3), load_from_dir=None):
        self.input_shape = input_shape
        if load_from_dir and os.path.exists(load_from_dir):
            self.load_models(load_from_dir)
        else:
            st.error("Model directory not found. Please ensure models are saved in the specified directory.")
            self.efficientnet_model = None
            self.mobilenetv3_model = None
            self.mobilenet_model = None
            self.meta_learner = None

    def load_models(self, load_dir):
        try:
            # Load models in .keras format
            self.efficientnet_model = load_model(os.path.join(load_dir, "efficientnet_model.keras"))
            self.mobilenetv3_model = load_model(os.path.join(load_dir, "mobilenetv3_model.keras"))
            self.mobilenet_model = load_model(os.path.join(load_dir, "mobilenet_model.keras"))
            self.meta_learner = load_model(os.path.join(load_dir, "meta_learner_model.keras"))
            
            st.success(f"Models loaded successfully from {load_dir}")
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
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

    def generate_meta_features(self, X, y):
        """Generate meta features for stacking ensemble"""
        X_processed = self.preprocess_input(X)
        efficientnet_preds = self.efficientnet_model.predict(X_processed[0], verbose=0)
        mobilenetv3_preds = self.mobilenetv3_model.predict(X_processed[1], verbose=0)
        mobilenet_preds = self.mobilenet_model.predict(X_processed[2], verbose=0)
        meta_features = np.hstack([efficientnet_preds, mobilenetv3_preds, mobilenet_preds])
        return meta_features, y

    def predict_single_image(self, image, selected_model):
        image = image.resize((224, 224), Image.Resampling.LANCZOS)
        X = np.array(image, dtype=np.float32)[np.newaxis, ...]
        X_processed = self.preprocess_input(X)

        if selected_model == "Stacking Ensemble" and self.meta_learner is not None:
            meta_features, _ = self.generate_meta_features(X, np.array([0]))
            probs = self.meta_learner.predict(meta_features, verbose=0)[0]
            result = {
                'class': 'Pneumonia' if np.argmax(probs) == 1 else 'Normal',
                'probability': probs[1]
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
        return result

def load_dataset(data_dir, target_size=(224, 224), max_per_class=1000):
    data_dir = Path(data_dir)
    train_dir = data_dir / 'train'
    classes = ['NORMAL', 'PNEUMONIA']
    images = {cls: [] for cls in classes}

    st.write("\nLoading dataset...")
    for cls in classes:
        cls_dir = train_dir / cls
        if not cls_dir.exists():
            st.error(f"Directory {cls_dir} not found!")
            continue

        count = 0
        image_files = list(cls_dir.glob('*.jpeg')) + list(cls_dir.glob('*.jpg'))
        np.random.shuffle(image_files)

        for img_path in image_files:
            if count >= max_per_class:
                break
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize(target_size, Image.Resampling.LANCZOS)
                images[cls].append((img_path, img))
                count += 1
                if count % 100 == 0:
                    st.write(f"Loaded {count}/{max_per_class} {cls} images")
            except Exception as e:
                st.error(f"Error loading {img_path}: {e}")
                continue

    st.write("\nDataset Statistics:")
    for cls in classes:
        st.write(f"{cls}: {len(images[cls])} images")
    return images

# Streamlit app
st.title("Pneumonia Detection using Pre-trained Models")
st.markdown("This application allows you to select an image from the dataset (NORMAL or PNEUMONIA) and predict using a pre-trained model from the saved models directory.")

# Sidebar for configuration
st.sidebar.header("Configuration")
data_dir_input = st.sidebar.text_input("Dataset Directory", "chest_xray")
model_dir = st.sidebar.text_input("Model Directory", "saved_models")
max_per_class = st.sidebar.slider("Max Images per Class", 100, 2000, 1000, 100)

# Load dataset
if st.button("Load Dataset"):
    with st.spinner("Downloading and loading dataset..."):
        try:
            dataset_path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
            data_dir = Path(dataset_path) / "chest_xray"
            if not data_dir.exists():
                for item in Path(dataset_path).iterdir():
                    if item.is_dir() and 'chest' in item.name.lower():
                        data_dir = item
                        break
        except Exception as e:
            st.error(f"Error downloading dataset: {e}")
            st.write("Using local dataset path instead...")
            data_dir = Path(data_dir_input)

        images = load_dataset(data_dir, target_size=(224, 224), max_per_class=max_per_class)
        st.session_state['images'] = images

# Image and model selection
if 'images' in st.session_state:
    st.header("Select Image and Model")
    class_choice = st.selectbox("Select Class", ["NORMAL", "PNEUMONIA"])
    image_files = [str(img_path) for img_path, _ in st.session_state['images'][class_choice]]
    selected_image = st.selectbox("Select Image", image_files)
    selected_model = st.selectbox("Select Model", ["EfficientNetB0", "MobileNetV3Small", "MobileNetV2", "Stacking Ensemble"])

    if selected_image:
        # Load and display the selected image
        image_path = Path(selected_image)
        image = [img for path, img in st.session_state['images'][class_choice] if str(path) == selected_image][0]
        st.image(image, caption=f"Selected Image: {image_path.name}", use_column_width=True)

        # Load models and predict
        if st.button("Predict"):
            with st.spinner("Loading models and making prediction..."):
                model = PneumoniaStackingEnsemble(input_shape=(224, 224, 3), load_from_dir=model_dir)
                if model.efficientnet_model is None:
                    st.error("Failed to load models. Please check the model directory.")
                else:
                    result = model.predict_single_image(image, selected_model)
                    if result:
                        st.subheader("Prediction Result")
                        st.write(f"**Model**: {selected_model}")
                        st.write(f"**Class**: {result['class']}")
                        st.write(f"**Probability of Pneumonia**: {result['probability']:.4f}")