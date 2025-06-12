import streamlit as st
import numpy as np
from pathlib import Path
import kagglehub
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

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
        self.model_performance = None

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

    def evaluate_models(self, X_test, y_test):
        """Evaluate all models and store performance metrics"""
        X_processed = self.preprocess_input(X_test)
        
        # Get predictions from all models
        models = {
            'EfficientNetB0': (self.efficientnet_model, 0),
            'MobileNetV3Small': (self.mobilenetv3_model, 1),
            'MobileNetV2': (self.mobilenet_model, 2),
            'Stacking Ensemble': (self.meta_learner, None)
        }
        
        performance = {}
        
        for model_name, (model, idx) in models.items():
            if model is None:
                continue
                
            if model_name == 'Stacking Ensemble':
                meta_features, _ = self.generate_meta_features(X_test, y_test)
                y_pred_probs = model.predict(meta_features, verbose=0)
            else:
                y_pred_probs = model.predict(X_processed[idx], verbose=0)
            
            y_pred = np.argmax(y_pred_probs, axis=1)
            y_true = np.argmax(y_test, axis=1) if len(y_test.shape) > 1 else y_test
            
            # Calculate metrics
            report = classification_report(y_true, y_pred, output_dict=True)
            cm = confusion_matrix(y_true, y_pred)
            
            performance[model_name] = {
                'accuracy': report['accuracy'],
                'precision': report['weighted avg']['precision'],
                'recall': report['weighted avg']['recall'],
                'f1_score': report['weighted avg']['f1-score'],
                'confusion_matrix': cm
            }
        
        self.model_performance = performance
        return performance

def load_dataset(data_dir, target_size=(224, 224), max_per_class=1000, test_split=0.2):
    data_dir = Path(data_dir)
    train_dir = data_dir / 'train'
    test_dir = data_dir / 'test'
    classes = ['NORMAL', 'PNEUMONIA']
    images = {cls: {'train': [], 'test': []} for cls in classes}

    st.write("\nLoading dataset...")
    
    # Load training data
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
                images[cls]['train'].append((img_path, img))
                count += 1
                if count % 100 == 0:
                    st.write(f"Loaded {count}/{max_per_class} {cls} training images")
            except Exception as e:
                st.error(f"Error loading {img_path}: {e}")
                continue
    
    # Load test data
    for cls in classes:
        cls_dir = test_dir / cls
        if not cls_dir.exists():
            st.error(f"Directory {cls_dir} not found!")
            continue

        count = 0
        image_files = list(cls_dir.glob('*.jpeg')) + list(cls_dir.glob('*.jpg'))
        np.random.shuffle(image_files)

        for img_path in image_files:
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize(target_size, Image.Resampling.LANCZOS)
                images[cls]['test'].append((img_path, img))
                count += 1
                if count % 50 == 0:
                    st.write(f"Loaded {count} {cls} test images")
            except Exception as e:
                st.error(f"Error loading {img_path}: {e}")
                continue

    st.write("\nDataset Statistics:")
    for cls in classes:
        st.write(f"{cls}:")
        st.write(f"  - Training: {len(images[cls]['train'])} images")
        st.write(f"  - Test: {len(images[cls]['test'])} images")
    
    return images

def prepare_test_data(images):
    """Prepare test data for evaluation"""
    X_test = []
    y_test = []
    
    for cls_idx, cls in enumerate(['NORMAL', 'PNEUMONIA']):
        for img_path, img in images[cls]['test']:
            X_test.append(np.array(img, dtype=np.float32))
            y_test.append(cls_idx)
    
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    return X_test, y_test

def plot_model_performance(performance):
    """Create visualizations of model performance"""
    if not performance:
        st.warning("No performance data available")
        return
    
    st.subheader("Model Performance Comparison")
    
    # Create a DataFrame for metrics
    metrics_df = pd.DataFrame.from_dict(performance, orient='index')
    metrics_df = metrics_df[['accuracy', 'precision', 'recall', 'f1_score']]
    metrics_df = metrics_df.reset_index().rename(columns={'index': 'Model'})
    
    # Melt for easier plotting
    metrics_melted = metrics_df.melt(id_vars='Model', var_name='Metric', value_name='Score')
    
    # Bar plot of all metrics
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=metrics_melted, x='Model', y='Score', hue='Metric', ax=ax)
    ax.set_title("Model Performance Metrics Comparison")
    ax.set_ylim(0, 1.1)
    ax.legend(loc='upper right')
    st.pyplot(fig)
    
    # Confusion matrices
    st.subheader("Confusion Matrices")
    cols = st.columns(len(performance))
    
    for idx, (model_name, metrics) in enumerate(performance.items()):
        cm = metrics['confusion_matrix']
        fig_cm, ax_cm = plt.subplots(figsize=(4, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Normal', 'Pneumonia'],
                    yticklabels=['Normal', 'Pneumonia'],
                    ax=ax_cm)
        ax_cm.set_title(f"{model_name}\nAccuracy: {metrics['accuracy']:.2f}")
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")
        cols[idx].pyplot(fig_cm)
        plt.close(fig_cm)

# Streamlit app
st.title("Pneumonia Detection using Pre-trained Models")
st.markdown("""
This application allows you to:
1. Select an image from the dataset (NORMAL or PNEUMONIA) and predict using different models
2. Compare model performance metrics
3. View confusion matrices for each model
""")

# Sidebar for configuration
st.sidebar.header("Configuration")
data_dir_input = st.sidebar.text_input("Dataset Directory", "chest_xray")
model_dir = st.sidebar.text_input("Model Directory", "saved_models")
max_per_class = st.sidebar.slider("Max Training Images per Class", 100, 2000, 1000, 100)

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
        
        # Prepare test data for evaluation
        X_test, y_test = prepare_test_data(images)
        st.session_state['X_test'] = X_test
        st.session_state['y_test'] = y_test

# Image and model selection
if 'images' in st.session_state:
    st.header("Image Prediction")
    class_choice = st.selectbox("Select Class", ["NORMAL", "PNEUMONIA"])
    image_files = [str(img_path) for img_path, _ in st.session_state['images'][class_choice]['train']]
    selected_image = st.selectbox("Select Image", image_files)
    selected_model = st.selectbox("Select Model", ["EfficientNetB0", "MobileNetV3Small", "MobileNetV2", "Stacking Ensemble"])

    if selected_image:
        # Load and display the selected image
        image_path = Path(selected_image)
        image = [img for path, img in st.session_state['images'][class_choice]['train'] if str(path) == selected_image][0]
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
                        
                        # Show confidence visualization
                        fig, ax = plt.subplots(figsize=(6, 2))
                        ax.barh(['Pneumonia Probability'], [result['probability']], color='skyblue')
                        ax.set_xlim(0, 1)
                        ax.set_title('Model Confidence')
                        st.pyplot(fig)

    # Model evaluation section
    st.header("Model Performance Evaluation")
    if st.button("Evaluate All Models"):
        if 'X_test' not in st.session_state or 'y_test' not in st.session_state:
            st.error("Test data not available for evaluation")
        else:
            with st.spinner("Evaluating models on test set..."):
                model = PneumoniaStackingEnsemble(input_shape=(224, 224, 3), load_from_dir=model_dir)
                if model.efficientnet_model is None:
                    st.error("Failed to load models. Please check the model directory.")
                else:
                    performance = model.evaluate_models(st.session_state['X_test'], st.session_state['y_test'])
                    st.session_state['performance'] = performance
                    plot_model_performance(performance)

    if 'performance' in st.session_state:
        plot_model_performance(st.session_state['performance'])