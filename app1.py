import streamlit as st
import numpy as np
from pathlib import Path
import kagglehub
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0, MobileNetV3Small, MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.mixed_precision import set_global_policy, Policy
import io
import base64

# Import the PneumoniaStackingEnsemble class (assuming it's in the same directory or module)
# For simplicity, we'll include the class directly in this file to avoid import issues
class PneumoniaStackingEnsemble:
    def __init__(self, input_shape=(224, 224, 3)):
        np.random.seed(42)
        self.input_shape = input_shape
        self.efficientnet_model = self.build_individual_model('efficientnet')
        self.mobilenetv3_model = self.build_individual_model('mobilenetv3small')
        self.mobilenet_model = self.build_individual_model('mobilenet')
        self.meta_learner = None
        print("\nIndividual Models Initialized")

    def build_individual_model(self, model_name):
        inputs = tf.keras.Input(shape=self.input_shape, name=f'input_{model_name}')
        if model_name == 'efficientnet':
            base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=self.input_shape)
        elif model_name == 'mobilenetv3small':
            base_model = MobileNetV3Small(weights='imagenet', include_top=False, input_shape=self.input_shape, alpha=0.75)
        elif model_name == 'mobilenet':
            base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=self.input_shape, alpha=0.75)
        
        for i, layer in enumerate(base_model.layers):
            layer._name = f'{model_name}_{layer.name}_{i}'
        
        x = base_model(inputs)
        x = GlobalAveragePooling2D(name=f'gap_{model_name}')(x)
        x = Dense(128, activation='relu', name=f'dense_128_{model_name}')(x)
        x = BatchNormalization(name=f'bn_128_{model_name}')(x)
        x = Dropout(0.3, name=f'dropout_128_{model_name}')(x)
        outputs = Dense(2, activation='softmax', dtype='float32', name=f'output_{model_name}')(x)
        
        model = Model(inputs=inputs, outputs=outputs, name=f'model_{model_name}')
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def build_meta_learner(self):
        inputs = Input(shape=(6,))
        x = Dense(32, activation='relu')(inputs)
        x = Dropout(0.3)(x)
        x = Dense(16, activation='relu')(x)
        outputs = Dense(2, activation='softmax', dtype='float32')(x)
        meta_learner = Model(inputs=inputs, outputs=outputs, name='meta_learner')
        meta_learner.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return meta_learner

    def preprocess_input(self, X):
        X = X.astype('float32')
        X_efficientnet = tf.keras.applications.efficientnet.preprocess_input(X.copy())
        X_mobilenetv3 = tf.keras.applications.mobilenet_v3.preprocess_input(X.copy())
        X_mobilenet = tf.keras.applications.mobilenet_v2.preprocess_input(X.copy())
        return [X_efficientnet, X_mobilenetv3, X_mobilenet]

    def get_data_augmentation(self):
        return ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )

    def train_base_models(self, X_train, y_train, X_val, y_val, epochs=30, batch_size=64):
        X_train_processed = self.preprocess_input(X_train)
        X_val_processed = self.preprocess_input(X_val)
        histories = {'EfficientNetB0': None, 'MobileNetV3Small': None, 'MobileNetV2': None}

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True
        )
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6
        )

        datagen = self.get_data_augmentation()

        print("Training EfficientNetB0...")
        efficientnet_history = self.efficientnet_model.fit(
            datagen.flow(X_train_processed[0], y_train, batch_size=batch_size),
            validation_data=(X_val_processed[0], y_val),
            epochs=epochs,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        histories['EfficientNetB0'] = efficientnet_history.history

        print("Training MobileNetV3Small...")
        mobilenetv3_history = self.mobilenetv3_model.fit(
            datagen.flow(X_train_processed[1], y_train, batch_size=batch_size),
            validation_data=(X_val_processed[1], y_val),
            epochs=epochs,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        histories['MobileNetV3Small'] = mobilenetv3_history.history

        print("Training MobileNetV2...")
        mobilenet_history = self.mobilenet_model.fit(
            datagen.flow(X_train_processed[2], y_train, batch_size=batch_size),
            validation_data=(X_val_processed[2], y_val),
            epochs=epochs,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        histories['MobileNetV2'] = mobilenet_history.history

        return histories

    def generate_meta_features(self, X, y):
        X_processed = self.preprocess_input(X)
        efficientnet_preds = self.efficientnet_model.predict(X_processed[0], verbose=0)
        mobilenetv3_preds = self.mobilenetv3_model.predict(X_processed[1], verbose=0)
        mobilenet_preds = self.mobilenet_model.predict(X_processed[2], verbose=0)
        meta_features = np.hstack([efficientnet_preds, mobilenetv3_preds, mobilenet_preds])
        return meta_features, y

    def train_meta_learner(self, X_train, y_train, X_val, y_val, epochs=20, batch_size=64):
        print("\nTraining Meta-Learner...")
        meta_train_features, meta_train_labels = self.generate_meta_features(X_train, y_train)
        meta_val_features, meta_val_labels = self.generate_meta_features(X_val, y_val)
        self.meta_learner = self.build_meta_learner()
        
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True
        )
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6
        )

        history = self.meta_learner.fit(
            meta_train_features, meta_train_labels,
            validation_data=(meta_val_features, meta_val_labels),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        return {'MetaLearner': history.history}

    def evaluate(self, X_test, y_test):
        X_test_processed = self.preprocess_input(X_test)
        models = {
            'EfficientNetB0': (self.efficientnet_model, 0),
            'MobileNetV3Small': (self.mobilenetv3_model, 1),
            'MobileNetV2': (self.mobilenet_model, 2)
        }
        results = {}

        for name, (model, idx) in models.items():
            print(f"\nEvaluating {name}...")
            y_pred = np.argmax(model.predict(X_test_processed[idx], verbose=0), axis=1)
            y_pred_proba = model.predict(X_test_processed[idx], verbose=0)[:, 1]
            results[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred),
                'auc_roc': roc_auc_score(y_test, y_pred_proba),
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }

        print("\nEvaluating Stacking Ensemble...")
        meta_features, _ = self.generate_meta_features(X_test, y_test)
        y_pred = np.argmax(self.meta_learner.predict(meta_features, verbose=0), axis=1)
        y_pred_proba = self.meta_learner.predict(meta_features, verbose=0)[:, 1]
        results['StackingEnsemble'] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'auc_roc': roc_auc_score(y_test, y_pred_proba),
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        return results

    def predict_single_image(self, image):
        image = image.resize((224, 224), Image.Resampling.LANCZOS)
        X = np.array(image, dtype=np.float32)[np.newaxis, ...]
        X_processed = self.preprocess_input(X)
        
        results = {}
        models = {
            'EfficientNetB0': (self.efficientnet_model, 0),
            'MobileNetV3Small': (self.mobilenetv3_model, 1),
            'MobileNetV2': (self.mobilenet_model, 2)
        }
        for name, (model, idx) in models.items():
            probs = model.predict(X_processed[idx], verbose=0)[0]
            results[name] = {
                'class': 'Pneumonia' if np.argmax(probs) == 1 else 'Normal',
                'probability': probs[1]
            }
        
        meta_features, _ = self.generate_meta_features(X, np.array([0]))
        probs = self.meta_learner.predict(meta_features, verbose=0)[0]
        results['StackingEnsemble'] = {
            'class': 'Pneumonia' if np.argmax(probs) == 1 else 'Normal',
            'probability': probs[1]
        }
        return results

def load_dataset(data_dir, target_size=(224, 224), max_per_class=1000, split_ratio=0.8):
    data_dir = Path(data_dir)
    train_dir = data_dir / 'train'
    classes = ['NORMAL', 'PNEUMONIA']
    X, y = [], []

    st.write("\nLoading dataset...")
    for cls_idx, cls in enumerate(classes):
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
                img_array = np.array(img, dtype=np.float32)
                X.append(img_array)
                y.append(cls_idx)
                count += 1
                if count % 100 == 0:
                    st.write(f"Loaded {count}/{max_per_class} {cls} images")
            except Exception as e:
                st.error(f"Error loading {img_path}: {e}")
                continue

    X = np.array(X)
    y = np.array(y)

    st.write("\nDataset Statistics:")
    st.write(f"Total images: {len(X)}")
    st.write(f"Normal: {np.sum(y == 0)}, Pneumonia: {np.sum(y == 1)}")
    st.write(f"Image shape: {X.shape[1:]}")
    st.write(f"Pixel range: [{X.min():.3f}, {X.max():.3f}]")

    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]
    n_train = int(len(X) * split_ratio)
    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]
    return X_train, y_train, X_val, y_val

def plot_results(histories, evaluation_results, y_val):
    fig = plt.figure(figsize=(18, 12))

    plt.subplot(2, 2, 1)
    for name, history in histories.items():
        plt.plot(history['val_accuracy'], label=name, linewidth=2)
    plt.title('Validation Accuracy Comparison', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    for name, history in histories.items():
        plt.plot(history['val_loss'], label=name, linewidth=2)
    plt.title('Validation Loss Comparison', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    metrics = ['accuracy', 'f1_score', 'auc_roc']
    model_names = list(evaluation_results.keys())
    x = np.arange(len(model_names))
    width = 0.25
    plt.subplot(2, 2, 3)
    for i, metric in enumerate(metrics):
        values = [evaluation_results[name][metric] for name in model_names]
        plt.bar(x + i * width, values, width, label=metric.replace('_', ' ').title())
    plt.xticks(x + width, model_names, fontsize=12)
    plt.title('Evaluation Metrics Comparison', fontsize=14)
    plt.ylabel('Score', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 4)
    ensemble_preds = evaluation_results['StackingEnsemble']['predictions']
    cm = confusion_matrix(y_val, ensemble_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Pneumonia'])
    disp.plot(cmap='Blues', ax=plt.gca())
    plt.title('Stacking Ensemble Confusion Matrix', fontsize=14)

    plt.tight_layout()
    
    # Convert plot to base64 for Streamlit
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    return img_str

# Streamlit app
st.title("Pneumonia Detection using Stacking Ensemble")
st.markdown("This application allows you to train a stacking ensemble model for pneumonia detection from chest X-ray images and make predictions on new images.")

# Check GPU availability
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    st.write(f"GPU is available: {physical_devices}")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    st.write("No GPU available, using CPU.")

# Enable mixed precision training
set_global_policy(Policy('mixed_float16'))

# Sidebar for hyperparameter inputs
st.sidebar.header("Model Configuration")
max_per_class = st.sidebar.slider("Max Images per Class", 100, 2000, 1000, 100)
split_ratio = st.sidebar.slider("Train-Test Split Ratio", 0.6, 0.9, 0.8, 0.05)
base_epochs = st.sidebar.slider("Base Model Epochs", 5, 50, 30, 5)
meta_epochs = st.sidebar.slider("Meta-Learner Epochs", 5, 50, 20, 5)
batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64, 128], index=2)

# Dataset loading
st.header("Dataset Loading")
data_dir_input = st.text_input("Dataset Directory", "chest_xray")
train_button = st.button("Load and Train Models")

if train_button:
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

        X_train, y_train, X_val, y_val = load_dataset(
            data_dir,
            target_size=(224, 224),
            max_per_class=max_per_class,
            split_ratio=split_ratio
        )

    with st.spinner("Training models..."):
        model = PneumoniaStackingEnsemble(input_shape=(224, 224, 3))
        base_histories = model.train_base_models(
            X_train, y_train,
            X_val, y_val,
            epochs=base_epochs,
            batch_size=batch_size
        )
        meta_history = model.train_meta_learner(
            X_train, y_train,
            X_val, y_val,
            epochs=meta_epochs,
            batch_size=batch_size
        )
        histories = {**base_histories, **meta_history}

    with st.spinner("Evaluating models..."):
        evaluation_results = model.evaluate(X_val, y_val)
        st.header("Evaluation Results")
        for name, metrics in evaluation_results.items():
            st.subheader(name)
            st.write(f"Accuracy: {metrics['accuracy']:.4f}")
            st.write(f"F1-Score: {metrics['f1_score']:.4f}")
            st.write(f"AUC-ROC: {metrics['auc_roc']:.4f}")

    st.header("Training and Evaluation Visualizations")
    img_str = plot_results(histories, evaluation_results, y_val)
    st.image(f"data:image/png;base64,{img_str}")

    # Save model to session state for predictions
    st.session_state['model'] = model
    st.session_state['histories'] = histories
    st.session_state['evaluation_results'] = evaluation_results
    st.session_state['y_val'] = y_val

# Single image prediction
st.header("Predict Pneumonia from a Single Image")
uploaded_file = st.file_uploader("Upload a chest X-ray image (jpg/jpeg)", type=["jpg", "jpeg"])
if uploaded_file and 'model' in st.session_state:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    with st.spinner("Making predictions..."):
        model = st.session_state['model']
        predictions = model.predict_single_image(image)
        
        st.subheader("Prediction Results")
        for name, result in predictions.items():
            st.write(f"**{name}**:")
            st.write(f"Class: {result['class']}")
            st.write(f"Probability of Pneumonia: {result['probability']:.4f}")

# Display saved visualizations if available
if 'histories' in st.session_state:
    st.header("Previous Training Visualizations")
    img_str = plot_results(
        st.session_state['histories'],
        st.session_state['evaluation_results'],
        st.session_state['y_val']
    )
    st.image(f"data:image/png;base64,{img_str}")