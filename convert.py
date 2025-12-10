import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras

print(f"TensorFlow version: {tf.__version__}")

model_dir = "saved_models"
models_to_convert = [
    "efficientnet_model.keras",
    "mobilenetv3_model.keras", 
    "mobilenet_model.keras",
    "meta_learner_model.keras"
]

for model_file in models_to_convert:
    filepath = os.path.join(model_dir, model_file)
    if os.path.exists(filepath):
        try:
            print(f"\nConverting {model_file}...")
            
            # Load with different methods
            try:
                model = keras.models.load_model(filepath, compile=False)
            except Exception as e1:
                print(f"Method 1 failed: {e1}")
                try:
                    # Try with safe_mode=False
                    model = keras.models.load_model(filepath, compile=False, safe_mode=False)
                except Exception as e2:
                    print(f"Method 2 failed: {e2}")
                    print(f"❌ Cannot load {model_file}")
                    continue
            
            # Save as H5
            h5_path = filepath.replace('.keras', '.h5')
            model.save(h5_path, save_format='h5')
            print(f"✅ Saved to {h5_path}")
            
        except Exception as e:
            print(f"❌ Error converting {model_file}: {e}")
    else:
        print(f"⚠️ File not found: {filepath}")

print("\n✅ Conversion complete!")