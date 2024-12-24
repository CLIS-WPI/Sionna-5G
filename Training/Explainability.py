import shap
import tensorflow as tf
import numpy as np
import h5py
import os

# Load the trained actor model
MODEL_PATH = "training/models/actor_final.h5"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Ensure the training script has saved the model.")

actor_model = tf.keras.models.load_model(MODEL_PATH)
print("Loaded trained actor model.")

# Load the dataset for explainability analysis
DATASET_PATH = "training/dataset/latest_dataset.h5"  # Adjust this path if needed
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}. Ensure the dataset exists.")

with h5py.File(DATASET_PATH, "r") as f:
    states = f["states"][:]  # Only states are required for explainability
    print(f"Loaded dataset from {DATASET_PATH} with shape: {states.shape}")

# Subset data for faster SHAP analysis (optional, to reduce computational cost)
states_sample = states[:1000]  # Modify the size as needed based on your GPU memory

# Initialize SHAP Explainer for TensorFlow models
print("Initializing SHAP explainer...")
explainer = shap.Explainer(actor_model, shap.sample(states_sample, 100))  # SHAP expects a sample of data to initialize

# Explain predictions
print("Generating SHAP values...")
shap_values = explainer(states_sample)

# Visualize the results
FEATURE_NAMES = [f"Feature {i}" for i in range(states.shape[1])]
print("Creating SHAP summary plot...")
shap.summary_plot(shap_values, states_sample, feature_names=FEATURE_NAMES)

print("Explainability analysis completed.")
