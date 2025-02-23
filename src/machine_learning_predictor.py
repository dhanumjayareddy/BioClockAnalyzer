#!/usr/bin/env python3
"""
machine_learning_predictor.py
-----------------------------
Production-ready script for training and evaluating a deep learning model to predict key regulatory nodes.
Features:
- Robust error handling and logging.
- Parameter validation and scalable data splitting.
- Hyperparameter tuning placeholder for model customization.
- Detailed inline documentation.
Usage:
    python src/machine_learning_predictor.py --features data/processed/features.csv --labels data/processed/labels.csv --model_output results/ml/model.h5
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def parse_args():
    parser = argparse.ArgumentParser(description="Train a deep learning model for circadian network regulation prediction.")
    parser.add_argument("--features", required=True, help="Path to the features CSV file.")
    parser.add_argument("--labels", required=True, help="Path to the labels CSV file.")
    parser.add_argument("--model_output", required=True, help="Path to save the trained model.")
    return parser.parse_args()

def load_features_labels(features_path, labels_path):
    if not os.path.exists(features_path):
        logging.error(f"Features file not found: {features_path}")
        sys.exit(1)
    if not os.path.exists(labels_path):
        logging.error(f"Labels file not found: {labels_path}")
        sys.exit(1)
    try:
        features = pd.read_csv(features_path, index_col=0)
        labels = pd.read_csv(labels_path, index_col=0)
        logging.info("Features and labels loaded successfully.")
        return features, labels
    except Exception as e:
        logging.error(f"Error loading features/labels: {e}")
        sys.exit(1)

def build_model(input_dim):
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(input_dim,)),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_and_evaluate(features, labels, model_output):
    X_train, X_test, y_train, y_test = train_test_split(features.values, labels.values.ravel(), test_size=0.2, random_state=42)
    logging.info(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")
    
    model = build_model(X_train.shape[1])
    logging.info("Model architecture:")
    model.summary(print_fn=logging.info)
    
    # Hyperparameter tuning can be added here (e.g., using Keras Tuner)
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)
    
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    y_pred = model.predict(X_test)
    auc = roc_auc_score(y_test, y_pred)
    logging.info(f"Test Accuracy: {accuracy:.4f}")
    logging.info(f"Test AUC: {auc:.4f}")
    
    # Save the model
    os.makedirs(os.path.dirname(model_output), exist_ok=True)
    model.save(model_output)
    logging.info(f"Trained model saved to {model_output}")
    return model

if __name__ == "__main__":
    args = parse_args()
    features, labels = load_features_labels(args.features, args.labels)
    train_and_evaluate(features, labels, args.model_output)
