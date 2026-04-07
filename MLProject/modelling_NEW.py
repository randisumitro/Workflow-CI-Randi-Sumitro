#!/usr/bin/env python3
"""
MLflow Project Compatible Titanic Model Training
Author: Randi Sumitro
Project: Membangun Sistem Machine Learning - Dicoding

This script is compatible with MLflow project structure and can be run
using 'mlflow run' command.
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(data_path):
    """Load Titanic dataset from file or download if not available"""
    try:
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            logger.info(f"Loaded data from: {data_path}")
        else:
            logger.info(f"Data file not found at {data_path}, downloading...")
            url = "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"
            df = pd.read_csv(url)
            
            # Basic preprocessing
            df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})
            df = df.fillna(df.median())
            
            # Scale features
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            feature_cols = ['Pclass', 'Sex', 'Age', 'Siblings/Spouses Aboard', 'Parents/Children Aboard', 'Fare']
            df[feature_cols] = scaler.fit_transform(df[feature_cols])
            
            # Save processed data
            os.makedirs(os.path.dirname(data_path), exist_ok=True)
            df.to_csv(data_path, index=False)
            logger.info(f"Processed data saved to: {data_path}")
            
        return df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def prepare_data(df):
    """Prepare features and target for training"""
    feature_columns = ['Pclass', 'Sex', 'Age', 'Siblings/Spouses Aboard', 'Parents/Children Aboard', 'Fare']
    target_column = 'Survived'
    
    X = df[feature_columns]
    y = df[target_column]
    
    return X, y, feature_columns

def train_model(X_train, y_train, X_test, y_test, max_depth=10, n_estimators=100, model_name="titanic-model"):
    """Train RandomForest model with MLflow logging"""
    
    with mlflow.start_run() as run:
        logger.info(f"Started MLflow run: {run.info.run_id}")
        
        # Initialize model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        # Train model
        logger.info("Training RandomForest model...")
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'train_accuracy': accuracy_score(y_train, y_pred_train),
            'train_precision': precision_score(y_train, y_pred_train, average='binary'),
            'train_recall': recall_score(y_train, y_pred_train, average='binary'),
            'train_f1_score': f1_score(y_train, y_pred_train, average='binary'),
            'test_accuracy': accuracy_score(y_test, y_pred_test),
            'test_precision': precision_score(y_test, y_pred_test, average='binary'),
            'test_recall': recall_score(y_test, y_pred_test, average='binary'),
            'test_f1_score': f1_score(y_test, y_pred_test, average='binary')
        }
        
        # Log parameters
        mlflow.log_params({
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'model_name': model_name
        })
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Log model
        mlflow.sklearn.log_model(model, "model", registered_model_name=model_name)
        
        # Print results
        logger.info("=" * 60)
        logger.info("MLFLOW PROJECT TRAINING RESULTS")
        logger.info("=" * 60)
        logger.info(f"Run ID: {run.info.run_id}")
        logger.info(f"Model: {model_name}")
        logger.info("\nTest Metrics:")
        for metric, value in metrics.items():
            if metric.startswith('test_'):
                logger.info(f"  {metric}: {value:.4f}")
        
        logger.info("=" * 60)
        
        return model, run.info.run_id

def main():
    """Main function for MLflow project"""
    parser = argparse.ArgumentParser(description='Titanic ML Project')
    parser.add_argument('--data_path', type=str, default='data_preprocessed/titanic_processed.csv',
                        help='Path to the Titanic dataset')
    parser.add_argument('--model_name', type=str, default='titanic-model',
                        help='Name of the model')
    parser.add_argument('--max_depth', type=int, default=10,
                        help='Maximum depth of the random forest')
    parser.add_argument('--n_estimators', type=int, default=100,
                        help='Number of trees in the random forest')
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("TITANIC MLFLOW PROJECT")
    logger.info("Author: Randi Sumitro")
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Model name: {args.model_name}")
    logger.info("=" * 60)
    
    try:
        # Load data
        df = load_data(args.data_path)
        
        # Prepare data
        X, y, feature_columns = prepare_data(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"Training set shape: {X_train.shape}")
        logger.info(f"Test set shape: {X_test.shape}")
        
        # Train model
        model, run_id = train_model(
            X_train, y_train, X_test, y_test,
            max_depth=args.max_depth,
            n_estimators=args.n_estimators,
            model_name=args.model_name
        )
        
        logger.info(f"Training completed successfully!")
        logger.info(f"Run ID: {run_id}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
