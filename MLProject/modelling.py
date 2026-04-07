#!/usr/bin/env python3
"""
Titanic Model Training for MLflow Project
Author: Randi Sumitro

This script is designed to work with MLflow projects.
"""

import sys
import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Set MLflow tracking
    mlflow.set_tracking_uri("https://dagshub.com/Randi Sumitro/titanic-mlflow.mlflow")
    mlflow.set_experiment("Titanic-CI-Pipeline")
    
    # Load data
    try:
        # Try to load from data_preprocessed directory
        data_path = os.path.join("data_preprocessed", "titanic_processed.csv")
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
        else:
            # Fallback to download
            url = "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"
            df = pd.read_csv(url)
            
            # Basic preprocessing
            if 'Name' in df.columns:
                df = df.drop('Name', axis=1)
            
            from sklearn.preprocessing import StandardScaler, LabelEncoder
            from sklearn.impute import SimpleImputer
            
            imputer_num = SimpleImputer(strategy='median')
            imputer_cat = SimpleImputer(strategy='most_frequent')
            
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            categorical_columns = df.select_dtypes(include=['object']).columns
            
            if len(numeric_columns) > 0:
                df[numeric_columns] = imputer_num.fit_transform(df[numeric_columns])
            
            if len(categorical_columns) > 0:
                df[categorical_columns] = imputer_cat.fit_transform(df[categorical_columns])
            
            le = LabelEncoder()
            for col in categorical_columns:
                if col in df.columns:
                    df[col] = le.fit_transform(df[col])
            
            scaler = StandardScaler()
            if len(numeric_columns) > 0:
                df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
        
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        sys.exit(1)
    
    # Prepare data
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model with MLflow
    with mlflow.start_run() as run:
        # Enable autologging
        mlflow.sklearn.autolog()
        
        # Train model
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        rf.fit(X_train, y_train)
        
        # Evaluate
        y_pred = rf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Model trained successfully. Accuracy: {accuracy:.4f}")
        logger.info(f"Run ID: {run.info.run_id}")
        
        # Log model manually as well
        mlflow.sklearn.log_model(rf, "model", registered_model_name="titanic-model")
        
        print(f"Model trained with accuracy: {accuracy:.4f}")
        print(f"Run ID: {run.info.run_id}")

if __name__ == "__main__":
    main()
