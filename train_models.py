import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_crop_recommendation():
    print("Training Crop Recommendation Model...")
    try:
        df = pd.read_csv('Crop_recommendation.csv')
    except FileNotFoundError:
        print("Error: 'Crop_recommendation.csv' not found. Please place it in this directory.")
        return

    # Features and Target
    X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = df['label']

    # Label Encoding
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Scaling
    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_test_scaled = sc.transform(X_test)

    # Model Training
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Evaluation
    y_pred = model.predict(X_test_scaled)
    print(f"Crop Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")

    # Save Models
    if not os.path.exists('models'):
        os.makedirs('models')
    
    with open('models/crop_recommendation_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('models/crop_label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    with open('models/crop_scaler.pkl', 'wb') as f:
        pickle.dump(sc, f)
    
    print("Crop models saved successfully.\n")

def train_irrigation_recommendation():
    print("Training Irrigation Recommendation Model...")
    try:
        df = pd.read_csv('Project_datasheet_2019-2020.csv')
    except FileNotFoundError:
        print("Error: 'Project_datasheet_2019-2020.csv' not found. Please place it in this directory.")
        return

    # Required columns based on app.py usage or typical dataset structure
    # Based on irrigation11.ipynb inspection earlier:
    # Columns: CropType, CropDays, Soil Moisture, Soil Temperature, Temperature, Humidity, Irrigation(Y/N)
    
    # Cleaning if needed (dropping Unnamed columns as seen in notebook preview)
    df = df.dropna(axis=1, how='all')
    
    X = df[['CropType', 'CropDays', 'Soil Moisture', 'Soil Temperature', 'Temperature', 'Humidity']]
    y = df['Irrigation(Y/N)']

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scaling
    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_test_scaled = sc.transform(X_test)

    # Model Training
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Evaluation
    y_pred = model.predict(X_test_scaled)
    print(f"Irrigation Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")

    # Save Models
    with open('models/irrigation_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('models/irrigation_scaler.pkl', 'wb') as f:
        pickle.dump(sc, f)

    print("Irrigation models saved successfully.\n")

if __name__ == "__main__":
    train_crop_recommendation()
    train_irrigation_recommendation()
    print("All tasks completed.")
