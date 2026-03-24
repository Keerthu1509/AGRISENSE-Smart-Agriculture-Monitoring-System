import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import warnings
warnings.filterwarnings('ignore')

crop_df = pd.read_csv('/kaggle/input/crop-recommendation-dataset/Crop_recommendation.csv')
irrigation_df = pd.read_csv('/kaggle/input/irrigation-dataset2/Project_datasheet_2019-2020.csv')

print("Crop Recommendation Dataset Shape:", crop_df.shape)
print("\nCrop Dataset Info:")
print(crop_df.info())
print("\nFirst 5 rows:")
print(crop_df.head())
print("\n" + "="*80 + "\n")

print("Irrigation Dataset Shape:", irrigation_df.shape)
print("\nIrrigation Dataset Info:")
print(irrigation_df.info())
print("\nFirst 5 rows:")
print(irrigation_df.head())
print("CROP RECOMMENDATION DATASET ANALYSIS")
print("="*80)
print("\nMissing Values:")
print(crop_df.isnull().sum())
print("\nDuplicate Rows:", crop_df.duplicated().sum())
print("\nCrop Distribution:")
print(crop_df['label'].value_counts())
print("\nStatistical Summary:")
print(crop_df.describe())

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
crop_df['label'].value_counts().plot(kind='bar', color='teal')
plt.title('Crop Distribution')
plt.xlabel('Crop')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')

plt.subplot(1, 2, 2)
sns.heatmap(crop_df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap - Crop Dataset')
plt.tight_layout()
plt.show()

print("\n" + "="*80)
print("IRRIGATION DATASET ANALYSIS")
print("="*80)

irrigation_df = irrigation_df.drop(columns=['Unnamed: 7', 'Unnamed: 8', 'Unnamed: 9', 'Unnamed: 10'])
print("\nCleaned Irrigation Dataset Shape:", irrigation_df.shape)
print("\nMissing Values:")
print(irrigation_df.isnull().sum())
print("\nDuplicate Rows:", irrigation_df.duplicated().sum())
print("\nIrrigation Distribution:")
print(irrigation_df['Irrigation(Y/N)'].value_counts())
print("\nCrop Type Distribution:")
print(irrigation_df['CropType'].value_counts())
print("\nStatistical Summary:")
print(irrigation_df.describe())

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
irrigation_df['Irrigation(Y/N)'].value_counts().plot(kind='bar', color='coral')
plt.title('Irrigation Distribution (0=No, 1=Yes)')
plt.xlabel('Irrigation')
plt.ylabel('Count')
plt.xticks(rotation=0)

plt.subplot(1, 2, 2)
sns.heatmap(irrigation_df.corr(numeric_only=True), annot=True, cmap='viridis', fmt='.2f')
plt.title('Correlation Heatmap - Irrigation Dataset')
plt.tight_layout()
plt.show()
print("CROP RECOMMENDATION MODEL - PREPROCESSING")
print("="*80)

X_crop = crop_df.drop('label', axis=1)
y_crop = crop_df['label']

le_crop = LabelEncoder()
y_crop_encoded = le_crop.fit_transform(y_crop)

print("Crop Classes:", le_crop.classes_)
print("Number of Crops:", len(le_crop.classes_))

scaler_crop = StandardScaler()
X_crop_scaled = scaler_crop.fit_transform(X_crop)

X_crop_train, X_crop_test, y_crop_train, y_crop_test = train_test_split(
    X_crop_scaled, y_crop_encoded, test_size=0.2, random_state=42, stratify=y_crop_encoded
)

print(f"\nTraining Set: {X_crop_train.shape}")
print(f"Testing Set: {X_crop_test.shape}")

print("\n" + "="*80)
print("IRRIGATION MODEL - PREPROCESSING & BALANCING")
print("="*80)

X_irrigation = irrigation_df.drop('Irrigation(Y/N)', axis=1)
y_irrigation = irrigation_df['Irrigation(Y/N)']

print("Before Balancing:")
print(y_irrigation.value_counts())

scaler_irrigation = StandardScaler()

from sklearn.utils import resample

df_majority = irrigation_df[irrigation_df['Irrigation(Y/N)'] == 0]
df_minority = irrigation_df[irrigation_df['Irrigation(Y/N)'] == 1]

df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)

df_balanced = pd.concat([df_majority, df_minority_upsampled])
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print("\nAfter Upsampling:")
print(df_balanced['Irrigation(Y/N)'].value_counts())

X_irrigation_balanced = df_balanced.drop('Irrigation(Y/N)', axis=1)
y_irrigation_balanced = df_balanced['Irrigation(Y/N)']

X_irrigation_balanced_scaled = scaler_irrigation.fit_transform(X_irrigation_balanced)

X_irr_train, X_irr_test, y_irr_train, y_irr_test = train_test_split(
    X_irrigation_balanced_scaled, y_irrigation_balanced, test_size=0.2, random_state=42, stratify=y_irrigation_balanced
)

print(f"\nTraining Set: {X_irr_train.shape}")
print(f"Testing Set: {X_irr_test.shape}")

plt.figure(figsize=(8, 4))
df_balanced['Irrigation(Y/N)'].value_counts().plot(kind='bar', color=['green', 'blue'])
plt.title('Balanced Irrigation Dataset')
plt.xlabel('Irrigation (0=No, 1=Yes)')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()
print("TRAINING CROP RECOMMENDATION MODEL")
print("="*80)

rf_crop = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_crop.fit(X_crop_train, y_crop_train)

y_crop_pred = rf_crop.predict(X_crop_test)
crop_accuracy = accuracy_score(y_crop_test, y_crop_pred)

print(f"Crop Model Training Accuracy: {rf_crop.score(X_crop_train, y_crop_train)*100:.2f}%")
print(f"Crop Model Testing Accuracy: {crop_accuracy*100:.2f}%")

print("\n" + "="*80)
print("TRAINING IRRIGATION MODEL")
print("="*80)

rf_irrigation = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_irrigation.fit(X_irr_train, y_irr_train)

y_irr_pred = rf_irrigation.predict(X_irr_test)
irrigation_accuracy = accuracy_score(y_irr_test, y_irr_pred)

print(f"Irrigation Model Training Accuracy: {rf_irrigation.score(X_irr_train, y_irr_train)*100:.2f}%")
print(f"Irrigation Model Testing Accuracy: {irrigation_accuracy*100:.2f}%")

feature_names_crop = ['N', 'P', 'K', 'Temperature', 'Humidity', 'pH', 'Rainfall']
feature_importance_crop = pd.DataFrame({
    'Feature': feature_names_crop,
    'Importance': rf_crop.feature_importances_
}).sort_values('Importance', ascending=False)

feature_names_irrigation = ['CropType', 'CropDays', 'Soil Moisture', 'Soil Temperature', 'Temperature', 'Humidity']
feature_importance_irrigation = pd.DataFrame({
    'Feature': feature_names_irrigation,
    'Importance': rf_irrigation.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nCrop Model - Top 5 Important Features:")
print(feature_importance_crop.head())

print("\nIrrigation Model - Top Features:")
print(feature_importance_irrigation)

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.barh(feature_importance_crop['Feature'], feature_importance_crop['Importance'], color='teal')
plt.xlabel('Importance')
plt.title('Crop Model Feature Importance')
plt.gca().invert_yaxis()

plt.subplot(1, 2, 2)
plt.barh(feature_importance_irrigation['Feature'], feature_importance_irrigation['Importance'], color='coral')
plt.xlabel('Importance')
plt.title('Irrigation Model Feature Importance')
plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()
from sklearn.metrics import precision_score, recall_score, f1_score

print("CROP RECOMMENDATION MODEL - DETAILED EVALUATION")
print("="*80)

crop_precision = precision_score(y_crop_test, y_crop_pred, average='weighted')
crop_recall = recall_score(y_crop_test, y_crop_pred, average='weighted')
crop_f1 = f1_score(y_crop_test, y_crop_pred, average='weighted')

print(f"Accuracy:  {crop_accuracy*100:.2f}%")
print(f"Precision: {crop_precision*100:.2f}%")
print(f"Recall:    {crop_recall*100:.2f}%")
print(f"F1-Score:  {crop_f1*100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_crop_test, y_crop_pred, target_names=le_crop.classes_))

print("\n" + "="*80)
print("IRRIGATION MODEL - DETAILED EVALUATION")
print("="*80)

irr_precision = precision_score(y_irr_test, y_irr_pred, average='weighted')
irr_recall = recall_score(y_irr_test, y_irr_pred, average='weighted')
irr_f1 = f1_score(y_irr_test, y_irr_pred, average='weighted')

print(f"Accuracy:  {irrigation_accuracy*100:.2f}%")
print(f"Precision: {irr_precision*100:.2f}%")
print(f"Recall:    {irr_recall*100:.2f}%")
print(f"F1-Score:  {irr_f1*100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_irr_test, y_irr_pred, target_names=['No Irrigation', 'Irrigate']))

cm_crop = confusion_matrix(y_crop_test, y_crop_pred)
cm_irrigation = confusion_matrix(y_irr_test, y_irr_pred)

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.heatmap(cm_crop, annot=True, fmt='d', cmap='Blues', xticklabels=le_crop.classes_, yticklabels=le_crop.classes_, cbar=False)
plt.title('Crop Model Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(rotation=0, fontsize=8)

plt.subplot(1, 2, 2)
sns.heatmap(cm_irrigation, annot=True, fmt='d', cmap='Greens', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.title('Irrigation Model Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')

plt.tight_layout()
plt.show()
print("CROP RECOMMENDATION MODEL - RANDOM PREDICTIONS")
print("="*80)

random_indices_crop = np.random.choice(len(X_crop_test), 5, replace=False)

for idx in random_indices_crop:
    actual_label = le_crop.inverse_transform([y_crop_test[idx]])[0]
    predicted_label = le_crop.inverse_transform([y_crop_pred[idx]])[0]
    probabilities = rf_crop.predict_proba([X_crop_test[idx]])[0]
    confidence = max(probabilities) * 100
    
    print(f"\nSample {idx}:")
    print(f"Actual Crop: {actual_label}")
    print(f"Predicted Crop: {predicted_label}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"Status: {'✓ Correct' if actual_label == predicted_label else '✗ Wrong'}")

print("\n" + "="*80)
print("IRRIGATION MODEL - RANDOM PREDICTIONS")
print("="*80)

random_indices_irr = np.random.choice(len(X_irr_test), 5, replace=False)

for idx in random_indices_irr:
    actual = 'Irrigate' if y_irr_test.iloc[idx] == 1 else 'No Irrigation'
    predicted = 'Irrigate' if y_irr_pred[idx] == 1 else 'No Irrigation'
    probabilities = rf_irrigation.predict_proba([X_irr_test[idx]])[0]
    confidence = max(probabilities) * 100
    
    print(f"\nSample {idx}:")
    print(f"Actual: {actual}")
    print(f"Predicted: {predicted}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"Status: {'✓ Correct' if actual == predicted else '✗ Wrong'}")

print("\n" + "="*80)
print("CUSTOM INPUT PREDICTIONS")
print("="*80)

sample_crop_input = np.array([[90, 42, 43, 20.87, 82.00, 6.50, 202.93]])
sample_crop_scaled = scaler_crop.transform(sample_crop_input)
crop_prediction = rf_crop.predict(sample_crop_scaled)
crop_proba = rf_crop.predict_proba(sample_crop_scaled)[0]
predicted_crop = le_crop.inverse_transform(crop_prediction)[0]
confidence_crop = max(crop_proba) * 100

print("\nCrop Recommendation Test:")
print(f"Input: N=90, P=42, K=43, Temp=20.87°C, Humidity=82%, pH=6.5, Rainfall=202.93mm")
print(f"Recommended Crop: {predicted_crop.upper()}")
print(f"Confidence: {confidence_crop:.2f}%")

sample_irr_input = np.array([[1, 50, 150, 24, 28, 55]])
sample_irr_scaled = scaler_irrigation.transform(sample_irr_input)
irr_prediction = rf_irrigation.predict(sample_irr_scaled)
irr_proba = rf_irrigation.predict_proba(sample_irr_scaled)[0]
predicted_irr = 'YES - Irrigate Now' if irr_prediction[0] == 1 else 'NO - Do Not Irrigate'
confidence_irr = max(irr_proba) * 100

print("\nIrrigation Advisory Test:")
print(f"Input: CropType=Paddy, CropDays=50, SoilMoisture=150, SoilTemp=24°C, Temp=28°C, Humidity=55%")
print(f"Irrigation Decision: {predicted_irr}")
print(f"Confidence: {confidence_irr:.2f}%")
print("SAVING MODELS AND ARTIFACTS")
print("="*80)

with open('crop_recommendation_model.pkl', 'wb') as f:
    pickle.dump(rf_crop, f)
print("✓ Crop Recommendation Model saved: crop_recommendation_model.pkl")

with open('irrigation_model.pkl', 'wb') as f:
    pickle.dump(rf_irrigation, f)
print("✓ Irrigation Model saved: irrigation_model.pkl")

with open('crop_label_encoder.pkl', 'wb') as f:
    pickle.dump(le_crop, f)
print("✓ Crop Label Encoder saved: crop_label_encoder.pkl")

with open('crop_scaler.pkl', 'wb') as f:
    pickle.dump(scaler_crop, f)
print("✓ Crop Scaler saved: crop_scaler.pkl")

with open('irrigation_scaler.pkl', 'wb') as f:
    pickle.dump(scaler_irrigation, f)
print("✓ Irrigation Scaler saved: irrigation_scaler.pkl")

print("\n" + "="*80)
print("LIBRARY VERSIONS")
print("="*80)

import sys
import sklearn

print(f"Python Version: {sys.version.split()[0]}")
print(f"NumPy Version: {np.__version__}")
print(f"Pandas Version: {pd.__version__}")
print(f"Scikit-learn Version: {sklearn.__version__}")
print(f"Matplotlib Version: {plt.matplotlib.__version__}")
print(f"Seaborn Version: {sns.__version__}")

print("\n" + "="*80)
print("MODEL SUMMARY")
print("="*80)

print("\nCROP RECOMMENDATION MODEL:")
print(f"  Algorithm: Random Forest Classifier")
print(f"  Training Samples: {len(X_crop_train)}")
print(f"  Testing Samples: {len(X_crop_test)}")
print(f"  Number of Classes: {len(le_crop.classes_)}")
print(f"  Features: {len(feature_names_crop)}")
print(f"  Test Accuracy: {crop_accuracy*100:.2f}%")
print(f"  F1-Score: {crop_f1*100:.2f}%")

print("\nIRRIGATION MODEL:")
print(f"  Algorithm: Random Forest Classifier")
print(f"  Training Samples: {len(X_irr_train)}")
print(f"  Testing Samples: {len(X_irr_test)}")
print(f"  Number of Classes: 2 (Yes/No)")
print(f"  Features: {len(feature_names_irrigation)}")
print(f"  Test Accuracy: {irrigation_accuracy*100:.2f}%")
print(f"  F1-Score: {irr_f1*100:.2f}%")

print("\n" + "="*80)
print("TRAINING COMPLETED SUCCESSFULLY!")
print("="*80)
!zip -r model.zip /kaggle/working
from IPython.display import FileLink
FileLink(r'model.zip')

