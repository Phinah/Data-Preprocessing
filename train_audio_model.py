"""
Voice Verification Model Training Script
"""
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score
import joblib
import numpy as np

# Load the CSV that audio_processing.py created
CSV_PATH = Path("audio_features.csv")
if not CSV_PATH.exists():
    raise FileNotFoundError("Run audio_processing.py first → audio_features.csv missing")

df = pd.read_csv(CSV_PATH)

print("="*60)
print("VOICE VERIFICATION MODEL TRAINING")
print("="*60)
print(f"Loaded {len(df)} rows – {df['member_name'].nunique() if 'member_name' in df.columns else 0} speakers")
if 'member_name' in df.columns:
    print(df['member_name'].value_counts())

# Encode speaker names
le = LabelEncoder()
if 'member_name' in df.columns:
    df["label"] = le.fit_transform(df["member_name"])
else:
    # If no member_name, use authorized column
    df["label"] = df.get("authorized", 0)

# Select numeric feature columns
exclude = {"member_name", "phrase", "augmentation", "audio_path", "label", "authorized", "user_id", "file_name"}
feature_cols = [c for c in df.columns if c not in exclude]

print(f"\nNumber of feature columns: {len(feature_cols)}")
print(f"First 10 features: {feature_cols[:10]}")

X = df[feature_cols].values.astype(np.float32)
y = df["label"].values

# Handle any NaN values
if np.isnan(X).any():
    print("Warning: NaN values found, filling with 0")
    X = np.nan_to_num(X)

# Train / validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"\nTraining set: {len(X_train)} samples")
print(f"Validation set: {len(X_val)} samples")

# Train Random Forest Classifier
print("\n" + "="*60)
print("Training Random Forest Classifier...")
print("="*60)

rf_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_split=2,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)
rf_model.fit(X_train, y_train)

# Evaluate Random Forest
rf_pred = rf_model.predict(X_val)
rf_acc = accuracy_score(y_val, rf_pred)
rf_f1 = f1_score(y_val, rf_pred, average='weighted')

print(f"\nRandom Forest Results:")
print(f"Validation Accuracy: {rf_acc:.1%}")
print(f"Validation F1-Score (weighted): {rf_f1:.4f}")
print("\nClassification Report:")
if 'member_name' in df.columns:
    print(classification_report(y_val, rf_pred, target_names=le.classes_))
else:
    print(classification_report(y_val, rf_pred))

# Train Logistic Regression (alternative model)
print("\n" + "="*60)
print("Training Logistic Regression Classifier...")
print("="*60)

lr_model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight="balanced",
    solver='lbfgs'
)
lr_model.fit(X_train, y_train)

# Evaluate Logistic Regression
lr_pred = lr_model.predict(X_val)
lr_acc = accuracy_score(y_val, lr_pred)
lr_f1 = f1_score(y_val, lr_pred, average='weighted')

print(f"\nLogistic Regression Results:")
print(f"Validation Accuracy: {lr_acc:.1%}")
print(f"Validation F1-Score (weighted): {lr_f1:.4f}")
print("\nClassification Report:")
if 'member_name' in df.columns:
    print(classification_report(y_val, lr_pred, target_names=le.classes_))
else:
    print(classification_report(y_val, lr_pred))

# Select best model and save
if rf_acc >= lr_acc:
    print("\n" + "="*60)
    print("Selected Random Forest as the best model (higher accuracy)")
    print("="*60)
    best_model = rf_model
    model_name = "RandomForest"
else:
    print("\n" + "="*60)
    print("Selected Logistic Regression as the best model (higher accuracy)")
    print("="*60)
    best_model = lr_model
    model_name = "LogisticRegression"

# Save model, encoder, and feature columns
model_dir = Path("models")
model_dir.mkdir(parents=True, exist_ok=True)

joblib.dump(best_model, model_dir / "voice_verification_model.pkl")
if 'member_name' in df.columns:
    joblib.dump(le, model_dir / "voice_label_encoder.pkl")
joblib.dump(feature_cols, model_dir / "voice_feature_columns.pkl")

print("\n" + "="*60)
print("Model saved successfully!")
print("="*60)
print(f"   -> {model_dir / 'voice_verification_model.pkl'}")
if 'member_name' in df.columns:
    print(f"   -> {model_dir / 'voice_label_encoder.pkl'}")
print(f"   -> {model_dir / 'voice_feature_columns.pkl'}")
print(f"\nSelected model: {model_name}")
print(f"Final Accuracy: {max(rf_acc, lr_acc):.1%}")
print(f"Final F1-Score: {max(rf_f1, lr_f1):.4f}")
print("="*60)

