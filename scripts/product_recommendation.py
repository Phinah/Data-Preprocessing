"""
Product Recommendation Model Training Script
Uses merged customer data to predict product category
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score, log_loss
import joblib
import os

# Load merged dataset
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

merged_path = Path('merge-output/merged_data.csv')
if not merged_path.exists():
    raise FileNotFoundError("Run scripts/merge_datasets.py first → merged_data.csv missing")

print("="*60)
print("PRODUCT RECOMMENDATION MODEL TRAINING")
print("="*60)

df = pd.read_csv(merged_path)
print(f"Loaded merged dataset: {df.shape}")

# Prepare features and target
# Use favorite_category as target (most frequent purchase)
target_col = 'favorite_category'
if target_col not in df.columns:
    # Fallback: use last_category_purchased
    target_col = 'last_category_purchased'

# Define features
categorical_features = ['social_media_platform', 'dominant_sentiment', 'platforms_used']
numerical_features = [
    'total_spent', 'avg_purchase_amount', 'avg_customer_rating',
    'avg_engagement_score', 'avg_purchase_interest_score',
    'total_purchases', 'platform_diversity', 'customer_value_score'
]

# Filter to available columns
available_num = [c for c in numerical_features if c in df.columns]
available_cat = [c for c in categorical_features if c in df.columns]
features_for_model = available_num + available_cat

if len(features_for_model) == 0:
    raise ValueError(f"No model features found in dataframe. Available columns: {df.columns.tolist()}")

print(f"\nUsing {len(features_for_model)} features:")
print(f"  Numerical: {available_num}")
print(f"  Categorical: {available_cat}")

# Prepare X and y
X = df[features_for_model].copy()
y = df[target_col].copy()

# Drop rows with missing target
notnull_mask = y.notnull()
X = X.loc[notnull_mask]
y = y.loc[notnull_mask]

print(f"\nDataset after removing null targets: {len(X)} samples")
print(f"Target distribution:")
print(y.value_counts())

# Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Create preprocessing pipelines
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Combine transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, available_num),
        ('cat', categorical_transformer, available_cat)
    ],
    remainder='drop'
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

print(f"\nTraining set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# Train Random Forest
print("\n" + "="*60)
print("Training Random Forest Classifier...")
print("="*60)

rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    ))
])

rf_pipeline.fit(X_train, y_train)
rf_pred = rf_pipeline.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred, average='weighted')
rf_loss = log_loss(y_test, rf_pipeline.predict_proba(X_test))

print(f"\nRandom Forest Results:")
print(f"Accuracy: {rf_acc:.4f}")
print(f"F1-Score (weighted): {rf_f1:.4f}")
print(f"Log Loss: {rf_loss:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, rf_pred, target_names=le.classes_))

# Train Logistic Regression
print("\n" + "="*60)
print("Training Logistic Regression Classifier...")
print("="*60)

lr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight='balanced',
        solver='lbfgs',
        multi_class='multinomial'
    ))
])

lr_pipeline.fit(X_train, y_train)
lr_pred = lr_pipeline.predict(X_test)
lr_acc = accuracy_score(y_test, lr_pred)
lr_f1 = f1_score(y_test, lr_pred, average='weighted')
lr_loss = log_loss(y_test, lr_pipeline.predict_proba(X_test))

print(f"\nLogistic Regression Results:")
print(f"Accuracy: {lr_acc:.4f}")
print(f"F1-Score (weighted): {lr_f1:.4f}")
print(f"Log Loss: {lr_loss:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, lr_pred, target_names=le.classes_))

# Try XGBoost if available
try:
    from xgboost import XGBClassifier
    
    print("\n" + "="*60)
    print("Training XGBoost Classifier...")
    print("="*60)
    
    xgb_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(
            use_label_encoder=False,
            eval_metric='mlogloss',
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    xgb_pipeline.fit(X_train, y_train)
    xgb_pred = xgb_pipeline.predict(X_test)
    xgb_acc = accuracy_score(y_test, xgb_pred)
    xgb_f1 = f1_score(y_test, xgb_pred, average='weighted')
    xgb_loss = log_loss(y_test, xgb_pipeline.predict_proba(X_test))
    
    print(f"\nXGBoost Results:")
    print(f"Accuracy: {xgb_acc:.4f}")
    print(f"F1-Score (weighted): {xgb_f1:.4f}")
    print(f"Log Loss: {xgb_loss:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, xgb_pred, target_names=le.classes_))
    
    xgb_available = True
except ImportError:
    print("\nXGBoost not available, skipping...")
    xgb_available = False
    xgb_acc = 0

# Select best model
models = {
    'RandomForest': (rf_pipeline, rf_acc, rf_f1, rf_loss),
    'LogisticRegression': (lr_pipeline, lr_acc, lr_f1, lr_loss)
}

if xgb_available:
    models['XGBoost'] = (xgb_pipeline, xgb_acc, xgb_f1, xgb_loss)

best_model_name = max(models.keys(), key=lambda k: models[k][1])  # Best accuracy
best_model, best_acc, best_f1, best_loss = models[best_model_name]

print("\n" + "="*60)
print(f"Selected {best_model_name} as the best model")
print("="*60)
print(f"Final Accuracy: {best_acc:.4f}")
print(f"Final F1-Score: {best_f1:.4f}")
print(f"Final Log Loss: {best_loss:.4f}")

# Save model and dependencies
model_dir = Path("models")
model_dir.mkdir(parents=True, exist_ok=True)

joblib.dump(best_model, model_dir / "product_recommendation_model.pkl")
joblib.dump(le, model_dir / "product_label_encoder.pkl")
joblib.dump(features_for_model, model_dir / "product_feature_columns.pkl")

print("\n" + "="*60)
print("Model saved successfully!")
print("="*60)
print(f"   -> {model_dir / 'product_recommendation_model.pkl'}")
print(f"   -> {model_dir / 'product_label_encoder.pkl'}")
print(f"   -> {model_dir / 'product_feature_columns.pkl'}")
print("="*60)

