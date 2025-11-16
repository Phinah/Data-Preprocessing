"""
Product Prediction Script
Standalone script to predict product category for a customer
"""
import os
import joblib
import pandas as pd
from pathlib import Path

MODEL_DIR = Path('models')
XGB_PATH = MODEL_DIR / 'product_model_xgb.joblib'
RF_PATH = MODEL_DIR / 'product_recommendation_model.pkl'  # Updated to match actual filename
LE_PATH = MODEL_DIR / 'product_label_encoder.pkl'
MERGED_CSV = Path('merge-output/merged_data.csv')

# Choose model: prefer XGBoost if available, then RandomForest
if XGB_PATH.exists():
    model_path = XGB_PATH
    print(f'Loading XGBoost model...')
elif RF_PATH.exists():
    model_path = RF_PATH
    print(f'Loading RandomForest model...')
else:
    raise FileNotFoundError('No trained product model found in models/. Run scripts/product_recommendation.py first.')

if not LE_PATH.exists():
    raise FileNotFoundError('Label encoder not found. Run scripts/product_recommendation.py to generate label encoder.')

model = joblib.load(model_path)
le = joblib.load(LE_PATH)
print(f'Loaded model from: {model_path}')

# Load merged data to build a realistic sample (use first non-null row)
if not MERGED_CSV.exists():
    raise FileNotFoundError('merge-output/merged_data.csv not found. Run scripts/merge_datasets.py first.')

df = pd.read_csv(MERGED_CSV)

# Load feature columns used during training
feature_cols_path = MODEL_DIR / 'product_feature_columns.pkl'
if feature_cols_path.exists():
    feature_cols = joblib.load(feature_cols_path)
    print(f'Using feature columns from training: {len(feature_cols)} features')
else:
    # Fallback: use candidate features
    print('Warning: Feature columns file not found, using candidate features')
    feature_cols = ['total_spent', 'avg_purchase_amount', 'avg_customer_rating',
                   'avg_engagement_score', 'avg_purchase_interest_score',
                   'total_purchases', 'platform_diversity', 'customer_value_score',
                   'social_media_platform', 'dominant_sentiment', 'platforms_used']
    # Filter to available columns
    feature_cols = [c for c in feature_cols if c in df.columns]

if len(feature_cols) == 0:
    raise ValueError('No candidate features found in merged_data.csv')

# Take the first row that has any non-null values for these features
sample_row = df[feature_cols].dropna(how='all').iloc[[0]].copy()

# Fill numeric NaNs with median and categorical with mode to match training preprocessing
for col in sample_row.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        median_val = df[col].median()
        sample_row[col] = sample_row[col].fillna(median_val)
    else:
        mode_val = df[col].mode(dropna=True)
        sample_row[col] = sample_row[col].fillna(mode_val[0] if not mode_val.empty else 'Unknown')

print('\n' + '='*60)
print('PRODUCT PREDICTION')
print('='*60)
print('\nUsing sample input:')
print(sample_row.to_dict(orient='records')[0])

# Predict
pred_encoded = model.predict(sample_row)

# If model produced a single encoded prediction, ensure it's iterable
try:
    pred_labels = le.inverse_transform(pred_encoded)
except Exception:
    # maybe model returned string classes already
    pred_labels = pred_encoded

print('\n' + '='*60)
print('PREDICTION RESULT')
print('='*60)
print(f'Predicted Product Category: {pred_labels[0] if hasattr(pred_labels, "__iter__") and not isinstance(pred_labels, str) else pred_labels}')

# Get probabilities if available
try:
    proba = model.predict_proba(sample_row)[0]
    print(f'\nConfidence scores:')
    for i, class_name in enumerate(le.classes_):
        print(f'  {class_name}: {proba[i]:.2%}')
except:
    pass

print('='*60)

