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

# Visualize product category distribution
print("\n" + "="*60)
print("PRODUCT CATEGORY DISTRIBUTION VISUALIZATION")
print("="*60)

import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Product Category Distribution Analysis', fontsize=16, fontweight='bold')

# 1. Bar chart - Category counts
category_counts = y.value_counts()
axes[0, 0].bar(category_counts.index, category_counts.values, color='steelblue', alpha=0.8, edgecolor='black')
axes[0, 0].set_title('Product Category Distribution (Count)', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Product Category', fontsize=12)
axes[0, 0].set_ylabel('Number of Customers', fontsize=12)
axes[0, 0].tick_params(axis='x', rotation=45)
axes[0, 0].grid(axis='y', alpha=0.3)
for i, v in enumerate(category_counts.values):
    axes[0, 0].text(i, v + 0.5, str(v), ha='center', va='bottom', fontweight='bold')

# 2. Pie chart - Category percentages
colors_pie = plt.cm.Set3(np.linspace(0, 1, len(category_counts)))
wedges, texts, autotexts = axes[0, 1].pie(category_counts.values, labels=category_counts.index,
                                          autopct='%1.1f%%', startangle=90, colors=colors_pie)
axes[0, 1].set_title('Product Category Distribution (Percentage)', fontsize=14, fontweight='bold')
for autotext in autotexts:
    autotext.set_color('black')
    autotext.set_fontweight('bold')

# 3. Horizontal bar - Top categories
top_categories = category_counts.head(10)
colors_hbar = plt.cm.viridis(np.linspace(0, 1, len(top_categories)))
axes[1, 0].barh(range(len(top_categories)), top_categories.values, color=colors_hbar, alpha=0.8, edgecolor='black')
axes[1, 0].set_yticks(range(len(top_categories)))
axes[1, 0].set_yticklabels(top_categories.index)
axes[1, 0].set_xlabel('Number of Customers', fontsize=12)
axes[1, 0].set_title('Top Product Categories', fontsize=14, fontweight='bold')
axes[1, 0].grid(axis='x', alpha=0.3)
for i, v in enumerate(top_categories.values):
    axes[1, 0].text(v + 0.5, i, str(v), va='center', fontweight='bold')

# 4. Category vs Spending (if available)
if 'total_spent' in df.columns:
    category_spending = df.groupby(target_col)['total_spent'].agg(['mean', 'sum']).sort_values('sum', ascending=False)
    x_pos = np.arange(len(category_spending))
    width = 0.35
    
    bars1 = axes[1, 1].bar(x_pos - width/2, category_spending['mean'], width,
                           label='Average Spending', color='steelblue', alpha=0.8, edgecolor='black')
    ax_twin = axes[1, 1].twinx()
    bars2 = ax_twin.bar(x_pos + width/2, category_spending['sum'], width,
                        label='Total Spending', color='coral', alpha=0.8, edgecolor='black')
    
    axes[1, 1].set_xlabel('Product Category', fontsize=12)
    axes[1, 1].set_ylabel('Average Spending ($)', fontsize=12, color='steelblue')
    ax_twin.set_ylabel('Total Spending ($)', fontsize=12, color='coral')
    axes[1, 1].set_title('Spending by Product Category', fontsize=14, fontweight='bold')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(category_spending.index, rotation=45, ha='right')
    axes[1, 1].tick_params(axis='y', labelcolor='steelblue')
    ax_twin.tick_params(axis='y', labelcolor='coral')
    axes[1, 1].grid(axis='y', alpha=0.3)
    axes[1, 1].legend(loc='upper left')
    ax_twin.legend(loc='upper right')
else:
    axes[1, 1].text(0.5, 0.5, 'Spending data not available', 
                    ha='center', va='center', fontsize=12, transform=axes[1, 1].transAxes)
    axes[1, 1].axis('off')

plt.tight_layout()

# Save visualization
output_dir = Path('merge-output')
output_dir.mkdir(exist_ok=True)
output_path = output_dir / 'product_category_distribution_training.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"✓ Product category distribution visualization saved to: {output_path}")
plt.close()  # Close to avoid displaying in non-interactive mode

# Plot 2: Engagement Score Distribution
print("\n" + "="*60)
print("ENGAGEMENT SCORE DISTRIBUTION VISUALIZATION")
print("="*60)

engagement_cols = [c for c in df.columns if 'engagement' in c.lower()]
if engagement_cols:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Engagement Score Distribution Analysis', fontsize=16, fontweight='bold')
    
    # Use the first engagement column found
    eng_col = engagement_cols[0]
    df_eng = df[df[eng_col].notna()].copy()
    
    if len(df_eng) > 0:
        # 1. Histogram of engagement scores
        axes[0, 0].hist(df_eng[eng_col], bins=20, color='steelblue', alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(df_eng[eng_col].mean(), color='red', linestyle='--', linewidth=2, 
                          label=f'Mean: {df_eng[eng_col].mean():.2f}')
        axes[0, 0].axvline(df_eng[eng_col].median(), color='green', linestyle='--', linewidth=2,
                          label=f'Median: {df_eng[eng_col].median():.2f}')
        axes[0, 0].set_title('Engagement Score Distribution', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Engagement Score', fontsize=12)
        axes[0, 0].set_ylabel('Frequency', fontsize=12)
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # 2. Box plot
        axes[0, 1].boxplot(df_eng[eng_col], vert=True, patch_artist=True,
                           boxprops=dict(facecolor='lightblue', alpha=0.7))
        axes[0, 1].set_title('Engagement Score Box Plot', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('Engagement Score', fontsize=12)
        axes[0, 1].grid(axis='y', alpha=0.3)
        # Add statistics text
        stats_text = f"Min: {df_eng[eng_col].min():.2f}\nQ1: {df_eng[eng_col].quantile(0.25):.2f}\nMedian: {df_eng[eng_col].median():.2f}\nQ3: {df_eng[eng_col].quantile(0.75):.2f}\nMax: {df_eng[eng_col].max():.2f}"
        axes[0, 1].text(1.15, df_eng[eng_col].median(), stats_text, 
                       va='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 3. Engagement by Product Category (if available)
        if target_col in df_eng.columns:
            category_engagement = df_eng.groupby(target_col)[eng_col].agg(['mean', 'std', 'count']).sort_values('mean', ascending=False)
            x_pos = np.arange(len(category_engagement))
            bars = axes[1, 0].bar(x_pos, category_engagement['mean'], 
                                 yerr=category_engagement['std'],
                                 color='coral', alpha=0.8, edgecolor='black', capsize=5)
            axes[1, 0].set_xticks(x_pos)
            axes[1, 0].set_xticklabels(category_engagement.index, rotation=45, ha='right')
            axes[1, 0].set_title('Average Engagement Score by Product Category', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Product Category', fontsize=12)
            axes[1, 0].set_ylabel('Average Engagement Score', fontsize=12)
            axes[1, 0].grid(axis='y', alpha=0.3)
            # Add value labels
            for i, (bar, mean_val) in enumerate(zip(bars, category_engagement['mean'])):
                axes[1, 0].text(bar.get_x() + bar.get_width()/2., mean_val + category_engagement['std'].iloc[i],
                               f'{mean_val:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        else:
            axes[1, 0].text(0.5, 0.5, 'Product category data not available', 
                           ha='center', va='center', fontsize=12, transform=axes[1, 0].transAxes)
            axes[1, 0].axis('off')
        
        # 4. Engagement Score vs Spending (if available)
        if 'total_spent' in df_eng.columns:
            axes[1, 1].scatter(df_eng[eng_col], df_eng['total_spent'], 
                              alpha=0.6, s=100, color='steelblue', edgecolors='black', linewidth=0.5)
            axes[1, 1].set_xlabel('Engagement Score', fontsize=12)
            axes[1, 1].set_ylabel('Total Spent ($)', fontsize=12)
            axes[1, 1].set_title('Engagement Score vs Total Spent', fontsize=14, fontweight='bold')
            axes[1, 1].grid(alpha=0.3)
            # Add correlation coefficient
            corr = df_eng[[eng_col, 'total_spent']].corr().iloc[0, 1]
            axes[1, 1].text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                           transform=axes[1, 1].transAxes, fontsize=12,
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                           verticalalignment='top')
        else:
            axes[1, 1].text(0.5, 0.5, 'Spending data not available', 
                           ha='center', va='center', fontsize=12, transform=axes[1, 1].transAxes)
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # Save visualization
        output_path_eng = output_dir / 'engagement_score_distribution.png'
        plt.savefig(output_path_eng, dpi=150, bbox_inches='tight')
        print(f"✓ Engagement score distribution visualization saved to: {output_path_eng}")
        
        # Print summary statistics
        print(f"\nEngagement Score Statistics:")
        print(f"  Mean: {df_eng[eng_col].mean():.2f}")
        print(f"  Median: {df_eng[eng_col].median():.2f}")
        print(f"  Std Dev: {df_eng[eng_col].std():.2f}")
        print(f"  Min: {df_eng[eng_col].min():.2f}")
        print(f"  Max: {df_eng[eng_col].max():.2f}")
        print(f"  Range: {df_eng[eng_col].max() - df_eng[eng_col].min():.2f}")
        
        plt.close()
    else:
        print(f"⚠ No valid engagement score data found in column: {eng_col}")
else:
    print("⚠ No engagement score columns found in the dataset.")

# Plot 3: Feature Correlation Heatmap
print("\n" + "="*60)
print("FEATURE CORRELATION HEATMAP")
print("="*60)

# Get numerical features for correlation
numerical_features_for_corr = [
    'total_spent', 'avg_purchase_amount', 'avg_customer_rating',
    'avg_engagement_score', 'avg_purchase_interest_score',
    'total_purchases', 'platform_diversity', 'customer_value_score'
]

# Filter to available columns
available_num_corr = [c for c in numerical_features_for_corr if c in df.columns]

if len(available_num_corr) > 1:
    # Calculate correlation matrix
    corr_matrix = df[available_num_corr].corr()
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(corr_matrix.columns)))
    ax.set_yticks(np.arange(len(corr_matrix.columns)))
    ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(corr_matrix.columns, fontsize=10)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Correlation Coefficient', fontsize=12, rotation=270, labelpad=20)
    
    # Add text annotations with correlation values
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            corr_value = corr_matrix.iloc[i, j]
            text_color = 'white' if abs(corr_value) > 0.5 else 'black'
            ax.text(j, i, f'{corr_value:.2f}',
                   ha='center', va='center', color=text_color, fontweight='bold', fontsize=9)
    
    ax.set_title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    # Save visualization
    output_path_corr = output_dir / 'feature_correlation_heatmap.png'
    plt.savefig(output_path_corr, dpi=150, bbox_inches='tight')
    print(f"✓ Feature correlation heatmap saved to: {output_path_corr}")
    
    # Print top correlations
    print(f"\nTop Positive Correlations:")
    corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_pairs.append((
                corr_matrix.columns[i],
                corr_matrix.columns[j],
                corr_matrix.iloc[i, j]
            ))
    corr_pairs_sorted = sorted(corr_pairs, key=lambda x: abs(x[2]), reverse=True)
    for feat1, feat2, corr_val in corr_pairs_sorted[:5]:
        print(f"  {feat1} ↔ {feat2}: {corr_val:.3f}")
    
    plt.close()
else:
    print("⚠ Not enough numerical features available for correlation analysis.")

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

