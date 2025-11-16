"""
Merge Customer Social Profiles and Transactions Datasets
Creates a merged dataset with feature engineering for product recommendation
"""
import os
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Create output directory
output_dir = Path('merge-output')
output_dir.mkdir(exist_ok=True)

# Paths to datasets
social_profiles_path = "customer_social_profiles - customer_social_profiles.csv"
transactions_path = "customer_transactions - customer_transactions.csv"

print("="*60)
print("DATA MERGE PIPELINE")
print("="*60)

# Load datasets
print("\nLoading datasets...")
social = pd.read_csv(social_profiles_path)
trans = pd.read_csv(transactions_path)

print(f"Social profiles shape: {social.shape}")
print(f"Transactions shape: {trans.shape}")

# Clean column names
social.columns = social.columns.str.strip()
trans.columns = trans.columns.str.strip()

print(f"\nSocial profiles columns: {social.columns.tolist()}")
print(f"Transactions columns: {trans.columns.tolist()}")

# Step 1: Create customer ID mapping (legacy to new format)
trans['customer_id_new'] = 'A' + trans['customer_id_legacy'].astype(str)

print(f"\nSample ID mapping:")
print(trans[['customer_id_legacy', 'customer_id_new']].head())

# Step 2: Aggregate transaction data by customer
print("\nAggregating transaction features...")
transaction_features = trans.groupby('customer_id_new').agg({
    'transaction_id': 'count',
    'purchase_amount': ['sum', 'mean', 'std'],
    'customer_rating': 'mean',
    'product_category': lambda x: x.mode()[0] if len(x.mode()) > 0 else None
}).reset_index()

# Flatten column names
transaction_features.columns = [
    'customer_id_new',
    'total_purchases',
    'total_spent',
    'avg_purchase_amount',
    'std_purchase_amount',
    'avg_customer_rating',
    'favorite_category'
]

# Get last purchase information
trans['purchase_date'] = pd.to_datetime(trans['purchase_date'], errors='coerce')
last_purchase = trans.sort_values('purchase_date').groupby('customer_id_new').last()[
    ['product_category', 'purchase_date']
].reset_index()
last_purchase.columns = ['customer_id_new', 'last_category_purchased', 'last_purchase_date']

transaction_features = transaction_features.merge(last_purchase, on='customer_id_new', how='left')

# Step 3: Aggregate social profile data by customer
print("Aggregating social profile features...")
social_features = social.groupby('customer_id_new').agg({
    'social_media_platform': ['count', lambda x: x.nunique()],
    'engagement_score': ['mean', 'std', 'max'],
    'purchase_interest_score': ['mean', 'std', 'max'],
    'review_sentiment': lambda x: x.mode()[0] if len(x.mode()) > 0 else None
}).reset_index()

# Flatten column names
social_features.columns = [
    'customer_id_new',
    'total_social_profiles',
    'platform_diversity',
    'avg_engagement_score',
    'std_engagement_score',
    'max_engagement_score',
    'avg_purchase_interest_score',
    'std_purchase_interest_score',
    'max_purchase_interest_score',
    'dominant_sentiment'
]

# Calculate sentiment distribution
sentiment_pivot = social.groupby(['customer_id_new', 'review_sentiment']).size().unstack(fill_value=0)
sentiment_pivot.columns = [f'sentiment_{col.lower()}_count' for col in sentiment_pivot.columns]
sentiment_pivot = sentiment_pivot.reset_index()

social_features = social_features.merge(sentiment_pivot, on='customer_id_new', how='left')

# Get platform list
platform_list = social.groupby('customer_id_new')['social_media_platform'].apply(
    lambda x: ', '.join(x.unique())
).reset_index()
platform_list.columns = ['customer_id_new', 'platforms_used']

social_features = social_features.merge(platform_list, on='customer_id_new', how='left')

# Step 4: Merge transaction and social features
print("\nMerging datasets...")
merged = social_features.merge(transaction_features, on='customer_id_new', how='inner')

# Step 5: Feature engineering
print("Engineering additional features...")
merged['purchase_date'] = pd.to_datetime(merged['last_purchase_date'], errors='coerce')
merged['purchase_month'] = merged['purchase_date'].dt.month
merged['purchase_day_of_week'] = merged['purchase_date'].dt.dayofweek
merged['is_weekend'] = merged['purchase_day_of_week'] >= 5

# Calculate purchase frequency (if we had more date data)
merged['purchase_frequency'] = merged['total_purchases']  # Simplified

# Customer value score
merged['customer_value_score'] = (
    merged['total_spent'] * 0.4 +
    merged['avg_engagement_score'] * 0.3 +
    merged['avg_purchase_interest_score'] * 0.3
)

# Step 6: Handle missing values
print("Handling missing values...")
# Fill numeric columns with median
numeric_cols = merged.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    merged[col] = merged[col].fillna(merged[col].median())

# Fill categorical columns with mode
categorical_cols = merged.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if col != 'customer_id_new':
        merged[col] = merged[col].fillna(merged[col].mode()[0] if len(merged[col].mode()) > 0 else 'Unknown')

# Step 7: Remove duplicates
dupes_before = merged.duplicated().sum()
merged = merged.drop_duplicates()
dupes_after = merged.duplicated().sum()

# Step 8: Save merged dataset
merged.to_csv(output_dir / 'merged_data.csv', index=False)

print(f"\n{'='*60}")
print("MERGE COMPLETE")
print(f"{'='*60}")
print(f"Merged dataset shape: {merged.shape}")
print(f"Duplicates removed: {dupes_before - dupes_after}")

# Step 9: Validation report
validation_file = output_dir / 'merge_validation.txt'
with open(validation_file, 'w') as f:
    f.write("MERGE VALIDATION REPORT\n")
    f.write("="*60 + "\n\n")
    f.write(f"Original transactions: {len(trans)}\n")
    f.write(f"Original social profiles: {len(social)}\n")
    f.write(f"Merged dataset rows: {len(merged)}\n")
    f.write(f"Unique customers in merged: {merged['customer_id_new'].nunique()}\n")
    f.write(f"Duplicates before: {dupes_before}\n")
    f.write(f"Duplicates after: {dupes_after}\n\n")
    
    f.write("Missing values:\n")
    nulls = merged.isnull().sum()
    for col, n in nulls.items():
        if n > 0:
            f.write(f"  {col}: {n}\n")
    
    f.write("\nProduct category distribution:\n")
    if 'favorite_category' in merged.columns:
        vc = merged['favorite_category'].value_counts()
        for k, v in vc.items():
            f.write(f"  {k}: {v}\n")

print(f"Validation report saved: {validation_file}")

# Step 10: EDA Visualizations
print("\nGenerating EDA visualizations...")

# Distribution of purchase amount
plt.figure(figsize=(10, 6))
sns.histplot(merged['total_spent'].dropna(), kde=True, bins=30)
plt.title('Distribution of Total Spent per Customer')
plt.xlabel('Total Spent')
plt.ylabel('Frequency')
plt.savefig(output_dir / 'plot_total_spent_dist.png', dpi=150, bbox_inches='tight')
plt.close()

# Purchase amount by category
if 'favorite_category' in merged.columns:
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='favorite_category', y='total_spent', data=merged)
    plt.title('Total Spent by Favorite Product Category')
    plt.xticks(rotation=45)
    plt.savefig(output_dir / 'plot_spent_by_category.png', dpi=150, bbox_inches='tight')
    plt.close()

# Correlation heatmap
numeric_cols_eda = ['total_spent', 'avg_purchase_amount', 'avg_customer_rating',
                   'avg_engagement_score', 'avg_purchase_interest_score', 'total_purchases']
available_numeric = [c for c in numeric_cols_eda if c in merged.columns]

if len(available_numeric) > 1:
    plt.figure(figsize=(10, 8))
    corr_matrix = merged[available_numeric].corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0)
    plt.title('Correlation Matrix of Key Features')
    plt.savefig(output_dir / 'plot_correlations.png', dpi=150, bbox_inches='tight')
    plt.close()

# Engagement vs Purchase Interest
plt.figure(figsize=(10, 6))
sns.scatterplot(data=merged, x='avg_engagement_score', y='avg_purchase_interest_score',
               hue='favorite_category' if 'favorite_category' in merged.columns else None)
plt.title('Engagement Score vs Purchase Interest Score')
plt.xlabel('Average Engagement Score')
plt.ylabel('Average Purchase Interest Score')
plt.savefig(output_dir / 'plot_engagement_vs_interest.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"EDA plots saved in {output_dir}/")
print("\n" + "="*60)
print("Data merge pipeline complete!")
print("="*60)

