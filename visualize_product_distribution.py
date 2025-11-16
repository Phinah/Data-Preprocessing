"""
Product Category Distribution Visualization
Creates graphs showing product category distribution from merged data
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Load merged dataset
merged_path = Path('merge-output/merged_data.csv')
if not merged_path.exists():
    raise FileNotFoundError("Run scripts/merge_datasets.py first → merged_data.csv missing")

print("="*60)
print("PRODUCT CATEGORY DISTRIBUTION ANALYSIS")
print("="*60)

df = pd.read_csv(merged_path)
print(f"\nLoaded merged dataset: {df.shape}")

# Check for product category columns
category_cols = []
if 'favorite_category' in df.columns:
    category_cols.append('favorite_category')
if 'last_category_purchased' in df.columns:
    category_cols.append('last_category_purchased')
if 'product_category' in df.columns:
    category_cols.append('product_category')

if not category_cols:
    print("\n⚠ No product category columns found in the dataset.")
    print("Available columns:", df.columns.tolist()[:10])
    exit(1)

# Use the first available category column
target_col = category_cols[0]
print(f"\nUsing '{target_col}' for category distribution")

# Filter out null values
df_clean = df[df[target_col].notna()].copy()
print(f"Valid category entries: {len(df_clean)} out of {len(df)}")

if len(df_clean) == 0:
    print("⚠ No valid category data found.")
    exit(1)

# Create comprehensive visualization
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. Product Category Distribution (Bar Chart)
ax1 = fig.add_subplot(gs[0, :2])
category_counts = df_clean[target_col].value_counts()
colors_bar = plt.cm.Set3(np.linspace(0, 1, len(category_counts)))
bars = ax1.bar(category_counts.index, category_counts.values, color=colors_bar, alpha=0.8, edgecolor='black')
ax1.set_title('Product Category Distribution', fontsize=16, fontweight='bold', pad=20)
ax1.set_xlabel('Product Category', fontsize=12)
ax1.set_ylabel('Number of Customers', fontsize=12)
ax1.tick_params(axis='x', rotation=45, labelsize=10)
ax1.grid(axis='y', alpha=0.3)
# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}',
             ha='center', va='bottom', fontweight='bold', fontsize=10)

# 2. Product Category Distribution (Pie Chart)
ax2 = fig.add_subplot(gs[0, 2])
colors_pie = plt.cm.Set3(np.linspace(0, 1, len(category_counts)))
wedges, texts, autotexts = ax2.pie(category_counts.values, labels=category_counts.index, 
                                    autopct='%1.1f%%', startangle=90, colors=colors_pie,
                                    textprops={'fontsize': 9})
ax2.set_title('Category Distribution\n(Percentage)', fontsize=14, fontweight='bold', pad=15)
for autotext in autotexts:
    autotext.set_color('black')
    autotext.set_fontweight('bold')

# 3. Category Count Statistics
ax3 = fig.add_subplot(gs[1, 0])
stats_data = {
    'Total Categories': len(category_counts),
    'Total Customers': len(df_clean),
    'Most Popular': category_counts.max(),
    'Least Popular': category_counts.min()
}
stats_df = pd.DataFrame(list(stats_data.items()), columns=['Metric', 'Value'])
ax3.axis('tight')
ax3.axis('off')
table = ax3.table(cellText=stats_df.values, colLabels=stats_df.columns,
                  cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2)
for i in range(len(stats_df) + 1):
    for j in range(len(stats_df.columns)):
        cell = table[(i, j)]
        if i == 0:
            cell.set_facecolor('#4CAF50')
            cell.set_text_props(weight='bold', color='white')
        else:
            cell.set_facecolor('#E8F5E9')
ax3.set_title('Distribution Statistics', fontsize=14, fontweight='bold', pad=10)

# 4. Top Categories (Horizontal Bar)
ax4 = fig.add_subplot(gs[1, 1:])
top_n = min(10, len(category_counts))
top_categories = category_counts.head(top_n)
colors_hbar = plt.cm.viridis(np.linspace(0, 1, len(top_categories)))
ax4.barh(range(len(top_categories)), top_categories.values, color=colors_hbar, alpha=0.8, edgecolor='black')
ax4.set_yticks(range(len(top_categories)))
ax4.set_yticklabels(top_categories.index, fontsize=10)
ax4.set_xlabel('Number of Customers', fontsize=12)
ax4.set_title(f'Top {top_n} Product Categories', fontsize=14, fontweight='bold')
ax4.grid(axis='x', alpha=0.3)
# Add value labels
for i, v in enumerate(top_categories.values):
    ax4.text(v + 0.5, i, f'{int(v)}', va='center', fontweight='bold', fontsize=10)

# 5. Category Distribution by Spending (if available)
if 'total_spent' in df_clean.columns:
    ax5 = fig.add_subplot(gs[2, :])
    category_spending = df_clean.groupby(target_col)['total_spent'].agg(['mean', 'sum', 'count']).sort_values('sum', ascending=False)
    
    x_pos = np.arange(len(category_spending))
    width = 0.35
    
    bars1 = ax5.bar(x_pos - width/2, category_spending['mean'], width, 
                    label='Average Spending', color='steelblue', alpha=0.8, edgecolor='black')
    ax5_twin = ax5.twinx()
    bars2 = ax5_twin.bar(x_pos + width/2, category_spending['sum'], width,
                         label='Total Spending', color='coral', alpha=0.8, edgecolor='black')
    
    ax5.set_xlabel('Product Category', fontsize=12)
    ax5.set_ylabel('Average Spending ($)', fontsize=12, color='steelblue')
    ax5_twin.set_ylabel('Total Spending ($)', fontsize=12, color='coral')
    ax5.set_title('Spending Analysis by Product Category', fontsize=14, fontweight='bold')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(category_spending.index, rotation=45, ha='right', fontsize=9)
    ax5.tick_params(axis='y', labelcolor='steelblue')
    ax5_twin.tick_params(axis='y', labelcolor='coral')
    ax5.grid(axis='y', alpha=0.3)
    
    # Add legends
    ax5.legend(loc='upper left')
    ax5_twin.legend(loc='upper right')
else:
    ax5 = fig.add_subplot(gs[2, :])
    ax5.text(0.5, 0.5, 'Spending data not available\nfor category analysis', 
             ha='center', va='center', fontsize=14, transform=ax5.transAxes)
    ax5.axis('off')

plt.suptitle('Product Category Distribution Analysis', fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()

# Save the figure
output_dir = Path('merge-output')
output_dir.mkdir(exist_ok=True)
output_path = output_dir / 'product_category_distribution.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Visualization saved to: {output_path}")

# Plot 2: Engagement Score Distribution
print("\n" + "="*60)
print("ENGAGEMENT SCORE DISTRIBUTION ANALYSIS")
print("="*60)

engagement_cols = [c for c in df.columns if 'engagement' in c.lower()]
if engagement_cols:
    fig_eng = plt.figure(figsize=(16, 12))
    gs_eng = fig_eng.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Use the first engagement column found
    eng_col = engagement_cols[0]
    df_eng = df[df[eng_col].notna()].copy()
    
    if len(df_eng) > 0:
        # 1. Histogram of engagement scores
        ax1_eng = fig_eng.add_subplot(gs_eng[0, 0])
        ax1_eng.hist(df_eng[eng_col], bins=20, color='steelblue', alpha=0.7, edgecolor='black')
        ax1_eng.axvline(df_eng[eng_col].mean(), color='red', linestyle='--', linewidth=2, 
                       label=f'Mean: {df_eng[eng_col].mean():.2f}')
        ax1_eng.axvline(df_eng[eng_col].median(), color='green', linestyle='--', linewidth=2,
                       label=f'Median: {df_eng[eng_col].median():.2f}')
        ax1_eng.set_title('Engagement Score Distribution', fontsize=14, fontweight='bold')
        ax1_eng.set_xlabel('Engagement Score', fontsize=12)
        ax1_eng.set_ylabel('Frequency', fontsize=12)
        ax1_eng.legend()
        ax1_eng.grid(alpha=0.3)
        
        # 2. Box plot
        ax2_eng = fig_eng.add_subplot(gs_eng[0, 1])
        ax2_eng.boxplot(df_eng[eng_col], vert=True, patch_artist=True,
                       boxprops=dict(facecolor='lightblue', alpha=0.7))
        ax2_eng.set_title('Engagement Score Box Plot', fontsize=14, fontweight='bold')
        ax2_eng.set_ylabel('Engagement Score', fontsize=12)
        ax2_eng.grid(axis='y', alpha=0.3)
        # Add statistics text
        stats_text = f"Min: {df_eng[eng_col].min():.2f}\nQ1: {df_eng[eng_col].quantile(0.25):.2f}\nMedian: {df_eng[eng_col].median():.2f}\nQ3: {df_eng[eng_col].quantile(0.75):.2f}\nMax: {df_eng[eng_col].max():.2f}"
        ax2_eng.text(1.15, df_eng[eng_col].median(), stats_text, 
                    va='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 3. Engagement by Product Category
        ax3_eng = fig_eng.add_subplot(gs_eng[1, 0])
        if target_col in df_eng.columns:
            category_engagement = df_eng.groupby(target_col)[eng_col].agg(['mean', 'std', 'count']).sort_values('mean', ascending=False)
            x_pos = np.arange(len(category_engagement))
            bars = ax3_eng.bar(x_pos, category_engagement['mean'], 
                             yerr=category_engagement['std'],
                             color='coral', alpha=0.8, edgecolor='black', capsize=5)
            ax3_eng.set_xticks(x_pos)
            ax3_eng.set_xticklabels(category_engagement.index, rotation=45, ha='right')
            ax3_eng.set_title('Average Engagement Score by Product Category', fontsize=14, fontweight='bold')
            ax3_eng.set_xlabel('Product Category', fontsize=12)
            ax3_eng.set_ylabel('Average Engagement Score', fontsize=12)
            ax3_eng.grid(axis='y', alpha=0.3)
            # Add value labels
            for i, (bar, mean_val) in enumerate(zip(bars, category_engagement['mean'])):
                ax3_eng.text(bar.get_x() + bar.get_width()/2., mean_val + category_engagement['std'].iloc[i],
                           f'{mean_val:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        else:
            ax3_eng.text(0.5, 0.5, 'Product category data not available', 
                        ha='center', va='center', fontsize=12, transform=ax3_eng.transAxes)
            ax3_eng.axis('off')
        
        # 4. Engagement Score vs Spending
        ax4_eng = fig_eng.add_subplot(gs_eng[1, 1])
        if 'total_spent' in df_eng.columns:
            ax4_eng.scatter(df_eng[eng_col], df_eng['total_spent'], 
                          alpha=0.6, s=100, color='steelblue', edgecolors='black', linewidth=0.5)
            ax4_eng.set_xlabel('Engagement Score', fontsize=12)
            ax4_eng.set_ylabel('Total Spent ($)', fontsize=12)
            ax4_eng.set_title('Engagement Score vs Total Spent', fontsize=14, fontweight='bold')
            ax4_eng.grid(alpha=0.3)
            # Add correlation coefficient
            corr = df_eng[[eng_col, 'total_spent']].corr().iloc[0, 1]
            ax4_eng.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                       transform=ax4_eng.transAxes, fontsize=12,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                       verticalalignment='top')
        else:
            ax4_eng.text(0.5, 0.5, 'Spending data not available', 
                        ha='center', va='center', fontsize=12, transform=ax4_eng.transAxes)
            ax4_eng.axis('off')
        
        fig_eng.suptitle('Engagement Score Distribution Analysis', fontsize=18, fontweight='bold', y=0.995)
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
        
        plt.show()
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
    fig_corr = plt.figure(figsize=(14, 12))
    ax_corr = fig_corr.add_subplot(111)
    im = ax_corr.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    
    # Set ticks and labels
    ax_corr.set_xticks(np.arange(len(corr_matrix.columns)))
    ax_corr.set_yticks(np.arange(len(corr_matrix.columns)))
    ax_corr.set_xticklabels(corr_matrix.columns, rotation=45, ha='right', fontsize=10)
    ax_corr.set_yticklabels(corr_matrix.columns, fontsize=10)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax_corr, fraction=0.046, pad=0.04)
    cbar.set_label('Correlation Coefficient', fontsize=12, rotation=270, labelpad=20)
    
    # Add text annotations with correlation values
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            corr_value = corr_matrix.iloc[i, j]
            text_color = 'white' if abs(corr_value) > 0.5 else 'black'
            ax_corr.text(j, i, f'{corr_value:.2f}',
                       ha='center', va='center', color=text_color, fontweight='bold', fontsize=9)
    
    ax_corr.set_title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
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
    
    plt.show()
else:
    print("⚠ Not enough numerical features available for correlation analysis.")

# Display summary
print("\n" + "="*60)
print("CATEGORY DISTRIBUTION SUMMARY")
print("="*60)
print(f"\nTotal categories: {len(category_counts)}")
print(f"Total customers: {len(df_clean)}")
print(f"\nTop 5 categories:")
for i, (cat, count) in enumerate(category_counts.head(5).items(), 1):
    pct = (count / len(df_clean)) * 100
    print(f"  {i}. {cat}: {count} customers ({pct:.1f}%)")

plt.show()
print("\n✓ Visualization complete!")

