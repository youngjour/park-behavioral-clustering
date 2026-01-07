"""
Revised Behavioral Clustering with Statistical Significance Testing

Implements two clustering approaches:
1. Major Features Only (parsimonious)
2. PCA-based Comprehensive (recommended)

Then tests statistical significance of all features across clusters.

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy.stats import f_oneway, kruskal
from scipy.cluster.hierarchy import dendrogram, linkage

import warnings
warnings.filterwarnings('ignore')

# Setup paths
interim_dir = Path('../interim')
results_dir = Path('../results')
docs_dir = Path('../docs')

print("=" * 80)
print("REVISED BEHAVIORAL CLUSTERING WITH SIGNIFICANCE TESTING")
print("=" * 80)

# =============================================================================
# STEP 1: LOAD DATA
# =============================================================================

print("\n1. Loading behavioral features from visitor data...")

# Load existing behavioral clustering results (31 features)
df_behavioral = pd.read_csv(results_dir / 'enhanced_behavioral_diagnostic_dataset_FINAL.csv')

print(f"   Loaded {len(df_behavioral)} parks")
print(f"   Total columns: {len(df_behavioral.columns)}")

# Identify behavioral features (from 10-minute visitor data)
behavioral_features = [
    # Temporal patterns (from visitor data)
    'avg_population', 'max_population', 'population_std', 'population_cv',
    'peak_hour', 'peak_intensity',
    'morning_usage', 'afternoon_usage', 'evening_usage', 'night_usage',
    'weekday_weekend_ratio', 'weekend_preference', 'usage_consistency',
    'morning_rush_intensity', 'evening_rush_intensity',
    # Demographic patterns (from visitor data)
    'male_dominance', 'gender_balance',
    'youth_orientation', 'senior_orientation', 'working_age_orientation',
    'age_diversity', 'local_dominance', 'tourist_attraction',
    'demographic_stability',
    # Spatial behavioral (from visitor data + park data)
    'area_km2', 'population_density', 'distance_to_center',
    'centrality_score', 'temporal_catchment', 'weekly_catchment',
    'regional_influence'
]

# Filter to available features
available_behavioral = [f for f in behavioral_features if f in df_behavioral.columns]
print(f"   Available behavioral features: {len(available_behavioral)}")

# =============================================================================
# STEP 2: APPROACH 1 - MAJOR FEATURES ONLY (PARSIMONIOUS)
# =============================================================================

print("\n" + "=" * 80)
print("APPROACH 1: MAJOR FEATURES ONLY (Parsimonious Clustering)")
print("=" * 80)

# Define major features
major_features = [
    'avg_population',          # Overall usage level
    'max_population',          # Peak capacity
    'peak_hour',               # Temporal preference
    'weekday_weekend_ratio',   # Weekly pattern
    'tourist_attraction'       # Visitor origin
]

# Check availability
available_major = [f for f in major_features if f in df_behavioral.columns]
print(f"\nUsing {len(available_major)} major features:")
for f in available_major:
    print(f"  - {f}")

# Extract and scale
X_major = df_behavioral[available_major].fillna(df_behavioral[available_major].median())
scaler_major = StandardScaler()
X_major_scaled = scaler_major.fit_transform(X_major)

# Determine optimal k
print("\nDetermining optimal number of clusters...")
k_range = range(2, 8)
silhouette_scores_major = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_major_scaled)
    score = silhouette_score(X_major_scaled, labels)
    silhouette_scores_major.append(score)
    print(f"  k={k}: silhouette = {score:.3f}")

optimal_k_major = k_range[np.argmax(silhouette_scores_major)]
print(f"\nOptimal k = {optimal_k_major} (silhouette = {max(silhouette_scores_major):.3f})")

# Apply clustering
kmeans_major = KMeans(n_clusters=optimal_k_major, random_state=42, n_init=10)
df_behavioral['cluster_major_features'] = kmeans_major.fit_predict(X_major_scaled)

print(f"\nCluster distribution:")
print(df_behavioral['cluster_major_features'].value_counts().sort_index())

# =============================================================================
# STEP 3: APPROACH 2 - PCA-BASED COMPREHENSIVE (RECOMMENDED)
# =============================================================================

print("\n" + "=" * 80)
print("APPROACH 2: PCA-BASED COMPREHENSIVE (Recommended)")
print("=" * 80)

# Extract all behavioral features
X_comprehensive = df_behavioral[available_behavioral].fillna(df_behavioral[available_behavioral].median())
scaler_comp = StandardScaler()
X_comprehensive_scaled = scaler_comp.fit_transform(X_comprehensive)

print(f"\nUsing {len(available_behavioral)} comprehensive behavioral features")

# Apply PCA
pca = PCA(random_state=42)
X_pca = pca.fit_transform(X_comprehensive_scaled)

# Determine number of components for 80-90% variance
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
n_components_80 = np.argmax(cumulative_variance >= 0.80) + 1
n_components_90 = np.argmax(cumulative_variance >= 0.90) + 1

print(f"\nPCA variance explained:")
print(f"  80% variance: {n_components_80} components")
print(f"  90% variance: {n_components_90} components")
print(f"  Total variance with {n_components_80} PCs: {cumulative_variance[n_components_80-1]:.1%}")

# Use n_components for 80% variance
n_components = n_components_80
X_pca_reduced = X_pca[:, :n_components]

print(f"\nUsing {n_components} principal components for clustering")

# Cluster on PCA components
print("\nDetermining optimal number of clusters (PCA)...")
silhouette_scores_pca = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_pca_reduced)
    score = silhouette_score(X_pca_reduced, labels)
    silhouette_scores_pca.append(score)
    print(f"  k={k}: silhouette = {score:.3f}")

optimal_k_pca = k_range[np.argmax(silhouette_scores_pca)]
print(f"\nOptimal k = {optimal_k_pca} (silhouette = {max(silhouette_scores_pca):.3f})")

# Apply clustering
kmeans_pca = KMeans(n_clusters=optimal_k_pca, random_state=42, n_init=10)
df_behavioral['cluster_pca_comprehensive'] = kmeans_pca.fit_predict(X_pca_reduced)

print(f"\nCluster distribution:")
print(df_behavioral['cluster_pca_comprehensive'].value_counts().sort_index())

# =============================================================================
# STEP 4: COMPARE CLUSTERING APPROACHES
# =============================================================================

print("\n" + "=" * 80)
print("CLUSTERING APPROACH COMPARISON")
print("=" * 80)

comparison = pd.DataFrame({
    'Approach': ['Major Features', 'PCA Comprehensive'],
    'N Features/Components': [len(available_major), n_components],
    'Feature:Sample Ratio': [f"{len(available_major)}:{len(df_behavioral)}",
                            f"{n_components}:{len(df_behavioral)}"],
    'Optimal K': [optimal_k_major, optimal_k_pca],
    'Silhouette Score': [f"{max(silhouette_scores_major):.3f}",
                        f"{max(silhouette_scores_pca):.3f}"],
    'Recommendation': ['Stakeholder reports', 'Publication/Analysis']
})

print("\n", comparison.to_string(index=False))

# =============================================================================
# STEP 5: STATISTICAL SIGNIFICANCE TESTING
# =============================================================================

print("\n" + "=" * 80)
print("STATISTICAL SIGNIFICANCE TESTING")
print("=" * 80)

# Use PCA-based clusters for significance testing (recommended approach)
cluster_labels = df_behavioral['cluster_pca_comprehensive'].values

print(f"\nTesting significance of features across {optimal_k_pca} clusters")
print("Method: One-way ANOVA + Kruskal-Wallis + Effect Size (η²)")

# Define feature categories
feature_categories = {
    'Behavioral': available_behavioral,
    'Environmental': [col for col in df_behavioral.columns
                     if any(x in col for x in ['facility', 'green', 'amenity', 'terrain'])],
    'Transportation': [col for col in df_behavioral.columns
                      if any(x in col for x in ['centrality', 'betweenness', 'closeness',
                                               'pagerank', 'accessibility', 'reachable',
                                               'network', 'nodes', 'walk', 'drive', 'transit'])],
    'Socioeconomic': [col for col in df_behavioral.columns
                     if any(x in col for x in ['income', 'employment', 'education'])]
}

# Test significance for all features
all_significance_results = []

for category, features in feature_categories.items():
    if len(features) == 0:
        continue

    print(f"\n{category} Features: {len(features)} features to test")

    for feature in features:
        if feature not in df_behavioral.columns:
            continue

        # Group by cluster
        groups = [df_behavioral[cluster_labels == c][feature].dropna().values
                 for c in np.unique(cluster_labels)]

        # Filter out small groups (need at least 2 samples for variance)
        valid_groups = [g for g in groups if len(g) >= 2]

        # Skip if fewer than 2 valid groups to compare
        if len(valid_groups) < 2:
            continue

        # ANOVA
        try:
            F_stat, p_anova = f_oneway(*valid_groups)
        except:
            F_stat, p_anova = np.nan, np.nan

        # Kruskal-Wallis (non-parametric)
        try:
            H_stat, p_kruskal = kruskal(*valid_groups)
        except:
            H_stat, p_kruskal = np.nan, np.nan

        # Effect size (eta-squared)
        try:
            grand_mean = df_behavioral[feature].mean()
            ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in valid_groups)
            ss_total = sum((df_behavioral[feature] - grand_mean)**2)
            eta_squared = ss_between / ss_total if ss_total > 0 else 0
        except:
            eta_squared = np.nan

        # Significance
        significant = (p_anova < 0.05) or (p_kruskal < 0.05)

        # Effect interpretation
        if eta_squared < 0.01:
            effect = 'Negligible'
        elif eta_squared < 0.06:
            effect = 'Small'
        elif eta_squared < 0.14:
            effect = 'Medium'
        else:
            effect = 'Large'

        all_significance_results.append({
            'feature': feature,
            'category': category,
            'F_statistic': F_stat,
            'p_anova': p_anova,
            'H_statistic': H_stat,
            'p_kruskal': p_kruskal,
            'eta_squared': eta_squared,
            'significant': significant,
            'effect_size': effect
        })

# Create results DataFrame
df_significance = pd.DataFrame(all_significance_results)

# Only sort if we have results
if len(df_significance) > 0:
    df_significance = df_significance.sort_values('eta_squared', ascending=False)
else:
    print("\nWarning: No significance results generated")
    df_significance = pd.DataFrame()  # Empty DataFrame with no columns

# =============================================================================
# STEP 6: GENERATE SIGNIFICANCE REPORT
# =============================================================================

print("\n" + "=" * 80)
print("FEATURE SIGNIFICANCE REPORT")
print("=" * 80)

# Overall summary
if len(df_significance) > 0 and 'significant' in df_significance.columns:
    n_total = len(df_significance)
    n_significant = len(df_significance[df_significance['significant']])
else:
    n_total = 0
    n_significant = 0
    print("\nNo features tested - significance testing skipped")

if n_total > 0:
    print(f"\nTotal features tested: {n_total}")
    print(f"Significant features (p < 0.05): {n_significant} ({100*n_significant/n_total:.1f}%)")

    # By category
    print("\n" + "-" * 80)
    print("SIGNIFICANCE BY CATEGORY")
    print("-" * 80)

    for category in ['Behavioral', 'Environmental', 'Transportation', 'Socioeconomic']:
        cat_data = df_significance[df_significance['category'] == category]
        if len(cat_data) == 0:
            continue

        n_cat = len(cat_data)
        n_sig = len(cat_data[cat_data['significant']])

        print(f"\n{category}:")
        print(f"  Features tested: {n_cat}")
        print(f"  Significant: {n_sig} ({100*n_sig/n_cat:.1f}%)" if n_cat > 0 else "  Significant: 0")

        # Top 5 by effect size
        top_features = cat_data.nlargest(min(5, len(cat_data)), 'eta_squared')
        if len(top_features) > 0:
            print(f"\n  Top features by effect size:")
            for idx, row in top_features.iterrows():
                sig_mark = "***" if row['p_anova'] < 0.001 else \
                          "**" if row['p_anova'] < 0.01 else \
                          "*" if row['p_anova'] < 0.05 else ""
                print(f"    {row['feature']:30s}  η²={row['eta_squared']:.3f} {sig_mark:3s} ({row['effect_size']})")
else:
    print("\nSignificance testing was skipped due to insufficient data or missing features.")

# =============================================================================
# STEP 7: SAVE RESULTS
# =============================================================================

print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

# Save clustering results
output_clustering = results_dir / 'behavioral_clustering_revised.csv'
df_behavioral.to_csv(output_clustering, index=False)
print(f"\nClustering results: {output_clustering}")

# Save significance testing results
if len(df_significance) > 0:
    output_significance = results_dir / 'feature_significance_results.csv'
    df_significance.to_csv(output_significance, index=False)
    print(f"Significance results: {output_significance}")
else:
    print("Significance results: (none - no features tested)")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Silhouette scores comparison
ax = axes[0, 0]
ax.plot(k_range, silhouette_scores_major, 'bo-', label='Major Features', linewidth=2)
ax.plot(k_range, silhouette_scores_pca, 'ro-', label='PCA Comprehensive', linewidth=2)
ax.set_xlabel('Number of Clusters (k)')
ax.set_ylabel('Silhouette Score')
ax.set_title('Clustering Quality: Major vs PCA Approach')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. PCA variance explained
ax = axes[0, 1]
ax.plot(range(1, len(pca.explained_variance_ratio_)+1),
       cumulative_variance, 'g-', linewidth=2)
ax.axhline(y=0.80, color='r', linestyle='--', label='80% threshold')
ax.axhline(y=0.90, color='b', linestyle='--', label='90% threshold')
ax.set_xlabel('Number of Components')
ax.set_ylabel('Cumulative Variance Explained')
ax.set_title('PCA Cumulative Variance')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. Significance by category
ax = axes[1, 0]
if len(df_significance) > 0 and 'category' in df_significance.columns:
    category_summary = df_significance.groupby('category').agg({
        'significant': 'sum',
        'eta_squared': 'mean'
    }).reset_index()
    category_summary.columns = ['category', 'n_significant', 'mean_eta_squared']

    x_pos = np.arange(len(category_summary))
    bars = ax.bar(x_pos, category_summary['n_significant'])
    ax.set_xticks(x_pos)
    ax.set_xticklabels(category_summary['category'], rotation=45, ha='right')
    ax.set_ylabel('Number of Significant Features')
    ax.set_title('Significant Features by Category (p < 0.05)')
    ax.grid(True, alpha=0.3, axis='y')
else:
    ax.text(0.5, 0.5, 'No significance data', ha='center', va='center', transform=ax.transAxes)
    ax.set_title('Significant Features by Category (p < 0.05)')

# 4. Effect size distribution
ax = axes[1, 1]
if len(df_significance) > 0 and 'significant' in df_significance.columns:
    sig_features = df_significance[df_significance['significant']]
    if len(sig_features) > 0:
        effect_counts = sig_features['effect_size'].value_counts()
        effect_order = ['Large', 'Medium', 'Small', 'Negligible']
        effect_counts = effect_counts.reindex(effect_order, fill_value=0)

        colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']
        bars = ax.bar(range(len(effect_counts)), effect_counts.values, color=colors)
        ax.set_xticks(range(len(effect_counts)))
        ax.set_xticklabels(effect_counts.index)
        ax.set_ylabel('Number of Features')
        ax.set_title('Effect Size Distribution (Significant Features)')
        ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.text(0.5, 0.5, 'No significant features', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Effect Size Distribution (Significant Features)')
else:
    ax.text(0.5, 0.5, 'No significance data', ha='center', va='center', transform=ax.transAxes)
    ax.set_title('Effect Size Distribution (Significant Features)')

plt.tight_layout()
output_viz = results_dir / 'clustering_and_significance_analysis.png'
plt.savefig(output_viz, dpi=300, bbox_inches='tight')
print(f"Visualization: {output_viz}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

if n_total > 0:
    sig_pct = 100*n_significant/n_total
    sig_summary = f"  Total features tested: {n_total}\n  Significant features: {n_significant} ({sig_pct:.1f}%)"
else:
    sig_summary = "  Total features tested: 0\n  Significant features: 0 (0.0%)"

print(f"""
SUMMARY:
--------
Clustering Approaches:
  1. Major Features (n={len(available_major)}): k={optimal_k_major}, silhouette={max(silhouette_scores_major):.3f}
  2. PCA Comprehensive (n={n_components} PCs): k={optimal_k_pca}, silhouette={max(silhouette_scores_pca):.3f}

Significance Testing:
{sig_summary}

Recommendation:
  - Use PCA-based clusters for publication/analysis
  - Report major features approach for stakeholder communication
  - Focus interpretation on significant features (p < 0.05, medium-large η²)

Next Steps:
  1. Review top significant features in each category
  2. Create cluster profiles using significant features
  3. Develop diagnostic framework linking features → behaviors
  4. Test predictive models for cluster membership
""")
