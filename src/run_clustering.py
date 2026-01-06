import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set paths
BASE_DIR = r"c:\Users\jour\Documents\GitHub\accessibility_park-1\park-behavioral-clustering"
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
RESULTS_TABLES_DIR = os.path.join(BASE_DIR, "results", "tables")
RESULTS_FIGURES_DIR = os.path.join(BASE_DIR, "results", "figures")
SRC_DIR = os.path.join(BASE_DIR, "src")

os.makedirs(RESULTS_TABLES_DIR, exist_ok=True)
os.makedirs(RESULTS_FIGURES_DIR, exist_ok=True)
os.makedirs(SRC_DIR, exist_ok=True)

# 1. Load Data
clusters_path = os.path.join(DATA_DIR, "enhanced_behavioral_clusters.csv")
features_path = os.path.join(RESULTS_TABLES_DIR, "Table_S1_Clustering_Features.csv")

df = pd.read_csv(clusters_path)
features_df = pd.read_csv(features_path)

# Extract feature names
feature_names = features_df['Feature Name'].tolist()
print(f"Loaded {len(feature_names)} features for clustering.")

# 2. Preprocess
X = df[feature_names]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. K-Means Clustering (k=5)
k = 5
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)

# 4. Evaluation Metrics & Plot
# Elbow & Silhouette
inertias = []
silhouettes = []
k_range = range(2, 10)

for n_k in k_range:
    km = KMeans(n_clusters=n_k, random_state=42, n_init=10)
    km_labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, km_labels))

fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:red'
ax1.set_xlabel('Number of clusters (k)')
ax1.set_ylabel('Inertia', color=color)
ax1.plot(k_range, inertias, marker='o', color=color, label='Inertia')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Silhouette Score', color=color)
ax2.plot(k_range, silhouettes, marker='s', color=color, label='Silhouette')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('K-Means Optimization: Elbow Method & Silhouette Score')
fig.tight_layout()
plt.savefig(os.path.join(RESULTS_FIGURES_DIR, 'Figure_1_Clustering_Optimization.png'))
plt.close()

# 5. Update Dataset
df['cluster'] = labels

# Save updated dataset
df.to_csv(clusters_path, index=False)
print(f"Updated clusters saved to {clusters_path}")

# 6. Generate Cluster Description Table
# Calculate mean of features per cluster
cluster_summary = df.groupby('cluster')[feature_names].mean().T
cluster_summary.columns = [f'Cluster {i}' for i in range(k)]
cluster_summary.to_csv(os.path.join(RESULTS_TABLES_DIR, 'Table_4_Cluster_Descriptions.csv'))
print("Cluster description table saved.")
