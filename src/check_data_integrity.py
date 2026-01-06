import pandas as pd
import os

# Set paths
BASE_DIR = r"c:\Users\jour\Documents\GitHub\accessibility_park-1\park-behavioral-clustering"
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
RESULTS_TABLES_DIR = os.path.join(BASE_DIR, "results", "tables")

# 1. Load Data
clusters_path = os.path.join(DATA_DIR, "enhanced_behavioral_clusters.csv")
dataset_path = os.path.join(DATA_DIR, "final_dataset_18parks.csv")
features_path = os.path.join(RESULTS_TABLES_DIR, "Table_S2_Predictive_Features.csv")

df_clusters = pd.read_csv(clusters_path)
if df_clusters.columns[0].strip() == '' or 'Unnamed' in df_clusters.columns[0]:
    df_clusters.rename(columns={df_clusters.columns[0]: 'park_name'}, inplace=True)

df_data = pd.read_csv(dataset_path)
df_features = pd.read_csv(features_path)

# Merge
merged_df = pd.merge(df_clusters[['park_name', 'cluster']], df_data, on='park_name', suffixes=('_target', '_data'))

# 2. Verify Features
feature_list = df_features['Feature Name'].tolist()
print(f"Total features in Table S2: {len(feature_list)}")

missing_features = [f for f in feature_list if f not in merged_df.columns]
if missing_features:
    print(f"CRITICAL: The following features from Table S2 are MISSING in the dataset:\n{missing_features}")
else:
    print("SUCCESS: All features from Table S2 are present in the dataset.")

# 3. Check for Missing Values or Anomalies
X = merged_df[feature_list]
null_counts = X.isnull().sum()
if null_counts.sum() > 0:
    print("\nCRITICAL: Missing values detected:")
    print(null_counts[null_counts > 0])
else:
    print("\nSUCCESS: No missing (NaN) values detected in predictor features.")

# Check for all-zeros (might be suspicious for some features)
zero_counts = (X == 0).sum()
print("\nFeature Summary (Min, Max, Zero-Count):")
print(X.describe().T[['min', 'max']].join(pd.DataFrame(zero_counts, columns=['zero_count'])))

# 4. Print Data for Visual Inspection
print("\nSample Data (First 5 rows of features):")
print(X.head())
