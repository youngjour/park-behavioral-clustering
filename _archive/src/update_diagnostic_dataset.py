
import pandas as pd
import os

# Paths
BASE_DIR = r"c:\Users\jour\Documents\GitHub\accessibility_park-1"
RESULTS_DIR = os.path.join(BASE_DIR, "results") # Try root results first
if not os.path.exists(RESULTS_DIR):
    RESULTS_DIR = os.path.join(BASE_DIR, "park-behavioral-clustering", "results") 

CACHE_DIR = os.path.join(BASE_DIR, "cache")

diagnostic_dataset_path = os.path.join(RESULTS_DIR, "enhanced_behavioral_diagnostic_dataset_FINAL.csv")
network_features_path = os.path.join(CACHE_DIR, "advanced_network_features.csv")

print("=== UPDATING DIAGNOSTIC DATASET ===")
print(f"Target: {diagnostic_dataset_path}")

# 1. Load Datasets
if not os.path.exists(diagnostic_dataset_path):
    # Try finding it in src/results or root/results
    print(f"File not found at {diagnostic_dataset_path}")
    # Search logic in python if needed, but for now lets assume standard paths
    import sys
    sys.exit(1)

df_diag = pd.read_csv(diagnostic_dataset_path)
print(f"Original Diagnostic Dataset: {df_diag.shape}")

df_network = pd.read_csv(network_features_path)
print(f"New Network Features: {df_network.shape}")

# 2. Map Korean Names to English (Same as before)
kor_to_eng = {
    '강서한강공원': 'Gangseo Hangang Park',
    '광나루한강공원': 'Gwangnaru Hangang Park',
    '국립중앙박물관·용산가족공원': 'The National Museum of Korea·Yongsan Family Park',
    '난지한강공원': 'Nanji Hangang Park',
    '남산공원': 'Namsan Park',
    '뚝섬한강공원': 'Ttukseom Hangang Park',
    '망원한강공원': 'Mangwon Hangang Park',
    '반포한강공원': 'Banpo Hangang Park',
    '북서울꿈의숲': 'Dream Forest',
    '서울대공원': 'Seoul Grand Park',
    '서울숲공원': 'Seoul Forest',
    '서울숲': 'Seoul Forest',
    '양화한강공원': 'Yanghwa Hangang Park',
    '어린이대공원': "Children's Grand Park",
    '여의도한강공원': 'Yeouido Hangang Park',
    '월드컵공원': 'World Cup Park',
    '이촌한강공원': 'Ichon Hangang Park',
    '잠실종합운동장': 'Jamsil Sports Complex',
    '잠실한강공원': 'Jamsil Hangang Park',
    '잠원한강공원': 'Jamwon Hangang Park'
}

df_network['park_name_en'] = df_network['park_name'].map(kor_to_eng)
df_network = df_network.dropna(subset=['park_name_en'])
df_network = df_network.drop(columns=['park_name'])
df_network = df_network.rename(columns={'park_name_en': 'park_name'})

# 3. Identify Columns to Update
# We only want to update network/transportation features
# Common features: centrality_score, distance_to_center, area_km2 (maybe?)
# Let's inspect overlap
overlap_cols = [c for c in df_network.columns if c in df_diag.columns and c != 'park_name']
print(f"Identified {len(overlap_cols)} overlapping features to update: {overlap_cols}")

# 4. Update
# Drop old columns from diag
if overlap_cols:
    df_diag = df_diag.drop(columns=overlap_cols)

# Merge
print("Merging datasets...")
df_updated = pd.merge(df_diag, df_network, on='park_name', how='left')

# 5. Save
print(f"Updated Dataset Shape: {df_updated.shape}")
df_updated.to_csv(diagnostic_dataset_path, index=False)
print("SUCCESS: Diagnostic dataset updated.")
