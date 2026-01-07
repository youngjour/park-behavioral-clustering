
import pandas as pd
import os

# Paths
BASE_DIR = r"c:\Users\jour\Documents\GitHub\accessibility_park-1"
CLUSTERING_DIR = os.path.join(BASE_DIR, "park-behavioral-clustering")
DATA_PROCESSED = os.path.join(CLUSTERING_DIR, "data", "processed")
CACHE_DIR = os.path.join(BASE_DIR, "cache")

final_dataset_path = os.path.join(DATA_PROCESSED, "final_dataset_18parks.csv")
network_features_path = os.path.join(CACHE_DIR, "advanced_network_features.csv")

print("=== UPDATING FINAL DATASET ===")

# 1. Load Datasets
print(f"Loading {final_dataset_path}...")
df_final = pd.read_csv(final_dataset_path)
print(f"Original Final Dataset: {df_final.shape}")

print(f"Loading {network_features_path}...")
df_network = pd.read_csv(network_features_path)
print(f"New Network Features: {df_network.shape}")

# 2. Map Korean Names to English
# Mismatch identified: df_final has English names, df_network has Korean names (from shapefile).
# We must map df_network['park_name'] (Korean) -> English matching df_final.

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
    '서울숲': 'Seoul Forest', # Just in case
    '양화한강공원': 'Yanghwa Hangang Park',
    '어린이대공원': "Children's Grand Park",
    '여의도한강공원': 'Yeouido Hangang Park',
    '월드컵공원': 'World Cup Park',
    '이촌한강공원': 'Ichon Hangang Park',
    '잠실종합운동장': 'Jamsil Sports Complex', # Might not be in 18 parks?
    '잠실한강공원': 'Jamsil Hangang Park',
    '잠원한강공원': 'Jamwon Hangang Park'
}

print("Mapping network feature names to English...")
# Create a new column 'park_name_en' using the map
df_network['park_name_en'] = df_network['park_name'].map(kor_to_eng)

# Check for unmapped parks
unmapped = df_network[df_network['park_name_en'].isna()]['park_name'].unique()
if len(unmapped) > 0:
    print(f"Warning: {len(unmapped)} parks in network features could not be mapped to English (might be extra parks): {unmapped}")

# Drop the original Korean 'park_name' and rename 'park_name_en' to 'park_name' for merge
df_network = df_network.dropna(subset=['park_name_en']) # Keep only those we can map
df_network = df_network.drop(columns=['park_name'])
df_network = df_network.rename(columns={'park_name_en': 'park_name'})

# 2. Identify Network Columns to Update
# Logic: Any column present in df_network (except park_name) should be updated in df_final
network_cols = [c for c in df_network.columns if c != 'park_name']
print(f"Identified {len(network_cols)} network features to update.")

# 3. Prepare for Merge
# Remove existing network columns from df_final to avoid duplicates like _x, _y
cols_to_drop = [c for c in network_cols if c in df_final.columns]
if cols_to_drop:
    print(f"Dropping {len(cols_to_drop)} existing network columns from final dataset before merge...")
    df_final = df_final.drop(columns=cols_to_drop)

# 4. Merge
# Left join on park_name to keep all parks in final dataset (though usually they match)
print("Merging datasets on 'park_name'...")
df_updated = pd.merge(df_final, df_network, on='park_name', how='left')

# 5. Verify Merge
print(f"Updated Dataset Shape: {df_updated.shape}")
# Check for NaNs after merge (which would suggest parks in final dataset were missing in network features)
nan_in_new_cols = df_updated[network_cols].isna().sum().sum()
if nan_in_new_cols > 0:
    print(f"WARNING: {nan_in_new_cols} missing values found in network columns after merge!")
    print(df_updated[df_updated[network_cols].isna().any(axis=1)][['park_name']])
else:
    print("Merge successful. No missing values introduced.")

# 6. Save
print(f"Overwriting {final_dataset_path}...")
df_updated.to_csv(final_dataset_path, index=False)
print("SUCCESS: Final dataset updated.")
