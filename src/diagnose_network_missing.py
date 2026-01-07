
import pandas as pd
import os
import sys

# Paths
BASE_DIR = r"c:\Users\jour\Documents\GitHub\accessibility_park-1"
CLUSTERING_DIR = os.path.join(BASE_DIR, "park-behavioral-clustering")
DATA_PROCESSED = os.path.join(CLUSTERING_DIR, "data", "processed")
CACHE_DIR = os.path.join(BASE_DIR, "cache")

final_dataset_path = os.path.join(DATA_PROCESSED, "final_dataset_18parks.csv")
network_features_path = os.path.join(CACHE_DIR, "advanced_network_features.csv")

print("=== DIAGNOSING MISSING AND ZERO VALUES ===")

# 1. Check Final Dataset
if os.path.exists(final_dataset_path):
    print(f"\nLoading {final_dataset_path}...")
    try:
        df_final = pd.read_csv(final_dataset_path)
        print(f"Shape: {df_final.shape}")
        
        # Check for NaNs
        nan_counts = df_final.isna().sum()
        cols_with_nans = nan_counts[nan_counts > 0]
        if not cols_with_nans.empty:
            print("\nColumns with Missing Values in Final Dataset:")
            print(cols_with_nans)
        else:
            print("No missing values (NaNs) found in Final Dataset.")
            
        # Check for Zeros
        cols_numeric = df_final.select_dtypes(include=['number']).columns
        zeros = (df_final[cols_numeric] == 0).sum()
        cols_with_zeros = zeros[zeros > 0]
        if not cols_with_zeros.empty:
            print("\nColumns with ZERO values in Final Dataset:")
            print(cols_with_zeros)
            
    except Exception as e:
        print(f"Error reading final dataset: {e}")
else:
    print(f"Final dataset not found: {final_dataset_path}")

# 2. Check Advanced Network Features Cache
if os.path.exists(network_features_path):
    print(f"\nLoading {network_features_path}...")
    try:
        df_network = pd.read_csv(network_features_path)
        print(f"Shape: {df_network.shape}")
        
        # Check for NaNs
        nan_counts_net = df_network.isna().sum()
        cols_with_nans_net = nan_counts_net[nan_counts_net > 0]
        if not cols_with_nans_net.empty:
            print("\nColumns with Missing Values (NaN) in Network Cache:")
            print(cols_with_nans_net)
        else:
            print("No missing values (NaNs) found in Network Cache.")

        # Check for Zeros in Network Columns
        network_cols = [c for c in df_network.columns if any(x in c for x in ['topo_', 'service_', 'density_', 'infra_'])]
        if network_cols:
            zeros_counts_net = (df_network[network_cols] == 0).sum()
            cols_with_zeros_net = zeros_counts_net[zeros_counts_net > 0]
            
            if not cols_with_zeros_net.empty:
                print("\nColumns with ZERO values in Network Cache (Possible Failures):")
                print(cols_with_zeros_net)
                
                # Identify parks with many zeros
                # We assume 'park_name' is the identifier
                if 'park_name' in df_network.columns:
                    df_network['zero_count'] = (df_network[network_cols] == 0).sum(axis=1)
                    # Create a threshold: if more than 20% of network features are zero
                    threshold = len(network_cols) * 0.2
                    high_zeros = df_network[df_network['zero_count'] > threshold]
                    
                    if not high_zeros.empty:
                        print(f"\nParks with extensive ZERO values (> {int(threshold)} features):")
                        for idx, row in high_zeros.iterrows():
                            print(f" - {row['park_name']}: {row['zero_count']}/{len(network_cols)} zero features")
                else:
                    print("Column 'park_name' not found in network cache, cannot identify specific parks.")
            else:
                 print("\nNo ZERO values found in network columns.")
        else:
            print("No specific network feature columns identified.")

        # Check Intersection
        if 'df_final' in locals() and 'park_name' in df_network.columns and 'park_name' in df_final.columns:
            final_parks = set(df_final['park_name'])
            network_parks = set(df_network['park_name'])
            
            print(f"\nParks in Final Dataset: {len(final_parks)}") 
            print(f"Parks in Network Cache: {len(network_parks)}")
            
            missing_in_network = final_parks - network_parks
            if missing_in_network:
                print(f"Parks in Final Dataset but MISSING from Network Cache: {missing_in_network}")
            else:
                print("All Final Dataset parks are present in Network Cache.")
                
    except Exception as e:
        print(f"Error reading network cache: {e}")
else:
    print(f"Network features cache not found: {network_features_path}")

print("\n=== DIAGNOSIS COMPLETE ===")
