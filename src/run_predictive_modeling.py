import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
import os

# Set paths
BASE_DIR = r"c:\Users\jour\Documents\GitHub\accessibility_park-1\park-behavioral-clustering"
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
RESULTS_TABLES_DIR = os.path.join(BASE_DIR, "results", "tables")
SRC_DIR = os.path.join(BASE_DIR, "src")

# 1. Load Data
clusters_path = os.path.join(DATA_DIR, "enhanced_behavioral_clusters.csv")
dataset_path = os.path.join(DATA_DIR, "final_dataset_18parks.csv")
features_path = os.path.join(RESULTS_TABLES_DIR, "Table_S2_Predictive_Features.csv")

df_clusters = pd.read_csv(clusters_path)
df_data = pd.read_csv(dataset_path)
df_features = pd.read_csv(features_path)

# Merge: enhanced_clusters has 'cluster', final_dataset has 'park_name' and features
# Note: final_dataset ALSO has a 'cluster' column which might be old. We should drop it or use the one from enhanced_clusters.
# enhanced_clusters has 'park_name' as index? No, let's check. 
# Step 14 output shows 'park_name' is NOT a column, it seems to be in the first column but unnamed in header?
# Wait, look at Step 14 output: "Gangseo Hangang Park" is the first value. The header line is:
# ",avg_ppltn,max_ppltn..." -> The first column name is empty string or space.
# I need to handle this.

# Reload with specific handling
df_clusters = pd.read_csv(clusters_path)
if df_clusters.columns[0].strip() == '' or 'Unnamed' in df_clusters.columns[0]:
    df_clusters.rename(columns={df_clusters.columns[0]: 'park_name'}, inplace=True)

# Dataset 18 parks might be fine.
df_data = pd.read_csv(dataset_path)

# Merge
# We want 'cluster' from df_clusters and FEATURES from df_data.
merged_df = pd.merge(df_clusters[['park_name', 'cluster']], df_data, on='park_name', suffixes=('_target', '_data'))

# Use the cluster from df_clusters as target (it deals with the re-run k=5)
target_col = 'cluster_target'

# Get Predictive Features
feature_list = df_features['Feature Name'].tolist()
# Filter out any that might be missing
available_features = [f for f in feature_list if f in merged_df.columns]
missing_features = [f for f in feature_list if f not in merged_df.columns]
if missing_features:
    print(f"Warning: Missing features from dataset: {missing_features}")

X = merged_df[available_features]
y = merged_df[target_col]
park_names = merged_df['park_name']

print(f"Data Loaded. Shape: {X.shape}. Target distribution:\n{y.value_counts()}")

# 2. Predictive Modeling with LOOCV
models = {
    'Logistic Regression': LogisticRegression(max_iter=2000, random_state=42),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
}

results = []
loo = LeaveOneOut()

print("\nStarting LOOCV...")

for name, model in models.items():
    y_true_all = []
    y_pred_all = []
    
    for train_index, test_index in loo.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        if name == 'XGBoost':
            # XGBoost requires 0..N-1 labels. 
            # If a class is missing in y_train, we need to remap.
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y_train_enc = le.fit_transform(y_train)
            
            # Check if we have enough classes
            if len(np.unique(y_train_enc)) < 2:
                 pred = [y_train.iloc[0]] # Fallback
            else:
                model.fit(X_train, y_train_enc)
                pred_enc = model.predict(X_test)
                # Map back. Note: if model predicts a class, it's an index in le.classes_
                # But wait, if X_test leads to a prediction, it gives an integer.
                # We need to inverse transform.
                pred = le.inverse_transform(pred_enc)
        
        else:
            # Standard Sklearn models
            if len(y_train.unique()) < 2:
                pred = [y_train.iloc[0]] 
            else:
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
        
        y_true_all.append(y_test.values[0])
        y_pred_all.append(pred[0])
        
    acc = accuracy_score(y_true_all, y_pred_all)
    f1 = f1_score(y_true_all, y_pred_all, average='macro')
    
    print(f"Model: {name} | Accuracy: {acc:.4f} | F1-Macro: {f1:.4f}")
    
    results.append({
        'Model': name,
        'Accuracy': acc,
        'F1-Score (Macro)': f1
    })

# 3. Save Results
results_df = pd.DataFrame(results)
output_path = os.path.join(RESULTS_TABLES_DIR, "Table_6_Model_Performance.csv")
results_df.to_csv(output_path, index=False)
print(f"\nResults saved to {output_path}")
