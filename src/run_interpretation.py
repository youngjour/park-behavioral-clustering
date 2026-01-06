import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import PartialDependenceDisplay
import os

# Set paths
BASE_DIR = r"c:\Users\jour\Documents\GitHub\accessibility_park-1\park-behavioral-clustering"
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
RESULTS_TABLES_DIR = os.path.join(BASE_DIR, "results", "tables")
RESULTS_FIGURES_DIR = os.path.join(BASE_DIR, "results", "figures")

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
# Need 'avg_ppltn' for Volume target (Efficiency Index)
merged_df = pd.merge(df_clusters[['park_name', 'cluster', 'avg_ppltn']], df_data, on='park_name', suffixes=('_target', '_data'))

feature_list = df_features['Feature Name'].tolist()
available_features = [f for f in feature_list if f in merged_df.columns]

X = merged_df[available_features]
y_cls = merged_df['cluster_target']
y_vol = np.log1p(merged_df['avg_ppltn']) # Target for regression

print(f"Data Loaded. X Shape: {X.shape}")

# ---------------------------------------------------------
# Part 1: Interpretation (XGBoost Classifier)
# ---------------------------------------------------------
print("Running Interpretation (SHAP & PDP)...")

# Encode labels
le = LabelEncoder()
y_cls_enc = le.fit_transform(y_cls)

# Train Classifier
model_cls = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
model_cls.fit(X, y_cls_enc)

# SHAP
# Explainer
explainer = shap.TreeExplainer(model_cls)
shap_values = explainer.shap_values(X)

# Summary Plot
plt.figure(figsize=(10, 8))
# For multiclass, shap_values is a list of arrays (one for each class).
# Summary plot for multiclass sums them up or we can pick one?
# Usually 'shap.summary_plot(shap_values, X)' handles multiclass by stacking or via color.
shap.summary_plot(shap_values, X, show=False)
plt.savefig(os.path.join(RESULTS_FIGURES_DIR, 'Figure_2_SHAP_Summary.png'), bbox_inches='tight')
plt.close()

# PDP
# Get Top 3 features by importance
importances = model_cls.feature_importances_
indices = np.argsort(importances)[::-1]
top_features = X.columns[indices[:3]].tolist()
print(f"Top features for PDP: {top_features}")

# Create PDP
# Note: multiclass PDP usually requires specifying the target class. 
# We'll plot for each class or just average?
# Sklearn's PartialDependenceDisplay allows 'target' param. 
# We'll generate PDP for class 0 (Evening Urban) as example, or iterate?
# Let's plot for the most populous class or just default (which might average for binary, or require target for multi).
# For XGBoost estimator in sklearn wrapper, we can use sklearn's display.
# We will generate one figure with subplots for the top features.
# We have to pick a target class for multiclass PDP. Let's pick all classes in one plot per feature? Too messy.
# Let's pick the 'Evening Urban' (Cluster 0) and 'Mega' (Cluster 3 or so). 
# Actually, let's just do it for Class 0 for now as an example, or loop through top features.
# If we don't specify target for multiclass, it might fail.
# Let's assume Class 0 is interesting.
target_class_idx = 0 
fig, ax = plt.subplots(figsize=(12, 4))
PartialDependenceDisplay.from_estimator(model_cls, X, top_features, target=target_class_idx, ax=ax)
plt.suptitle(f"Partial Dependence Plots (Target: Class {le.inverse_transform([target_class_idx])[0]})")
plt.savefig(os.path.join(RESULTS_FIGURES_DIR, 'Figure_3_PDP.png'), bbox_inches='tight')
plt.close()


# ---------------------------------------------------------
# Part 2: Diagnosis (XGBoost Regressor)
# ---------------------------------------------------------
print("Running Structural Diagnosis (Efficiency Index)...")

model_reg = xgb.XGBRegressor(random_state=42)
model_reg.fit(X, y_vol)
y_pred_vol = model_reg.predict(X)

# Calculate Efficiency
residuals = y_vol - y_pred_vol
# Efficiency Index: Let's use Residual directly. Positive = More efficient than expected.

merged_df['log_actual'] = y_vol
merged_df['log_predicted'] = y_pred_vol
merged_df['efficiency_residual'] = residuals

# Define Outliers
mean_res = residuals.mean()
std_res = residuals.std()
threshold = 1.0 * std_res # Using 1.0 std for small sample to highlight SOME outliers

merged_df['efficiency_status'] = 'Normal'
merged_df.loc[merged_df['efficiency_residual'] > mean_res + threshold, 'efficiency_status'] = 'Positive Outlier'
merged_df.loc[merged_df['efficiency_residual'] < mean_res - threshold, 'efficiency_status'] = 'Negative Outlier'

# Save Table
out_table = merged_df[['park_name', 'cluster_target', 'avg_ppltn', 'log_actual', 'log_predicted', 'efficiency_residual', 'efficiency_status']]
out_table.to_csv(os.path.join(RESULTS_TABLES_DIR, 'Table_Efficiency_Diagnosis.csv'), index=False)

# Plot Diagnosis
plt.figure(figsize=(10, 8))
sns.scatterplot(data=merged_df, x='log_predicted', y='log_actual', hue='efficiency_status', style='cluster_target', s=100)
# Add diagonal line
min_val = min(y_vol.min(), y_pred_vol.min())
max_val = max(y_vol.max(), y_pred_vol.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)

# Label outliers
for i, row in merged_df.iterrows():
    if row['efficiency_status'] != 'Normal':
        plt.text(row['log_predicted'], row['log_actual'], row['park_name'], fontsize=9)

plt.title('Structural Diagnosis: Actual vs Predicted Volume')
plt.xlabel('Predicted Log Volume (Potential)')
plt.ylabel('Actual Log Volume (Usage)')
plt.savefig(os.path.join(RESULTS_FIGURES_DIR, 'Figure_4_Efficiency_Diagnosis.png'))
plt.close()

print("Interpretation and Diagnosis Complete.")
