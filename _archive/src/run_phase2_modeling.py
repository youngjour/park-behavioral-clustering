"""
Phase 2: Predictive Model Development & Validation
Run all three models with LOOCV and compare performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

# ML libraries
import xgboost as xgb
import lightgbm as lgb
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("=" * 80)
print("PHASE 2: PREDICTIVE MODEL DEVELOPMENT & VALIDATION")
print("=" * 80)

# Load 18-park complete dataset
dataset_dir = Path('../dataset')
df = pd.read_csv(dataset_dir / 'training_dataset_18parks_complete.csv')

print(f"\n1. Training data: {len(df)} parks × {len(df.columns)} columns")

# Define target and input features
target_cols = [f'hour_{str(h).zfill(2)}' for h in range(24)]
exclude_cols = ['park_name', 'poi_code', 'has_comprehensive_features',
                'kmeans_cluster', 'new_kmeans_cluster', 'behavioral_cluster',
                'num_fragments', 'num_facility_fragments'] + target_cols

all_cols = df.columns.tolist()
input_cols = [col for col in all_cols if col not in exclude_cols]

# Remove features with >50% missing values
missing_pct = df[input_cols].isnull().sum() / len(df) * 100
valid_features = missing_pct[missing_pct < 50].index.tolist()

print(f"2. Input features: {len(valid_features)} (after removing high-missing features)")

# Handle remaining missing values with median imputation
X = df[valid_features].copy()
for col in X.columns:
    if X[col].isnull().any():
        X[col].fillna(X[col].median(), inplace=True)

y = df[target_cols].copy()

print(f"3. Final dataset: X={X.shape}, y={y.shape}")
print(f"   Feature-to-sample ratio: {X.shape[1]/X.shape[0]:.2f}")

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

# LOOCV setup
loo = LeaveOneOut()
n_samples = len(X)

# Storage for all models
xgb_predictions = np.zeros((n_samples, 24))
lgb_predictions = np.zeros((n_samples, 24))
gp_predictions = np.zeros((n_samples, 24))
gp_std = np.zeros((n_samples, 24))

xgb_mae_per_hour = np.zeros(24)
lgb_mae_per_hour = np.zeros(24)
gp_mae_per_hour = np.zeros(24)

# ============================================================================
# MODEL A: XGBoost
# ============================================================================
print("\n" + "=" * 80)
print("MODEL A: XGBoost with LOOCV")
print("=" * 80)

xgb_params = {
    'max_depth': 3,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'objective': 'reg:squarederror',
    'random_state': 42,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0
}

print(f"\nTraining 24 XGBoost models...")
for hour_idx, hour_col in enumerate(target_cols):
    hour_num = int(hour_col.split('_')[1])

    y_true = []
    y_pred = []

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[hour_col].iloc[train_idx], y[hour_col].iloc[test_idx]

        model = xgb.XGBRegressor(**xgb_params)
        model.fit(X_train, y_train, verbose=False)

        pred = model.predict(X_test)[0]
        y_pred.append(pred)
        y_true.append(y_test.values[0])

    xgb_predictions[:, hour_idx] = y_pred
    xgb_mae_per_hour[hour_idx] = mean_absolute_error(y_true, y_pred)

    if (hour_idx + 1) % 6 == 0 or hour_idx == 0:
        print(f"  Completed hours 00-{hour_num:02d}... MAE: {xgb_mae_per_hour[hour_idx]:.1f}")

xgb_rmse_per_hour = np.array([np.sqrt(mean_squared_error(
    y[f'hour_{h:02d}'], xgb_predictions[:, h])) for h in range(24)])
xgb_r2_per_hour = np.array([r2_score(
    y[f'hour_{h:02d}'], xgb_predictions[:, h]) for h in range(24)])

print(f"\nXGBoost Results:")
print(f"  MAE:  {xgb_mae_per_hour.mean():6.1f} ± {xgb_mae_per_hour.std():5.1f}")
print(f"  RMSE: {xgb_rmse_per_hour.mean():6.1f} ± {xgb_rmse_per_hour.std():5.1f}")
print(f"  R²:   {xgb_r2_per_hour.mean():6.3f} ± {xgb_r2_per_hour.std():5.3f}")

# ============================================================================
# MODEL B: LightGBM
# ============================================================================
print("\n" + "=" * 80)
print("MODEL B: LightGBM with LOOCV")
print("=" * 80)

lgb_params = {
    'max_depth': 3,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'objective': 'regression',
    'random_state': 42,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'verbose': -1
}

print(f"\nTraining 24 LightGBM models...")
for hour_idx, hour_col in enumerate(target_cols):
    hour_num = int(hour_col.split('_')[1])

    y_true = []
    y_pred = []

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[hour_col].iloc[train_idx], y[hour_col].iloc[test_idx]

        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(X_train, y_train)

        pred = model.predict(X_test)[0]
        y_pred.append(pred)
        y_true.append(y_test.values[0])

    lgb_predictions[:, hour_idx] = y_pred
    lgb_mae_per_hour[hour_idx] = mean_absolute_error(y_true, y_pred)

    if (hour_idx + 1) % 6 == 0 or hour_idx == 0:
        print(f"  Completed hours 00-{hour_num:02d}... MAE: {lgb_mae_per_hour[hour_idx]:.1f}")

lgb_rmse_per_hour = np.array([np.sqrt(mean_squared_error(
    y[f'hour_{h:02d}'], lgb_predictions[:, h])) for h in range(24)])
lgb_r2_per_hour = np.array([r2_score(
    y[f'hour_{h:02d}'], lgb_predictions[:, h]) for h in range(24)])

print(f"\nLightGBM Results:")
print(f"  MAE:  {lgb_mae_per_hour.mean():6.1f} ± {lgb_mae_per_hour.std():5.1f}")
print(f"  RMSE: {lgb_rmse_per_hour.mean():6.1f} ± {lgb_rmse_per_hour.std():5.1f}")
print(f"  R²:   {lgb_r2_per_hour.mean():6.3f} ± {lgb_r2_per_hour.std():5.3f}")

# ============================================================================
# MODEL C: Gaussian Process (sample first 6 hours due to computational cost)
# ============================================================================
print("\n" + "=" * 80)
print("MODEL C: Gaussian Process with LOOCV (first 6 hours)")
print("=" * 80)
print("Note: Running GP on subset due to computational cost...")

kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)

gp_sample_hours = [0, 6, 12, 18]  # Sample key hours
print(f"\nTraining GP for hours: {gp_sample_hours}")

for hour_idx in gp_sample_hours:
    hour_col = f'hour_{hour_idx:02d}'

    y_true = []
    y_pred = []
    y_uncertainty = []

    for train_idx, test_idx in loo.split(X_scaled):
        X_train, X_test = X_scaled.iloc[train_idx], X_scaled.iloc[test_idx]
        y_train, y_test = y[hour_col].iloc[train_idx], y[hour_col].iloc[test_idx]

        gp = GaussianProcessRegressor(kernel=kernel, random_state=42,
                                     n_restarts_optimizer=2, normalize_y=True)
        gp.fit(X_train, y_train)

        pred, std = gp.predict(X_test, return_std=True)
        y_pred.append(pred[0])
        y_uncertainty.append(std[0])
        y_true.append(y_test.values[0])

    gp_predictions[:, hour_idx] = y_pred
    gp_std[:, hour_idx] = y_uncertainty
    gp_mae_per_hour[hour_idx] = mean_absolute_error(y_true, y_pred)

    print(f"  Hour {hour_idx:02d}: MAE={gp_mae_per_hour[hour_idx]:.1f}, Avg σ={np.mean(y_uncertainty):.1f}")

print(f"\nGaussian Process Results (sample hours):")
gp_sample_mae = gp_mae_per_hour[gp_sample_hours]
print(f"  MAE:  {gp_sample_mae.mean():6.1f} ± {gp_sample_mae.std():5.1f}")

# ============================================================================
# MODEL COMPARISON
# ============================================================================
print("\n" + "=" * 80)
print("MODEL COMPARISON SUMMARY")
print("=" * 80)

comparison = pd.DataFrame({
    'Model': ['XGBoost', 'LightGBM'],
    'MAE_mean': [xgb_mae_per_hour.mean(), lgb_mae_per_hour.mean()],
    'MAE_std': [xgb_mae_per_hour.std(), lgb_mae_per_hour.std()],
    'RMSE_mean': [xgb_rmse_per_hour.mean(), lgb_rmse_per_hour.mean()],
    'RMSE_std': [xgb_rmse_per_hour.std(), lgb_rmse_per_hour.std()],
    'R2_mean': [xgb_r2_per_hour.mean(), lgb_r2_per_hour.mean()],
    'R2_std': [xgb_r2_per_hour.std(), lgb_r2_per_hour.std()]
})

print("\nPerformance Metrics (averaged across 24 hours):")
print(comparison.to_string(index=False))

best_idx = comparison['MAE_mean'].idxmin()
best_model = comparison.loc[best_idx, 'Model']

print(f"\n{'=' * 80}")
print(f"WINNER: {best_model} (Lowest MAE: {comparison.loc[best_idx, 'MAE_mean']:.1f})")
print(f"{'=' * 80}")

# Save results
comparison.to_csv(dataset_dir / 'model_comparison_phase2.csv', index=False)
print(f"\nModel comparison saved to: {dataset_dir / 'model_comparison_phase2.csv'}")

# ============================================================================
# SAVE PREDICTIONS
# ============================================================================
results = df[['park_name']].copy()

for hour in range(24):
    results[f'actual_hour_{hour:02d}'] = y[f'hour_{hour:02d}'].values
    results[f'pred_xgb_hour_{hour:02d}'] = xgb_predictions[:, hour]
    results[f'pred_lgb_hour_{hour:02d}'] = lgb_predictions[:, hour]

results['actual_daily_total'] = y.sum(axis=1).values
results['xgb_daily_total'] = xgb_predictions.sum(axis=1)
results['lgb_daily_total'] = lgb_predictions.sum(axis=1)

output_file = dataset_dir / 'loocv_predictions_phase2.csv'
results.to_csv(output_file, index=False)

print(f"Prediction results saved to: {output_file}")

# ============================================================================
# VISUALIZATION
# ============================================================================
print(f"\nCreating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: MAE per hour
ax = axes[0, 0]
hours = range(24)
ax.plot(hours, xgb_mae_per_hour, 'o-', label='XGBoost', linewidth=2, markersize=6)
ax.plot(hours, lgb_mae_per_hour, 's-', label='LightGBM', linewidth=2, markersize=6)
ax.set_xlabel('Hour of Day', fontsize=12)
ax.set_ylabel('Mean Absolute Error', fontsize=12)
ax.set_title('MAE by Hour of Day', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xticks(range(0, 24, 2))

# Plot 2: R² per hour
ax = axes[0, 1]
ax.plot(hours, xgb_r2_per_hour, 'o-', label='XGBoost', linewidth=2, markersize=6)
ax.plot(hours, lgb_r2_per_hour, 's-', label='LightGBM', linewidth=2, markersize=6)
ax.set_xlabel('Hour of Day', fontsize=12)
ax.set_ylabel('R² Score', fontsize=12)
ax.set_title('R² Score by Hour of Day', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xticks(range(0, 24, 2))
ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)

# Plot 3: Model comparison
ax = axes[1, 0]
x_pos = np.arange(len(comparison))
ax.bar(x_pos, comparison['MAE_mean'], yerr=comparison['MAE_std'],
       capsize=5, alpha=0.7, color=['#1f77b4', '#ff7f0e'])
ax.set_xticks(x_pos)
ax.set_xticklabels(comparison['Model'])
ax.set_ylabel('MAE (visitors)', fontsize=12)
ax.set_title('Average MAE Comparison', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Plot 4: Sample prediction
ax = axes[1, 1]
park_idx = 0
park_name = df.iloc[park_idx]['park_name']
y_true_sample = y.iloc[park_idx].values
y_pred_sample = xgb_predictions[park_idx]

ax.plot(hours, y_true_sample, 'o-', label='Actual', linewidth=2, markersize=8, color='black')
ax.plot(hours, y_pred_sample, 's-', label=f'Predicted ({best_model})', linewidth=2, markersize=6, color='red')
ax.set_xlabel('Hour of Day', fontsize=12)
ax.set_ylabel('Visitors', fontsize=12)
ax.set_title(f'Sample Prediction: {park_name}', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xticks(range(0, 24, 2))

plt.tight_layout()
plt.savefig(dataset_dir / 'model_performance_phase2.png', dpi=300, bbox_inches='tight')
print(f"Visualization saved to: {dataset_dir / 'model_performance_phase2.png'}")

print("\n" + "=" * 80)
print("PHASE 2: COMPLETE")
print("=" * 80)
print(f"\nBest Model: {best_model}")
print(f"  Average MAE: {comparison.loc[best_idx, 'MAE_mean']:.1f} visitors")
print(f"  Average R²: {comparison.loc[best_idx, 'R2_mean']:.3f}")
print(f"\nReady for Phase 3: Optimization Framework")
print("=" * 80)
