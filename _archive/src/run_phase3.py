import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# 1. Load Data
try:
    dataset_path = r'../park-behavioral-clustering/data/processed/final_dataset_18parks.csv'
    df = pd.read_csv(dataset_path)
    print(f"Loaded dataset: {df.shape}")

    # Define X and y
    X = df.drop(columns=['park_name', 'cluster'])
    y = df['cluster']

    feature_names = X.columns.tolist()
    print(f"Target Classes: {y.unique()}")
    print("Class Counts:")
    print(y.value_counts())
    print(f"Features ({len(feature_names)}): {feature_names}")

    # 2. Preprocessing
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_names)

    # 3. Model Definition
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42, n_estimators=50, max_depth=3)
    }

    # 4. LOOCV Evaluation
    results = []
    loo = LeaveOneOut()

    print("\nStarting LOOCV...")

    for name, model in models.items():
        y_true = []
        y_pred = []
        
        for train_index, test_index in loo.split(X_scaled):
            X_train, X_test = X_scaled.iloc[train_index], X_scaled.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            try:
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                
                y_true.append(y_test.values[0])
                y_pred.append(pred[0])
            except Exception as e_model:
                print(f"Error training {name} on fold: {e_model}")
                # Assign a wrong prediction just to continue? Or skip?
                # For LOOCV, if it fails, maybe we count it as a miss?
                y_true.append(y_test.values[0])
                y_pred.append(-1) # Impossible class

            
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        
        results.append({'Model': name, 'Accuracy': acc, 'F1 Score (Macro)': f1})
        print(f"{name}: Accuracy={acc:.3f}, F1={f1:.3f}")

    # Create Summary Table
    results_df = pd.DataFrame(results).sort_values(by='F1 Score (Macro)', ascending=False)
    print("\n=== Model Performance (LOOCV) ===")
    print(results_df)

    # 5. Save Results
    # 5. Save Results
    output_path = '../results/phase3_model_comparison_18parks.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

except Exception as e:
    print(f"An error occurred: {e}")
