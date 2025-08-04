import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb

# Load data
repo_root = Path(__file__).resolve().parent.parent
splits_dir_reduced = repo_root / "Datasets" / "Datasplits" / "Reduced Features"
X = pd.concat([
    pd.read_csv(splits_dir_reduced / "X_train.csv"),
    pd.read_csv(splits_dir_reduced / "X_test.csv")
], axis=0).reset_index(drop=True)
y = pd.concat([
    pd.read_csv(splits_dir_reduced / "y_train.csv"),
    pd.read_csv(splits_dir_reduced / "y_test.csv")
], axis=0).reset_index(drop=True)
y = y['HeLa_label'] if 'HeLa_label' in y.columns else y

X = np.array(X).copy()
y = np.array(y).copy()

xgb_params = {
    'colsample_bytree': 0.3137816316738625, 
    'gamma': 0.00592667549172721, 
    'learning_rate': 0.13513958939715948, 
    'max_depth': 6, 
    'min_child_weight': 1.813906379741374, 
    'reg_alpha': 2.6859916838491023, 
    'reg_lambda': 6.606437062753413, 
    'subsample': 0.9991868010545087,
    'random_state': 42,
    'early_stopping_rounds': 30
}

results = {
    'Seed': [],
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'F1 Score': [],
    'AUC': []
}

for seed in range(100):
    print(f"Running model {seed + 1}/100 with random seed: {seed}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    xgb_opt = xgb.XGBClassifier(
        **xgb_params, eval_metric='auc', use_label_encoder=False, n_estimators=500
    )
    xgb_opt.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False
    )
    y_pred_test = xgb_opt.predict(X_test)
    y_pred_proba_test = xgb_opt.predict_proba(X_test)[:, 1]
    results['Seed'].append(seed)
    results['Accuracy'].append(accuracy_score(y_test, y_pred_test))
    results['Precision'].append(precision_score(y_test, y_pred_test))
    results['Recall'].append(recall_score(y_test, y_pred_test))
    results['F1 Score'].append(f1_score(y_test, y_pred_test))
    results['AUC'].append(roc_auc_score(y_test, y_pred_proba_test))

results_df = pd.DataFrame(results)

# Save results to Experiments folder
experiments_dir = repo_root / "Classifier" / "Experiments"
experiments_dir.mkdir(parents=True, exist_ok=True)
results_df.to_csv(experiments_dir / "model_stability_xgb.csv", index=False)

mean_results = results_df.mean()
std_results = results_df.std()

print("\nAverage Performance Across 100 Models:")
for metric in results_df.columns[1:]:
    print(f'{metric}: Mean = {mean_results[metric]:.6f}, Std = {std_results[metric]:.6f}')