import pandas as pd
import pickle
from pathlib import Path

import lazypredict
from lazypredict.Supervised import LazyClassifier

from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import xgboost as xgb
from xgboost import XGBClassifier

repo_root = Path(__file__).resolve().parent.parent

# Output directories
experiments_dir = repo_root / "Experiments"
experiments_dir.mkdir(parents=True, exist_ok=True)
models_dir = repo_root / "Models"
models_dir.mkdir(parents=True, exist_ok=True)

# Load the Train-Test-Splits feature reduced
splits_dir_reduced = repo_root / "Datasets" / "Datasplits" / "Reduced Features"
X_train = pd.read_csv(splits_dir_reduced / "X_train.csv")
X_test = pd.read_csv(splits_dir_reduced / "X_test.csv")
y_train = pd.read_csv(splits_dir_reduced / "y_train.csv").squeeze()
y_test = pd.read_csv(splits_dir_reduced / "y_test.csv").squeeze()

# Load the Train-Test-Splits scaled
splits_dir_all = repo_root / "Datasets" / "Datasplits" / "All Features"
X_train_unfiltered = pd.read_csv(splits_dir_all / "X_train_unfiltered.csv")
X_test_unfiltered = pd.read_csv(splits_dir_all / "X_test_unfiltered.csv")
y_train_unfiltered = pd.read_csv(splits_dir_all / "y_train_unfiltered.csv").squeeze()
y_test_unfiltered = pd.read_csv(splits_dir_all / "y_test_unfiltered.csv").squeeze()

# Full X and y for feature reduced and scaled
X = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)
y = pd.concat([y_train, y_test], axis=0).reset_index(drop=True)
X_unfiltered = pd.concat([X_train_unfiltered, X_test_unfiltered], axis=0).reset_index(drop=True)
y_unfiltered = pd.concat([y_train_unfiltered, y_test_unfiltered], axis=0).reset_index(drop=True)

#############################
# Default Model Comparison  #
#############################

lazy = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
train, test = lazy.fit(X_train, X_test, y_train, y_test)
train_unfiltered, test_unfiltered = lazy.fit(X_train_unfiltered, X_test_unfiltered, y_train_unfiltered, y_test_unfiltered)

# Save LazyPredict results
test.to_csv(experiments_dir / "LazyPredict_reduced_features.csv")
test_unfiltered.to_csv(experiments_dir / "LazyPredict_all_features.csv")

##############################################
# Optimized Model Comparison on scaled data  #
##############################################

# XGBoost
xgb_params = {'colsample_bytree': 0.3137816316738625, 
          'gamma': 0.00592667549172721, 
          'learning_rate': 0.13513958939715948, 
          'max_depth': 6, 
          'min_child_weight': 1.813906379741374, 
          'reg_alpha': 2.6859916838491023, 
          'reg_lambda': 6.606437062753413, 
          'subsample': 0.9991868010545087,
          'random_state': 42}

xgb_opt = xgb.XGBClassifier(**xgb_params, eval_metric='auc', use_label_encoder=False, n_estimators=500)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = cross_val_score(xgb_opt, X_train_unfiltered, y_train_unfiltered, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=10)
xgb_opt.fit(X_train_unfiltered, y_train_unfiltered, 
            eval_set=[(X_train_unfiltered, y_train_unfiltered), (X_test_unfiltered, y_test_unfiltered)], early_stopping_rounds=30,
            verbose=10)
y_pred_test = xgb_opt.predict(X_test_unfiltered)
y_pred_proba_test = xgb_opt.predict_proba(X_test_unfiltered)[:, 1]
accuracy_test = accuracy_score(y_test_unfiltered, y_pred_test)
precision_test = precision_score(y_test_unfiltered, y_pred_test)
recall_test = recall_score(y_test_unfiltered, y_pred_test)
f1_test = f1_score(y_test_unfiltered, y_pred_test)
auc_test = roc_auc_score(y_test_unfiltered, y_pred_proba_test)

# RandomForest
best_rf_params = {
    'n_estimators': 100,
    'max_depth': 9,
    'min_samples_split': 6,
    'min_samples_leaf': 4,
    'max_features': 'log2',
    'bootstrap': False,
    'random_state': 42
}
rf_opt = RandomForestClassifier(**best_rf_params, n_jobs=-1)
cv_rf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_rf_results = cross_val_score(rf_opt, X_train_unfiltered, y_train_unfiltered, cv=cv_rf, scoring='roc_auc', n_jobs=1, verbose=10)
rf_opt.fit(X_train_unfiltered, y_train_unfiltered)
y_rf_pred_test = rf_opt.predict(X_test_unfiltered)
y_rf_pred_proba_test = rf_opt.predict_proba(X_test_unfiltered)[:, 1]
rf_accuracy_test = accuracy_score(y_test_unfiltered, y_rf_pred_test)
rf_precision_test = precision_score(y_test_unfiltered, y_rf_pred_test)
rf_recall_test = recall_score(y_test_unfiltered, y_rf_pred_test)
rf_f1_test = f1_score(y_test_unfiltered, y_rf_pred_test)
rf_auc_test = roc_auc_score(y_test_unfiltered, y_rf_pred_proba_test)

# LogisticRegression
best_lr_params = {
    'penalty': 'l1',
    'C': 10.435891033240313,
    'solver': 'saga',
    'l1_ratio': 0.9888854133510901,
    'random_state': 42,
    'max_iter': 1000
}
lr_opt = LogisticRegression(**{k: v for k, v in best_lr_params.items() if v is not None})
cv_lr = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_lr_results = cross_val_score(lr_opt, X_train_unfiltered, y_train_unfiltered, cv=cv_lr, scoring='roc_auc', n_jobs=1, verbose=10)
lr_opt.fit(X_train_unfiltered, y_train_unfiltered)
y_lr_pred_test = lr_opt.predict(X_test_unfiltered)
y_lr_pred_proba_test = lr_opt.predict_proba(X_test_unfiltered)[:, 1]
lr_accuracy_test = accuracy_score(y_test_unfiltered, y_lr_pred_test)
lr_precision_test = precision_score(y_test_unfiltered, y_lr_pred_test)
lr_recall_test = recall_score(y_test_unfiltered, y_lr_pred_test)
lr_f1_test = f1_score(y_test_unfiltered, y_lr_pred_test)
lr_auc_test = roc_auc_score(y_test_unfiltered, y_lr_pred_proba_test)

##############################################
# Optimized Model Comparison on RFE data     #
##############################################

# XGBoost RFE
xgb_opt_rfe = xgb.XGBClassifier(**xgb_params, eval_metric='auc', use_label_encoder=False, n_estimators=500)
cv_rfe = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results_rfe = cross_val_score(xgb_opt_rfe, X_train, y_train, cv=cv_rfe, scoring='roc_auc', n_jobs=-1, verbose=10)
xgb_opt_rfe.fit(X_train, y_train, 
            eval_set=[(X_train, y_train), (X_test, y_test)], early_stopping_rounds=30,
            verbose=10)
y_pred_test_rfe = xgb_opt_rfe.predict(X_test)
y_pred_proba_test_rfe = xgb_opt_rfe.predict_proba(X_test)[:, 1]
accuracy_test_rfe = accuracy_score(y_test, y_pred_test_rfe)
precision_test_rfe = precision_score(y_test, y_pred_test_rfe)
recall_test_rfe = recall_score(y_test, y_pred_test_rfe)
f1_test_rfe = f1_score(y_test, y_pred_test_rfe)
auc_test_rfe = roc_auc_score(y_test, y_pred_proba_test_rfe)

# RandomForest RFE
rf_opt_rfe = RandomForestClassifier(**best_rf_params, n_jobs=-1)
cv_rf_rfe = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_rf_results_rfe = cross_val_score(rf_opt_rfe, X_train, y_train, cv=cv_rf_rfe, scoring='roc_auc', n_jobs=1, verbose=10)
rf_opt_rfe.fit(X_train, y_train)
y_rf_pred_test_rfe = rf_opt_rfe.predict(X_test)
y_rf_pred_proba_test_rfe = rf_opt_rfe.predict_proba(X_test)[:, 1]
rf_accuracy_test_rfe = accuracy_score(y_test, y_rf_pred_test_rfe)
rf_precision_test_rfe = precision_score(y_test, y_rf_pred_test_rfe)
rf_recall_test_rfe = recall_score(y_test, y_rf_pred_test_rfe)
rf_f1_test_rfe = f1_score(y_test, y_rf_pred_test_rfe)
rf_auc_test_rfe = roc_auc_score(y_test, y_rf_pred_proba_test_rfe)

# LogisticRegression RFE
lr_opt_rfe = LogisticRegression(**{k: v for k, v in best_lr_params.items() if v is not None})
cv_lr_rfe = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_lr_results_rfe = cross_val_score(lr_opt_rfe, X_train, y_train, cv=cv_lr_rfe, scoring='roc_auc', n_jobs=1, verbose=10)
lr_opt_rfe.fit(X_train, y_train)
y_lr_pred_test_rfe = lr_opt_rfe.predict(X_test)
y_lr_pred_proba_test_rfe = lr_opt_rfe.predict_proba(X_test)[:, 1]
lr_accuracy_test_rfe = accuracy_score(y_test, y_lr_pred_test_rfe)
lr_precision_test_rfe = precision_score(y_test, y_lr_pred_test_rfe)
lr_recall_test_rfe = recall_score(y_test, y_lr_pred_test_rfe)
lr_f1_test_rfe = f1_score(y_test, y_lr_pred_test_rfe)
lr_auc_test_rfe = roc_auc_score(y_test, y_lr_pred_proba_test_rfe)

#################################
# Train Final Model on all Data #
#################################

xgb_final = xgb.XGBClassifier(**xgb_params, eval_metric='auc', use_label_encoder=False, n_estimators=500)
cv_final = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results_final = cross_val_score(xgb_final, X, y, cv=cv_final, scoring='roc_auc', n_jobs=-1, verbose=10)
xgb_final.fit(X, y)

# Save all models as pickle files
with open(models_dir / "xgb_opt.pkl", "wb") as f:
    pickle.dump(xgb_opt, f)
with open(models_dir / "rf_opt.pkl", "wb") as f:
    pickle.dump(rf_opt, f)
with open(models_dir / "lr_opt.pkl", "wb") as f:
    pickle.dump(lr_opt, f)
with open(models_dir / "xgb_opt_rfe.pkl", "wb") as f:
    pickle.dump(xgb_opt_rfe, f)
with open(models_dir / "rf_opt_rfe.pkl", "wb") as f:
    pickle.dump(rf_opt_rfe, f)
with open(models_dir / "lr_opt_rfe.pkl", "wb") as f:
    pickle.dump(lr_opt_rfe, f)
with open(models_dir / "xgb_final.pkl", "wb") as f:
    pickle.dump(xgb_final, f)

# Save optimized model metrics to CSV
metrics = [
    # All features
    {
        "Model": "XGBoost (All Features)",
        "Accuracy": accuracy_test,
        "Precision": precision_test,
        "Recall": recall_test,
        "F1": f1_test,
        "AUC": auc_test,
        "CV_AUC_Mean": cv_results.mean(),
        "CV_AUC_Std": cv_results.std()
    },
    {
        "Model": "RandomForest (All Features)",
        "Accuracy": rf_accuracy_test,
        "Precision": rf_precision_test,
        "Recall": rf_recall_test,
        "F1": rf_f1_test,
        "AUC": rf_auc_test,
        "CV_AUC_Mean": cv_rf_results.mean(),
        "CV_AUC_Std": cv_rf_results.std()
    },
    {
        "Model": "LogisticRegression (All Features)",
        "Accuracy": lr_accuracy_test,
        "Precision": lr_precision_test,
        "Recall": lr_recall_test,
        "F1": lr_f1_test,
        "AUC": lr_auc_test,
        "CV_AUC_Mean": cv_lr_results.mean(),
        "CV_AUC_Std": cv_lr_results.std()
    },
    # Reduced features
    {
        "Model": "XGBoost (Reduced Features)",
        "Accuracy": accuracy_test_rfe,
        "Precision": precision_test_rfe,
        "Recall": recall_test_rfe,
        "F1": f1_test_rfe,
        "AUC": auc_test_rfe,
        "CV_AUC_Mean": cv_results_rfe.mean(),
        "CV_AUC_Std": cv_results_rfe.std()
    },
    {
        "Model": "RandomForest (Reduced Features)",
        "Accuracy": rf_accuracy_test_rfe,
        "Precision": rf_precision_test_rfe,
        "Recall": rf_recall_test_rfe,
        "F1": rf_f1_test_rfe,
        "AUC": rf_auc_test_rfe,
        "CV_AUC_Mean": cv_rf_results_rfe.mean(),
        "CV_AUC_Std": cv_rf_results_rfe.std()
    },
    {
        "Model": "LogisticRegression (Reduced Features)",
        "Accuracy": lr_accuracy_test_rfe,
        "Precision": lr_precision_test_rfe,
        "Recall": lr_recall_test_rfe,
        "F1": lr_f1_test_rfe,
        "AUC": lr_auc_test_rfe,
        "CV_AUC_Mean": cv_lr_results_rfe.mean(),
        "CV_AUC_Std": cv_lr_results_rfe.std()
    }
]
metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv(experiments_dir / "Optimized_Model_Metrics.csv", index=False)

# Save final model metrics to CSV
final_metrics = {
    "Model": "XGBoost Final",
    "CV_AUC_Mean": cv_results_final.mean(),
    "CV_AUC_Std": cv_results_final.std()
}
pd.DataFrame([final_metrics]).to_csv(experiments_dir / "Final_Model_Metrics.csv", index=False)

print("All results and metrics have been saved.")