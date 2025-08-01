import pandas as pd
import pickle

import lazypredict
from lazypredict.Supervised import LazyClassifier

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split

import xgboost as xgb
from xgboost import XGBClassifier

# Load the Train-Test-Splits feature reduced
with open("data_splits_feature_reduced.pkl", "rb") as f:
    X_test, y_test, X_train, y_train, X, y = pickle.load(f)

# Load the Train-Test-Splits scaled
with open("data_splits_scaled.pkl", "rb") as f:
    X_train_unfiltered, X_test_unfiltered, y_train_unfiltered, y_test_unfiltered, X_unfiltered , y_unfiltered = pickle.load(f)

# Ensure X_train and y_train are dataframes
X_train = pd.DataFrame(X_train)
y_train = pd.DataFrame(y_train)
X_test = pd.DataFrame(X_test)
y_test = pd.DataFrame(y_test)
X = pd.DataFrame(X)
y = pd.DataFrame(y)

# Ensure X_train and y_train are dataframes
X_train_unfiltered = pd.DataFrame(X_train_unfiltered)
y_train_unfiltered = pd.DataFrame(y_train_unfiltered)
X_test_unfiltered = pd.DataFrame(X_test_unfiltered)
y_test_unfiltered = pd.DataFrame(y_test_unfiltered)

#############################
# Default Model Comparison #
#############################

lazy=LazyClassifier(verbose=0,ignore_warnings=True,custom_metric=None)

# Fit the models and evaluate their performance on the test set
train, test = lazy.fit(X_train, X_test, y_train, y_test)

print(test)

lazy_unfiltered=LazyClassifier(verbose=0,ignore_warnings=True,custom_metric=None)

# Fit the models and evaluate their performance on the test set
train_unfiltered, test_unfiltered = lazy.fit(X_train_unfiltered, X_test_unfiltered, y_train_unfiltered, y_test_unfiltered)

print(test_unfiltered)

##############################################
# Optimized Model Comaparison on scaled data #
##############################################

# Selecting best parameters from optimization
xgb_params = {'colsample_bytree': 0.3137816316738625, 
          'gamma': 0.00592667549172721, 
          'learning_rate': 0.13513958939715948, 
          'max_depth': 6, 
          'min_child_weight': 1.813906379741374, 
          'reg_alpha': 2.6859916838491023, 
          'reg_lambda': 6.606437062753413, 
          'subsample': 0.9991868010545087,
          'random_state': 42}

# Train the optimized model with early stopping on the full training set
xgb_opt = xgb.XGBClassifier(**xgb_params, eval_metric='auc', use_label_encoder=False, n_estimators=500)

# Set up stratified cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation on the training data
cv_results = cross_val_score(xgb_opt, X_train_unfiltered, y_train_unfiltered, cv=cv, scoring='roc_auc',n_jobs=-1, verbose=10)

# Fit the model on the full training set
xgb_opt.fit(X_train_unfiltered, y_train_unfiltered, 
            eval_set=[(X_train_unfiltered, y_train_unfiltered), (X_test_unfiltered, y_test_unfiltered)], early_stopping_rounds=30,
            verbose=10)

# Predict on the test set
y_pred_test = xgb_opt.predict(X_test_unfiltered)
y_pred_proba_test = xgb_opt.predict_proba(X_test_unfiltered)[:, 1]

# Calculate performance metrics
accuracy_test = accuracy_score(y_test_unfiltered, y_pred_test)
precision_test = precision_score(y_test_unfiltered, y_pred_test)
recall_test = recall_score(y_test_unfiltered, y_pred_test)
f1_test = f1_score(y_test_unfiltered, y_pred_test)
auc_test = roc_auc_score(y_test_unfiltered, y_pred_proba_test)

# Print the cross-validation results (mean and std of AUC score across folds)
print(f'Cross-Validation AUC: {cv_results.mean():.6f} ± {cv_results.std():.6f}')

# Print model performance on the test set
print(f'XGBoost - Test Set: Accuracy: {accuracy_test:.6f}, Precision: {precision_test:.6f}, Recall: {recall_test:.6f}, F1 score: {f1_test:.6f}, AUC: {auc_test:.6f}')

# Selecting best parameters
best_rf_params = {
    'n_estimators': 100,
    'max_depth': 9,
    'min_samples_split': 6,
    'min_samples_leaf': 4,
    'max_features': 'log2',
    'bootstrap': False,
    'random_state': 42
}

# Train the optimized RandomForest model on the full training set
rf_opt = RandomForestClassifier(**best_rf_params, n_jobs=-1)

# Set up stratified cross-validation
cv_rf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation on the training data
cv_rf_results = cross_val_score(rf_opt, X_train_unfiltered, y_train_unfiltered, cv=cv_rf, scoring='roc_auc',n_jobs=1, verbose=10)

# Fit the model on the full training set
rf_opt.fit(X_train_unfiltered, y_train_unfiltered)

# Predict on the test set
y_rf_pred_test = rf_opt.predict(X_test_unfiltered)
y_rf_pred_proba_test = rf_opt.predict_proba(X_test_unfiltered)[:, 1]

# Calculate performance metrics
rf_accuracy_test = accuracy_score(y_test_unfiltered, y_rf_pred_test)
rf_precision_test = precision_score(y_test_unfiltered, y_rf_pred_test)
rf_recall_test = recall_score(y_test_unfiltered, y_rf_pred_test)
rf_f1_test = f1_score(y_test_unfiltered, y_rf_pred_test)
rf_auc_test = roc_auc_score(y_test_unfiltered, y_rf_pred_proba_test)

# Print the cross-validation results (mean and std of AUC score across folds)
print(f'Cross-Validation AUC (RF): {cv_rf_results.mean():.6f} ± {cv_rf_results.std():.6f}')

# Print model performance on the test set
print(f'RandomForest - Test Set: Accuracy: {rf_accuracy_test:.6f}, Precision: {rf_precision_test:.6f}, Recall: {rf_recall_test:.6f}, F1 score: {rf_f1_test:.6f}, AUC: {rf_auc_test:.6f}')

# Selecting best parameters
best_lr_params = {
    'penalty': 'l1',
    'C': 10.435891033240313,
    'solver': 'saga',
    'l1_ratio': 0.9888854133510901,
    'random_state': 42,
    'max_iter': 1000
}

# Train the optimized Logistic Regression model on the full training set
lr_opt = LogisticRegression(**{k: v for k, v in best_lr_params.items() if v is not None})

# Set up stratified cross-validation
cv_lr = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation on the training data
cv_lr_results = cross_val_score(lr_opt, X_train_unfiltered, y_train_unfiltered, cv=cv_lr, scoring='roc_auc', n_jobs=1, verbose=10)

# Fit the model on the full training set
lr_opt.fit(X_train_unfiltered, y_train_unfiltered)

# Predict on the test set
y_lr_pred_test = lr_opt.predict(X_test_unfiltered)
y_lr_pred_proba_test = lr_opt.predict_proba(X_test_unfiltered)[:, 1]

# Calculate performance metrics
lr_accuracy_test = accuracy_score(y_test_unfiltered, y_lr_pred_test)
lr_precision_test = precision_score(y_test_unfiltered, y_lr_pred_test)
lr_recall_test = recall_score(y_test_unfiltered, y_lr_pred_test)
lr_f1_test = f1_score(y_test_unfiltered, y_lr_pred_test)
lr_auc_test = roc_auc_score(y_test_unfiltered, y_lr_pred_proba_test)

# Print the cross-validation results (mean and std of AUC score across folds)
print(f'Cross-Validation AUC (LR): {cv_lr_results.mean():.6f} ± {cv_lr_results.std():.6f}')

# Print model performance on the test set
print(f'LogisticRegression - Test Set: Accuracy: {lr_accuracy_test:.6f}, Precision: {lr_precision_test:.6f}, Recall: {lr_recall_test:.6f}, F1 score: {lr_f1_test:.6f}, AUC: {lr_auc_test:.6f}')


###########################################
# Optimized Model Comaparison on RFE data #
###########################################

# Selecting best parameters
params = {'colsample_bytree': 0.3137816316738625, 
          'gamma': 0.00592667549172721, 
          'learning_rate': 0.13513958939715948, 
          'max_depth': 6, 
          'min_child_weight': 1.813906379741374, 
          'reg_alpha': 2.6859916838491023, 
          'reg_lambda': 6.606437062753413, 
          'subsample': 0.9991868010545087,
          'random_state': 42}

# Train the optimized model with early stopping on the full training set
xgb_opt_rfe = xgb.XGBClassifier(**params, eval_metric='auc', use_label_encoder=False, n_estimators=500)

# Set up stratified cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation on the training data
cv_results = cross_val_score(xgb_opt, X_train, y_train, cv=cv, scoring='roc_auc',n_jobs=-1, verbose=10)

# Fit the model on the full training set
xgb_opt_rfe.fit(X_train, y_train, 
            eval_set=[(X_train, y_train), (X_test, y_test)], early_stopping_rounds=30,
            verbose=10)

# Predict on the test set
y_pred_test = xgb_opt_rfe.predict(X_test)
y_pred_proba_test = xgb_opt_rfe.predict_proba(X_test)[:, 1]

# Calculate performance metrics
accuracy_test = accuracy_score(y_test, y_pred_test)
precision_test = precision_score(y_test, y_pred_test)
recall_test = recall_score(y_test, y_pred_test)
f1_test = f1_score(y_test, y_pred_test)
auc_test = roc_auc_score(y_test, y_pred_proba_test)

# Print the cross-validation results (mean and std of AUC score across folds)
print(f'Cross-Validation AUC: {cv_results.mean():.6f} ± {cv_results.std():.6f}')

# Print model performance on the test set
print(f'XGBoost_rfe - Test Set: Accuracy: {accuracy_test:.6f}, Precision: {precision_test:.6f}, Recall: {recall_test:.6f}, F1 score: {f1_test:.6f}, AUC: {auc_test:.6f}')

# Selecting best parameters
best_rf_params = {
    'n_estimators': 100,
    'max_depth': 9,
    'min_samples_split': 6,
    'min_samples_leaf': 4,
    'max_features': 'log2',
    'bootstrap': False,
    'random_state': 42
}

# Train the optimized RandomForest model on the full training set
rf_opt_rfe = RandomForestClassifier(**best_rf_params, n_jobs=-1)

# Set up stratified cross-validation
cv_rf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation on the training data
cv_rf_results = cross_val_score(rf_opt_rfe, X_train, y_train, cv=cv_rf, scoring='roc_auc',n_jobs=1, verbose=10)

# Fit the model on the full training set
rf_opt_rfe.fit(X_train, y_train)

# Predict on the test set
y_rf_pred_test = rf_opt_rfe.predict(X_test)
y_rf_pred_proba_test = rf_opt_rfe.predict_proba(X_test)[:, 1]

# Calculate performance metrics
rf_accuracy_test = accuracy_score(y_test, y_rf_pred_test)
rf_precision_test = precision_score(y_test, y_rf_pred_test)
rf_recall_test = recall_score(y_test, y_rf_pred_test)
rf_f1_test = f1_score(y_test, y_rf_pred_test)
rf_auc_test = roc_auc_score(y_test, y_rf_pred_proba_test)

# Print the cross-validation results (mean and std of AUC score across folds)
print(f'Cross-Validation AUC (RF): {cv_rf_results.mean():.6f} ± {cv_rf_results.std():.6f}')

# Print model performance on the test set
print(f'RandomForest_rfe - Test Set: Accuracy: {rf_accuracy_test:.6f}, Precision: {rf_precision_test:.6f}, Recall: {rf_recall_test:.6f}, F1 score: {rf_f1_test:.6f}, AUC: {rf_auc_test:.6f}')

# Selecting best parameters
best_lr_params = {
    'penalty': 'l1',  # Converted based on index 0 to 'l1'
    'C': 10.435891033240313,  # Regularization strength
    'solver': 'saga',  # Converted based on index 1 to 'saga'
    'l1_ratio': 0.9888854133510901,  # l1_ratio value for elasticnet
    'random_state': 42,
    'max_iter': 1000
}

# Train the optimized Logistic Regression model on the full training set
lr_opt_rfe = LogisticRegression(**{k: v for k, v in best_lr_params.items() if v is not None})

# Set up stratified cross-validation
cv_lr = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation on the training data
cv_lr_results = cross_val_score(lr_opt_rfe, X_train, y_train, cv=cv_lr, scoring='roc_auc', n_jobs=1, verbose=10)

# Fit the model on the full training set
lr_opt_rfe.fit(X_train, y_train)

# Predict on the test set
y_lr_pred_test = lr_opt_rfe.predict(X_test)
y_lr_pred_proba_test = lr_opt_rfe.predict_proba(X_test)[:, 1]

# Calculate performance metrics
lr_accuracy_test = accuracy_score(y_test, y_lr_pred_test)
lr_precision_test = precision_score(y_test, y_lr_pred_test)
lr_recall_test = recall_score(y_test, y_lr_pred_test)
lr_f1_test = f1_score(y_test, y_lr_pred_test)
lr_auc_test = roc_auc_score(y_test, y_lr_pred_proba_test)

# Print the cross-validation results (mean and std of AUC score across folds)
print(f'Cross-Validation AUC (LR): {cv_lr_results.mean():.6f} ± {cv_lr_results.std():.6f}')

# Print model performance on the test set
print(f'LogisticRegression_rfe - Test Set: Accuracy: {lr_accuracy_test:.6f}, Precision: {lr_precision_test:.6f}, Recall: {lr_recall_test:.6f}, F1 score: {lr_f1_test:.6f}, AUC: {lr_auc_test:.6f}')

#################################
# Train Final Model on all Data #
#################################

# Selecting best parameters from optimization
xgb_params = {'colsample_bytree': 0.3137816316738625, 
          'gamma': 0.00592667549172721, 
          'learning_rate': 0.13513958939715948, 
          'max_depth': 6, 
          'min_child_weight': 1.813906379741374, 
          'reg_alpha': 2.6859916838491023, 
          'reg_lambda': 6.606437062753413, 
          'subsample': 0.9991868010545087,
          'random_state': 42}

# Train the optimized model with early stopping on the full training set
xgb_final = xgb.XGBClassifier(**xgb_params, eval_metric='auc', use_label_encoder=False, n_estimators=500)

# Set up stratified cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation on the training data
cv_results = cross_val_score(xgb_final, X, y, cv=cv, scoring='roc_auc',n_jobs=-1, verbose=10)

# Fit the model on the full training set
xgb_final.fit(X, y)

# Save the optimized Logistic Regression model
with open('xgb_final.pkl', 'wb') as f:
    pickle.dump(xgb_final, f)