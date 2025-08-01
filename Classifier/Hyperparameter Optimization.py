import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
import xgboost as xgb
from xgboost import XGBClassifier

from typing import Any, Dict, Union, Callable

# Load the Train-Test-Splits
with open("data_splits_feature_reduced.pkl", "rb") as f:
    X_test, y_test, X_train, y_train, X, y = pickle.load(f)

# Ensure X_train and y_train are dataframes
X_train = pd.DataFrame(X_train)
y_train = pd.DataFrame(y_train)
X_test = pd.DataFrame(X_test)
y_test = pd.DataFrame(y_test)

#######################################
# Hyperparameter Optimization XGBoost #
#######################################

def hyperparameter_tuning(space: Dict[str, Union[float, int]],
                          X: pd.DataFrame, y: pd.Series,
                          n_splits: int=5,
                          metric: callable=roc_auc_score) -> Dict[str, Any]:

    """
    Perform hyperparameter tuning for an XGBoost classifier.
    This function takes a dictionary of hyperparameters, training
    and test data, and an optional value for early stopping rounds,
    and returns a dictionary with the loss and model resulting from
    the tuning process. The model is trained using the training
    data and evaluated on the test data. The loss is computed as
    the negative of the accuracy score.

    Returns
    -------
    Dict[str, Any]
    A dictionary with the loss and model resulting from the
    tuning process. The loss is a float, and the model is an
    XGBoost classifier.
    """
    int_vals = ['max_depth', 'reg_alpha']

    space = {k: (int(val) if k in int_vals else val)
             for k, val in space.items()}

    # Initialize the model with the hyperparameters (without early stopping)
    model = xgb.XGBClassifier(**space)

    # Perform Stratified K-Fold Cross-Validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Evaluate the model using cross-validation and return the negative roc_auc_score for minimization
    cv_scores = cross_val_score(model, X, y, cv=skf, scoring='roc_auc', n_jobs=-1)

    # Return the mean ROC AUC score across all folds
    return {'loss': -np.mean(cv_scores), 'status': STATUS_OK, 'model': model}

# Hyperparameter dictionary
options = {
    'max_depth': hp.quniform('max_depth', 3, 12, 1),
    'min_child_weight': hp.loguniform('min_child_weight', -3, 3),
    'subsample': hp.uniform('subsample', 0.3, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1),
    'reg_alpha': hp.uniform('reg_alpha', 0, 10),
    'reg_lambda': hp.uniform('reg_lambda', 0, 10),
    'gamma': hp.loguniform('gamma', -10, 0),
    'learning_rate': hp.loguniform('learning_rate', -5, -2),
    'random_state': 42
}

trials = Trials()

best = fmin(
    fn=lambda space: hyperparameter_tuning(space, X_train, y_train),  # Objective function
    space=options,  # Hyperparameter search space
    algo=tpe.suggest,  # Optimization algorithm Tree-structured Parzen Estimator (Expected Improvement)
    max_evals=10,  # Maximum evaluations
    trials=trials,  # Trials object
)

print(best)

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
xgb_opt = xgb.XGBClassifier(**params, eval_metric='auc', use_label_encoder=False, n_estimators=500)

# Set up stratified cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation on the training data
cv_results = cross_val_score(xgb_opt, X_train, y_train, cv=cv, scoring='roc_auc',n_jobs=-1, verbose=10)

# Fit the model on the full training set
xgb_opt.fit(X_train, y_train, 
            eval_set=[(X_train, y_train), (X_test, y_test)], early_stopping_rounds=30,
            verbose=10)

# Predict on the test set
y_pred_test = xgb_opt.predict(X_test)
y_pred_proba_test = xgb_opt.predict_proba(X_test)[:, 1]

# Calculate performance metrics
accuracy_test = accuracy_score(y_test, y_pred_test)
precision_test = precision_score(y_test, y_pred_test)
recall_test = recall_score(y_test, y_pred_test)
f1_test = f1_score(y_test, y_pred_test)
auc_test = roc_auc_score(y_test, y_pred_proba_test)

# Print the cross-validation results (mean and std of AUC score across folds)
print(f'Cross-Validation AUC: {cv_results.mean():.6f} ± {cv_results.std():.6f}')

# Print model performance on the test set
print(f'XGBoost - Test Set: Accuracy: {accuracy_test:.6f}, Precision: {precision_test:.6f}, Recall: {recall_test:.6f}, F1 score: {f1_test:.6f}, AUC: {auc_test:.6f}')

# Save the optimized model
with open('xgb_opti_model.pkl', 'wb') as f:
    pickle.dump(xgb_opt, f)

#############################################
# Hyperparameter Optimization Random Forest #
#############################################

def hyperparameter_tuning_rf(space: Dict[str, Union[float, int, str]],
                          X: pd.DataFrame, y: pd.Series,
                          n_splits: int=5,
                          metric: callable=roc_auc_score) -> Dict[str, Any]:

    """
    Perform hyperparameter tuning for a RandomForest classifier.
    This function takes a dictionary of hyperparameters, training
    and test data, and returns a dictionary with the loss and model
    resulting from the tuning process.
    
    Returns
    -------
    Dict[str, Any]
    A dictionary with the loss and model resulting from the
    tuning process. The loss is a float, and the model is a
    RandomForest classifier.
    """
    
    # Parameters that need to be converted to integers
    int_vals = ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf']

    # Ensure integer values are passed for specific parameters, excluding strings like 'log2' for max_features
    space = {k: (int(val) if k in int_vals else val) for k, val in space.items() if k in int_vals or isinstance(val, (int, float))}

    # Initialize the model with the hyperparameters
    model = RandomForestClassifier(**space, random_state=42, n_jobs=-1)

    # Perform Stratified K-Fold Cross-Validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Evaluate the model using cross-validation and return the negative roc_auc_score for minimization
    cv_scores = cross_val_score(model, X, y, cv=skf, scoring='roc_auc', n_jobs=-1)

    # Return the mean ROC AUC score across all folds
    return {'loss': -np.mean(cv_scores), 'status': STATUS_OK, 'model': model}

# Hyperparameter dictionary for RandomForest
options_rf = {
    'n_estimators': hp.quniform('n_estimators', 100, 500, 50),
    'max_depth': hp.quniform('max_depth', 5, 20, 1),
    'min_samples_split': hp.quniform('min_samples_split', 2, 10, 1),
    'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 10, 1),
    'max_features': hp.choice('max_features', [None, 'sqrt', 'log2']),  # Categorical string choice
    'bootstrap': hp.choice('bootstrap', [True, False])  # Categorical boolean choice
}

trials_rf = Trials()

# Convert y_train and y_test to 1D arrays
y_train = y_train.values.ravel() if isinstance(y_train, pd.DataFrame) else y_train.ravel()
y_test = y_test.values.ravel() if isinstance(y_test, pd.DataFrame) else y_test.ravel()

best_rf = fmin(
    fn=lambda space: hyperparameter_tuning_rf(space, X_train, y_train),  # Objective function
    space=options_rf,  # Hyperparameter search space
    algo=tpe.suggest,  # Optimization algorithm Tree-structured Parzen Estimator (Expected Improvement)
    max_evals=1000,  # Maximum evaluations
    trials=trials_rf,  # Trials object
)

print(best_rf)

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
cv_rf_results = cross_val_score(rf_opt, X_train, y_train, cv=cv_rf, scoring='roc_auc',n_jobs=1, verbose=10)

# Fit the model on the full training set
rf_opt.fit(X_train, y_train)

# Predict on the test set
y_rf_pred_test = rf_opt.predict(X_test)
y_rf_pred_proba_test = rf_opt.predict_proba(X_test)[:, 1]

# Calculate performance metrics
rf_accuracy_test = accuracy_score(y_test, y_rf_pred_test)
rf_precision_test = precision_score(y_test, y_rf_pred_test)
rf_recall_test = recall_score(y_test, y_rf_pred_test)
rf_f1_test = f1_score(y_test, y_rf_pred_test)
rf_auc_test = roc_auc_score(y_test, y_rf_pred_proba_test)

# Print the cross-validation results (mean and std of AUC score across folds)
print(f'Cross-Validation AUC (RF): {cv_rf_results.mean():.6f} ± {cv_rf_results.std():.6f}')

# Print model performance on the test set
print(f'RandomForest - Test Set: Accuracy: {rf_accuracy_test:.6f}, Precision: {rf_precision_test:.6f}, Recall: {rf_recall_test:.6f}, F1 score: {rf_f1_test:.6f}, AUC: {rf_auc_test:.6f}')

# Save the optimized RandomForest model
with open('rf_opti_model.pkl', 'wb') as f:
    pickle.dump(rf_opt, f)

##################################################
# Hyperparameter Optimization LogisticRegression #
##################################################

def hyperparameter_tuning_lr(space: Dict[str, Union[float, int, str]],
                             X: pd.DataFrame, y: pd.Series,
                             n_splits: int=5,
                             metric: callable=roc_auc_score) -> Dict[str, Any]:

    """
    Perform hyperparameter tuning for a Logistic Regression classifier.
    This function takes a dictionary of hyperparameters, training
    and test data, and returns a dictionary with the loss and model
    resulting from the tuning process.
    
    Returns
    -------
    Dict[str, Any]
    A dictionary with the loss and model resulting from the
    tuning process. The loss is a float, and the model is a
    Logistic Regression classifier.
    """
    
    # Initialize the model with the hyperparameters
    model = LogisticRegression(**space, random_state=42, max_iter=5000)

    # Perform Stratified K-Fold Cross-Validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Evaluate the model using cross-validation and return the negative roc_auc_score for minimization
    cv_scores = cross_val_score(model, X, y, cv=skf, scoring='roc_auc', n_jobs=-1)

    # Return the mean ROC AUC score across all folds
    return {'loss': -np.mean(cv_scores), 'status': STATUS_OK, 'model': model}

# Hyperparameter dictionary for LogisticRegression
options_lr = {
    'penalty': hp.choice('penalty', ['l1', 'l2', 'elasticnet']),  # Removed 'none'
    'C': hp.loguniform('C', np.log(0.001), np.log(100)),  # Regularization strength (inverse of lambda)
    'solver': hp.choice('solver', ['liblinear', 'saga']),  # Solver choice (compatible with l1 and elasticnet)
    'l1_ratio': hp.uniform('l1_ratio', 0, 1)  # Used for elasticnet penalty
}

# Filter solvers based on penalty
def logistic_regression_filter_space(space):
    if space['penalty'] == 'l2':
        del space['l1_ratio']
    elif space['penalty'] == 'l1':
        space['solver'] = 'liblinear'
        del space['l1_ratio']
    elif space['penalty'] == 'elasticnet':
        space['solver'] = 'saga'
    return space

trials_lr = Trials()

best_lr = fmin(
    fn=lambda space: hyperparameter_tuning_lr(logistic_regression_filter_space(space), X_train, y_train),  # Objective function
    space=options_lr,  # Hyperparameter search space
    algo=tpe.suggest,  # Optimization algorithm Tree-structured Parzen Estimator (Expected Improvement)
    max_evals=1000,  # Maximum evaluations
    trials=trials_lr,  # Trials object
)

print(best_lr)

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
lr_opt = LogisticRegression(**{k: v for k, v in best_lr_params.items() if v is not None})

# Set up stratified cross-validation
cv_lr = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation on the training data
cv_lr_results = cross_val_score(lr_opt, X_train, y_train, cv=cv_lr, scoring='roc_auc', n_jobs=1, verbose=10)

# Fit the model on the full training set
lr_opt.fit(X_train, y_train)

# Predict on the test set
y_lr_pred_test = lr_opt.predict(X_test)
y_lr_pred_proba_test = lr_opt.predict_proba(X_test)[:, 1]

# Calculate performance metrics
lr_accuracy_test = accuracy_score(y_test, y_lr_pred_test)
lr_precision_test = precision_score(y_test, y_lr_pred_test)
lr_recall_test = recall_score(y_test, y_lr_pred_test)
lr_f1_test = f1_score(y_test, y_lr_pred_test)
lr_auc_test = roc_auc_score(y_test, y_lr_pred_proba_test)

# Print the cross-validation results (mean and std of AUC score across folds)
print(f'Cross-Validation AUC (LR): {cv_lr_results.mean():.6f} ± {cv_lr_results.std():.6f}')

# Print model performance on the test set
print(f'LogisticRegression - Test Set: Accuracy: {lr_accuracy_test:.6f}, Precision: {lr_precision_test:.6f}, Recall: {lr_recall_test:.6f}, F1 score: {lr_f1_test:.6f}, AUC: {lr_auc_test:.6f}')

# Save the optimized Logistic Regression model
with open('lr_opti_model.pkl', 'wb') as f:
    pickle.dump(lr_opt, f)