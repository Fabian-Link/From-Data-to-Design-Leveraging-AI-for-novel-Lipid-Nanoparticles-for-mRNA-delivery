import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from matplotlib.colors import ListedColormap
from yellowbrick.model_selection import LearningCurve

from plot_style import set_plot_style
set_plot_style()

repo_root = Path(__file__).resolve().parent.parent

# Load train data
splits_dir_reduced = repo_root / "Datasets" / "Datasplits" / "Reduced Features"
X_train = pd.read_csv(splits_dir_reduced / "X_train.csv")
y_train = pd.read_csv(splits_dir_reduced / "y_train.csv")
y_train = y_train['HeLa_label'] if 'HeLa_label' in y_train.columns else y_train

# Load test data
splits_dir_reduced = repo_root / "Datasets" / "Datasplits" / "Reduced Features"
X_test = pd.read_csv(splits_dir_reduced / "X_test.csv")
y_test = pd.read_csv(splits_dir_reduced / "y_test.csv")
y_true = y_test['HeLa_label'] if 'HeLa_label' in y_test.columns else y_test

# Load XGBoost model
model_path = repo_root / "Classifier" / "Models" / "xgb_opt_rfe.pkl"
with open(model_path, "rb") as f:
    xgb_model = pickle.load(f)

# Predict probabilities and labels
y_proba = xgb_model.predict_proba(X_test)[:, 1]
y_pred = xgb_model.predict(X_test)

# Get colors from plot_style
colors = [c['color'] for c in plt.rcParams['axes.prop_cycle']]
custom_cmap = ListedColormap([colors[0], colors[2]])

# --- ROC Curve ---
fpr, tpr, thresholds = roc_curve(y_true, y_proba)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color=colors[2], lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color=colors[0], lw=1, linestyle="--", label="Random chance")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate", fontsize=30, fontweight='bold')
plt.ylabel("True Positive Rate", fontsize=30, fontweight='bold')
plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=30, fontweight='bold')
plt.legend(fontsize=30, prop={'weight': 'bold'})
plt.tight_layout()

pdf_dir = repo_root / "Figures" / "PDF"
pdf_dir.mkdir(parents=True, exist_ok=True)
roc_path = pdf_dir / "XGB_ROC_Curve.pdf"
plt.savefig(roc_path)
plt.close()

print(f"ROC curve saved as PDF at: {roc_path}")

# --- Confusion Matrix ---
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
fig, ax = plt.subplots()
disp.plot(ax=ax, cmap=custom_cmap, colorbar=False)
plt.title("Confusion Matrix", fontsize=30, fontweight='bold')
plt.tight_layout()

cm_path = pdf_dir / "XGB_Confusion_Matrix.pdf"
plt.savefig(cm_path)
plt.close()

print(f"Confusion matrix saved as PDF at: {cm_path}")

# Create the learning curve visualizer
visualizer = LearningCurve(
    xgb_model, 
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='roc_auc',
    cv=5,
    n_jobs=-1,
    train_color=colors[0],
    test_color=colors[2],
)

# Fit the visualizer and the model
visualizer.fit(X_train, y_train)

# Apply unified style to the plot
set_plot_style()
plt.title("Learning Curve XGBClassifier", fontsize=30, fontweight='bold')
plt.xlabel("Training Instances", fontsize=30, fontweight='bold')
plt.ylabel("Score", fontsize=30, fontweight='bold')
plt.legend(["Training Score", "CV Score"], fontsize=30, prop={'weight': 'bold'})
plt.tight_layout()

# Save the plot as PDF
lc_path = pdf_dir / "XGB_Learning_Curve.pdf"
plt.savefig(lc_path)
plt.close()

print(f"Learning curve saved as PDF at: {lc_path}")

# Load model stability results
stability_csv = repo_root / "Classifier" / "Experiments" / "model_stability_xgb.csv"
stability_df = pd.read_csv(stability_csv)

plt.figure()
plt.scatter(
    stability_df['Seed'], stability_df['AUC'],
    color=colors[0], alpha=0.7, label='ROC AUC'
)
mean_auc = stability_df['AUC'].mean()
plt.axhline(
    mean_auc, color=colors[2], linestyle='--', linewidth=2,
    label=f'Mean AUC ({mean_auc:.2f})'
)
plt.xlabel("Train-test Split (Seed)", fontsize=30, fontweight='bold')
plt.ylabel("ROC AUC Score", fontsize=30, fontweight='bold')
plt.title("Model Stability - XGBClassifier", fontsize=30, fontweight='bold')
plt.ylim(0.7, 1.05)
plt.legend(fontsize=30, prop={'weight': 'bold'})
plt.tight_layout()

stability_path = pdf_dir / "Model_Stability_XGBClassifier.pdf"
plt.savefig(stability_path)
plt.close()

print(f"Model stability plot saved as PDF at: {stability_path}")