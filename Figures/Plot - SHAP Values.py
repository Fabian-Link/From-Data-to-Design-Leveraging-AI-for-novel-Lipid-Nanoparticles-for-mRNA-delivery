import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import seaborn as sns

from plot_style import set_plot_style
set_plot_style()

repo_root = Path(__file__).resolve().parent.parent

# Load train data and model
splits_dir_reduced = repo_root / "Datasets" / "Datasplits" / "Reduced Features"
X_train = pd.read_csv(splits_dir_reduced / "X_train.csv")
model_path = repo_root / "Classifier" / "Models" / "xgb_final.pkl"
with open(model_path, "rb") as f:
    xgb_model = pickle.load(f)

# SHAP analysis
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_train)

# Top 10 features by mean absolute SHAP value
mean_shap_values = np.abs(shap_values).mean(axis=0)
top_10_indices = np.argsort(mean_shap_values)[-10:]

custom_cmap = sns.diverging_palette(250, 30, l=65, center="dark", as_cmap=True)

plt.figure(figsize=(12, 8))
shap.summary_plot(
    shap_values[:, top_10_indices],
    X_train.iloc[:, top_10_indices],
    plot_type="dot",
    color_bar_label="Feature Value",
    cmap=custom_cmap,
    show=False
)

plt.axvline(x=0, color='black', linestyle='-', lw=2)

# Style axes and spines
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color('grey')

plt.grid(False)
plt.xlabel('Shapley Values', fontsize=30, fontweight='bold')
plt.ylabel('Features', fontsize=30, fontweight='bold')
plt.title('SHAP Beeswarm Plot', fontsize=30, fontweight='bold')
plt.tight_layout()

# Save as PDF
pdf_dir = repo_root / "Figures" / "PDF"
pdf_dir.mkdir(parents=True, exist_ok=True)
pdf_path = pdf_dir / "SHAP_Beeswarm.pdf"
plt.savefig(pdf_path)
plt.close()

print(f"SHAP beeswarm plot saved as PDF at: {pdf_path}")