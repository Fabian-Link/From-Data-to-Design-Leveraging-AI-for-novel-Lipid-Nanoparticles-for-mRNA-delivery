import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from plot_style import set_plot_style
set_plot_style()

def jitter(df: pd.DataFrame, col: str, amount: float=1) -> pd.Series:
    vals = np.random.uniform(low=-amount/2, high=amount/2, size=df.shape[0])
    return df[col] + vals

repo_root = Path(__file__).resolve().parent.parent
trials_path = repo_root / "Classifier" / "Experiments" / "xgb_hyperopt_trials.csv"
df_trials = pd.read_csv(trials_path)

# If loss is negative ROC AUC, convert to positive ROC AUC
if (df_trials['loss'].max() <= 0.0) and (df_trials['loss'].min() >= -1.0):
    df_trials['roc_auc'] = -df_trials['loss']
elif (df_trials['loss'].max() <= 1.0) and (df_trials['loss'].min() >= 0.0):
    df_trials['roc_auc'] = 1 - df_trials['loss']
else:
    df_trials['roc_auc'] = df_trials['loss']

pdf_dir = repo_root / "Figures" / "PDF"
pdf_dir.mkdir(parents=True, exist_ok=True)

hyperparams = [col for col in df_trials.columns if col not in ['loss', 'roc_auc', 'tid']]

for param in hyperparams:
    plt.figure(figsize=(12, 8))
    # Jitter for integer/discrete params
    if df_trials[param].dtype in [np.int64, np.int32]:
        x_vals = jitter(df_trials, param, amount=0.5)
    else:
        x_vals = df_trials[param]
    sc = plt.scatter(
        x_vals, df_trials['roc_auc'],
        c=df_trials['tid'], cmap='viridis', alpha=0.6
    )
    plt.xlabel(param)
    plt.ylabel("ROC AUC Score")
    plt.title(f"Hyperopt Trials: {param} vs ROC AUC")
    plt.colorbar(sc, label="Trial Iteration")
    plt.tight_layout()
    pdf_path = pdf_dir / f"XGB_Hyperopt_{param}_Scatter.pdf"
    plt.savefig(pdf_path)
    plt.close()

print(f"Hyperopt scatter plots saved as PDF in: {pdf_dir}")