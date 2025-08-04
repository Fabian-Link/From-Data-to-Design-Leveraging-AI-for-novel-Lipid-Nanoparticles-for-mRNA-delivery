import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import umap

from plot_style import set_plot_style
set_plot_style()

repo_root = Path(__file__).resolve().parent.parent

# --- Reduced Features ---
splits_dir_reduced = repo_root / "Datasets" / "Datasplits" / "Reduced Features"
X_train = pd.read_csv(splits_dir_reduced / "X_train.csv")
X_test = pd.read_csv(splits_dir_reduced / "X_test.csv")
y_train = pd.read_csv(splits_dir_reduced / "y_train.csv")
y_test = pd.read_csv(splits_dir_reduced / "y_test.csv")
X_reduced = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)
y_reduced = pd.concat([y_train, y_test], axis=0).reset_index(drop=True)
labels_reduced = y_reduced['HeLa_label'] if 'HeLa_label' in y_reduced.columns else y_reduced

# --- All Features ---
splits_dir_all = repo_root / "Datasets" / "Datasplits" / "All Features"
X_train_all = pd.read_csv(splits_dir_all / "X_train_unfiltered.csv")
X_test_all = pd.read_csv(splits_dir_all / "X_test_unfiltered.csv")
y_train_all = pd.read_csv(splits_dir_all / "y_train_unfiltered.csv")
y_test_all = pd.read_csv(splits_dir_all / "y_test_unfiltered.csv")
X_all = pd.concat([X_train_all, X_test_all], axis=0).reset_index(drop=True)
y_all = pd.concat([y_train_all, y_test_all], axis=0).reset_index(drop=True)
labels_all = y_all['HeLa_label'] if 'HeLa_label' in y_all.columns else y_all

# Get pastel colors from plot_style
colors = [c['color'] for c in plt.rcParams['axes.prop_cycle']]
label_colors = {0: colors[0], 1: colors[2]}

# --- UMAP Reduced Features ---
umap_model_reduced = umap.UMAP(
    n_components=2,
    random_state=42,
    n_neighbors=50,
    min_dist=0.7,
    metric='euclidean',
    n_jobs=-1
)
embedding_reduced = umap_model_reduced.fit_transform(X_reduced)

plt.figure()
for label in sorted(set(labels_reduced)):
    idx = labels_reduced == label
    plt.scatter(
        embedding_reduced[idx, 0], embedding_reduced[idx, 1],
        c=label_colors[label], label=f"HeLa Label {label}",
        alpha=1, edgecolor='none', s=40
    )
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.title('UMAP - Reduced Features')
plt.legend()
plt.tight_layout()
pdf_dir = repo_root / "Figures" / "PDF"
pdf_dir.mkdir(parents=True, exist_ok=True)
pdf_path_reduced = pdf_dir / "UMAP_HeLa_Label_Reduced.pdf"
plt.savefig(pdf_path_reduced)
plt.close()

# --- UMAP All Features ---
umap_model_all = umap.UMAP(
    n_components=2,
    random_state=42,
    n_neighbors=50,
    min_dist=0.7,
    metric='euclidean',
    n_jobs=-1
)
embedding_all = umap_model_all.fit_transform(X_all)

plt.figure()
for label in sorted(set(labels_all)):
    idx = labels_all == label
    plt.scatter(
        embedding_all[idx, 0], embedding_all[idx, 1],
        c=label_colors[label], label=f"HeLa Label {label}",
        alpha=1, edgecolor='none', s=40
    )
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.title('UMAP - All Features')
plt.legend()
plt.tight_layout()
pdf_path_all = pdf_dir / "UMAP_HeLa_Label_All.pdf"
plt.savefig(pdf_path_all)
plt.close()

print(f"UMAP plots saved as PDF at:\n{pdf_path_reduced}\n{pdf_path_all}")