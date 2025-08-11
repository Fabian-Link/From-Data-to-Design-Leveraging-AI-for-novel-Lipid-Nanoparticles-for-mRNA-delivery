import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from pathlib import Path
import umap
import tempfile
import os
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import trustworthiness

from plot_style import set_plot_style
set_plot_style()

repo_root = Path(__file__).resolve().parent.parent
sys.path.append(str(repo_root / "Classifier"))
from Molecular_Descriptor_Calculation import calculate_descriptors

# 1. Load all descriptors for All Features set
splits_dir_all = repo_root / "Datasets" / "Datasplits" / "All Features"
X_train = pd.read_csv(splits_dir_all / "X_train_unfiltered.csv")
X_test = pd.read_csv(splits_dir_all / "X_test_unfiltered.csv")
X_all = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)

# Remove non-numeric columns (e.g., SMILES, NAME)
numeric_cols = X_all.select_dtypes(include=[np.number]).columns.tolist()
X_all_filtered = X_all[numeric_cols].dropna(axis=1, how='any')
X_all_filtered = X_all_filtered.loc[:, X_all_filtered.nunique() > 1]  # Drop constant columns

# 2. Define STONED files and colors
stoned_dir = repo_root / "Generation of novel lipids"
stoned_files = [
    stoned_dir / "STONED_generated_ionizablelipids_positive_sim_04_mut10_smiles.csv",
    stoned_dir / "STONED_generated_ionizablelipids_positive_sim_06_mut10_smiles.csv",
    stoned_dir / "STONED_generated_ionizablelipids_positive_sim_08_mut10_smiles.csv"
]
stoned_labels = ["STONED 0.4", "STONED 0.6", "STONED 0.8"]
colors = [c['color'] for c in plt.rcParams['axes.prop_cycle']]

for stoned_file, stoned_label in zip(stoned_files, stoned_labels):
    # --- Prepare descriptor output path ---
    desc_out_path = stoned_dir / f"descriptors_{stoned_label.replace(' ', '_')}_all_features.csv"

    # --- Sample subgroups of SMILES for every combination of seed and mutation depth ---
    stoned_df = pd.read_csv(stoned_file)
    smiles_list = stoned_df['SMILES'].tolist()

    n_seeds = 100
    n_mut_depths = 10
    n_per_group = len(smiles_list) // (n_seeds * n_mut_depths)

    sampled_smiles = []
    for seed_idx in range(n_seeds):
        for mut_depth in range(n_mut_depths):
            group_start = (seed_idx * n_mut_depths + mut_depth) * n_per_group
            group_end = group_start + n_per_group
            group = smiles_list[group_start:group_end]
            if group:
                sampled_smiles.append(random.choice(group))  # or group[0] for deterministic

    stoned_sampled_smiles = pd.DataFrame({'SMILES': sampled_smiles})

    # --- Calculate or load descriptors (save immediately after calculation) ---
    if desc_out_path.exists():
        stoned_desc = pd.read_csv(desc_out_path)
        print(f"Loaded existing descriptors: {desc_out_path}")
    else:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
            stoned_sampled_smiles.to_csv(tmp.name, index=False)
            smiles_csv_path = tmp.name

        desc_csv_path = calculate_descriptors(smiles_csv_path)
        stoned_desc = pd.read_csv(desc_csv_path)
        stoned_desc.to_csv(desc_out_path, index=False)
        os.remove(smiles_csv_path)
        print(f"Calculated and saved descriptors: {desc_out_path}")

    # --- Filter STONED descriptors to numeric columns and drop NaNs/constant columns ---
    stoned_desc_filtered = stoned_desc[numeric_cols].dropna(axis=1, how='any')
    stoned_desc_filtered = stoned_desc_filtered.loc[:, stoned_desc_filtered.nunique() > 1]
    # Align columns with X_all_filtered
    common_cols = [col for col in X_all_filtered.columns if col in stoned_desc_filtered.columns]
    X_all_final = X_all_filtered[common_cols].apply(pd.to_numeric, errors='coerce').dropna(axis=0, how='any').reset_index(drop=True)
    stoned_desc_final = stoned_desc_filtered[common_cols].apply(pd.to_numeric, errors='coerce').dropna(axis=0, how='any').reset_index(drop=True)

    # --- Concatenate and label ---
    X_umap = pd.concat([X_all_final, stoned_desc_final], axis=0).reset_index(drop=True)
    labels = np.array(['Training Set'] * len(X_all_final) + [stoned_label] * len(stoned_desc_final))
    labels = labels[X_umap.index]

    # --- Fit new scaler and scale ---
    scaler = StandardScaler()
    X_umap_scaled = scaler.fit_transform(X_umap)

    # --- Run UMAP with default params ---
    umap_model = umap.UMAP(n_components=2, random_state=42, n_neighbors=100, min_dist=1.0, metric='manhattan')
    embedding = umap_model.fit_transform(X_umap_scaled)

    # --- Print UMAP metrics ---
    trust = trustworthiness(X_umap_scaled, embedding, n_neighbors=15)
    print(f"Trustworthiness: {trust:.3f}")

    # --- Plot ---
    plt.figure(figsize=(12, 8))
    for group, color in zip(['Training Set', stoned_label], [colors[0], colors[2]]):
        idx = labels == group
        plt.scatter(
            embedding[idx, 0], embedding[idx, 1],
            c=color, label=group, alpha=0.7, edgecolor='none', s=40
        )
    plt.xlabel('UMAP 1', fontsize=30, fontweight='bold')
    plt.ylabel('UMAP 2', fontsize=30, fontweight='bold')
    plt.title(f'UMAP: Training set vs {stoned_label}', fontsize=30, fontweight='bold')
    plt.legend(fontsize=30, prop={'weight': 'bold'})
    plt.tight_layout()
    pdf_dir = repo_root / "Figures" / "PDF"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = pdf_dir / f"UMAP_TrainingSet_vs_{stoned_label.replace(' ', '_')}.pdf"
    plt.savefig(pdf_path)
    plt.close()
    print(f"UMAP plot saved as PDF at: {pdf_path}")