import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- config ---
repo_root = Path(__file__).resolve().parent.parent
data_dir   = repo_root / "Generation of novel lipids"   # adjust if needed
out_dir    = repo_root / "Figures" / "PDF"
out_dir.mkdir(parents=True, exist_ok=True)

# If you have a training-set nearest-neighbor CSV, put it here; otherwise it’ll be skipped.
files = {
    "Training NN": data_dir / "tanimoto_training_nn.csv",          # optional
    "STONED 0.8":  data_dir / "tanimoto_STONED_0.8.csv",
    "STONED 0.6":  data_dir / "tanimoto_STONED_0.6.csv",
    "STONED 0.4":  data_dir / "tanimoto_STONED_0.4.csv",
}

# subsample factor: take every 100th value
SUBSAMPLE = 100

# columns we’ll try to read from each CSV (first match wins)
candidate_cols = ("tanimoto", "nn_tanimoto", "similarity", "NearestNeighbor")

palette = {
    "Training NN": "#CFD8DC",
    "STONED 0.8":  "#EDC07B",
    "STONED 0.6":  "#F3DF9A",
    "STONED 0.4":  "#9FC5C8",
}

def load_similarities(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    for c in candidate_cols:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce").dropna().values
            return s[::SUBSAMPLE]
    # fallback: first numeric column
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) == 0:
        raise ValueError(f"No numeric columns found in {path}")
    s = pd.to_numeric(df[num_cols[0]], errors="coerce").dropna().values
    return s[::SUBSAMPLE]

# --- plot ---
plt.figure(figsize=(10, 6))
bins = np.linspace(0.0, 1.0, 41)   # old version: full range 0–1

for label, fpath in files.items():
    if not fpath.exists():
        print(f"[skip] {label}: file not found -> {fpath}")
        continue
    vals = load_similarities(fpath)
    plt.hist(vals, bins=bins, density=True, alpha=0.85, edgecolor="none",
             color=palette.get(label, None), label=label)

plt.xlabel("Nearest-neighbor ECFP4 Tanimoto to training set")
plt.ylabel("Density")
plt.title("Domain overlap and breadth by similarity bin")
plt.legend(frameon=False, loc="upper left")
plt.tight_layout()
plt.savefig(out_dir / "NN_Tanimoto_overlap_bins_subsampled.pdf")
plt.show()