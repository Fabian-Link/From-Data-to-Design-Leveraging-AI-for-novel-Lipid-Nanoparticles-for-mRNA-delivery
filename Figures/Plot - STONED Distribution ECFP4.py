import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
data_dir  = repo_root / "Generation of novel lipids"
out_dir   = repo_root / "Figures" / "PDF"
out_dir.mkdir(parents=True, exist_ok=True)

# One CSV per set (single numeric column with similarities)
files = {
    "Training NN": data_dir / "tanimoto_training_nn.csv",     # leave-one-out NN on training set
    "STONED 0.8":  data_dir / "tanimoto_STONED_0.8.csv",
    "STONED 0.6":  data_dir / "tanimoto_STONED_0.6.csv",
    "STONED 0.4":  data_dir / "tanimoto_STONED_0.4.csv",
}

def load_sims(path: Path, every=100):
    s = pd.read_csv(path, header=0).select_dtypes(include=[np.number]).iloc[:,0]
    return pd.to_numeric(s, errors="coerce").dropna().values[::every]

bins = np.linspace(0.0, 1.0, 41)

plt.figure(figsize=(12, 7))

# plot training first so others overlay it
order = ["Training NN", "STONED 0.8", "STONED 0.6", "STONED 0.4"]
for label in order:
    f = files[label]
    if not f.exists():
        print(f"[skip] {label}: {f} not found")
        continue
    vals = load_sims(f, every=100)  # subsample every 100th value
    alpha = 0.35 if label == "Training NN" else 0.8
    plt.hist(vals, bins=bins, density=True, alpha=alpha, edgecolor="none", label=label)

plt.xlabel("Nearest-neighbor ECFP4 Tanimoto to training set")
plt.ylabel("Density")
plt.title("Domain overlap and breadth by similarity bin")
plt.legend(frameon=False, loc="upper left")
plt.tight_layout()
plt.savefig(out_dir / "NN_Tanimoto_overlap_bins_with_training.pdf")
plt.show()
