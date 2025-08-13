import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from pathlib import Path
import textwrap

from plot_style import set_plot_style
set_plot_style()

repo_root = Path(__file__).resolve().parent.parent
exp_dir = repo_root / "Classifier" / "Experiments"

# Load LazyPredict results
df_all = pd.read_csv(exp_dir / "LazyPredict_all_features.csv")
df_reduced = pd.read_csv(exp_dir / "LazyPredict_reduced_features.csv")

colors = [c['color'] for c in plt.rcParams['axes.prop_cycle']]
pdf_dir = repo_root / "Figures" / "PDF"
pdf_dir.mkdir(parents=True, exist_ok=True)

def wrap_labels(labels, width=18):
    return [textwrap.fill(str(l), width) for l in labels]

def plot_lazypredict(df, title, color, outpath, top_n=None, wrap_width=18):
    # keep only what we need and drop NAs
    df = df[['Model', 'ROC AUC']].dropna()
    if top_n is not None:
        df = df.nlargest(top_n, 'ROC AUC')
    df = df.sort_values('ROC AUC', ascending=True)

    # increase height for more space between bars
    fig_h = max(6, 0.6 * len(df) + 2)
    fig, ax = plt.subplots(figsize=(11, fig_h))

    bars = ax.barh(wrap_labels(df['Model'], wrap_width), df['ROC AUC'], color=color)

    ax.set_xlim(0, 1.0)
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.grid(axis='x', linestyle='--', alpha=0.25)
    ax.set_xlabel("ROC AUC", fontsize=30, fontweight='bold')
    ax.set_ylabel("")
    ax.set_title(title, fontsize=30, fontweight='bold', pad=12)
    ax.invert_yaxis()

    ax.tick_params(axis='y', labelsize=18)
    ax.tick_params(axis='x', labelsize=18)

    # reduce font size for bar labels and add more padding
    ax.bar_label(
        bars,
        labels=[f"{v:.2f}" for v in df['ROC AUC']],
        padding=10,
        fontsize=14,
        fontweight='bold'
    )

    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)

# Plot both figures
plot_lazypredict(
    df_all,
    "LazyPredict Model Screening (All Features)",
    colors[0],
    pdf_dir / "LazyPredict_Model_Screening_AllFeatures.pdf",
    top_n=None  # or e.g. 20 to show top-20
)

plot_lazypredict(
    df_reduced,
    "LazyPredict Model Screening (Reduced Features)",
    colors[2],
    pdf_dir / "LazyPredict_Model_Screening_ReducedFeatures.pdf",
    top_n=None
)

print("Saved polished figures to:", pdf_dir)
