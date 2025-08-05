import matplotlib.pyplot as plt
import seaborn as sns

def set_plot_style():
    # General matplotlib style
    plt.rcParams.update({
        "figure.figsize": (12, 8),
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "axes.labelweight": "bold",
        "axes.linewidth": 1.5,
        "axes.grid": False,
        "grid.alpha": 0.3,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "legend.frameon": True,
        "legend.loc": "best",
        "font.size": 14,
        "font.family": "sans-serif",
        "savefig.dpi": 300,
        "lines.linewidth": 2,
        "lines.markersize": 8,
        "axes.prop_cycle": plt.cycler(color=[
            "#3A5F75",  # muted blue
            "#81B3B0",  # pastel teal
            "#EABA6B",  # pastel yellow
            "#F7A399",  # pastel coral
            "#B7A9E4",  # pastel lavender
            "#B7E4C7",  # pastel mint
            "#F7CAC9",  # pastel pink
            "#A3C9A8",  # pastel green
        ])
    })

    # Seaborn style
    sns.set_style("white")
    sns.set_context("talk", font_scale=1.2)

# Usage in other scripts:
# from plot_style import set_plot_style
# set_plot_style()