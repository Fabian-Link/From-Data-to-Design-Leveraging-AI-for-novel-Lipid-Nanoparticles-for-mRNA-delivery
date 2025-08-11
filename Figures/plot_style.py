import matplotlib.pyplot as plt
import seaborn as sns

def set_plot_style():
    plt.rcParams.update({
        "figure.figsize": (12, 8),
        "axes.titlesize": 24,
        "axes.titleweight": "bold",
        "axes.labelsize": 24,
        "axes.labelweight": "bold",
        "axes.linewidth": 1.5,
        "axes.grid": False,
        "grid.alpha": 0.3,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
        "legend.fontsize": 24,
        "legend.frameon": True,
        "legend.loc": "best",
        "legend.title_fontsize": 24,
        "font.size": 24,
        "font.family": "arial",
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

    sns.set_style("white")
    sns.set_context("talk", font_scale=1.2)