import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from plot_style import set_plot_style
set_plot_style()

# Load the AGILE_Dataset
repo_root = Path(__file__).resolve().parent.parent
csv_path = repo_root / "Datasets" / "All Data" / "AGILE_Dataset.csv"
df = pd.read_csv(csv_path)

# Plot histogram of HeLa Expression values
plt.figure()
plt.hist(df['expt_Hela'], bins=50, edgecolor=None)
plt.xlabel('HeLa Expression Value')
plt.ylabel('Frequency')
plt.title('Distribution of HeLa Expression Values')
plt.tight_layout()

# Save the plot as PDF
pdf_dir = repo_root / "Figures" / "PDF"
pdf_dir.mkdir(parents=True, exist_ok=True)
pdf_path = pdf_dir / "HeLa_Expression_Distribution.pdf"
plt.savefig(pdf_path)
plt.close()

print(f"Plot saved as PDF at: {pdf_path}")

# Calculate mean of HeLa expression values
hela_mean = df['expt_Hela'].mean()

# Plot histogram with mean marker
plt.figure()
plt.hist(df['expt_Hela'], bins=50, edgecolor=None)
plt.axvline(x=hela_mean, color='#EABA6B', linestyle='--', linewidth=2, label=f'Class Threshold')
plt.xlabel('HeLa Expression Value')
plt.ylabel('Frequency')
plt.title('Distribution of HeLa Expression Values')
plt.legend()
plt.tight_layout()

# Save the plot as PDF
pdf_dir = repo_root / "Figures" / "PDF"
pdf_dir.mkdir(parents=True, exist_ok=True)
pdf_path = pdf_dir / "HeLa_Expression_Distribution_with_Mean.pdf"
plt.savefig(pdf_path)
plt.close()

print(f"Plot with mean marker saved as PDF at: {pdf_path}")