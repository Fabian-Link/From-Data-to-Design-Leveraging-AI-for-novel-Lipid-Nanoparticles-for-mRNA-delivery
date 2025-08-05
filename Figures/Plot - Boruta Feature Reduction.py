import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from plot_style import set_plot_style
set_plot_style()

# Load the Boruta progress CSV
repo_root = Path(__file__).resolve().parent.parent
csv_path = repo_root / "Classifier" / "Experiments" / "Boruta_Feature_Reduction_Progress.csv"
df = pd.read_csv(csv_path)

# Plot the feature reduction process
plt.figure()
plt.plot(df['iteration'], df['confirmed'], label='Confirmed')
plt.plot(df['iteration'], df['tentative'], label='Tentative')
plt.plot(df['iteration'], df['rejected'], label='Rejected')
plt.xlabel('Iteration', fontsize=30, fontweight='bold')
plt.ylabel('Number of Features', fontsize=30, fontweight='bold')
plt.title('Boruta Feature Reduction', fontsize=30, fontweight='bold')
plt.legend(fontsize=30, prop={'weight': 'bold'})
plt.tight_layout()

# Save the plot as PDF
pdf_dir = repo_root / "Figures" / "PDF"
pdf_dir.mkdir(parents=True, exist_ok=True)
pdf_path = pdf_dir / "Boruta_Feature_Reduction.pdf"
plt.savefig(pdf_path)
plt.close()

print(f"Plot saved as PDF at: {pdf_path}")