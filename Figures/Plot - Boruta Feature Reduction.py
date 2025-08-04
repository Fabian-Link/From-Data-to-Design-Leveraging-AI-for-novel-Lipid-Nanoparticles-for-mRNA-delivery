import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load the Boruta progress CSV
repo_root = Path(__file__).resolve().parent.parent
csv_path = repo_root / "Classifier" / "Experiments" / "Boruta_Feature_Reduction_Progress.csv"
df = pd.read_csv(csv_path)

# Plot the feature reduction process
plt.figure(figsize=(8, 6))
plt.plot(df['iteration'], df['confirmed'], label='Confirmed', color='green')
plt.plot(df['iteration'], df['tentative'], label='Tentative', color='orange')
plt.plot(df['iteration'], df['rejected'], label='Rejected', color='red')
plt.xlabel('Iteration')
plt.ylabel('Number of Features')
plt.title('Boruta Feature Reduction Process')
plt.legend()
plt.tight_layout()

# Save the plot as PDF
pdf_dir = repo_root / "Figures" / "PDF"
pdf_dir.mkdir(parents=True, exist_ok=True)
pdf_path = pdf_dir / "Boruta_Feature_Reduction_Process.pdf"
plt.savefig(pdf_path)
plt.close()

print(f"Plot saved as PDF at: {pdf_path}")