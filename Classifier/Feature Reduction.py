import pandas as pd
import numpy as np
import sys
import io
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from boruta import BorutaPy
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import pickle
from pathlib import Path
from Molecular_Descriptor_Calculation import calculate_descriptors

###################
# Datapreprocessing
###################

repo_root = Path(__file__).resolve().parent.parent

# Calculate descriptors from AGILE_Dataset.csv and save to All Data
agile_dataset_path = repo_root / "Datasets" / "All Data" / "AGILE_Dataset.csv"
descriptor_output_path = repo_root / "Datasets" / "All Data" / "AGILE_Dataset_Mordred_2d_descriptors.csv"
calculate_descriptors(agile_dataset_path, output_filename=descriptor_output_path)

# Loading descriptors
df_descriptors = pd.read_csv(descriptor_output_path, low_memory=False)

# Drop the SMILES column
data = df_descriptors.drop(columns=['SMILES'])

# Filtering out numerical values as well as booleans to remove missing/non-numerical values
column_num = []
column_bool = []
for column in data.columns:
    column_type = data[column].dtype
    if column_type == 'object':
        pass
    elif column_type == 'bool':
        column_bool.append(column)
    else:
        column_num.append(column)
data[column_bool] = data[column_bool].astype(int)
data = data[column_num + column_bool]

# Remove constant values
constant_columns = [col for col in data.columns if data[col].nunique() == 1]
data.drop(columns=constant_columns, inplace=True)

# Adding Labels
df_labels = pd.read_csv(agile_dataset_path, low_memory=False)
data_processed = pd.concat([df_labels[['expt_Hela']], data], axis=1)

#################
# Standartscaling
#################

hela_threshold = data_processed['expt_Hela'].mean()
data_processed['HeLa_label'] = (data_processed['expt_Hela'] >= hela_threshold).astype(int)
data_processed.drop(columns=['expt_Hela'], inplace=True)

features = data_processed.drop(columns=['HeLa_label'])
target = data_processed['HeLa_label'].astype('int')

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
scaled_features_df = pd.DataFrame(scaled_features, columns=features.columns)

# Split the data into training, validation, and testing sets (80-20 split)
X_unfiltered = scaled_features_df
y_unfiltered = target
X_train_unfiltered, X_test_unfiltered, y_train_unfiltered, y_test_unfiltered = train_test_split(
    X_unfiltered, y_unfiltered, test_size=0.2, random_state=42, stratify=y_unfiltered
)

# Save each split as a CSV file
splits_dir = repo_root / "Datasets" / "Datasplits" / "All Features"
splits_dir.mkdir(parents=True, exist_ok=True)
X_train_unfiltered.to_csv(splits_dir / "X_train_unfiltered.csv", index=False)
X_test_unfiltered.to_csv(splits_dir / "X_test_unfiltered.csv", index=False)
y_train_unfiltered.to_csv(splits_dir / "y_train_unfiltered.csv", index=False)
y_test_unfiltered.to_csv(splits_dir / "y_test_unfiltered.csv", index=False)
print("Data splits saved as CSV files in:", splits_dir)

##########################
# Boruta Feature Selection
##########################

np.int = np.int32
np.float = np.float64
np.bool = np.bool_

rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
feat_selector = BorutaPy(rf, n_estimators='auto', perc=100, alpha=0.05, two_step=True, max_iter=100, random_state=42, verbose=2)

# Capture stdout during BorutaPy fit
old_stdout = sys.stdout
sys.stdout = mystdout = io.StringIO()

feat_selector.fit(scaled_features_df.values, target.values)

sys.stdout = old_stdout
boruta_output = mystdout.getvalue()

# Parse BorutaPy output for iteration info
boruta_progress = []
lines = boruta_output.strip().split('\n')
for i in range(0, len(lines), 4):
    block = lines[i:i+4]
    if len(block) == 4 and 'Iteration:' in block[0]:
        try:
            iteration = int(re.search(r'Iteration:\s*(\d+)', block[0]).group(1))
            confirmed = int(re.search(r'Confirmed:\s*(\d+)', block[1]).group(1))
            tentative = int(re.search(r'Tentative:\s*(\d+)', block[2]).group(1))
            rejected = int(re.search(r'Rejected:\s*(\d+)', block[3]).group(1))
            boruta_progress.append({
                "iteration": iteration,
                "confirmed": confirmed,
                "tentative": tentative,
                "rejected": rejected
            })
        except Exception:
            continue

# Save Boruta progress to CSV for visualization
experiments_dir = repo_root / "Classifier" / "Experiments"
experiments_dir.mkdir(parents=True, exist_ok=True)
boruta_progress_df = pd.DataFrame(boruta_progress)
boruta_progress_df.to_csv(experiments_dir / "Boruta_Feature_Reduction_Progress.csv", index=False)
print(f"Boruta feature reduction progress saved at: {experiments_dir / 'Boruta_Feature_Reduction_Progress.csv'}")

###########################################################
# Filtering original Mordred dataframe to selected features
###########################################################

selected_features = features.columns[feat_selector.support_]
X_filtered = feat_selector.transform(features.values)

scaler_filtered = StandardScaler()
scaled_features = scaler_filtered.fit_transform(X_filtered)
scaled_X_filtered_df = pd.DataFrame(scaled_features, columns=selected_features)
X_filtered_df = pd.DataFrame(scaled_X_filtered_df, columns=selected_features)

# Add back the "SMILES" column from the original dataframe
X_filtered_with_smiles = pd.concat([df_descriptors[['SMILES']], X_filtered_df], axis=1)

# Save the molecular descriptors in a .csv file
output_filename = repo_root / "Datasets" / "All Data" / "AGILE_Dataset_Boruta_feature_reduction.csv"
X_filtered_with_smiles.to_csv(output_filename, index=False)
print("CSV file saved successfully at:", output_filename)

# Split the data into training, validation, and testing sets (80-20 split)
X = X_filtered_df
y = target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Save the data splits as CSV files
splits_dir = repo_root / "Datasets" / "Datasplits" / "Reduced Features"
splits_dir.mkdir(parents=True, exist_ok=True)
X_train.to_csv(splits_dir / "X_train.csv", index=False)
X_test.to_csv(splits_dir / "X_test.csv", index=False)
y_train.to_csv(splits_dir / "y_train.csv", index=False)
y_test.to_csv(splits_dir / "y_test.csv", index=False)
print("Reduced feature data splits saved as CSV files in:", splits_dir)

# Save the StandardScaler object using pickle
scaler_path = repo_root / "Classifier" / "scaler.pkl"
with open(scaler_path, "wb") as f:
    pickle.dump(scaler_filtered, f)
print(f"Scaler saved successfully at: {scaler_path}")