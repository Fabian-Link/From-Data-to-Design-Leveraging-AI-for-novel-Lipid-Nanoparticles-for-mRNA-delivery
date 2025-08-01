import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from boruta import BorutaPy
from sklearn.model_selection import train_test_split
import pickle
from pathlib import Path
from Molecular_Descriptor_Calculation import calculate_descriptors

###################
#Datapreprocessing#
###################

repo_root = Path(__file__).resolve().parent.parent

# Calculate descriptors from AGILE_Dataset.csv and save to All Data
agile_dataset_path = repo_root / "Datasets" / "All Data" / "AGILE_Dataset.csv"
descriptor_output_path = repo_root / "Datasets" / "All Data" / "AGILE_Dataset_Mordred_2d.csv"
calculate_descriptors(agile_dataset_path, output_filename=descriptor_output_path)

# Loading descriptors
df_descriptors = pd.read_csv(descriptor_output_path, low_memory=False)

# Drop the SMILES column
data = df_descriptors.drop(columns=['SMILES'])

#Filtering out numerical values as well as booleans to remove missing/non-numerical values
column_num = []
column_bool = []

for column in data.columns:
    column_type = data[column].dtype
    # Skip non-numerical values
    if column_type == 'object':
        pass
    elif column_type == 'bool':
        column_bool.append(column)
    else:
        column_num.append(column)

# Convert boolean columns to integers
data[column_bool] = data[column_bool].astype(int)

# Concatinate the columns to a new dataframe without missing values/non-numerical values
data = data[column_num + column_bool]

# Remove constant values
constant_columns = [col for col in data.columns if data[col].nunique() == 1]
data.drop(columns=constant_columns, inplace=True)

# Adding Labels
df_labels = pd.read_csv(agile_dataset_path, low_memory=False)
data_processed = pd.concat([df_labels[['expt_Hela']], data], axis=1)

#################
#Standartscaling#
#################

# Calculate the threshold for binary classification based on 'expt_Hela'
hela_threshold = data_processed['expt_Hela'].mean()

# Create binary labels based on the threshold for 'expt_Hela'
data_processed['HeLa_label'] = (data_processed['expt_Hela'] >= hela_threshold).astype(int)

# Remove 'expt_Hela' column
data_processed.drop(columns=['expt_Hela'], inplace=True)

# Separate features from the target label
features = data_processed.drop(columns=['HeLa_label'])
target = data_processed['HeLa_label'].astype('int')

# Scale the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Convert the scaled features back to a DataFrame
scaled_features_df = pd.DataFrame(scaled_features, columns=features.columns)

# Split the data into training, validation, and testing sets (80-20 split)
X_unfiltered = scaled_features_df
y_unfiltered = target
X_train_unfiltered, X_test_unfiltered, y_train_unfiltered, y_test_unfiltered = train_test_split(X_unfiltered, y_unfiltered, test_size=0.2, random_state=42, stratify=y_unfiltered)

# Define output directory for splits
splits_dir = repo_root / "Datasets" / "Datasplits" / "All Features"
splits_dir.mkdir(parents=True, exist_ok=True)

# Save each split as a CSV file
X_train_unfiltered.to_csv(splits_dir / "X_train_unfiltered.csv", index=False)
X_test_unfiltered.to_csv(splits_dir / "X_test_unfiltered.csv", index=False)
y_train_unfiltered.to_csv(splits_dir / "y_train_unfiltered.csv", index=False)
y_test_unfiltered.to_csv(splits_dir / "y_test_unfiltered.csv", index=False)

print("Data splits saved as CSV files in:", splits_dir)

##########################
#Boruta Feature Selection#
##########################

#Update Numpy Library values to resolve error with Boruta
np.int = np.int32
np.float = np.float64
np.bool = np.bool_

# define random forest classifier, with utilising all cores and ampling in proportion to y labels
rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)

# define Boruta feature selection method
feat_selector = BorutaPy(rf, n_estimators='auto', perc=100, alpha=0.05, two_step=True, max_iter=100, random_state=42, verbose=2)

# find all relevant features
feat_selector.fit(scaled_features_df.values, target.values)

# check selected features
feat_selector.support_

###########################################################
#Filtering original Mordred dataframe to selected features#
###########################################################

# Extract the names of the selected features
selected_features = features.columns[feat_selector.support_]

# call transform() on X to filter it down to selected features
X_filtered = feat_selector.transform(features.values)

# Scale the features
scaler_filtered = StandardScaler()
scaled_features = scaler_filtered.fit_transform(X_filtered)

# Convert the scaled features back to a DataFrame
scaled_X_filtered_df = pd.DataFrame(scaled_features, columns=selected_features)

# Convert the filtered array (X_filtered) into a DataFrame using the selected feature names
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
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler_filtered , f)

print("Data preprocessed successfully.")