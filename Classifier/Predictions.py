import pandas as pd
import pickle
from pathlib import Path

###################################
# Preprocessing the Descriptor set#
###################################

# Define repo root
repo_root = Path(__file__).resolve().parent.parent

# Load training data and extract features
train_path = repo_root / "Datasets" / "All Data" / "AGILE_Dataset_Boruta_feature_reduction.csv"
df_train = pd.read_csv(train_path)
train_features = df_train.drop(columns=["SMILES"]).columns

# Load prediction data and filter using training features
pred_path = repo_root / "Predictions" / "Validation" / "XY-Lipids_all_Mordred_2d.csv"
df_pred = pd.read_csv(pred_path)
df_filtered = df_pred[train_features].copy()

# Add the SMILES column back to the filtered dataframe
df_filtered['SMILES'] = df_pred['SMILES']

# Filter out SMILES already present in the training dataset
df_filtered_pred = df_filtered[~df_filtered['SMILES'].isin(df_train['SMILES'])].copy()

# Drop the SMILES column from the features
X = df_filtered_pred.drop(columns=['SMILES'])

# Identify object columns
object_columns = X.select_dtypes(include=['object']).columns

# Convert object columns to numeric, if possible
for column in object_columns:
    X[column] = pd.to_numeric(X[column], errors='coerce')
    
# Fill any remaining NaN values resulting from the conversion
X.fillna(0, inplace=True)

# Load the saved scaler
scaler_filename = Path(__file__).parent / "scaler.pkl"
with open(scaler_filename, "rb") as f:
    scaler = pickle.load(f)

# Apply the scaler to the new features
X_scaled = scaler.transform(X)

# Convert the scaled array back to a DataFrame
df_X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

##############
# Predictions#
##############

# Load the saved model
best_model_filename = Path(__file__).parent / "Models" / "xgb_final.pkl"
with open(best_model_filename, "rb") as f:
    best_model = pickle.load(f)

# Perform predictions and get prediction probabilities
predictions = best_model.predict(df_X_scaled)
probabilities = best_model.predict_proba(df_X_scaled)

# Create a dataframe with SMILES, predicted class, and probabilities
result_df = pd.DataFrame({
    'SMILES': df_filtered_pred['SMILES'],
    'Class': predictions,
    'Probability': probabilities.max(axis=1)  # Assuming you want the max probability for the predicted class
})

# Sort by 'Class' first (with '1' at the top) and then by 'Probability' within each class
result_df = result_df.sort_values(by=['Class', 'Probability'], ascending=[False, False])

# Save the molecular descriptors in a .csv file
output_filename = repo_root / "Predictions" / "Validation" / "XY-Lipids_Predictions.xlsx"
result_df.to_excel(output_filename, index=False)

print("CSV file saved successfully at:", output_filename)