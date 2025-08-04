import pandas as pd
import pickle
from pathlib import Path
from Molecular_Descriptor_Calculation import calculate_descriptors
from Classifier.Tanimoto import run_tanimoto
from Classifier.Merge_Tanimoto_with_Predictions import merge_predictions_with_tanimoto

###################################
# User input
###################################

repo_root = Path(__file__).resolve().parent.parent

# 1. Training set for feature filtering
train_path = repo_root / "Datasets" / "All Data" / "AGILE_Dataset_Boruta_feature_reduction.csv"

# 2. File to predict on
smiles_excel_path = repo_root / 'Predictions' / 'Extrapolation' / 'Homoserine_smiles.xlsx'

###################################
# Automated output naming
###################################

input_stem = smiles_excel_path.stem.replace("_smiles", "")  # Remove "_smiles" for cleaner output names

descriptor_output_filename = smiles_excel_path.with_name(f"{input_stem}_all_Mordred_2d.csv")
output_predictions = smiles_excel_path.with_name(f"{input_stem}_Predictions.xlsx")
output_tanimoto_path = smiles_excel_path.with_name(f"{input_stem}_Tanimoto_Distances.xlsx")
final_output_path = smiles_excel_path.with_name(f"{input_stem}_Predictions_with_Tanimotot.xlsx")

###################################
# Main workflow
###################################

# Load training data and extract features
df_train = pd.read_csv(train_path)
train_features = df_train.drop(columns=["SMILES"]).columns

# Calculate descriptors and get the output file path
descriptor_csv_path = calculate_descriptors(smiles_excel_path, output_filename=descriptor_output_filename)

# Load prediction data and filter using training features
df_pred = pd.read_csv(descriptor_csv_path)
df_filtered = df_pred[train_features].copy()
df_filtered['SMILES'] = df_pred['SMILES']
df_filtered_pred = df_filtered[~df_filtered['SMILES'].isin(df_train['SMILES'])].copy()
X = df_filtered_pred.drop(columns=['SMILES'])
object_columns = X.select_dtypes(include=['object']).columns
for column in object_columns:
    X[column] = pd.to_numeric(X[column], errors='coerce')
X.fillna(0, inplace=True)

# Load the saved scaler
scaler_filename = repo_root / "Classifier" / "scaler.pkl"
with open(scaler_filename, "rb") as f:
    scaler = pickle.load(f)
X_scaled = scaler.transform(X)
df_X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Load the saved model
best_model_filename = Path(__file__).parent / "Models" / "xgb_final.pkl"
with open(best_model_filename, "rb") as f:
    best_model = pickle.load(f)

# Perform predictions and get prediction probabilities
predictions = best_model.predict(df_X_scaled)
probabilities = best_model.predict_proba(df_X_scaled)
result_df = pd.DataFrame({
    'SMILES': df_filtered_pred['SMILES'],
    'Class': predictions,
    'Probability': probabilities.max(axis=1)
})
result_df = result_df.sort_values(by=['Class', 'Probability'], ascending=[False, False])
result_df.to_excel(output_predictions, index=False)

# Run Tanimoto calculation
training_csv_path = train_path  # Use same training set for tanimoto
run_tanimoto(training_csv_path, smiles_excel_path, output_tanimoto_path)

# Merge predictions with Tanimoto
merge_predictions_with_tanimoto(output_predictions, output_tanimoto_path, final_output_path)

print("All steps completed. Final file at:", final_output_path)