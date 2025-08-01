import pandas as pd

# Load the two Excel sheets
sheet_one_path = 'Homoserine_all_Predictions.xlsx'
sheet_two_path = 'Homoserine_all_Tanimoto_Distances.xlsx'

# Read the data from both sheets
sheet_one_df = pd.read_excel(sheet_one_path)
sheet_two_df = pd.read_excel(sheet_two_path)

# Merge the dataframes on the 'SMILES' column
merged_df = pd.merge(sheet_one_df, sheet_two_df[['SMILES', 'Average_Tanimoto_Distance']], on='SMILES', how='left')

# Save the result to a new Excel file
merged_df.to_excel('Homoserine_all_Predictions_with_Tanimoto.xlsx', index=False)

print("csv file saved successfully.")

# Display the result
print(merged_df.head())