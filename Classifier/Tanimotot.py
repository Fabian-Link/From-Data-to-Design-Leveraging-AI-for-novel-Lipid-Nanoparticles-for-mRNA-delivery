import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from tqdm import tqdm

def calculate_tanimoto_distance(fp1, fp2):
    """Compute the Tanimoto distance (1 - similarity) between two fingerprints."""
    return 1 - DataStructs.TanimotoSimilarity(fp1, fp2)

def get_morgan_fingerprint(smiles, radius=2, n_bits=2048):
    """Convert SMILES to RDKit molecule and compute Morgan fingerprint."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)

def calculate_average_tanimoto_distances(training_csv, new_molecules_excel, output_file):
    # Load training molecules
    training_df = pd.read_csv(training_csv, usecols=['SMILES'])
    training_smiles = training_df['SMILES'].dropna().tolist()

    # Load new molecules
    new_df = pd.read_excel(new_molecules_excel, usecols=['SMILES'])
    new_smiles = new_df['SMILES'].dropna().tolist()
    
    # Compute fingerprints for training set
    training_fps = [get_morgan_fingerprint(sm) for sm in training_smiles]
    training_fps = [fp for fp in training_fps if fp is not None]  # Remove None values
    
    average_distances = []
    
    for new_smile in tqdm(new_smiles, desc='Processing new molecules', unit='molecule'):
        new_fp = get_morgan_fingerprint(new_smile)
        if new_fp is None:
            average_distances.append(1.0)  # Assign max distance if invalid SMILES
            continue
        
        # Compute distances with training set
        distances = [calculate_tanimoto_distance(new_fp, train_fp) for train_fp in training_fps]
        avg_distance = np.mean(distances) if distances else 1.0
        average_distances.append(avg_distance)
    
    # Add results to DataFrame
    new_df['Average_Tanimoto_Distance'] = average_distances
    new_df.to_excel(output_file, index=False)
    
    return new_df

# File paths
training_csv_path = 'AGILE_Dataset.csv'
new_molecules_excel_path = 'Homoserine_all.xlsx'
output_excel_path = 'Homoserine_all_Tanimoto_Distances.xlsx'

# Run analysis
updated_df = calculate_average_tanimoto_distances(training_csv_path, new_molecules_excel_path, output_excel_path)

# Display results
print(updated_df.head())