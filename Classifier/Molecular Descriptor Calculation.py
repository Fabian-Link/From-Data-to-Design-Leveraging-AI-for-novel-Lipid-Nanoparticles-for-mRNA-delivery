from rdkit import Chem, RDLogger
import mordred
from mordred import Calculator, descriptors
import pandas as pd
from pathlib import Path

def main():
    # Load Dataset
    repo_root = Path(__file__).resolve().parent.parent
    smiles_path = repo_root / 'Predictions' / 'Validation' / 'XY-Lipids_smiles.xlsx'
    df = pd.read_excel(smiles_path)

    # Extract the SMILES column
    dataset = df[['SMILES']]

    # Transform the SMILES into mol Objects
    mol_list = []

    for smile in dataset['SMILES']:
        mol = Chem.MolFromSmiles(smile)
        mol_list.append(mol)

    dataset_mol = pd.concat([dataset, pd.DataFrame(mol_list, columns = (['Mol']))], axis=1)

    # Initializing the Calculator class from the Mordred library
    calc = Calculator(descriptors, ignore_3D=True)

    # Calculating the 2D Descriptors of the Dataset
    all_desc = calc.pandas(dataset_mol['Mol'])

    # Appending the calculated descriptors to the corresponding SMILES
    df_mordred_2d = pd.concat([dataset['SMILES'], all_desc], axis=1)

    # Save the molecular descriptors in a .csv file
    output_filename =  smiles_path.parent / 'XY-Lipids_all_Mordred_2d.csv'
    df_mordred_2d.to_csv(output_filename, index=False)

    print("csv file saved successfully at:", output_filename)

if __name__ == "__main__":
    main()