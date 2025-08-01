from rdkit import Chem, RDLogger
import mordred
from mordred import Calculator, descriptors
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def calculate_descriptors(smiles_excel_path, output_filename=None):
    repo_root = Path(__file__).resolve().parent.parent
    smiles_path = Path(smiles_excel_path)
    df = pd.read_excel(smiles_path)

    # Extract the SMILES column
    dataset = df[['SMILES']]

    # Transform the SMILES into mol Objects
    mol_list = []
    for smile in tqdm(dataset['SMILES'], desc="Parsing SMILES"):
            mol = Chem.MolFromSmiles(smile)
            mol_list.append(mol)

    dataset_mol = pd.concat([dataset, pd.DataFrame(mol_list, columns=['Mol'])], axis=1)

    # Initializing the Calculator class from the Mordred library
    calc = Calculator(descriptors, ignore_3D=True)

    # Calculating the 2D Descriptors of the Dataset
    all_desc = calc.pandas(dataset_mol['Mol'], nproc=1)

    # Appending the calculated descriptors to the corresponding SMILES
    df_mordred_2d = pd.concat([dataset['SMILES'], all_desc], axis=1)

    # Set output filename dynamically
    if output_filename is None:
        output_filename = smiles_path.with_name(smiles_path.stem + '_all_Mordred_2d.csv')
    else:
        output_filename = Path(output_filename)

    df_mordred_2d.to_csv(output_filename, index=False)

    print("csv file saved successfully at:", output_filename)
    return output_filename