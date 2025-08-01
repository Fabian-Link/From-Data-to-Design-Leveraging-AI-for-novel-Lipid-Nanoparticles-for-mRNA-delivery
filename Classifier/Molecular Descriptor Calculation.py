from rdkit import Chem, RDLogger
import mordred
from mordred import Calculator, descriptors
import pandas as pd

def main():
    # Load Dataset
    df = pd.read_excel('Homoserine_all.xlsx')

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
    output_filename = 'Homoserine_all_Mordred_2d.csv'
    df_mordred_2d.to_csv(output_filename, index=False)

    print("csv file saved successfully.")

if __name__ == "__main__":
    main()