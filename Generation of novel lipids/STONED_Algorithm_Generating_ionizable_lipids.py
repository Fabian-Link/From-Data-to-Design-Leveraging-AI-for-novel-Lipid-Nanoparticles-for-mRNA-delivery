import time 
import selfies
import rdkit
import random
import numpy as np
import random
import pandas as pd
from rdkit import Chem
from selfies import encoder, decoder
from rdkit.Chem import MolFromSmiles as smi2mol
from rdkit.Chem import AllChem
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity
from rdkit.Chem import Mol
from rdkit.Chem.AtomPairs.Sheridan import GetBPFingerprint, GetBTFingerprint
from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D
from rdkit.Chem import Draw
from syba.syba import SybaClassifier

from rdkit.Chem import MolToSmiles as mol2smi
from rdkit import RDLogger

##############################################################################################
#Modifying the SELFIE alphabet to only include tokens that are present in the seed structures#
##############################################################################################

df = pd.read_csv("AGILE_Dataset.csv")

# Extract the SMILES column
smiles_column = 'SMILES'
smiles_list = df[smiles_column].tolist()

# Define the SMILES for the common functional groups
functional_groups = {
    'Ester': 'CC(=O)OC',
    'Ether': 'CCOC',
    'Thioester': 'CC(=O)SC',
    'Thioether': 'CCSC',
    'Amide': 'CC(=O)NC',
    'Ethan': 'CC',
    'Propyl': 'CCC',
    'Hydroxy': 'CO',
    'Methylhydroxy': 'CCO',
    'Ethylhydroxy': 'CCCO',
    'Propylhydroxy': 'CCCCO',
    'Ethen': 'C=C',
    'Prop-1-ene': 'CC=C'
}

# Encode each functional group into SELFIES
functional_group_selfies = []
for group in functional_groups.values():
    try:
        selfies_sf = selfies.encoder(group)
        functional_group_selfies.append(selfies_sf)
        smiles_list.append(group)  # Append to original SMILES list
    except selfies.EncoderError:
        pass  # Handle encoding error, if any

# Encode each SMILES string into SELFIES for the original list
selfies_list = []
symbols_ionizablelipids = []
for smiles in smiles_list:
    try:
        # Encoding SMILES into SELFIE
        selfies_sf = selfies.encoder(smiles)
        selfies_list.append(selfies_sf)
        symbols = list(selfies.split_selfies(selfies_sf))
        symbols_ionizablelipids.extend(symbols)
    except selfies.EncoderError:
        pass  # Handle encoding error, if any

# Generate the alphabet from all SELFIES symbols
ionizablelipids_alphabet = selfies.get_alphabet_from_selfies(symbols_ionizablelipids)

# Append the functional group SELFIES directly to the alphabet
for selfies_sf in functional_group_selfies:
    ionizablelipids_alphabet.add(selfies_sf)

############################################################################################################
#In this section of the Script I import the diffrent librarys and functions from the Aspuru-Guzik et al.####
#GitHub:https://github.com/aspuru-guzik-group/stoned-selfies/blob/main/stoned_selfies_tut.ipynb#############
############################################################################################################

def randomize_smiles(mol):
    '''Returns a random (dearomatized) SMILES given an rdkit mol object of a molecule.
    Parameters:
    mol (rdkit.Chem.rdchem.Mol) :  RdKit mol object (None if invalid smile string smi)
    
    Returns:
    mol (rdkit.Chem.rdchem.Mol) : RdKit mol object  (None if invalid smile string smi)
    '''
    if not mol:
        return None

    Chem.Kekulize(mol)
    return rdkit.Chem.MolToSmiles(mol, canonical=False, doRandom=True, isomericSmiles=False,  kekuleSmiles=True) 

def sanitize_smiles(smi):
    '''Return a canonical smile representation of smi
    
    Parameters:
    smi (string) : smile string to be canonicalized 
    
    Returns:
    mol (rdkit.Chem.rdchem.Mol) : RdKit mol object                          (None if invalid smile string smi)
    smi_canon (string)          : Canonicalized smile representation of smi (None if invalid smile string smi)
    conversion_successful (bool): True/False to indicate if conversion was  successful 
    '''
    try:
        mol = smi2mol(smi, sanitize=True)
        smi_canon = mol2smi(mol, isomericSmiles=False, canonical=True)
        return (mol, smi_canon, True)
    except:
        return (None, None, False)
    
def get_selfie_chars(selfie):
    '''Obtain a list of all selfie characters in string selfie
    
    Parameters: 
    selfie (string) : A selfie string - representing a molecule 
    
    Example: 
    >>> get_selfie_chars('[C][=C][C][=C][C][=C][Ring1][Branch1_1]')
    ['[C]', '[=C]', '[C]', '[=C]', '[C]', '[=C]', '[Ring1]', '[Branch1_1]']
    
    Returns:
    chars_selfie: list of selfie characters present in molecule selfie
    '''
    chars_selfie = [] # A list of all SELFIE sybols from string selfie
    while selfie != '':
        chars_selfie.append(selfie[selfie.find('['): selfie.find(']')+1])
        selfie = selfie[selfie.find(']')+1:]
    return chars_selfie

class _FingerprintCalculator:
    ''' Calculate the fingerprint for a molecule, given the fingerprint type
    Parameters: 
        mol (rdkit.Chem.rdchem.Mol) : RdKit mol object (None if invalid smile string smi)
        fp_type (string)            :Fingerprint type  (choices: AP/PHCO/BPF,BTF,PAT,ECFP4,ECFP6,FCFP4,FCFP6)  
    Returns:
        RDKit fingerprint object
    '''

    def get_fingerprint(self, mol: Mol, fp_type: str):
        method_name = 'get_' + fp_type
        method = getattr(self, method_name)
        if method is None:
            raise Exception(f'{fp_type} is not a supported fingerprint type.')
        return method(mol)

    def get_AP(self, mol: Mol):
        return AllChem.GetAtomPairFingerprint(mol, maxLength=10)

    def get_PHCO(self, mol: Mol):
        return Generate.Gen2DFingerprint(mol, Gobbi_Pharm2D.factory)

    def get_BPF(self, mol: Mol):
        return GetBPFingerprint(mol)

    def get_BTF(self, mol: Mol):
        return GetBTFingerprint(mol)

    def get_PATH(self, mol: Mol):
        return AllChem.RDKFingerprint(mol)

    def get_ECFP4(self, mol: Mol):
        return AllChem.GetMorganFingerprint(mol, 2)

    def get_ECFP6(self, mol: Mol):
        return AllChem.GetMorganFingerprint(mol, 3)

    def get_FCFP4(self, mol: Mol):
        return AllChem.GetMorganFingerprint(mol, 2, useFeatures=True)

    def get_FCFP6(self, mol: Mol):
        return AllChem.GetMorganFingerprint(mol, 3, useFeatures=True)


def get_fingerprint(mol: Mol, fp_type: str):
    ''' Fingerprint getter method. Fingerprint is returned after using object of 
        class '_FingerprintCalculator'
        
    Parameters: 
        mol (rdkit.Chem.rdchem.Mol) : RdKit mol object (None if invalid smile string smi)
        fp_type (string)            :Fingerprint type  (choices: AP/PHCO/BPF,BTF,PAT,ECFP4,ECFP6,FCFP4,FCFP6)  
    Returns:
        RDKit fingerprint object
        
    '''
    return _FingerprintCalculator().get_fingerprint(mol=mol, fp_type=fp_type)

def mutate_selfie(selfie, max_molecules_len, write_fail_cases=False):
    '''Return a mutated selfie string (only one mutation on slefie is performed)
    
    Mutations are done until a valid molecule is obtained 
    Rules of mutation: With a 33.3% propbabily, either: 
        1. Add a random SELFIE character in the string
        2. Replace a random SELFIE character with another
        3. Delete a random character
    
    Parameters:
    selfie            (string)  : SELFIE string to be mutated 
    max_molecules_len (int)     : Mutations of SELFIE string are allowed up to this length
    write_fail_cases  (bool)    : If true, failed mutations are recorded in "selfie_failure_cases.txt"
    
    Returns:
    selfie_mutated    (string)  : Mutated SELFIE string
    smiles_canon      (string)  : canonical smile of mutated SELFIE string
    '''
    valid=False
    fail_counter = 0
    chars_selfie = get_selfie_chars(selfie)
    
    while not valid:
        fail_counter += 1
                
        #alphabet = list(selfies.get_semantic_robust_alphabet()) # 34 SELFIE characters
        alphabet = list(ionizablelipids_alphabet) # SELFIE Characters that are present in typical ionizable lipids

        choice_ls = [1, 2, 3] # 1=Insert; 2=Replace; 3=Delete
        random_choice = np.random.choice(choice_ls, 1)[0]
        
        # Insert a character in a Random Location
        if random_choice == 1: 
            random_index = np.random.randint(len(chars_selfie)+1)
            random_character = np.random.choice(alphabet, size=1)[0]
            
            selfie_mutated_chars = chars_selfie[:random_index] + [random_character] + chars_selfie[random_index:]

        # Replace a random character 
        elif random_choice == 2:                         
            random_index = np.random.randint(len(chars_selfie))
            random_character = np.random.choice(alphabet, size=1)[0]
            if random_index == 0:
                selfie_mutated_chars = [random_character] + chars_selfie[random_index+1:]
            else:
                selfie_mutated_chars = chars_selfie[:random_index] + [random_character] + chars_selfie[random_index+1:]
                
        # Delete a random character
        elif random_choice == 3: 
            random_index = np.random.randint(len(chars_selfie))
            if random_index == 0:
                selfie_mutated_chars = chars_selfie[random_index+1:]
            else:
                selfie_mutated_chars = chars_selfie[:random_index] + chars_selfie[random_index+1:]
                
        else: 
            raise Exception('Invalid Operation trying to be performed')

        selfie_mutated = "".join(x for x in selfie_mutated_chars)
        sf = "".join(x for x in chars_selfie)
        
        try:
            smiles = decoder(selfie_mutated)
            mol, smiles_canon, done = sanitize_smiles(smiles)
            if len(selfie_mutated_chars) > max_molecules_len or smiles_canon=="":
                done = False
            if done:
                valid = True
            else:
                valid = False
        except:
            valid=False
            if fail_counter > 1 and write_fail_cases == True:
                f = open("selfie_failure_cases.txt", "a+")
                f.write('Tried to mutate SELFIE: '+str(sf)+' To Obtain: '+str(selfie_mutated) + '\n')
                f.close()
    
    return (selfie_mutated, smiles_canon)

def get_mutated_SELFIES(selfies_ls, num_mutations): 
    ''' Mutate all the SELFIES in 'selfies_ls' 'num_mutations' number of times. 
    
    Parameters:
    selfies_ls   (list)  : A list of SELFIES 
    num_mutations (int)  : number of mutations to perform on each SELFIES within 'selfies_ls'
    
    Returns:
    selfies_ls   (list)  : A list of mutated SELFIES
    
    '''
    for _ in range(num_mutations): 
        selfie_ls_mut_ls = []
        for str_ in selfies_ls: 
            
            str_chars = get_selfie_chars(str_)
            max_molecules_len = len(str_chars) + num_mutations
            
            selfie_mutated, _ = mutate_selfie(str_, max_molecules_len)
            selfie_ls_mut_ls.append(selfie_mutated)
        
        selfies_ls = selfie_ls_mut_ls.copy()
    return selfies_ls

def get_fp_scores(smiles_back, target_smi, fp_type): 
    '''Calculate the Tanimoto fingerprint (using fp_type fingerint) similarity between a list 
       of SMILES and a known target structure (target_smi). 
       
    Parameters:
    smiles_back   (list) : A list of valid SMILES strings 
    target_smi (string)  : A valid SMILES string. Each smile in 'smiles_back' will be compared to this stucture
    fp_type (string)     : Type of fingerprint  (choices: AP/PHCO/BPF,BTF,PAT,ECFP4,ECFP6,FCFP4,FCFP6) 
    
    Returns: 
    smiles_back_scores (list of floats) : List of fingerprint similarities
    '''
    smiles_back_scores = []
    target    = Chem.MolFromSmiles(target_smi)

    fp_target = get_fingerprint(target, fp_type)

    for item in smiles_back: 
        mol    = Chem.MolFromSmiles(item)
        fp_mol = get_fingerprint(mol, fp_type)
        score  = TanimotoSimilarity(fp_mol, fp_target)
        smiles_back_scores.append(score)
    return smiles_back_scores

##################################################
#Creating a chemical subsapce of ionizable lipids#
##################################################

total_time = time.time()

# Define the parameters for structure generation
fp_type = 'ECFP4'
num_random_samples = 100
num_mutation_ls = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

all_smiles_collect = []
all_smiles_collect_broken = []
canon_smi_ls = []
mols_similarity_8 = []
mols_similarity_6 = []
mols_similarity_4 = []

for smi in smiles_list:
    start_time = time.time()
    print(f"Processing seed structure: {smi}")

    # Convert the seed SMILES to an RDKit molecule object
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise Exception('Invalid starting structure encountered')

    # Randomize the seed molecule
    randomized_smile_orderings = [randomize_smiles(mol) for _ in range(num_random_samples)]

    # Convert the randomized molecules to SELFIES
    selfies_ls = [encoder(x) for x in randomized_smile_orderings]
    print('Randomized molecules (in SELFIES) time: ', time.time() - start_time)
    
    seed_smiles_collect = []
    seed_smiles_collect_broken = []

    for num_mutations in num_mutation_ls:
        # Mutate the SELFIES
        selfies_mut = get_mutated_SELFIES(selfies_ls.copy(), num_mutations=num_mutations)

        # Convert back to SMILES
        smiles_back = [decoder(x) for x in selfies_mut]
        all_smiles_collect += smiles_back
        seed_smiles_collect += smiles_back
        all_smiles_collect_broken.append(smiles_back)
        seed_smiles_collect_broken.append(smiles_back)

    # Work on:  seed_smiles_collect
    seed_canon_smi_ls = []
    for item in seed_smiles_collect:
        mol, smi_canon, did_convert = sanitize_smiles(item)
        if mol is None or smi_canon == '' or not did_convert:
            raise Exception('Invalid SMILES string found')
        seed_canon_smi_ls.append(smi_canon)
    seed_canon_smi_ls = list(set(seed_canon_smi_ls))
    canon_smi_ls.extend(seed_canon_smi_ls)
    print('Unique mutated structure obtainment time: ', time.time() - start_time)

    # Compute similarity scores for this set of mutated molecules
    canon_smi_ls_scores = get_fp_scores(seed_canon_smi_ls, target_smi=smi, fp_type=fp_type)
    print('Fingerprint calculation time: ', time.time() - start_time)

    # Molecules with fingerprint similarity > 0.8
    indices_thresh_8 = [i for i, x in enumerate(canon_smi_ls_scores) if x > 0.8]
    mols_8 = [Chem.MolFromSmiles(seed_canon_smi_ls[idx]) for idx in indices_thresh_8]
    mols_similarity_8 += mols_8
    print("Number of Molecules with similarity of x > 0.8 is:", len(mols_8))

    # Molecules with fingerprint similarity > 0.6
    indices_thresh_6 = [i for i, x in enumerate(canon_smi_ls_scores) if 0.6 < x < 0.8]
    mols_6 = [Chem.MolFromSmiles(seed_canon_smi_ls[idx]) for idx in indices_thresh_6]
    mols_similarity_6 += mols_6
    print("Number of Molecules with similarity of 0.6 < x < 0.8 is:", len(mols_6))

    # Molecules with fingerprint similarity > 0.4
    indices_thresh_4 = [i for i, x in enumerate(canon_smi_ls_scores) if 0.4 < x < 0.6]
    mols_4 = [Chem.MolFromSmiles(seed_canon_smi_ls[idx]) for idx in indices_thresh_4]
    mols_similarity_4 += mols_4
    print("Number of Molecules with similarity of 0.4 < x < 0.6 is:", len(mols_4))

    print('Number of generated molecules:', len(seed_smiles_collect))
    print('Number of canonical SMILES:', len(seed_canon_smi_ls))
    print('Time for Seed: ', time.time() - start_time)

# Print the total number of generated molecules
print("Total number of Molecules with similarity of x > 0.8 is:", len(mols_similarity_8))
print("Total number of Molecules with similarity of 0.6 < x < 0.8 is:", len(mols_similarity_6))
print("Total number of Molecules with similarity of 0.4 < x < 0.6 is:", len(mols_similarity_4))
print("Total number of generated molecules:", len(all_smiles_collect))
print("Total number of canonical SMILES:", len(canon_smi_ls))
print('Total time: ', time.time() - total_time)

###############################################################################
#Filtering out un-synthetic structures with SYBA synthetic accessibility score#
#GitHub: https://github.com/lich-uct/syba######################################
###############################################################################

# Initialize SYBA classifier
syba = SybaClassifier()
syba.fitDefaultScore()

def filter_molecules_by_syba(syba, all_smiles, *molecule_sets):
    """
    Filter molecules by SYBA score.

    Parameters:
    syba (SybaClassifier): The SYBA classifier object.
    all_smiles (list): The complete list of generated molecules.
    *molecule_sets: Variable length argument list of molecule sets to be filtered.

    Returns:
    tuple: Filtered molecule sets in the same order as provided in the input.
    """
    # Calculate SYBA scores for all generated molecules
    syba_scores_all = [syba.predict(smi=item) for item in all_smiles]

    # Filter out molecules with positive SYBA scores for the complete list
    positive_syba_indices_all = [i for i, score in enumerate(syba_scores_all) if score > 0]
    filtered_smiles_syba_all = [all_smiles[i] for i in positive_syba_indices_all]

    print('Number of original generated molecules with positive SYBA score:', len(filtered_smiles_syba_all))
    print('Number of molecules filtered out based on SYBA score:', len(all_smiles) - len(filtered_smiles_syba_all))

    filtered_molecule_sets = []

    # Process each molecule set
    for mol_set in molecule_sets:
        syba_scores_set = [syba.predict(mol=mol) for mol in mol_set]
        positive_syba_indices_set = [i for i, score in enumerate(syba_scores_set) if score > 0]
        filtered_mols_syba_set = [mol_set[i] for i in positive_syba_indices_set]

        filtered_molecule_sets.append(filtered_mols_syba_set)

        print(f'Number of molecules in set with positive SYBA score: {len(filtered_mols_syba_set)}')
        print(f'Number of molecules filtered out in set based on SYBA score: {len(mol_set) - len(filtered_mols_syba_set)}')

    return (filtered_smiles_syba_all, *filtered_molecule_sets)

def save_filtered_sets(filtered_mols_syba_4, filtered_mols_syba_6, filtered_mols_syba_8):
    """
    Save the filtered molecule sets to CSV files.

    Parameters:
    filtered_mols_syba_4 (list): Filtered molecules with similarity > 0.4.
    filtered_mols_syba_6 (list): Filtered molecules with similarity > 0.6.
    filtered_mols_syba_8 (list): Filtered molecules with similarity > 0.8.
    """
    smiles_4 = [mol2smi(mol) for mol in filtered_mols_syba_4]
    smiles_6 = [mol2smi(mol) for mol in filtered_mols_syba_6]
    smiles_8 = [mol2smi(mol) for mol in filtered_mols_syba_8]

    df_4 = pd.DataFrame(smiles_4, columns=['SMILES'])
    df_6 = pd.DataFrame(smiles_6, columns=['SMILES'])
    df_8 = pd.DataFrame(smiles_8, columns=['SMILES'])

    df_4.to_csv("AIRMAIL_STONED_generated_ionizablelipids_positive_sim_04_mut10_smiles.csv", index=False)
    df_6.to_csv("AIRMAIL_STONED_generated_ionizablelipids_positive_sim_06_mut10_smiles.csv", index=False)
    df_8.to_csv("AIRMAIL_STONED_generated_ionizablelipids_positive_sim_08_mut10_smiles.csv", index=False)

    print("Filtered molecule sets saved to CSV files.")

# Use the function to filter molecules in one go
filtered_smiles_syba_all, filtered_mols_syba_4, filtered_mols_syba_6, filtered_mols_syba_8 = filter_molecules_by_syba(
    syba, all_smiles_collect, mols_similarity_4, mols_similarity_6, mols_similarity_8
)

# Save the filtered sets to CSV files
save_filtered_sets(filtered_mols_syba_4, filtered_mols_syba_6, filtered_mols_syba_8)

# Print the total time
print('Total time: ', time.time() - total_time)