import numpy as np
import pandas as pd
import copy
import stat

from rdkit import Chem, RDConfig, rdBase, DataStructs
from rdkit.Chem import PandasTools, AllChem, Draw, rdMolDescriptors, GraphDescriptors, Descriptors, rdFMCS
from rdkit.Chem.Draw import IPythonConsole, rdDepictor, rdMolDraw2D
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import MACCSkeys

import matplotlib.pyplot as plt
from rdkit.Chem import rdMolTransforms
from rdkit.Chem.Draw import rdMolDraw2D, rdDepictor, IPythonConsole
rdDepictor.SetPreferCoordGen(True)

# Read CSV file with error handling
try:
    model_csv_file = pd.read_csv('chemistry/hiv_dataset_3.csv', sep=',')  # Adjust the delimiter if needed
except pd.errors.ParserError:
    print("Error reading CSV file. Check for formatting issues in the file.")
    model_csv_file = None  # You can choose to handle this differently based on your needs

# Continue with the rest of your code, if applicable
# For example, you can print the DataFrame or perform further operations
if model_csv_file is not None:
    print("CSV File Found!")

print(model_csv_file.columns)
# first, we need to pop compounds that do not have a standard value (I saw some of them here)
model_csv_file_retained = model_csv_file.loc[model_csv_file['Standard Value'] != None]

# then we are taking smiles and standard value to a new dataframe
column_names = ["Standard Value", "Smiles", "Molecule ChEMBL ID"]
model_csv_file_retained_only = model_csv_file_retained[column_names]

model_csv_file_retained_only
model_csv_file_retained = model_csv_file.loc[model_csv_file['Standard Value'] != "NaN"]
# Assuming model_csv_file_retained_only is your DataFrame
model_csv_file_retained_only['Smiles'] = model_csv_file_retained_only['Smiles'].astype(str)

# Create 'Mol' column
model_csv_file_retained_only['Mol'] = [Chem.MolFromSmiles(smile) if smile is not None else None for smile in model_csv_file_retained_only['Smiles']]

mol_object_list = []
bad_smiles = 0
for smi in model_csv_file_retained_only['Smiles']:
    try:  # Try converting the SMILES to a mol object
        mol = Chem.MolFromSmiles(smi)
        mol_object_list.append(mol)
    except:  # Print the SMILES if there was an error in converting
        mol_object_list.append('nan')
        bad_smiles += 1
        print(smi)
print(bad_smiles)
model_csv_file_retained_only['Mol'] = mol_object_list
