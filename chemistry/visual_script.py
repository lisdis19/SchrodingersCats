import numpy as np
import pandas as pd
import copy
import stat

from rdkit import Chem, RDConfig, rdBase, DataStructs
from rdkit.Chem import PandasTools, AllChem, Draw, rdMolDescriptors, GraphDescriptors, Descriptors, rdFMCS
from rdkit.Chem.Draw import rdDepictor, rdMolDraw2D
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import MACCSkeys

from rdkit.Chem import rdMolTransforms
from rdkit.Chem.Draw import rdMolDraw2D, rdDepictor
from rdkit.Chem.MolStandardize import rdMolStandardize

import numpy as np
import pandas as pd
import copy
import stat

from rdkit import Chem, RDConfig, rdBase, DataStructs
from rdkit.Chem import PandasTools, AllChem, Draw, rdMolDescriptors, GraphDescriptors, Descriptors, rdFMCS
from rdkit.Chem.Draw import rdDepictor, rdMolDraw2D
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import MACCSkeys

from rdkit.Chem import rdMolTransforms
from rdkit.Chem.Draw import rdMolDraw2D, rdDepictor
from rdkit.Chem.MolStandardize import rdMolStandardize

rdDepictor.SetPreferCoordGen(True)

# Read CSV file with error handling
def upload_csv(uploaded_data_input):
    try:
        uploaded_data = pd.read_csv('chemistry/hiv_dataset_3.csv', sep=',')  # Adjust the delimiter if needed
    except pd.errors.ParserError:
        print("Error reading CSV file. Check for formatting issues in the file.")
        uploaded_data = None  # You can choose to handle this differently based on your needs
    # Continue with the rest of your code, if applicable
    # For example, you can print the DataFrame or perform further operations
    if uploaded_data is not None:
        print("CSV Model File Found!")

def parse_data(uploaded_data):
    mol_object_list = []
    bad_smiles = 0
    for smi in uploaded_data['Smiles']:
        try:  # Try converting the SMILES to a mol object
            rdMolStandardize.StandardizeSmiles(smi)
            mol = Chem.MolFromSmiles(smi)
            mol_object_list.append(mol)
        except:  # Print the SMILES if there was an error in converting
            mol_object_list.append('bad_molecule')
            bad_smiles += 1
        #print(smi)
#print(bad_smiles)
    uploaded_data['Mol'] = mol_object_list
    uploaded_data = uploaded_data.loc[uploaded_data['Mol'] != "bad_molecule"]

def visualize_molecules(uploaded_data):
    number_molecules = len(uploaded_data)
    if number_molecules > 20:
        Draw.MolsToGridImage(uploaded_data['Mol'].iloc[0:20]) #choose first 20 molecules
        Draw.MolToImage(mol, size=(300, 300), kekulize=True)
    else:
        Draw.MolsToGridImage(uploaded_data['Mol'])








# create Morgan fingerprint structure representation
morgan_finger = []
bit_morgan = [{}] #label fingerprints
i = 0
for mol in uploaded_data['Mol']:
  bit_morgan.append({})
  morgan_finger.append(
      rdMolDescriptors.GetMorganFingerprintAsBitVect(mol,
                                                     radius = 2, nBits = 1024, bitInfo=bit_morgan[i] )
  )
  i += 1
  # radius, more more options, nBits, the same
morgan_np = np.array(morgan_finger)
morgan_df = pd.DataFrame(morgan_np)
morgan_df.to_csv('morgan_df_features_forprediction.csv')
