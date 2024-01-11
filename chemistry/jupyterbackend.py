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

try:
    model_csv_file = pd.read_csv('chemistry/hiv_dataset_3.csv', sep=',')  # Adjust the delimiter if needed
except pd.errors.ParserError:
    print("Error reading CSV file. Check for formatting issues in the file.")
    model_csv_file = None  # You can choose to handle this differently based on your needs

# Continue with the rest of your code, if applicable
# For example, you can print the DataFrame or perform further operations
if model_csv_file is not None:
    print("CSV Model File Found!")



# first, we need to pop compounds that do not have a standard value (I saw some of them here)
model_csv_file_retained = model_csv_file.loc[model_csv_file['Standard Value'] != None]
model_csv_file_retained_2 = model_csv_file_retained.loc[model_csv_file_retained['Smiles'] != None]
model_csv_file_retained_3 = model_csv_file_retained_2.loc[model_csv_file_retained_2['Standard Value'] != "10000"]
#print(len(model_csv_file))
#print(len(model_csv_file_retained))
#print(len(model_csv_file_retained_2))
#print(len(model_csv_file_retained_3))

# then we are taking smiles and standard value to a new dataframe
column_names = ["Standard Value", "Smiles", "Molecule ChEMBL ID"]
model_csv_file_retained_only = model_csv_file_retained_3[column_names]

# Assuming model_csv_file_retained_only is your DataFrame
#model_csv_file_retained_only['Smiles'] = model_csv_file_retained_only['Smiles'].astype(str)
# Create 'Mol' column
#model_csv_file_retained_only['Mol'] = [Chem.MolFromSmiles(smile) if smile is not None else None for smile in model_csv_file_retained_only['Smiles']]

mol_object_list = []
bad_smiles = 0
for smi in model_csv_file_retained_only['Smiles']:
    try:  # Try converting the SMILES to a mol object
        rdMolStandardize.StandardizeSmiles(smi)
        mol = Chem.MolFromSmiles(smi)
        mol_object_list.append(mol)
    except:  # Print the SMILES if there was an error in converting
        mol_object_list.append('bad_molecule')
        bad_smiles += 1
        #print(smi)
#print(bad_smiles)
model_csv_file_retained_only['Mol'] = mol_object_list
model_csv_file_retained_only_2 = model_csv_file_retained_only.loc[model_csv_file_retained_only['Mol'] != "bad_molecule"]

activity = []
for row in range(0, len(model_csv_file_retained_only_2)):
    try:
    #print("...")
      if float(model_csv_file_retained_only_2['Standard Value'][row]) < 1000:
          activity.append('1')
      else:
          activity.append('0')
    except:
      activity.append('0')
      #print("You lost")

model_csv_file_retained_only_2['Activity'] = activity
model_csv_file_retained_only_2['Activity'] = model_csv_file_retained_only_2['Activity'].astype(float)

# create Morgan fingerprint structure representation
morgan_finger = []
bit_morgan = [{}] #label fingerprints
i = 0
for mol in model_csv_file_retained_only_2['Mol']:
  bit_morgan.append({})
  morgan_finger.append(
      rdMolDescriptors.GetMorganFingerprintAsBitVect(mol,
                                                     radius = 2, nBits = 1024, bitInfo=bit_morgan[i] )
  )
  i += 1
  # radius, more more options, nBits, the same
morgan_np = np.array(morgan_finger)
morgan_df = pd.DataFrame(morgan_np)
morgan_df.to_csv('morgan_df_features.csv')
model_csv_file_retained_only_2.to_csv('activities.csv')


import numpy as np
import pandas as pd
import copy
import stat
from rdkit import Chem, RDConfig, rdBase, DataStructs
from rdkit.Chem import PandasTools, AllChem, Draw, rdMolDescriptors, GraphDescriptors, Descriptors, rdFMCS
from rdkit.Chem.Draw import rdDepictor, rdMolDraw2D
from rdkit.ML.Descriptors import MoleculeDescriptors

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

from sklearn import tree



features = morgan_df
labels = model_csv_file_retained_only_2['Activity']

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, shuffle=True)
rf=RandomForestClassifier()
rf.fit(X_train, y_train)
predict=rf.predict(X_test)
cm = confusion_matrix(np.asarray(y_test).reshape(-1), np.asarray(predict))
print(cm)

#second model

#clf = tree.DecisionTreeClassifier()
#clf = clf.fit(X_train, y_train)
#predicted = clf.predict(X_test)
#cm = confusion_matrix(np.asarray(y_test).reshape(-1), np.asarray(predicted))
#print(cm)

#dot_data = tree.export_graphviz(clf, out_file=None)

