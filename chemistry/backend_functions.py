import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
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

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score

from sklearn import tree

from jupyterbackend import process_csv, make_predictions

morgan_df = pd.read_csv('morgan_df_features.csv')
model_csv_file_retained_only_2 = pd.read_csv('activities.csv')

features = morgan_df
labels = model_csv_file_retained_only_2['Activity']

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, shuffle=True)
rf=RandomForestClassifier()
rf.fit(X_train, y_train)
predict=rf.predict(X_test)
cm = confusion_matrix(np.asarray(y_test).reshape(-1), np.asarray(predict))
print(cm)
print(dir(cm))

new_picture = make_predictions('user_input.csv')
