import numpy as np
import pandas as pd
import copy
import stat
from rdkit import Chem, RDConfig, rdBase, DataStructs
from rdkit.Chem import PandasTools, AllChem, Draw, rdMolDescriptors, GraphDescriptors, Descriptors, rdFMCS
from rdkit.Chem.Draw import IPythonConsole, rdDepictor, rdMolDraw2D
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


features = pd.read_csv('chemistry/hiv_dataset_3.csv', sep=',')

features = 
labels = 

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, shuffle=True)
rf=RandomForestClassifier()
rf.fit(X_train, y_train)
predict=rf.predict(X_test)
cm = ConfusionMatrix(np.asarray(y_test).reshape(-1), np.asarray(predict))
print(cm)


