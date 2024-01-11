import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Define the mol_to_vector function
def mol_to_vector(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    properties = rdMolDescriptors.Properties()
    # To consider: which properties are the most helpful in predicting your
    # desired properties?
    return list(properties.ComputeProperties(mol))

# Load your molecular CSV file into a pandas DataFrame
file_path = 'chemistry/hiv_dataset_3.csv'
molecular_data = pd.read_csv(file_path)

# Assuming your molecular data has features (X) and labels (y)
# Replace 'your_feature_columns' with the actual column names for features
X = molecular_data[['Smiles']]
# Replace 'your_label_column' with the actual column name for labels
y = molecular_data['Molecule ChEMBL ID']

# Handle NaN values and convert 'Smiles' column to strings
X['Smiles'] = X['Smiles'].astype(str)

# Label encode the 'Molecule ChEMBL ID'
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and test sets
test_size = 0.2
random_state = 42
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=test_size, random_state=random_state)

# Process training data
x_training = []
y_training = []

for ind in range(len(X_train)):
    mol_vec = mol_to_vector(X_train.iloc[ind, 0])  # 'Smiles' is at column index 0
    if mol_vec is None:
        continue
    x_training.append(mol_vec)
    y_training.append(y_train[ind])

# Process testing data
x_testing = []
y_testing = []

for ind in range(len(X_test)):
    mol_vec = mol_to_vector(X_test.iloc[ind, 0])  # 'Smiles' is at column index 0
    if mol_vec is None:
        continue
    x_testing.append(mol_vec)
    y_testing.append(y_test[ind])

# Use a classification model
classification_model = LogisticRegression()
classification_model.fit(x_training, y_training)

# Predictions on testing data for classification
test_output_classification = classification_model.predict(x_testing)

# Evaluate the classification performance
print(f'Accuracy: {accuracy_score(y_testing, test_output_classification)}')
print('Classification Report:')
print(classification_report(y_testing, test_output_classification))
print('Confusion Matrix:')
print(confusion_matrix(y_testing, test_output_classification))

# Use a regression model
regression_model = linear_model.BayesianRidge()
regression_model.fit(x_training, y_training)

# Predictions on testing data for regression
test_output_regression = regression_model.predict(x_testing)

print(f'Mean squared error (MSE): {mean_squared_error(y_testing, test_output_regression)}')
print(f'R^2: {r2_score(y_testing, test_output_regression)}')

abs_error = abs(y_testing - test_output_regression)
plt.plot(abs_error, '*')
plt.show()
