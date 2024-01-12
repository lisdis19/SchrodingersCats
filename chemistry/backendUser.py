import pandas as pd
from sklearn.model_selection import train_test_split
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

# Load your molecular CSV file into a pandas DataFrame
file_path = 'chemistry/hiv_dataset_3.csv'
molecular_data = pd.read_csv(file_path)

# Assuming your molecular data has features (X) and labels (y)
# Replace 'your_feature_columns' with the actual column names for features
X = molecular_data[['Smiles']]
# Replace 'your_label_column' with the actual column name for labels
y = molecular_data['Molecule ChEMBL ID']

# Split the data into training and test sets
test_size = 0.2  # Adjust the test_size as needed
random_state = 42  # Set a random seed for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

# Select a random SMILES value from the DataFrame
random_smiles = X_train.sample(n=1).iloc[0]['Smiles']

# Create a molecule from the random SMILES
m = Chem.MolFromSmiles(random_smiles)

# Calculate and print molecular properties
properties = rdMolDescriptors.Properties()
names = properties.GetPropertyNames()
values = properties.ComputeProperties(m)
for name, value in zip(names, values):
    print(f"{name}: {value}")
