import pandas as pd
from sklearn.model_selection import train_test_split
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

def process_molecular_data(file_path, test_size=0.2, random_state=42):
    # Read molecular data from CSV file
    molecular_data = pd.read_csv(file_path)

    # Extract features (X) and labels (y)
    X = molecular_data[['Smiles']]
    y = molecular_data['Molecule ChEMBL ID']

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Select a random SMILES value from the DataFrame
    random_smiles = X_train.sample(n=1).iloc[0]['Smiles']

    # Create a molecule from the random SMILES
    m = Chem.MolFromSmiles(random_smiles)

    # Calculate and print molecular properties
    properties = rdMolDescriptors.Properties()
    names = properties.GetPropertyNames()
    values = properties.ComputeProperties(m)
    output_string = ""
    for name, value in zip(names, values):
        #print(f"{name}: {value}")
        output_string += f"{name}: {value}\n"
    return output_string

# Example usage
file_path = 'chemistry/hiv_dataset_3.csv'
process_molecular_data(file_path)

