import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Function to convert SMILES to molecular fingerprint
def smiles_to_fingerprint(smiles, radius=2, n_bits=1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fp)
    else:
        return np.zeros(n_bits)

# Function to generate fingerprint features for a DataFrame
def get_fingerprint_features(df, sub_1_col, sub_2_col, prod_col):
    sub_1_fp = np.array([smiles_to_fingerprint(smiles) for smiles in df[sub_1_col]])
    sub_2_fp = np.array([smiles_to_fingerprint(smiles) for smiles in df[sub_2_col]])
    prod_fp = np.array([smiles_to_fingerprint(smiles) for smiles in df[prod_col]])

    features = np.hstack([sub_1_fp, sub_2_fp, prod_fp])
    return features

# Load the datasets
# Only use 'test' as test set
# Only use 'one' as the test set
# Use non-train as test set
df = pd.read_csv('Data/DCC_with_mid_splited.csv')
train1 = df[df['class'] == 'train']
test1 = df[df['class'] == 'test']

# Extract features and labels
X_train = get_fingerprint_features(train1, 'sub_1_smiles', 'sub_2_smiles', 'product_smiles')
X_test = get_fingerprint_features(test1, 'sub_1_smiles', 'sub_2_smiles', 'product_smiles')
y_train = train1['yield']
y_test = test1['yield']

# Train the XGBoost model
model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1)  # Adjust hyperparameters as needed
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'xgboost_model.pkl')

# Load the model
model = joblib.load('xgboost_model.pkl')

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'R-squared (R2): {r2}')
