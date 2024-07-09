import logging
from rdkit import Chem
from rdkit.Chem import AllChem

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

templates = {
    'DCC': '[#6:1](=[#8:2])[#8:3].[O]>>[#6:1]([#8:3]C(NC1CCCCC1)=[NH+]C2CCCCC2)=[#8:2]',
    'EDC': '[#6:1](=[#8:2])[#8:3].[O]>>[#6:1]([#8:3]C(NCCC[NH+](C)C)=NCC)=[#8:2]',
    'HATU': '[#6:1](=[#8:2])[#8:3].[O]>>[#6:1]([#8:3]N1N=NC2=CC=CN=C21)=[#8:2]',
    'HBTU': '[#6X3:1](=[#8:2])[#8:3].[O]>>CN(C)C(N(C)C)([#8:3][#6:1]=[#8:2])N1N=[N+]([O-])C2=CC=CC=C21',
    'PyBOP': '[#6:1](=[#8:2])[#8:3].[O]>>[#6:1]([#8:3]N1N=NC2=CC=CN=C21)=[#8:2]',
    'TBTU': '[#6:1](=[#8:2])[#8:3].[O]>>[#6:1]([#8:3]N1N=NC2=CC=CN=C21)=[#8:2]'
}

def return_mid(row):
    try:
        reactant1_smiles = row['sub_2_smiles']
        reactant2_smiles = 'O'
        condition = row['condition']

        logging.info(f'Processing reaction for condition: {condition}')
        
        if condition not in templates:
            raise ValueError(f"Condition '{condition}' is not in the template list.")
        
        rxn_smarts = templates[condition]
        rxn = AllChem.ReactionFromSmarts(rxn_smarts)
        
        reactant1_mol = Chem.MolFromSmiles(reactant1_smiles)
        reactant2_mol = Chem.MolFromSmiles(reactant2_smiles)

        if reactant1_mol is None or reactant2_mol is None:
            raise ValueError("Invalid SMILES string for reactants.")
        
        products = rxn.RunReactants((reactant1_mol, reactant2_mol))
        
        if not products or not products[0]:
            raise ValueError("Reaction did not yield any products.")
        
        product_mol = products[0][0]
        product_smiles = Chem.MolToSmiles(product_mol)
        
        logging.info(f'Reaction successful: {reactant1_smiles} + {reactant2_smiles} -> {product_smiles}')
        return product_smiles
    except Exception as e:
        logging.error(f"Error processing row: {e}")
        return ''