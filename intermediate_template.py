import logging
from rdkit import Chem
from rdkit.Chem import AllChem

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

templates = {
    'DCC': '[#6;X3:1](=[OX1:2])[#8X2;H1:3].[#6]1(-[#6]-[#6]-[#6]-[#6]-[#6]-1)-[#7:5]=[#6:4]=[#7]-[#6]1-[#6]-[#6]-[#6]-[#6]-[#6]-1>>[#6:1](-[#8:3]-[#6:4](-[#7X3;H1:5]-[#6]1-[#6]-[#6]-[#6]-[#6]-[#6]-1)=[#7]-[#6]1-[#6]-[#6]-[#6]-[#6]-[#6]-1)=[#8:2]',
    'EDC': '[#6;X3:1](=[OX1:2])[#8X2;H1:3].[#6]-[#6]-[#7]=[#6:4]=[#7:5]-[#6]-[#6]-[#6]-[#7](-[#6])-[#6]>>[#6]-[#6]-[#7]=[#6:4](-[#8:2]-[#6:1]=[#8:3])-[#7;H1:5]-[#6]-[#6]-[#6]-[#7](-[#6])-[#6]',
    'HATU': '[#6;X3:1](=[OX1:2])[#8X2;H1:3].[#6]-[#7+](-[#6])=[#6](-[#7](-[#6])-[#6])-[#7:4]1:[#7:5]:[#7+:6](-[#8-:7]):[#6]2:[#7]:[#6]:[#6]:[#6]:[#6]:1:2>>[#8:2]=[#6:1]-[#8:3]-[#7]1[#6]2:[#7]:[#6]:[#6]:[#6]:[#6]:2-[#7:4]=[#7:5]1',
    'HBTU': '[#6;X3:1](=[OX1:2])[#8X2;H1:3].[#6]-[#7+](-[#6])=[#6](-[#7](-[#6])-[#6])-[#7:4]1:[#7:5]:[#7+:6](-[#8-:7]):[#6]2:[#6]:[#6]:[#6]:[#6]:[#6]:1:2>>[#8:2]=[#6:1]-[#8:3]-[#7]1[#6]2:[#6]:[#6]:[#6]:[#6]:[#6]:2-[#7:4]=[#7:5]1',
    'PyBOP': '[#6;X3:1](=[OX1:2])[#8X2;H1:3].[#6]12:[#6]:[#6]:[#6]:[#6]:[#6]:1:[#7:4]:[#7:5]:[#7:6]:2-[#8:7]-[#15+](-[#7]1-[#6]-[#6]-[#6]-[#6]-1)(-[#7]1-[#6]-[#6]-[#6]-[#6]-1)-[#7]1-[#6]-[#6]-[#6]-[#6]-1>>[#8:2]=[#6:1]-[#8:7]-[#7:6]1:[#6]2:[#6]:[#6]:[#6]:[#6]:[#6]:2:[#7:4]:[#7:5]:1',
    'TBTU': '[#6;X3:1](=[OX1:2])[#8X2;H1:3].[#6]-[#7+](-[#6])=[#6](-[#7](-[#6])-[#6])-[#7:4]1:[#7:5]:[#7+:6](-[#8-:7]):[#6]2:[#6]:[#6]:[#6]:[#6]:[#6]:1:2>>[#8:2]=[#6:1]-[#8:3]-[#7]1[#6]2:[#6]:[#6]:[#6]:[#6]:[#6]:2-[#7:4]=[#7:5]1'
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