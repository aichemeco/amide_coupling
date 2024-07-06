from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd


templates = {
        'DCC':'[#6:1](=[#8:2])[#8:3].[O]>>[#6:1]([#8:3]C(NC1CCCCC1)=[NH+]C2CCCCC2)=[#8:2]',
        'EDC':'[#6:1](=[#8:2])[#8:3].[O]>>[#6:1]([#8:3]C(NCCC[NH+](C)C)=NCC)=[#8:2]',
        'HATU':'[#6:1](=[#8:2])[#8:3].[O]>>[#6:1]([#8:3]N1N=NC2=CC=CN=C21)=[#8:2]',
        'HBTU':'[#6X3:1](=[#8:2])[#8:3].[#7+]=[#6;!a:5]>>[#6][#7]([#6])[#6:5][#8:3][#6:1]=[#8:2]',
        'PyBOP':'[#6:1](=[#8:2])[#8:3].[O]>>[#6:1]([#8:3]N1N=NC2=CC=CN=C21)=[#8:2]',
        'TBTU':'[#6:1](=[#8:2])[#8:3].[O]>>[#6:1]([#8:3]N1N=NC2=CC=CN=C21)=[#8:2]'
     }

def return_mid(sub2):
    reactant1_smiles = sub2
    reactant2_smiles = 'O'
    #reactant2_smiles = "CN(C)C(=[N+](C)C)ON1C2=CC=CC=C2N=N1" is for HBTU


    rxn_smarts = templates['DCC']
    rxn = AllChem.ReactionFromSmarts(rxn_smarts)
    
    reactant1_mol = Chem.MolFromSmiles(reactant1_smiles)
    reactant2_mol = Chem.MolFromSmiles(reactant2_smiles)

    p = rxn.RunReactants((reactant1_mol, reactant2_mol))[0][0]
        
    return Chem.MolToSmiles(p)
df = pd.read_csv('Dataset_path')
#use acid to get Intermediate
df['Intermediate'] = temp['sub_2_SMILES'].apply(return_mid)
