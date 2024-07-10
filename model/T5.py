import pandas as pd
from rdkit import Chem
import subprocess
def can(x):
    mol = Chem.MolFromSmiles(x)
    return Chem.MolToSmiles(mol)
def return_reaction(row, sub1_column, sub2_column, product_column):
    return f"{can(row[sub1_column])}.{can(row[sub2_column])}>>{can(row[product_column])}"

def T5_train(df, sub1_column, sub2_column, product_column):
    df['text'] = df.apply(lambda row: return_reaction(row, sub1_column, sub2_column, product_column), axis=1)
    df.fillna(0, inplace=True)
    df['labels']= df['yield']
    train_df = df[df['train_or_test']=='train']
    val_df = df[df['train_or_test']!='train']
    test_df = df[df['train_or_test']=='test']
    train_df.to_csv("T5_train.csv", encoding='urf-8')
    

    # Define the command as a list of arguments

    command = [
        "t5chem",
        "train",
        "--data_dir", "model/",
        "--output_dir", "model/T5_saved_model/",
        "--task_type", "regression",
        "--pretrain", "models/T5_pretrained_model/",
        "--num_epoch", "30"
    ]

    result = subprocess.run(command, capture_output=True, text=True)

    print("Output:\n", result.stdout)

    print("Error:\n", result.stderr)

    if result.returncode == 0:
        print("Command executed successfully!")
    else:
        print("Command failed with return code", result.returncode)

