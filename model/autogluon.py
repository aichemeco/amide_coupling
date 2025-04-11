import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor

test_all_df = pd.read_csv("./full_hte_split.csv")

train_df = test_all_df[test_all_df['class'] == 'train']
test_df = test_all_df[test_all_df['class'] == 'test']
test_one_df = test_all_df[test_all_df['class'] == 'one']
test_remaining_df = test_all_df[test_all_df['class'] == 'remaining']

test_one_df_predict=test_one_df[['condition_id', 'sub_1_smiles', 'sub_2_smiles', 'product_smiles',
       'Activation_ID', 'Additive_ID', 'Base_ID', 'solvent_id', 'yield',
       'yield_clf', 'condition_SMILES']]
test_remaining_df_predict=test_remaining_df[['condition_id', 'sub_1_smiles', 'sub_2_smiles', 'product_smiles',
       'Activation_ID', 'Additive_ID', 'Base_ID', 'solvent_id', 'yield',
       'yield_clf', 'condition_SMILES']]
train_df_pure=train_df[['condition_id', 'sub_1_smiles', 'sub_2_smiles', 'product_smiles',
       'Activation_ID', 'Additive_ID', 'Base_ID', 'solvent_id', 'yield',
       'yield_clf', 'condition_SMILES']]
test_df_pure=test_df[['condition_id', 'sub_1_smiles', 'sub_2_smiles', 'product_smiles',
       'Activation_ID', 'Additive_ID', 'Base_ID', 'solvent_id', 'yield',
       'yield_clf', 'condition_SMILES']]

train_data = TabularDataset(train_df_pure)

predictor = TabularPredictor(label='yield').fit(train_data=train_data)  

testdf_all = pd.concat([test_df_pure,test_one_df_predict,test_remaining_df_predict])

print(predictor.evaluate(testdf_all))
print(predictor.evaluate(test_one_df_predict))
print(predictor.evaluate(test_remaining_df_predict))
