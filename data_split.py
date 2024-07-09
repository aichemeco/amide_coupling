import pandas as pd
from sklearn.model_selection import train_test_split

def data_split(data, sub1_column, sub2_column, index_column):
    try:
        # Check if required columns exist in the data
        assert sub1_column in data.columns, f"Column '{sub1_column}' not found in the DataFrame."
        assert sub2_column in data.columns, f"Column '{sub2_column}' not found in the DataFrame."
        assert index_column in data.columns, f"Column '{index_column}' not found in the DataFrame."

        unique_sub_1_smiles = set(data[sub1_column].unique())
        unique_sub_2_smiles = set(data[sub2_column].unique())

        unique_smiles = unique_sub_1_smiles.union(unique_sub_2_smiles)

        train_smiles, test_smiles = train_test_split(list(unique_smiles), test_size=0.1, random_state=42)

        train_smiles_set = set(train_smiles)
        test_smiles_set = set(test_smiles)

        train_data = data[(data[sub1_column].isin(train_smiles_set)) & (data[sub2_column].isin(train_smiles_set))]
        test_data = data[(data[sub1_column].isin(test_smiles_set)) & (data[sub2_column].isin(test_smiles_set))]

        # Validation asserts to ensure proper splitting
        assert train_data[sub1_column].isin(test_smiles_set).sum() == 0, "Train and test data overlap in sub1_column."
        assert train_data[sub2_column].isin(test_smiles_set).sum() == 0, "Train and test data overlap in sub2_column."
        assert test_data[sub1_column].isin(train_smiles_set).sum() == 0, "Train and test data overlap in sub1_column."
        assert test_data[sub2_column].isin(train_smiles_set).sum() == 0, "Train and test data overlap in sub2_column."

        remaining_data = pd.concat([train_data, test_data])

        # Finding differences between original data and split data
        difference_remaining_to_df = remaining_data.merge(data, how='left', indicator=True)
        difference_remaining_to_df = difference_remaining_to_df[difference_remaining_to_df['_merge'] == 'left_only'].drop(columns=['_merge'])

        difference_df_to_remaining = data.merge(remaining_data, how='left', indicator=True)
        difference_df_to_remaining = difference_df_to_remaining[difference_df_to_remaining['_merge'] == 'left_only'].drop(columns=['_merge'])

        # Adding class labels to distinguish different subsets
        train_data['class'] = 'train'
        test_data['class'] = 'test'
        one = difference_df_to_remaining[~(difference_df_to_remaining[sub1_column].isin(set(train_data[sub1_column]))) & (difference_df_to_remaining[sub2_column].isin(set(train_data[sub2_column])))]

        one['class'] = 'one'
        difference_df_to_remaining['class'] = 'remaining'

        final_res = pd.concat([train_data, test_data, one, difference_df_to_remaining])
        final_res = final_res.drop_duplicates(subset=[index_column])

        return final_res

    except AssertionError as e:
        # Log the error and return None or an empty DataFrame
        print(f"Error in data_split function: {str(e)}")
        return pd.DataFrame()

    except Exception as e:
        # Handle other unexpected exceptions
        print(f"An error occurred in data_split function: {str(e)}")
        return pd.DataFrame()
