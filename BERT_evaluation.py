import torch
import pandas as pd
from model.Bertmodel import SmilesClassificationModel
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
import numpy as np
import logging
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(condition):
    """Load data for the given condition."""
    try:
        df = pd.read_csv(f'data/{condition}_with_mid_splited.csv')
        return df
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return pd.DataFrame()

def preprocess_data(df):
    """Preprocess the data and prepare the datasets."""
    try:
        # Split the data into train, validation, and test sets
        train_df = df[df['class'] == 'train']
        val_test_df = df[df['class'] != 'train']
        remain = df[df['class'] == 'remaining']
        val_df = df[df['class'] == 'one']
        test_df = df[df['class'] == 'test']  # Assuming 'test' class exists in the DataFrame

        # Standardize the labels
        mean_hte = train_df.labels.mean()
        std_hte = train_df.labels.std()

        for df_ in [train_df, val_test_df, remain, val_df, test_df]:
            df_['labels'] = (df_['labels'] - mean_hte) / std_hte

        return train_df, val_test_df, remain, val_df, test_df, mean_hte, std_hte
    except KeyError as e:
        logging.error(f"Column missing: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None, None
    except Exception as e:
        logging.error(f"Error in data preprocessing: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None, None

def train_and_evaluate(model_path, train_df, val_test_df, val_df, test_df, mean_hte, std_hte):
    """Train the model and evaluate on various datasets."""
    yield_bert = SmilesClassificationModel("bert", model_path, num_labels=1, use_cuda=torch.cuda.is_available(), freeze_all_but_one=False)
    std = std_hte
    mean = mean_hte

    datasets = {
        'train': train_df,
        'val_test': val_test_df,
        'val': val_df,
        'test': test_df
    }

    for dataset_name, df in datasets.items():
        texts = list(df.text)
        true_labels = df.labels.values
        true_labels = true_labels * std + mean

        yield_predicted = yield_bert.predict(texts)[0]
        yield_predicted = yield_predicted * std + mean

        mse = mean_squared_error(true_labels, yield_predicted)
        mae = mean_absolute_error(true_labels, yield_predicted)
        rmse = np.sqrt(mse)
        r2 = r2_score(true_labels, yield_predicted)
        pearson_corr, _ = pearsonr(true_labels, yield_predicted)
        spearman_corr, _ = spearmanr(true_labels, yield_predicted)

        logging.info(f"{dataset_name.upper()} SET EVALUATION:")
        logging.info(f"MSE: {mse}")
        logging.info(f"MAE: {mae}")
        logging.info(f"RMSE: {rmse}")
        logging.info(f"R2 Score: {r2}")
        logging.info(f"Pearson Correlation: {pearson_corr}")
        logging.info(f"Spearman Correlation: {spearman_corr}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model evaluation on different text types and conditions.")
    parser.add_argument('--condition', type=str, required=True, help='The reaction condition.')
    parser.add_argument('--text_type', type=int, choices=[1, 2, 3], required=True, help='The text type to use (1, 2, or 3).')
    
    args = parser.parse_args()
    condition = args.condition
    text_type = args.text_type
    model_path = f"BERT_model/{condition}{text_type}/"

    df = load_data(condition)
    df['text'] = df[f'text{text_type}']

    if df.empty:
        logging.error("Data loading failed. Exiting.")
    else:
        train_df, val_test_df, remain, val_df, test_df, mean_hte, std_hte = preprocess_data(df)
        if train_df.empty or val_test_df.empty or val_df.empty or test_df.empty:
            logging.error("Data preprocessing failed. Exiting.")
        else:
            train_and_evaluate(model_path, train_df, val_test_df, val_df, test_df, mean_hte, std_hte)
