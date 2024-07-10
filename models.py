from model.SVM import SVM
from model.XGBoost import XGBoost
from model.RandomForest import RandomForest
from model.Bertmodel import BERT_train
from model.T5 import T5_train
model_mapping = {
    "SVM": SVM,
    "XGBoost": XGBoost,
    "RandomForest": RandomForest,
    "BERT": BERT_train,
    "T5":T5_train
}


def model_dispatcher(func):
    def wrapper(model_type, df, sub1_column, sub2_column, product_column):
        if model_type in model_mapping:
            model_mapping[model_type](df, sub1_column, sub2_column, product_column)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    return wrapper

@model_dispatcher
def train_machine_learning_model(model_type, df, sub1_column, sub2_column, product_column):
    pass 
