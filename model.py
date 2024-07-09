from SVM import SVM
from XGBoost import XGBoost
from RandomForest import RandomForest


def train_machine_learning_model(model_type,df,sub1_column, sub2_column, product_column)
    if model_type == "SVM":
        SVM(df, sub1_column, sub2_column, product_column) #df is yourdataset, sub1_column is the tiele of ammonia column, sub2_column is the title of acid column and product_column is the title of product column.
    elif model_type == "XGBoost":
        XGBoost(df, sub1_column, sub2_column, product_column)
    elif model_type == "RandomForest":
        RandomForest(df, sub1_column, sub2_column, product_column)
    return
