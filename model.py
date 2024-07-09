from SVM import SVM
from XGBoost import XGBoost
from RandomForest import RandomForest


def train_machine_learning_model(model,df,sub1_column, sub2_column, product_column)
    if model == "SVM":
        SVM(df, sub1_column, sub2_column, product_column) #df is yourdataset, sub1_column is the tiele of ammonia column, sub2_column is the title of acid column and product_column is the title of product column.
    elif model == "XGBoost":
        XGBoost(df, sub1_column, sub2_column, product_column)
    elif model == "RandomForest":
        RandomForest(df, sub1_column, sub2_column, product_column)
    return