import torch
import pandas as pd
from rxnfp.models import SmilesClassificationModel
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import numpy as np
from scipy.stats import pearsonr
from scipy.stats import spearmanr

condition = "DCC"

# Load the datasets
#model name with ○○１ is used to test text1.
#model name with ○○2 is used to test text2.
#model name with ○○3 is used to test text3.
df = pd.read_csv(f'Data/{condition}_with_mid_splited.csv')
model_path = f"Bert_model/{condition}1/"
df['text'] = df['text1']

# Only use 'test' as test set
# Only use 'one' as the test set
# Use non-train as test set
train_df = df[df['class']=='train']
val_test_df = df[df['class']!='train']
remain = df[df['class']=='remaining']
val_df = df[df['class']=='one']
test_df = temp
test_df['labels'] = test_df['yield']
mean_hte = train_df.labels.mean()
std_hte = train_df.labels.std()

train_df['labels'] = (train_df['labels'] - mean_hte) / std_hte
val_test_df['labels'] = (val_test_df['labels'] - mean_hte) / std_hte
remain['labels'] = (remain['labels'] - mean_hte) / std_hte
val_df['labels'] = (val_df['labels'] - mean_hte) / std_hte
test_df['labels'] = (test_df['labels'] - mean_hte) / std_hte


yield_bert = SmilesClassificationModel("bert", model_path, num_labels=1, use_cuda=torch.cuda.is_available(), freeze_all_but_one = False)

std = std_hte
mean = mean_hte
tra = list(train_df.text)
vt = list(val_test_df.text)
val = list(val_df.text)
test = list(test_df.text)

#Train data
yield_predicted_train = yield_bert.predict(tra)[0]
yield_predicted_train = yield_predicted_train * std + mean
yield_true_train = train_df.labels.values
yield_true_train = yield_true_train * std + mean
MSE = mean_squared_error(yield_true_train,yield_predicted_train)
MAE = mean_absolute_error(yield_true_train,yield_predicted_train)
RMSE = np.sqrt(mean_squared_error(yield_predicted_train,yield_true_train))
R2 = r2_score(yield_true_train,yield_predicted_train)
Pearson = pearsonr(yield_true_train,yield_predicted_train)
Spearman = spearmanr(yield_true_train,yield_predicted_train)

print("MSE:", MSE, '\n', "MAE:", MAE, '\n',"RMSE:", RMSE, '\n',"R2_score:", R2, '\n', "Pearson:", Pearson, '\n', "Spearman", Spearman)

#Random split test data
yield_predicted = yield_bert.predict(vt)[0]
yield_predicted = yield_predicted * std + mean
yield_true = val_test_df.labels.values
yield_true = yield_true * std + mean
MSE = mean_squared_error(yield_true,yield_predicted)
MAE = mean_absolute_error(yield_true,yield_predicted)
RMSE = np.sqrt(mean_squared_error(yield_true,yield_predicted))
R2 = r2_score(yield_true,yield_predicted)
Pearson = pearsonr(yield_true,yield_predicted)
Spearman = spearmanr(yield_true,yield_predicted)

print("MSE:", MSE, '\n', "MAE:", MAE, '\n',"RMSE:", RMSE, '\n',"R2_score:", R2, '\n', "Pearson:", Pearson, '\n', "Spearman", Spearman)

#Known one substrate test data
yield_predicted = yield_bert.predict(val)[0]
yield_predicted = yield_predicted * std + mean
yield_true = val_df.labels.values
yield_true = yield_true * std + mean
MSE = mean_squared_error(yield_true,yield_predicted)
MAE = mean_absolute_error(yield_true,yield_predicted)
RMSE = np.sqrt(mean_squared_error(yield_true,yield_predicted))
R2 = r2_score(yield_true,yield_predicted)
Pearson = pearsonr(yield_true,yield_predicted)
Spearman = spearmanr(yield_true,yield_predicted)

print("MSE:", MSE, '\n', "MAE:", MAE, '\n',"RMSE:", RMSE, '\n',"R2_score:", R2, '\n', "Pearson:", Pearson, '\n', "Spearman", Spearman)

#Unknown both substrates test data
yield_predicted_test = yield_bert.predict(test)[0]
yield_predicted_test = yield_predicted_test * std + mean
yield_true_test = test_df.labels.values
yield_true_test = yield_true_test * std + mean
MSE = mean_squared_error(yield_true_test,yield_predicted_test)
MAE = mean_absolute_error(yield_true_test,yield_predicted_test)
RMSE = np.sqrt(mean_squared_error(yield_true_test,yield_predicted_test))
R2 = r2_score(yield_true_test,yield_predicted_test)
Pearson = pearsonr(yield_true_test,yield_predicted_test)
Spearman = spearmanr(yield_true_test,yield_predicted_test)

print("MSE:", MSE, '\n', "MAE:", MAE, '\n',"RMSE:", RMSE, '\n',"R2_score:", R2, '\n', "Pearson:", Pearson, '\n', "Spearman", Spearman)