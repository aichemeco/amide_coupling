# Environment install

```shell
conda create -n Acylation python=3.9 -y
conda activate Acylation

# Install PyTorch on CPU or GPU environment
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 \
    --extra-index-url https://download.pytorch.org/whl/cu117

cd ..
pip install -r requirements.txt
```

# Data process
Use the following command for preprocessing
```shell
python main.py your_dataset_path EDC
```
Use the following command for data split and 'train' class is for training, 'one' class is for Partial substrate novelty dataset, 'test' class is for Full substrate novelty dataset. 
Data outside the 'train' class can be considered a randomly divided test set.
```python
from data_split import data_split

```
# Model training
## BERT and T5
reposotory for yield BERT: https://github.com/rxn4chemistry/rxn_yields
<br />
reposotory for T5-Chem: https://github.com/HelloJocelynLu/t5chem
<br />
When we build a multi-condition model, we construct the SMILES reaction in the following form:
"ammonia.acid.condition>>product"
## Machine learning
```python
from model import train_machine_learning_model
#Use model to select model("SVM", "XGBoost" or "RandomForest"). df is yourdataset, sub1_column is the tiele of ammonia column, sub2_column is the title of acid column and product_column is the title of product column.
train_machine_learning_model(model,df,sub1_column, sub2_column, product_column)
```
# Evaluation
```shell
python BERT_evaluation.py --condition DCC --text_type 1
```
text_type can be set as 1, 2 or 3.
<br />
1 is for ammonia.acid>>product
<br />
2 is for ammonia.acid>intermediate>product
<br />
3 is for ammonia.intermediate>>product