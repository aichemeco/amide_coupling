# N-acylation reaction model
This is the code for the "Intermediate Knowledge Enhanced the Performance of N-Acylation Yield Prediction Model" paper.
Preprint of this paper can be found in [[Paper](url_to_be_anounced)].

## Intermediate Knowledge Enhanced the Performance of N-Acylation Yield Prediction Model
Authors: Chonghuan Zhang, Qianghua Lin, Hao Deng, Yaxian Kong, Zhunzhun Yu,* Kuangbiao Liao*
Abstract: Acylation is an important reaction widely applied in medicinal chemistry. However, yield optimization remains a challenging issue due to the broad conditions space. Recently, accurate condition recommendations via machine learning have emerged as a novel and efficient method to achieve the desired transformations without a trial-and-error process. Nonetheless, accurately predicting yields is challenging due to the complex relationships involved. Herein, we present our strategy to address this problem. Two steps were taken to ensure the quality of the dataset. First, we skillfully selected substrates to ensure diversity and representativeness. Second, experiments were conducted using our in-house high-throughput experimentation (HTE) platform to minimize the influence of human factors. Additionally, we proposed an intermediate knowledge-embedded strategy to enhance the model's robustness. The performance of the model was first evaluated at three different levels—random split, partial substrate novelty, and full substrate novelty. All model metrics in these cases improved dramatically, achieving an $R^2$ of 0.89, MAE of 6.1\%, and RMSE of 8.0\%. Moreover, the generalization of our strategy was assessed using external datasets from reported literature. The prediction error for nine reactions among 30 was less than 5\%, and the model was able to identify which reaction in a reaction pair with a reactivity cliff had a higher yield. In summary, our research demonstrated the feasibility of achieving accurate yield predictions through the combination of HTE and embedding intermediate knowledge into the model. This approach also has the potential to facilitate other related machine learning tasks.
![Overview of the acylation reaction model](https://github.com/aichemeco/Acylation/blob/main/overview.png?raw=true, "Overview of the acylation reaction model")


## Environment install

```shell
conda create -n Acylation python=3.9 -y
conda activate Acylation

# Install PyTorch on CPU or GPU environment
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 \
    --extra-index-url https://download.pytorch.org/whl/cu117

cd ..
pip install -r requirements.txt
```

## Data process
Use the following command for preprocessing
```shell
python main.py your_dataset_path EDC
```
Use the following command for data split and 'train' class is for training, 'one' class is for Partial substrate novelty dataset, 'test' class is for Full substrate novelty dataset. 
Data outside the 'train' class can be considered a randomly divided test set.
```python
from data_split import data_split

```
## Model training
```python
from model import train_machine_learning_model
#Use model_type to select model("BERT", "T5", "SVM", "XGBoost" or "RandomForest"). df is yourdataset, sub1_column is the tiele of ammonia column, sub2_column is the title of acid column and product_column is the title of product column.
train_machine_learning_model(model_type,df,sub1_column, sub2_column, product_column)
```
### BERT and T5
reposotory for yield BERT: https://github.com/rxn4chemistry/rxn_yields
<br />
reposotory for T5-Chem: https://github.com/HelloJocelynLu/t5chem
<br />
First, we prepared 95 different conditions for HTE, excluding acyl chloride due to its incompatibility with DMF solvent and we construct the SMILES reaction in the following form for a multi-condition model:
<br />
<<<<<<< HEAD
"amine.carboxylic_acid.condition_contexts>>product"
=======
"ammonia.acid.condition>>product"
>>>>>>> 48353f99573b21fee63434c4f38b56190f3f490c
## Machine learning
If trying to build machine learning model with data under different conditions, each of the different reaction conditions should be encoded as a unique integer ranging from 1 to 95. This encoding allowed the model to differentiate between various reaction setups.
```python
df_one_hot = pd.get_dummies(df, columns=[condition_id_column])
```
If trying to build machine learning model with data under different conditions, each of the different reaction conditions should be encoded as a unique integer ranging from 1 to 95. This encoding allowed the model to differentiate between various reaction setups.
```python
df_one_hot = pd.get_dummies(df, columns=[condition_id_column])
```
# Evaluation
```shell
python BERT_evaluation.py --condition DCC --text_type 1
```
text_type can be set as 1, 2 or 3.
<br />
1 is for amine.carboxylic_acid>>product
<br />
2 is for amine.carboxylic_acid>intermediate>product
<br />
3 is for amine.intermediate>>product

## Citation
Please kindly cite our papers if you use the data/code/model.
```
bibtex to be announced.
```

## License
This project is licensed under the terms of the MIT license. See [LICENSE](https://github.com/aichemeco/Acylation/blob/main/LICENSE.md) for additional details.
