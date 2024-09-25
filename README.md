# amide-coupling reaction model
This is the code for the "Intermediate Knowledge Enhanced the Performance of N-Acylation Yield Prediction Model" paper.
Preprint of this paper can be found in [[Paper](https://chemrxiv.org/engage/chemrxiv/article-details/66bdd53fa4e53c4876744a85)].  

You can access our model interface using the following URL link: [amide-coupling model](https://www.aichemeco.com/acylation)

## Intermediate Knowledge Enhanced the Performance of amide-coupling Yield Prediction Model
Authors: Chonghuan Zhang, Qianghua Lin, Hao Deng, Chenxi Yang, Yaxian Kong, Zhunzhun Yu,* Kuangbiao Liao*

Abstract: Amide coupling is an important reaction widely applied in medicinal chemistry. However, condition recommendation remains a challenging issue due to the broad conditions space. Recently, accurate condition recommendations via machine learning have emerged as a novel and efficient method to find a suitable condition to achieve the desired transformations. Nonetheless, accurately predicting yields is challenging due to the complex relationships involved. Herein, we present our strategy to address this problem. Two steps were taken to ensure the quality of the dataset. First, we selected a diverse and representative set of substrates to capture a broad spectrum of substrate structures and reaction conditions using an unbiased machine-based sampling approach. Second, experiments were conducted using our in-house high-throughput experimentation (HTE) platform to minimize the influence of human factors. Additionally, we proposed an intermediate knowledge-embedded strategy to enhance the model's robustness. The performance of the model was first evaluated at three different levels—random split, partial substrate novelty, and full substrate novelty. All model metrics in these cases improved dramatically, achieving an $R^2$ of 0.89, MAE of 6.1\%, and RMSE of 8.0\% in full substrate novelty test dataset. Moreover, the generalization of our strategy was assessed using external datasets from reported literature. The prediction error for 18 reactions among 88 was less than or equal to  5\%. Meanwhile, the model could recommend suitable conditions for some reactions to elevate the reaction yields. Besides, the model was able to identify which reaction in a reaction pair with a reactivity cliff had a higher yield. In summary, our research demonstrated the feasibility of achieving accurate yield predictions through the combination of HTE and embedding intermediate knowledge into the model. This approach also has the potential to facilitate other related machine learning tasks.

![Overview of the amide couling reaction yield prediction model](https://github.com/aichemeco/amide_coupling/blob/main/amide_coupling_overview.png?raw=true, "Overview of the amide coupling reaction yield prediction model")

## Environment Setup

To set up your environment, run the following commands:

```bash
# Create and activate the Conda environment
conda create -n Acylation python=3.9 -y
conda activate Acylation

# Install PyTorch (with CPU or GPU support)
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 \
    --extra-index-url https://download.pytorch.org/whl/cu117

# Install project dependencies
pip install -r requirements.txt
pip install t5chem
```

---

## Data and Preprocessing

Most of the data used in this work is available in the `data` folder.

- **HTE Substrate and Product Structures**:  
  See `HTE Data-Substrate and Products.xlsx` for substrate structures used in high-throughput experimentation (HTE).
  
- **USPTO Data**:  
  Access the USPTO dataset in `USPTO.csv`.

- **HTE Reaction Data**:  
  The data for six single conditions used to build our model are available in the following files:
  - `DCC_with_mid_splited.csv`
  - `EDC_with_mid_splited.csv`
  - `HATU_with_mid_splited.csv`
  - `HBTU_with_mid_splited.csv`
  - `PyBOP_with_mid_splited.csv`
  - `TBTU_with_mid_splited.csv`

- **Virtual Compounds**:  
  Our virtual compounds used for Min-Max Sampling can be found in `virtual_compounds.xlsx`.

### Preprocessing

Run the following command to preprocess your dataset:

```bash
python main.py your_dataset_path EDC
```

### Data Splitting

Use the `data_split` function for splitting the dataset:

```python
from data_split import data_split
```

- `'train'` class: for training.
- `'one'` class: for the partial substrate novelty dataset.
- `'test'` class: for the full substrate novelty dataset.  
  Data outside the `'train'` class serves as the randomly divided test set.

---

## Model Training

Train machine learning models using the following:

```python
from models import train_machine_learning_model

# Use the model_type parameter to choose the model ('BERT', 'T5', 'SVM', 'XGBoost', or 'RandomForest').
# 'df' is your dataset, and you should specify columns for sub1 (ammonia), sub2 (acid), and the product.
train_machine_learning_model(model_type, df, sub1_column, sub2_column, product_column)
```

---

### BERT and T5 Models

- **Yield BERT Repository**: [rxn_yields](https://github.com/rxn4chemistry/rxn_yields)
- **T5-Chem Repository**: [t5chem](https://github.com/HelloJocelynLu/t5chem)

For high-throughput experimentation (HTE), we prepared 95 unique conditions, excluding acyl chloride due to incompatibility with DMF solvent. The SMILES reactions are formatted as:

```plaintext
amine.carboxylic_acid.condition_contexts>>product
```

---

## Machine Learning Model Setup

For machine learning under different reaction conditions, each condition should be encoded as a unique integer (1 to 95). This encoding helps the model differentiate between various setups.

```python
df_one_hot = pd.get_dummies(df, columns=[condition_id_column])
```

---

## Evaluation

To evaluate the model, use:

```bash
python BERT_evaluation.py --condition EDC --text_type 1
```

- **text_type** values:
  - `1`: `amine.carboxylic_acid>>product`
  - `2`: `amine.carboxylic_acid>intermediate>product`
  - `3`: `amine.intermediate>>product`

---

## Citation
Please kindly cite our papers if you use the data/code/model.
```
@article{amide_coupling, title={Intermediate Knowledge Enhanced the Performance of N-Acylation Yield Prediction Model}, DOI={10.26434/chemrxiv-2024-tzsnq}, journal={ChemRxiv}, author={Zhang, Chonghuan and Lin, Qianghua and Deng, Hao and Kong, Yaxian and Yu, Zhunzhun and Liao, Kuangbiao}, year={2024}}
```
---

## License
This project is licensed under the terms of the MIT license. See [LICENSE](https://github.com/aichemeco/Acylation/blob/main/LICENSE.md) for additional details.
