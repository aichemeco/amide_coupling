# amide-coupling reaction model
This is the code for the "Intermediate Knowledge Enhanced the Performance of N-Acylation Yield Prediction Model" paper.
Preprint of this paper can be found in [[Paper](https://chemrxiv.org/engage/chemrxiv/article-details/66bdd53fa4e53c4876744a85)].  
And you can use our models directly at the following website:  
[www.aichemeco.com/acylation](https://www.aichemeco.com/acylation)

## Intermediate Knowledge Enhanced the Performance of amide-coupling Yield Prediction Model
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

pip install -r requirements.txt
pip install t5chem
```

## Data and data process
In data folder, we upload most data we used in our work.  
The structure of substrates we used for HTE is shown in  
HTE Data-Substrate and Products.xlsx  
Also, we uploaded our USPTO data in  
USPTO.csv  
The HTE data we used to establish our model for six single condition is saved in the following files,  
DCC_with_mid_splited.csv  
EDC_with_mid_splited.csv  
HATU_with_mid_splited.csv  
HBTU_with_mid_splited.csv  
PyBOP_with_mid_splited.csv  
TBTU_with_mid_splited.csv  
Use the following command for preprocessing
```shell
python main.py your_dataset_path EDC
```
Use the following command for data split and 'train' class is for training, 'one' class is for Partial substrate novelty dataset, 'test' class is for Full substrate novelty dataset. 
Data outside the 'train' class can be considered a randomly divided test set.
```python
from data_split import data_split

```
## Explanation of the dataset labels
### Data split
In our data files, we labeled different datasets we used when training the model in column 'class'.  
'train' means the data used to train our model.  
'one' means the partial substrate novelty data used for model test.  
'test' means the full substrate novelty data used for model test.  
'remaining' means the test data which does not in partial substrate novelty data or full substrate novelty data, both substrates in these data have appeared in training data.
### Training
In our work, we used three patterns to train our model:
1. no intermediate
2. amine + acid + intermediate → amide
3. amine + intermediate → amide  
The input reaction SMILES of the first pattern is labeled as 'text1', the input reaction SMILES of the second pattern is labeled as 'text2' and the input reaction SMILES of the third pattern is labeled as 'text3'. These three columns could be found in our datasets. These reactions could be used when training or prediction.
### 5-fold validation
Also, we labeled the test data we used to train our data in 5-fold validation in column 'n-fold'.  
In this column, 'fold-1-test' is the test data we used the first time.  
'fold-2-test' is the test data we used the second time.  
'fold-3-test' is the test data we used the third time.  
'fold-4-test' is the test data we used the fourth time.  
'fold-5-test' is the test data we used the fifth time.  
## Visualization
Use dimensionality_visualization.ipynb to do visualization.  
In our work, we employed four different methods to compare the differences between HTE data and USPTO data.
## BERT Model introduction
BERT (Bidirectional Encoder Representations from Transformers) is a revolutionary model in the field of natural language processing (NLP). Developed by researchers at Google AI Language, BERT was introduced in a paper titled "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" in 2018. This model has significantly advanced the state of the art in NLP tasks like text classification, question answering, and more.  
### The Evolution of NLP and BERT’s Impact
Before BERT, NLP models were primarily unidirectional, meaning they processed text in one direction (either left-to-right or right-to-left). This unidirectional approach limited the understanding of the context since it couldn’t fully capture the interdependence between words in a sentence. Models like RNNs (Recurrent Neural Networks) and LSTMs (Long Short-Term Memory networks) tried to address this by using sequential processing, but they still struggled with long-range dependencies and parallelization issues.  
The advent of the Transformer model, introduced by Vaswani et al. in 2017, marked a significant shift in NLP. Transformers allowed for parallel processing of words in a sentence, using a mechanism called "self-attention" to weigh the importance of different words in relation to each other. However, early Transformer-based models like OpenAI’s GPT (Generative Pre-trained Transformer) were still unidirectional, limiting their contextual understanding.  
BERT changed the game by being the first deeply bidirectional model, meaning it considers the context from both directions (left and right) at once. This bidirectional approach enables BERT to capture richer contextual relationships in the text, leading to superior performance on a variety of NLP tasks.  
### How BERT Works
BERT is built on the Transformer architecture, specifically the encoder part of the Transformer. The key innovation of BERT lies in how it is pre-trained. There are two main tasks that BERT is trained on:  
1. Masked Language Model (MLM): In this task, some of the words in the input sequence are randomly masked, and the model is trained to predict these masked words. For instance, in the sentence "The cat sat on the [MASK]," BERT would learn to predict that the masked word is "mat." This task forces BERT to learn bidirectional context because it needs to consider both the left and right contexts to predict the masked word.  
2. Next Sentence Prediction (NSP): This task involves training the model to understand the relationship between two sentences. The model is given pairs of sentences and must predict whether the second sentence follows the first in the original text. For example, given the sentences "The cat sat on the mat" and "It started to rain," the model would learn to understand that these two sentences are not sequentially related.  
The combination of these two tasks during pre-training allows BERT to develop a deep understanding of language, making it highly effective for a wide range of NLP tasks.  
### BERT’s Architecture
BERT comes in two main sizes: BERT-Base and BERT-Large. BERT-Base has 12 layers (also known as Transformer blocks), 768 hidden units, and 12 self-attention heads, resulting in 110 million parameters. BERT-Large is much bigger, with 24 layers, 1024 hidden units, and 16 self-attention heads, totaling 340 million parameters.  
Each layer in BERT’s architecture is composed of two main parts:  

1. Self-Attention Mechanism: This mechanism allows the model to focus on different parts of the input sentence at different layers, assigning different attention weights to each word based on its relevance to other words in the sentence.  

2. Feedforward Neural Network: After applying self-attention, the output is passed through a feedforward neural network, which is the same across all positions in the sequence. This network processes each position independently but with the contextual information provided by the self-attention mechanism.  

The outputs from these layers are then fed into the next layer, creating a deep, layered understanding of the text.  
### Fine-Tuning BERT for Specific Tasks
After BERT is pre-trained on large amounts of text, it can be fine-tuned for specific NLP tasks. Fine-tuning involves taking the pre-trained BERT model and training it further on a smaller, task-specific dataset. This is done by adding a small output layer on top of BERT and training the model on the new task.  

For example, in a sentiment analysis task, the output layer might be a softmax layer that classifies text as positive or negative. The fine-tuning process is relatively quick and requires significantly less data than training a model from scratch, as BERT has already learned a general understanding of language during pre-training.  
## Model training
```python
from models import train_machine_learning_model
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
"amine.carboxylic_acid.condition_contexts>>product"

## Machine learning
If trying to build machine learning model with data under different conditions, each of the different reaction conditions should be encoded as a unique integer ranging from 1 to 95. This encoding allowed the model to differentiate between various reaction setups.
```python
df_one_hot = pd.get_dummies(df, columns=[condition_id_column])
```
## Evaluation
```shell
python BERT_evaluation.py --condition EDC --text_type 1
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
@article{amide_coupling, title={Intermediate Knowledge Enhanced the Performance of N-Acylation Yield Prediction Model}, DOI={10.26434/chemrxiv-2024-tzsnq}, journal={ChemRxiv}, author={Zhang, Chonghuan and Lin, Qianghua and Deng, Hao and Kong, Yaxian and Yu, Zhunzhun and Liao, Kuangbiao}, year={2024}}
```

## License
This project is licensed under the terms of the MIT license. See [LICENSE](https://github.com/aichemeco/Acylation/blob/main/LICENSE.md) for additional details.
