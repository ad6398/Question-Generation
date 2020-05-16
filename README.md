# Question-Generation
Question Generation using transformers
Given a paragraph/text. Generate all the possible question related to this paragraph. Context(basically answer) may or may not be provided to generate question. If context is provided then genrate that question from that paragraph whose answer would be that context. 


## Dataset and Metrices
Whole model was trained on Squad 1.0 Dataset and BLUE scoring metrices were used.
## Models
### Simple Decoder above BERT Encoder
`model.py > class QG_SimpleBertModel(BertPreTrainedModel):` This was a simple architect which was a inital trial to tackle this problem. A simple MLP layer was used as decoder above output of BERT Model (i.e hidden state of Encoder). This model didn't give satisfiable BLUE score.


### BERT Encoder-Decoder Model
Implementation of this model was directly taken from Transformers Library by hugging face. BERT tansformer model was used as encoder as well as decoder. This model gave a appropriate BLUE score. Only training on half of SQuaD data (50K datapoint) for 5 epochs gave BLUE score of 36.91 on SQUAD v1.0 Dev data. This would increase on increasing number of epochs and trianing on whole SQUAD Data. 

## Walk Through
### Setup:
1. Python >=3.0
2. Transformers >=2.0
3. Spacy
4. NLTK
5. Pytorch
6. other Data science and general  libraries like Numpy, json, etc.

### scripts
`models.py`: Contains all classes for Encode Decoder Model, based on Transformers library

`train.py` contains all function to train Models and prediction

`process.py` : Dataset class based on torch, code to tokenize and conver into BERT input format

`inference.py`: code to load saved model and make inference

`arguments.py`: 

