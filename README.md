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
    this script initialize all the required arguments needs to make train, valid Dataset, for training QG model and for infernce.There is a class for each of these

    `class trainDataArgs/ validDataArgs:` a class to create training/valid Dataset
        bert_model= type of BERT model, for example 'bert-base-uncased'

        max_seq_len = max sequence length for trainig/validation

        squad_path= file path to training/validation Data in format of SQUAD.

        inferenceMode= True if we want to give input data other than SQUAD else False

        inferenceFile= if inferenceMode = True: a path to json file consisting of list of dict 
          [{'paragraph':' a string', 'context_list': ['list' , 'of', 'context', 'i.e answers'] }, {}, . . .]  context_list could be a optional in this json file

        occu = 30000 to restrict number of datapoints in this dataset 

    `class inferenceArgs:`
        all other argument are same as above class.
        infereceModelPath= path to saved pytorch Model


    `class trainingConfig:`
        bert_model = bert_model
        bert_config = config for BERT model , a optional argument. BertConfig.from_pretrained(bert_model)
        max_seq_len= max_seq_len
        train_batch_size= 16
        valid_batch_size= 16
        ignore_label =0 label to ignore, in genral padded token
        num_workers=0
        epochs =3 number of epochs
        learningRate = 5e-5
        save_dir = directory to save pytorch model. Whole model is saved not only weight

