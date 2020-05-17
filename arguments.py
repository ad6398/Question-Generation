
from transformers import BertConfig 

bert_model= 'bert-base-uncased'
max_seq_len =256

class trainDataArgs:
    bert_model= bert_model
    max_seq_len = max_seq_len
    squad_path= 'train-v1.1.json' #CR
    inferenceMode= False  # T/F for inference
    inferenceFile= None  # if nferenceMode = True: a path to json file consisting of list of dict with keys 'paragraph' as mandatory key and 'context_list' as option  [{'paragraph':' a string', 'context_list': ['list' , 'of', 'context', 'i.e answers'] }, {}, . . .]

    occu = None

class validDataArgs:
    bert_model= bert_model
    max_seq_len =max_seq_len
    squad_path= 'dev-v1.1.json' #CR
    inferenceMode= False  # T/F for inference
    inferenceFile= None  # if nferenceMode = True: a path to json file consisting of list of dict with keys 'paragraph' as mandatory key and 'context_list' as option  [{'paragraph':' a string', 'context_list': ['list' , 'of', 'context', 'i.e answers'] }, {}, . . .]
    occu= 10000

class inferenceArgs:
    bert_model= bert_model
    max_seq_len =max_seq_len
    squad_path= 'dev-v1.1.json' #CR
    inferenceMode= False  # T/F for inference
    inferenceFile= None  # CR
    # if nferenceMode = True: a path to json file consisting of list of dict with keys 'paragraph' as mandatory key and 'context_list' as option  [{'paragraph':' a string', 'context_list': ['list' , 'of', 'context', 'i.e answers'] }, {}, . . .]
    occu= None
    infereceModelPath= 'model_9'


class trainingConfig:
    bert_model = bert_model
    bert_config = BertConfig.from_pretrained(bert_model)
    max_seq_len= max_seq_len
    train_batch_size= 16
    valid_batch_size= 16
    ignore_label =0
    num_workers=0
    epochs =10
    learningRate = 5e-5
    save_dir = ''
