import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from transformers import BertConfig
import json
import spacy


def get_tokenizer(model_type_path):
    tok = BertTokenizer.from_pretrained(model_type_path, do_lower_case=True)
    return tok

def get_mask_ids(tokens, max_seq_length):
    """attention Mask id for padding 1 for original 0 for padded"""
    if len(tokens)>max_seq_length:
        raise IndexError("Token length more than max seq length!")
    return [1]*len(tokens) + [0] * (max_seq_length - len(tokens))

def get_segment_ids(tokens, max_seq_length):
    """Segments id : 0 for the first sequence, 1 for the second"""
    if len(tokens)>max_seq_length:
        raise IndexError("Token length more than max seq length!")
    segments = []
    first_sep = True
    current_segment_id = 0
    for token in tokens:
        segments.append(current_segment_id)
        if token == "[SEP]":
            if first_sep:
                first_sep = False 
                current_segment_id = 1
    assert current_segment_id ==1
    return segments + [0] * (max_seq_length - len(tokens))

def get_token_ids(tokens, tokenizer, max_seq_length):
    """Token ids from Tokenizer vocab"""
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = token_ids + [0] * (max_seq_length-len(token_ids))
    return input_ids



def process_input(text, ans, ques, tokenizer, max_seq_len):
    text_token = tokenizer.tokenize(text)
    ans_token= tokenizer.tokenize(ans)
    ques_token= tokenizer.tokenize(ques)

    if len(text_token) > max_seq_len-3 -len(ans_token):
        text_token = text_token[:max_seq_len- 3-len(ans_token)]

    return text_token, ans_token, ques_token
    
def pad_ques(ques, max_seq_len, padding =0):
    if len(ques)> max_seq_len:
        raise IndexError("len of ques {} greater than max_seq_len{}".format(len(ques), max_seq_len))

    req_len= max_seq_len- len(ques)
    ques += [padding]* req_len

    return ques



def convert_to_input(text, ans, ques, tokenizer, max_seq_len):
    text_token, ans_token, ques_token= process_input(text,ans,ques, tokenizer, max_seq_len)

    allToken= ["[CLS]"] + text_token  + ["[SEP]"] + ans_token + ["[SEP]"]
    ques_token= ["[CLS]"] + ques_token  + ["[SEP]"]
    ids= get_token_ids(allToken, tokenizer, max_seq_len)
    mask_ids = get_mask_ids(allToken, max_seq_len)
    segment_ids = get_segment_ids(allToken, max_seq_len)

    que_ids= get_token_ids(ques_token, tokenizer, len(ques_token))

    que_ids= pad_ques(que_ids, max_seq_len) #CR padding
    assert len(que_ids)== max_seq_len
    return ids, mask_ids, segment_ids, que_ids


def generate_context_list(text):
    ''' Extract noun chunks with spacy from text paragraph to get context
    '''
    nlp= spacy.load('en_core_web_sm') #this will download spacy english core model
    nlpDoc= nlp(text)
    lst= []
    for chunk in nlpDoc.noun_chunks:
        lst.append(chunk)
    return lst


class QGDataset(Dataset):
    def __init__(self, args):
        super().__init__()
        self.args= args
        self.examples= []
        self.tokenizer= get_tokenizer(args.bert_model)
        self.max_seq_len= self.args.max_seq_len

        if args.squad_path== None and args.inferenceMode == False:
            raise ValueError("invaild path to squad 1.0 data or wrong inference mode")

        if args.squad_path != None:
            with open(args.squad_path) as f:
                json_data= json.load(f)
                json_data = json_data['data']

            for data in json_data:
                for para in data['paragraphs']:
                    con = para['context']
                    qas= para['qas']
                    for xs in qas:
                        cur_ans= xs['answers'][0]['text']
                        cur_ans_offset= xs['answers'][0]['answer_start']
                        que= xs['question']
                        ex= {
                            'text': con,
                            'ans': cur_ans,
                            'ans_offset': cur_ans_offset,
                            'ques': que
                        }
                        self.examples.append(ex)
            del json_data
        
        else:
            if args.inferenceFile == None:
                raise ValueError("wrong file for inference")
            
            with open(args.inferenceFile) as f:
                json_data= json.load(f)
            
            for item in json_data:
                paragraph= None
                contextList= None
                if 'paragraph' in item.keys():
                    paragraph = item['paragraph']
                
                else:
                    raise KeyError("no text para graph is found. worong format of inference file")
                
                if 'context_list' in  item.keys():
                    contextList= item['context_list']
                
                else:
                    contextList = generate_context_list(paragraph)
                
                for context in contextList:
                    cur_ex= {
                        'text': paragraph,
                        'ans': context,
                        'ques': "a dummy ques to avoid None error"
                    }

                    self.examples.append(cur_ex)
                

        if args.occu:
            self.examples= self.examples[:args.occu]


    def __len__(self):
        return len(self.examples)

    def __getitem__(self,idx):
        cur_ex= self.examples[idx]
        ids, mask_id, seg_id, ques= convert_to_input(cur_ex['text'], cur_ex['ans'], cur_ex['ques'], self.tokenizer, self.max_seq_len)

        exm= {
            'ids': torch.tensor(ids, dtype= torch.long),
            'mask_ids': torch.tensor(mask_id, dtype= torch.long),
            'segment_ids': torch.tensor(seg_id, dtype= torch.long),
            'ques': torch.tensor(ques, dtype= torch.long)
        }
#         print(exm)
        return exm
    



        

        



