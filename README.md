# Question-Generation
Question Generation using transformers
Given a paragraph/text. Generate all the possible question related to this paragraph. Context(basically answer) may or may not be provided to generate question. If context is provided then genrate that question from that paragraph whose answer would be that context. 


## Dataset and Metrices
Whole model was trained on Squad 1.0 Dataset and BLUE scoring metrices were used.
## Models
### Simple Decoder above BERT Encoder
`model.py > class QG_SimpleBertModel(BertPreTrainedModel):` This was a simple architect which was a inital trial to tackle this problem. A simple MLP layer was used as decoder above output of BERT Model (i.e hidden state of Encoder). This model didn't give satisfiable BLUE score.


### BERT Encoder-Decoder Model
Implementation of this model was directly taken from Transformers Library by hugging face. BERT tansformer model was used as encoder as well as decoder. This model gave a appropriate BLUE score. This Score 
