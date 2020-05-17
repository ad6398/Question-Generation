import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_bert import BertLayerNorm, BertPreTrainedModel, BertModel
from transformers.activations import get_activation


class MLPWithLayerNorm(nn.Module):
    ''' MLP layer with layer regularization
    '''  
    def __init__(self, config, input_size):
        super(MLPWithLayerNorm, self).__init__()
        self.config = config
        self.linear1 = nn.Linear(input_size, config.hidden_size)
        self.non_lin1 = get_activation(self.config.hidden_act)
        self.layer_norm1 = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.linear2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.non_lin2 = get_activation(self.config.hidden_act)
        self.layer_norm2 = BertLayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden):
        return self.layer_norm2(self.non_lin2(self.linear2(self.layer_norm1(self.non_lin1(self.linear1(hidden))))))


class QG_SimpleBertModel(BertPreTrainedModel):
    ''' model for QG with bert as encoder and a simple decoder of MLP layers
    '''
    def __init__(self,config,bert_type_or_path, vocab_size):
        super().__init__(config)
        self.config= config
        self.vocab_size= vocab_size
        self.main_encoder= BertModel.from_pretrained(bert_type_or_path)  # out: last_HS, pooled_out, all_HS, attention(opt)
        self.mlp_input_size=  config.hidden_size  # CR 
        self.mlp = MLPWithLayerNorm(config, self.mlp_input_size) 
        self.decoder = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.decoder.weight= self.main_encoder.get_input_embeddings().weight #initalize decoder with encoder embedding
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        ques= None
        ):
        last_hs, pooled= self.main_encoder(input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds)
        hidden_states= self.mlp(last_hs)        
        logits= self.decoder(hidden_states)
        assert logits.size(-1)== self.vocab_size 
        loss = None
        if ques!= None:
            pre_labels= logits.view(-1, self.vocab_size)
            ques_labels= ques.view(-1)
            loss= F.cross_entropy(
                pre_labels,
                ques_labels,
                size_average=False,
                ignore_index= 0, 
                reduce=True 
            )
        return loss, logits