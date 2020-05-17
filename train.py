import os
import torch
from tqdm.notebook import tqdm
import pandas as pd
import torch.nn as nn
import numpy as np
import time, random

from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

from transformers import AdamW, BertConfig 
from transformers import get_linear_schedule_with_warmup
from transformers import EncoderDecoderModel

from process import QGDataset
from model import QG_SimpleBertModel
from arguments import trainDataArgs, trainingConfig, validDataArgs
from nltk.translate.bleu_score import sentence_bleu

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_blue_score(orig, pre):
#     print("orignal blue ->>>>:" , orig,"\n Predicted blue->>>>", pre)
    orig_tok= orig.split()
    pre_tok= pre.split()[:len(orig_tok)]
#     print("orignal blue ->>>>:" , orig_tok,"\n Predicted blue->>>>", pre_tok)
    ref= [orig_tok]
    score= sentence_bleu(ref, pre_tok)
    return score


def predict(evalData, batch_size, device, model,  ignore_label=-1, worker=0):
    model.eval()
    tokenizer = evalData.tokenizer
    vocab_size = tokenizer.vocab_size
    evalDataLoader = DataLoader(evalData,batch_size=batch_size, num_workers= worker)
    
    tdl = tqdm(evalDataLoader, total= len(evalDataLoader))
    total_acc= AverageMeter()
    predictions= []
    for idx,batch in enumerate(tdl):
        ids = batch['ids'].to(device, dtype=torch.long)
        mask_ids = batch['mask_ids'].to(device, dtype=torch.long)
        seg_ids = batch['segment_ids'].to(device, dtype=torch.long)
        ques = batch['ques'].to(device, dtype=torch.long)
        with torch.no_grad():
            loss, logits= model(
                input_ids= ids,
                attention_mask= mask_ids,
                decoder_input_ids= ids,
                # decoder_inputs_embeds= model.get_input_embeddings().weight,
                token_type_ids= seg_ids,
                masked_lm_labels = ques
            )[:2]

        logits= logits.view(-1, vocab_size)
        # orig_ques= ques.view(-1)
        logits = logits.detach().cpu().numpy()
        orig_ques = ques.detach().cpu().numpy()
        pred_ques = np.argmax(logits, axis=1).flatten().squeeze()
        pred_ques = np.reshape(pred_ques,(batch_size,-1))
        cur_pre= []
        for i in range(orig_ques.shape[0]):
            cur_orignal_ques= tokenizer.decode(list(orig_ques[i]), skip_special_tokens=True)
            cur_pred_ques= list(pred_ques[i])
            try:
                cur_len= cur_pred_ques.index(102) # find first sep token
            except ValueError:
                cur_len= len(cur_pred_ques)-1

            cur_pred_ques = cur_pred_ques[:cur_len+1]
            cur_pred_ques= tokenizer.decode(cur_pred_ques, skip_special_tokens=True)
            # print("orignal ->>>>:" , cur_orignal_ques,"\n Predicted->>>>", cur_pred_ques)
            cur_acc= get_blue_score(cur_orignal_ques, cur_pred_ques)
            cur_pre.append(cur_pred_ques)
            total_acc.update(cur_acc)

        predictions += cur_pre
        tdl.set_postfix(accu= total_acc.avg)

    return predictions


def train(trainData, validData, device, train_config):
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    tokenizer= trainData.tokenizer
    vocab_size= tokenizer.vocab_size
    model = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-uncased', 'bert-base-uncased')
    batch_size= train_config.train_batch_size
    trainDataloader= DataLoader(trainData, batch_size= train_config.train_batch_size, num_workers= train_config.num_workers)

    param_optimizer = list(model.named_parameters())  #get parameter of models
    no_decay = [
        "bias",
        "LayerNorm.bias",
        "LayerNorm.weight"
    ] ##doubt layers to be not decayed #issue
    optimizer_parameters = [
        {
            'params': [
                p for n, p in param_optimizer if not any(
                    nd in n for nd in no_decay
                )
            ], 
         'weight_decay': 0.001
        },
        {
            'params': [
                p for n, p in param_optimizer if any(
                    nd in n for nd in no_decay
                )
            ], 
            'weight_decay': 0.0
        },
    ]    
    optimizer =AdamW(
        optimizer_parameters, 
        lr= train_config.learningRate
    )
    total_len= trainData.__len__()

    num_steps= total_len/train_config.train_batch_size*train_config.epochs
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_steps
    )
    
    model.to(device)
    
    for epoch_i in range(0, train_config.epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, train_config.epochs))
        print('Training...')
        t0 = time.time()
        total_loss = AverageMeter()
        total_acc = AverageMeter()
        model.train()

        tdl = tqdm(trainDataloader, total=len(trainDataloader))
        for idx,batch in enumerate(tdl):

            ids= batch['ids'].to(device, dtype= torch.long)
            mask_ids= batch['mask_ids'].to(device, dtype= torch.long)
            seg_ids= batch['segment_ids'].to(device, dtype= torch.long)
            ques= batch['ques'].to(device, dtype= torch.long)
            
            model.zero_grad()

            loss, logits= model(
                input_ids= ids,
                attention_mask= mask_ids,
                decoder_input_ids= ids,
#                 decoder_inputs_embeds= model.get_input_embeddings().weight,
                token_type_ids= seg_ids,
                masked_lm_labels = ques
            )[:2]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            logits= logits.view(-1, vocab_size)
            # orig_ques= ques.view(-1)
            logits = logits.detach().cpu().numpy()
            orig_ques = ques.detach().cpu().numpy()
            pred_ques = np.argmax(logits, axis=1).flatten().squeeze()
            pred_ques = np.reshape(pred_ques,(batch_size,-1))
#             print("shape of orig and pred batch: ",orig_ques.shape, pred_ques.shape)
            for i in range(orig_ques.shape[0]):
                cur_orignal_ques= tokenizer.decode(list(orig_ques[i]), skip_special_tokens=True)
                cur_pred_ques= tokenizer.decode(list(pred_ques[i]), skip_special_tokens=True)
                cur_acc= get_blue_score(cur_orignal_ques, cur_pred_ques)
                total_acc.update(cur_acc)

            total_loss.update(loss.item(), mask_ids.size(0))
            tdl.set_postfix(accu= total_acc.avg)
            tdl.set_postfix(loss= total_loss.avg, accu= total_acc.avg)

        if validData:
            prediciton= predict(validData, train_config.valid_batch_size, device, model, ignore_label= train_config.ignore_label, worker= train_config.num_workers)

        torch.save(model, train_config.save_dir+"model_{}".format(epoch_i)) #save whole model after epoch


if __name__ == "__main__":
    if torch.cuda.is_available():    
    # Tell PyTorch to use the GPU.    
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    trainData= QGDataset(trainDataArgs)
    validData= QGDataset(validDataArgs)
    train(trainData, validData, device, trainingConfig)