from arguments import inferenceArgs
from process import QGDataset
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


def inference(args):
    if torch.cuda.is_available():
        device= torch.device("cuda")  
    else:
        device= torch.device("cpu")
        print("using CPU as device, may cause Out of memory error")

    evalData= QGDataset(args)
    batch_size = min(16,evalData.__len__())
    ignore_label= -1 #check
    worker=0
    model= torch.load(args.infereceModelPath)
    model.to(device)
    model.eval()
    
    tokenizer = evalData.tokenizer
    vocab_size = tokenizer.vocab_size
    evalDataLoader = DataLoader(evalData,batch_size=batch_size, num_workers= worker)
    
    tdl = tqdm(evalDataLoader, total= len(evalDataLoader))
    for idx,batch in enumerate(tdl):
        ids = batch['ids'].to(device, dtype=torch.long)
        mask_ids = batch['mask_ids'].to(device, dtype=torch.long)
        seg_ids = batch['segment_ids'].to(device, dtype=torch.long)
        ques = batch['ques'].to(device, dtype=torch.long)
        with torch.no_grad():
            logits= model(
                input_ids= ids,
                attention_mask= mask_ids,
                decoder_input_ids= ids,
                # decoder_inputs_embeds= model.get_input_embeddings().weight,
                token_type_ids= seg_ids,
                masked_lm_labels = None
            )[0]
        logits= logits.view(-1, vocab_size)
        # orig_ques= ques.view(-1)
        logits = logits.detach().cpu().numpy()
        pred_ques = np.argmax(logits, axis=1).flatten().squeeze()
        pred_ques = np.reshape(pred_ques,(batch_size,-1))
        for i in range(ids.shape[0]):
            cur_pred_ques= list(pred_ques[i])
            try:
                cur_len= cur_pred_ques.index(102) # find first sep token
            except ValueError:
                cur_len= len(cur_pred_ques)-1
            cur_pred_ques = cur_pred_ques[:cur_len+1]
            cur_pred_ques= tokenizer.decode(cur_pred_ques, skip_special_tokens=True)
            # print("orignal ->>>>:" , cur_orignal_ques,"\n Predicted->>>>", cur_pred_ques)
            print(cur_pred_ques)


if __name__ == "__main__":
    inference(inferenceArgs)