import os
import numpy as np 
import torch



def onehot_data_collate(batch):
    x, y = zip(*batch)
    max_length = max([len(onehot) for onehot in x])
    ## padding
    x = [np.pad(onehot, ((0, max_length - len(onehot)), (0, 0)), 'constant', constant_values=0) for onehot in x]
    x = np.array(x)
    
    data_dict = {}
    data_dict['input'] = torch.FloatTensor(x)
    data_dict['label'] = torch.FloatTensor(y).view(-1,1)
    return data_dict

def default_lm_data_collate(batch):
    batch, y = zip(*batch)
    labels = torch.FloatTensor(y).view(-1,1)
    
    inputs = torch.cat([item['input_ids'] for item in batch], dim=0)
    attention_mask = torch.cat([item['attention_mask'] for item in batch], dim=0)
    
    return {"input_ids": inputs,
            "attention_mask": attention_mask,
            "label": labels}

def emb_data_collate(batch):
    batch, y = zip(*batch)
        
    pad_token_id = batch[0]['pad_token_id']
    max_length = max([len(item['input_ids'][0]) for item in batch])
    
    ## padding
    inputs = []
    attention_mask = []
    for item in batch:
        size = len(item['input_ids'][0])
        inputs.append(torch.cat([item['input_ids'], torch.ones((1, max_length - size))*pad_token_id], dim=1).long())
        attention_mask.append(torch.cat([item['attention_mask'], torch.zeros((1, max_length - size))], dim=1).long())
    inputs = torch.cat(inputs, dim=0)
    attention_mask = torch.cat(attention_mask, dim=0)
    
    labels = torch.FloatTensor(y).view(-1,1)
    
    return {"input_ids": inputs,
            "attention_mask": attention_mask,
            "label": labels}
    
    
def go_data_collate(batch):
    batch, bp_y, mf_y, cc_y = zip(*batch)
    
    pad_token_id = batch[0]['pad_token_id']
    max_length = max([len(item['input_ids'][0]) for item in batch])
    
    ## padding
    inputs = []
    attention_mask = []
    for item in batch:
        size = len(item['input_ids'][0])
        inputs.append(torch.cat([item['input_ids'], torch.ones((1, max_length - size))*pad_token_id], dim=1).long())
        attention_mask.append(torch.cat([item['attention_mask'], torch.zeros((1, max_length - size))], dim=1).long())
    inputs = torch.cat(inputs, dim=0)
    attention_mask = torch.cat(attention_mask, dim=0)
    
    bp_y = torch.stack(bp_y)
    mf_y = torch.stack(mf_y)
    cc_y = torch.stack(cc_y)
    
    return {"input_ids": inputs,
            "attention_mask": attention_mask,
            "bp_label": bp_y,
            "mf_label": mf_y,
            "cc_label": cc_y
            }
  
def go_single_data_collate(batch):
    batch, y = zip(*batch)
    
    pad_token_id = batch[0]['pad_token_id']
    max_length = max([len(item['input_ids'][0]) for item in batch])
    
    ## padding
    inputs = []
    attention_mask = []
    for item in batch:
        size = len(item['input_ids'][0])
        inputs.append(torch.cat([item['input_ids'], torch.ones((1, max_length - size))*pad_token_id], dim=1).long())
        attention_mask.append(torch.cat([item['attention_mask'], torch.zeros((1, max_length - size))], dim=1).long())
    inputs = torch.cat(inputs, dim=0)
    attention_mask = torch.cat(attention_mask, dim=0)
    
    y = torch.stack(y)
    
    return {"input_ids": inputs,
            "attention_mask": attention_mask,
            "label": y,
            }  
    
def emb_ddg_collate(batch):
    wt_batch, mut_batch, position, y = zip(*batch)
        
    pad_token_id = wt_batch[0]['pad_token_id']
    max_length = max([len(item['input_ids'][0]) for item in wt_batch])
    
    ## padding
    wt_inputs = []
    wt_attention_mask = []
    for item in wt_batch:
        size = len(item['input_ids'][0])
        wt_inputs.append(torch.cat([item['input_ids'], torch.ones((1, max_length - size))*pad_token_id], dim=1).long())
        wt_attention_mask.append(torch.cat([item['attention_mask'], torch.zeros((1, max_length - size))], dim=1).long())
       
    wt_inputs = torch.cat(wt_inputs, dim=0)
    wt_attention_mask = torch.cat(wt_attention_mask, dim=0)
    
    
    mut_inputs = [] 
    mut_attention_mask = []
    for item in mut_batch:
        size = len(item['input_ids'][0])
        mut_inputs.append(torch.cat([item['input_ids'], torch.ones((1, max_length - size))*pad_token_id], dim=1).long())
        mut_attention_mask.append(torch.cat([item['attention_mask'], torch.zeros((1, max_length - size))], dim=1).long())
        
    mut_inputs = torch.cat(mut_inputs, dim=0)
    mut_attention_mask = torch.cat(mut_attention_mask, dim=0)
    
    labels = torch.FloatTensor(y).view(-1,1)
    positions = torch.LongTensor(position).view(-1)
    return {"wt_input_ids": wt_inputs,
            "wt_attention_mask": wt_attention_mask,
            "mut_input_ids": mut_inputs,
            "mut_attention_mask": mut_attention_mask,
            "position": positions,
            "label": labels}
    
    
def protein_mol_collate(batch):
    batch, y = zip(*batch)
        
    prot_pad_token_id = batch[0]['prot_pad_token_id']
    prot_max_length = max([len(item['prot_input_ids'][0]) for item in batch])
    
    mol_pad_token_id = batch[0]['mol_pad_token_id']
    mol_max_length = max([len(item['mol_input_ids'][0]) for item in batch])
    
    ## padding
    prot_inputs = []
    prot_attention_mask = []
    for item in batch:
        size = len(item['prot_input_ids'][0])
        prot_inputs.append(torch.cat([item['prot_input_ids'], torch.ones((1, prot_max_length - size))*prot_pad_token_id], dim=1).long())
        prot_attention_mask.append(torch.cat([item['prot_attention_mask'], torch.zeros((1, prot_max_length - size))], dim=1).long())
    prot_inputs = torch.cat(prot_inputs, dim=0)
    prot_attention_mask = torch.cat(prot_attention_mask, dim=0)
    
    mol_inputs = []
    mol_attention_mask = []
    for item in batch:
        size = len(item['mol_input_ids'][0])
        mol_inputs.append(torch.cat([item['mol_input_ids'], torch.ones((1, mol_max_length - size))*mol_pad_token_id], dim=1).long())
        mol_attention_mask.append(torch.cat([item['mol_attention_mask'], torch.zeros((1, mol_max_length - size))], dim=1).long())
    mol_inputs = torch.cat(mol_inputs, dim=0)
    mol_attention_mask = torch.cat(mol_attention_mask, dim=0)
    
    labels = torch.FloatTensor(y).view(-1,1)
    
    return {"prot_input_ids": prot_inputs,
            "prot_attention_mask": prot_attention_mask,
            "mol_input_ids": mol_inputs,
            "mol_attention_mask": mol_attention_mask,
            "label": labels}
    

class LM_Dynamic_Seqlen_collate(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        x, y = zip(*batch)
        max_length = max([len(seq) for seq in x]) + 2
        ## padding
        x = self.tokenizer.batch_encode_plus(x, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
        
        data_dict = {}
        data_dict['input_ids'] = x['input_ids']
        data_dict['attention_mask'] = x['attention_mask']
        data_dict['label'] = torch.FloatTensor(y).view(-1,1)
        return data_dict
    
def codon_optimized_data_collate(batch):
    batch, batch_label = zip(*batch)
        
    pad_token_id = batch[0]['pad_token_id']
    max_length = max([len(item['input_ids'][0]) for item in batch])
    
    ## padding
    inputs = []
    attention_mask = []

    for item in batch:
        size = len(item['input_ids'][0])
        inputs.append(torch.cat([item['input_ids'], torch.ones((1, max_length - size))*pad_token_id], dim=1).long())
        attention_mask.append(torch.cat([item['attention_mask'], torch.zeros((1, max_length - size))], dim=1).long())
        
    label_inputs = []
    for item in batch_label:
        size = len(item['input_ids'][0])
        label_inputs.append(torch.cat([item['input_ids'], torch.ones((1, max_length - size))*-100], dim=1).long())


    inputs = torch.cat(inputs, dim=0)
    attention_mask = torch.cat(attention_mask, dim=0)
    label_inputs = torch.cat(label_inputs, dim=0)    
    return {"input_ids": inputs,
            "attention_mask": attention_mask,
            "label": label_inputs}