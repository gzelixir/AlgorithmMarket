"""
Prepare the Shakespeare dataset for character-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""
import os
import pickle
import requests
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='argument parser')
# complusory settings
parser.add_argument('--data', type=str, help='data', default = "input_origin")
args = parser.parse_args()


external_input =  os.path.join(__file__.split("pgmlflow")[0],'data', f'{args.data}')
print("external_input:",external_input)
if os.path.exists(external_input):
    input_file_path = external_input
    print(f"load {external_input} to prepare.py")
else:
    # download the tiny shakespeare dataset
    input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
    print("load input.txt in git")
# if not os.path.exists(input_file_path):
#     data_url = 'https://raw.githubusercontent.com/HaochenW/Deep_promoter/master/seq/sequence_data.txt'
#     with open(input_file_path, 'w') as f:
#         f.write(requests.get(data_url).text)

with open(input_file_path, 'r') as f:
    data = f.read()
print(f"length of dataset in characters: {len(data):,}")

# get all the unique characters that occur in this text
chars = sorted(list(set(data)))
vocab_size = len(chars)
print("all the unique characters:", ''.join(chars))
print(f"vocab size: {vocab_size:,}")

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# create the train and test splits
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode both to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)

train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))


# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

# length of dataset in characters: 718,998
# all the unique characters: 
# ACGT
# vocab size: 5
# train has 647,098 tokens
# val has 71,900 tokens

# length of dataset in characters: 9,178,916
# all the unique characters: 
# ACGTacgntwy
# vocab size: 12
# train has 8,261,024 tokens
# val has 917,892 tokens
