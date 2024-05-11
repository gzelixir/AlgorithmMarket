import os
import warnings
import numpy as np
from collections import defaultdict
from transformers import EsmTokenizer, AutoTokenizer
from transformers import AutoTokenizer, T5ForConditionalGeneration, T5EncoderModel
from transformers import BertTokenizer
import torch


class MoleculeSequence(object):
    """
    Molecule with sequence.
    """
    
    ## esm2 lm tokenizer for protein sequence
    model_names = ["molt5-base", "molt5-small", "smole-bert"]

    def __init__(self, lm_model_name=None, model_path=None):
        """
        Args:
            lm_model_name: str, protein language model name, e.g. esm2_8m, esm2_35m, esm2_150m, esm2_650m
        """
        self.lm_model_name = lm_model_name
        self.model_path = model_path
        self.tokenizer = None
        if self.lm_model_name is not None:
            self.pad_token_id = self.get_lm_pad_tokenidx()
        else:
            self.pad_token_id = None
            
    
    # def onehot_featurize(self, sequence):
    #     """
    #     Args:
    #         sequence: str, protein sequence
    #     Returns:
    #         features: np.array, one-hot encoded protein sequence
    #     """
    #     feature = np.zeros((len(sequence), len(self.aa2id)))
    #     tokensids = np.array([self.aa2id[token] for token in sequence])
    #     feature[np.arange(len(sequence)), tokensids] = 1
    #     return feature
    
    def featurize(self, sequence):
        sequence = sequence.upper()
        return self.lm_featurize(sequence)
        
    def lm_featurize(self, sequence):
        """
        Args:
            sequence: str, molecule sequence
        Returns:
            features: np.array, esm encoded protein sequence
        """
        if self.tokenizer is None:
            if self.lm_model_name not in self.model_names:
                raise ValueError(f"lm_model_name {self.lm_model_name} not supported")
            if 't5' in self.lm_model_name:
                self.tokenizer = AutoTokenizer.from_pretrained(f"{self.model_path}/mol_LM/{self.lm_model_name}/", model_max_length=512)
            elif 'bert' in self.lm_model_name:
                self.tokenizer = BertTokenizer.from_pretrained(f"{self.model_path}/mol_LM/{self.lm_model_name}/")
        feature = self.tokenizer(sequence, return_tensors="pt")
        return feature

    def get_lm_pad_tokenidx(self):
        """
        Returns:
            pad_token_idx: int, pad token index
        """
        if self.tokenizer is None:
            if self.lm_model_name not in self.model_names:
                raise ValueError(f"lm_model_name {self.lm_model_name} not supported")
            self.tokenizer = AutoTokenizer.from_pretrained(f"{self.model_path}/mol_LM/{self.lm_model_name}/", model_max_length=512)
        
        pad_token_id = self.tokenizer.pad_token_id
        
        return pad_token_id