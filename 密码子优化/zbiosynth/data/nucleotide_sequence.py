import os
import warnings
import numpy as np
from collections import defaultdict
from transformers import EsmTokenizer
import torch


class NucleotideSequence(object):
    """
    Proteins with sequence.
    Support both residue-level and atom-level operations and ensure consistency between two views.
    """
    
    ## nucleotide to id
    base2id = {'A': 0, 'C':1, 'G':2, 'T':3, 'N': 4}
    id2base = {v: k for k, v in base2id.items()}
    
    ## rnalm tokenizer for Nucleotide Sequence
    model_names = ["rnalm_8m", "rnalm_35m", "rnalm_150m", "rnalm_650m"]
    # tokenizer_8m = EsmTokenizer.from_pretrained(f"/share/liufuxu/zBioSynth/resources/pretrained_weights/RNA_LM/rnalm_8m/")  
    # tokenizer_35m = EsmTokenizer.from_pretrained(f"/share/liufuxu/zBioSynth/resources/pretrained_weights/RNA_LM/rnalm_35m/")
    # tokenizer_150m = EsmTokenizer.from_pretrained(f"/share/liufuxu/zBioSynth/resources/pretrained_weights/RNA_LM/rnalm_150m/")
    # tokenizer_650m = EsmTokenizer.from_pretrained(f"/share/liufuxu/zBioSynth/resources/pretrained_weights/RNA_LM/rnalm_650m/")
    
    
    def __init__(self, lm_model_name=None, model_path=None):
        """
        Args:
            lm_model_name: str, protein language model name, e.g. esm2_8m, esm2_35m, esm2_150m, esm2_650m
        """
        self.lm_model_name = lm_model_name
        self.model_path = model_path
        self.tokenizer = None
        if self.lm_model_name is not None:
            self.pad_token_id = self.get_rnalm_pad_tokenidx()
        else:
            self.pad_token_id = None
            
    
    def featurize(self, sequence):
        sequence = sequence.upper()
        sequence = sequence.replace('U', 'T')
        if self.lm_model_name == None:
            return self.onehot_featurize(sequence)
        else:
            return self.rnalm_featurize(sequence)
        
    def onehot_featurize(self, sequence):
        """
        Args:
            sequence: str, protein sequence
        Returns:
            features: np.array, one-hot encoded protein sequence
        """
        feature = np.zeros((len(sequence), len(self.base2id)))
        tokensids = np.array([self.base2id[token] for token in sequence])
        feature[np.arange(len(sequence)), tokensids] = 1
        ## return first 4 columns
        feature = feature[:, :4]
        return feature
    
    def rnalm_featurize(self, sequence):
        """
        Args:
            sequence: str, protein sequence
        Returns:
            features: np.array, esm encoded protein sequence
        """
        if self.tokenizer is None:
            if self.lm_model_name not in self.model_names:
                raise ValueError(f"lm_model_name {self.lm_model_name} not supported")
            self.tokenizer = EsmTokenizer.from_pretrained(f"{self.model_path}/RNA_LM/{self.lm_model_name}/")  

        feature = self.tokenizer(sequence, return_tensors='pt')
        
        return feature

    def get_rnalm_pad_tokenidx(self):
        """
        Returns:
            pad_token_idx: int, pad token index
        """
        if self.tokenizer is None:
            if self.lm_model_name not in self.model_names:
                raise ValueError(f"lm_model_name {self.lm_model_name} not supported")
            self.tokenizer = EsmTokenizer.from_pretrained(f"{self.model_path}/RNA_LM/{self.lm_model_name}/")  

        pad_token_id = self.tokenizer.pad_token_id
        
        return pad_token_id