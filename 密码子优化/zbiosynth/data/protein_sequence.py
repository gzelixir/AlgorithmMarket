import os
import warnings
import numpy as np
from collections import defaultdict
from transformers import EsmTokenizer, AutoTokenizer, T5Tokenizer
import torch


class ProteinSequence(object):
    """
    Proteins with sequence.
    Support both residue-level and atom-level operations and ensure consistency between two views.
    """
    
    ## amino acid to id    
    aa2id  = {'A': 0,'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11,
              'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19, 'X': 20}
    id2aa = {v: k for k, v in aa2id.items()}
    ## esm2 lm tokenizer for protein sequence
    model_names = ["esm2_8m", "esm2_35m", "esm2_150m", "esm2_650m", "prot_t5_xl"]
    
    # tokenizer_8m = EsmTokenizer.from_pretrained(f"/share/liufuxu/zBioSynth/resources/pretrained_weights/hub/esm2_8m/")  
    # tokenizer_35m = EsmTokenizer.from_pretrained(f"/share/liufuxu/zBioSynth/resources/pretrained_weights/hub/esm2_35m/")
    # tokenizer_150m = EsmTokenizer.from_pretrained(f"/share/liufuxu/zBioSynth/resources/pretrained_weights/hub/esm2_150m/")
    # tokenizer_650m = EsmTokenizer.from_pretrained(f"/share/liufuxu/zBioSynth/resources/pretrained_weights/hub/esm2_650m/")
    # tokenizer_esmfold = AutoTokenizer.from_pretrained(f"/share/liufuxu/zBioSynth/resources/pretrained_weights/hub/esmfold_v1/")

    
    def __init__(self, lm_model_name=None, model_path=None):
        """
        Args:
            lm_model_name: str, protein language model name, e.g. esm2_8m, esm2_35m, esm2_150m, esm2_650m
        """
        self.lm_model_name = lm_model_name
        self.model_path = model_path
        self.tokenizer = None
        if self.lm_model_name is not None:
            self.pad_token_id = self.get_esm2_pad_tokenidx()
        else:
            self.pad_token_id = None
            
    
    def featurize(self, sequence):
        sequence = sequence.upper()
        if self.lm_model_name == None:
            return self.onehot_featurize(sequence)
        else:
            return self.esm2_featurize(sequence)
        
    def onehot_featurize(self, sequence):
        """
        Args:
            sequence: str, protein sequence
        Returns:
            features: np.array, one-hot encoded protein sequence
        """
        feature = np.zeros((len(sequence), len(self.aa2id)))
        tokensids = np.array([self.aa2id[token] for token in sequence])
        feature[np.arange(len(sequence)), tokensids] = 1
        return feature
    
    def esm2_featurize(self, sequence):
        """
        Args:
            sequence: str, protein sequence
        Returns:
            features: np.array, esm encoded protein sequence
        """
        if self.tokenizer is None:
            if self.lm_model_name not in self.model_names:
                raise ValueError(f"lm_model_name {self.lm_model_name} not supported")
            if 'esm2' in self.lm_model_name:
                self.tokenizer = EsmTokenizer.from_pretrained(f"{self.model_path}/prot_LM/{self.lm_model_name}/")  
            elif 't5' in self.lm_model_name:
                self.tokenizer = T5Tokenizer.from_pretrained(f"{self.model_path}/prot_LM/{self.lm_model_name}/", do_lower_case=False)
                
        feature = self.tokenizer(sequence, return_tensors='pt')
        return feature

    def get_esm2_pad_tokenidx(self):
        """
        Returns:
            pad_token_idx: int, pad token index
        """
        if self.tokenizer is None:
            if self.lm_model_name not in self.model_names:
                raise ValueError(f"lm_model_name {self.lm_model_name} not supported")
            if 'esm2' in self.lm_model_name:
                self.tokenizer = EsmTokenizer.from_pretrained(f"{self.model_path}/prot_LM/{self.lm_model_name}/")  
            elif 't5' in self.lm_model_name:
                self.tokenizer = T5Tokenizer.from_pretrained(f"{self.model_path}/prot_LM/{self.lm_model_name}/", do_lower_case=False)
                 
        pad_token_id = self.tokenizer.pad_token_id
   
        return pad_token_id