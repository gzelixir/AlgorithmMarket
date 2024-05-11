import os
import warnings
import numpy as np
import random
from collections import defaultdict
from transformers import EsmTokenizer
import torch
from dnachisel import DnaOptimizationProblem
from dnachisel import *
from dnachisel.biotools import gc_content



codon2aminoacid = {'TTT': 'F', 'CTT': 'L', 'ATT': 'I', 'GTT': 'V', 
                   'TTC': 'F', 'CTC': 'L', 'ATC': 'I', 'GTC': 'V', 
                   'TTA': 'L', 'CTA': 'L', 'ATA': 'I', 'GTA': 'V', 
                   'TTG': 'L', 'CTG': 'L', 'ATG': 'M', 'GTG': 'V', 
                   'TCT': 'S', 'CCT': 'P', 'ACT': 'T', 'GCT': 'A',
                   'TCC': 'S', 'CCC': 'P', 'ACC': 'T', 'GCC': 'A', 
                   'TCA': 'S', 'CCA': 'P', 'ACA': 'T', 'GCA': 'A', 
                   'TCG': 'S', 'CCG': 'P', 'ACG': 'T', 'GCG': 'A', 
                   'TAT': 'Y', 'CAT': 'H', 'AAT': 'N', 'GAT': 'D', 
                   'TAC': 'Y', 'CAC': 'H', 'AAC': 'N', 'GAC': 'D', 
                   'TAA': '*', 'CAA': 'Q', 'AAA': 'K', 'GAA': 'E',
                   'TAG': '*', 'CAG': 'Q', 'AAG': 'K', 'GAG': 'E',
                   'TGT': 'C', 'CGT': 'R', 'AGT': 'S', 'GGT': 'G', 
                   'TGC': 'C', 'CGC': 'R', 'AGC': 'S', 'GGC': 'G', 
                   'TGA': '*', 'CGA': 'R', 'AGA': 'R', 'GGA': 'G', 
                   'TGG': 'W', 'CGG': 'R', 'AGG': 'R', 'GGG': 'G'}

aminoacid2codon = {'F': ['TTT', 'TTC'], 
                   'L': ['TTA', 'TTG', 'CTT', 'CTC', 'CTA', 'CTG'], 
                   'I': ['ATT', 'ATC', 'ATA'], 
                   'M': ['ATG'], 
                   'V': ['GTT', 'GTC', 'GTA', 'GTG'], 
                   'S': ['TCT', 'TCC', 'TCA', 'TCG', 'AGT', 'AGC'], 
                   'P': ['CCT', 'CCC', 'CCA', 'CCG'], 
                   'T': ['ACT', 'ACC', 'ACA', 'ACG'], 
                   'A': ['GCT', 'GCC', 'GCA', 'GCG'], 
                   'Y': ['TAT', 'TAC'], 
                   'H': ['CAT', 'CAC'], 
                   'Q': ['CAA', 'CAG'], 
                   'N': ['AAT', 'AAC'], 
                   'K': ['AAA', 'AAG'], 
                   'D': ['GAT', 'GAC'], 
                   'E': ['GAA', 'GAG'], 
                   'C': ['TGT', 'TGC'], 
                   'W': ['TGG'], 
                   'R': ['CGT', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'],
                   'G': ['GGT', 'GGC', 'GGA', 'GGG'],
                   '*': ['TAA', 'TAG', 'TGA']}


high_frequency_aa2codon_dict = {
    "A": ['GCG'],
    "R": ['CGC'],
    "N": ['AAC'],
    "D": ["GAT"],
    "C": ["TGC"], 
    '*': ["TAA"],
    'Q': ["CAG"],
    "E": ["GAA"], 
    "G": ["GGC"], 
    "H": ["CAT"], 
    "I": ["ATT"], 
    "L": ["CTG"], 
    "K": ["AAA"], 
    "M": ["ATG"], 
    "F": ["TTT"], 
    "P": ["CCG"], 
    "S": ["AGC"], 
    "T": ["ACC"], 
    "W": ["TGG"], 
    "Y": ["TAT"], 
    "V": ["GTG"], 
}

class CodonSequence(object):
    """
    Proteins with sequence.
    Support both residue-level and atom-level operations and ensure consistency between two views.
    """
    
    ## nucleotide to id
    # base2id = {'A': 0, 'C':1, 'G':2, 'T':3, 'N': 4}
    # id2base = {v: k for k, v in base2id.items()}
    
    ## codonlm tokenizer for codon Sequence
    model_names = ["codonlm_8m", "codonlm_35m", "codonlm_650m"]
    
    def __init__(self, lm_model_name=None, model_path=None):
        """
        Args:
            lm_model_name: str, codon language model name, e.g. "codonlm_8m", "codonlm_35m", "codonlm_650m"
        """
        self.lm_model_name = lm_model_name
        self.model_path = model_path
        self.tokenizer = None
        if self.lm_model_name is not None:
            self.pad_token_id = self.get_codonlm_pad_tokenidx()
        else:
            self.pad_token_id = None
            
    
    def featurize(self, sequence):
        # if self.lm_model_name == None:
        #     return self.onehot_featurize(sequence)
        # else:
        sequence = sequence.upper()
        return self.codonlm_featurize(sequence)
        
    # def onehot_featurize(self, sequence):
    #     """
    #     Args:
    #         sequence: str, protein sequence
    #     Returns:
    #         features: np.array, one-hot encoded protein sequence
    #     """
    #     feature = np.zeros((len(sequence), len(self.base2id)))
    #     tokensids = np.array([self.base2id[token] for token in sequence])
    #     feature[np.arange(len(sequence)), tokensids] = 1
    #     ## return first 4 columns
    #     feature = feature[:, :4]
    #     return feature
    
    def codonlm_featurize(self, sequence):
        """
        Args:
            sequence: str, protein sequence
        Returns:
            features: np.array, esm encoded protein sequence
        """
        if self.tokenizer is None:
            if self.lm_model_name not in self.model_names:
                raise ValueError(f"lm_model_name {self.lm_model_name} not supported")
            self.tokenizer = EsmTokenizer.from_pretrained(f"{self.model_path}/codon_LM/{self.lm_model_name}/")  

        feature = self.tokenizer(sequence, return_tensors='pt')
        
        return feature

    def codonlm_decode(self, ids):
        """
        Args:
            sequence: str, protein sequence
        Returns:
            ids: np.array, list, torch.tensor
        """
        if self.tokenizer is None:
            if self.lm_model_name not in self.model_names:
                raise ValueError(f"lm_model_name {self.lm_model_name} not supported")
            self.tokenizer = EsmTokenizer.from_pretrained(f"{self.model_path}/codon_LM/{self.lm_model_name}/")  

        sequences = self.tokenizer.batch_decode(ids)
        
        return sequences

    def get_codonlm_pad_tokenidx(self):
        """
        Returns:
            pad_token_idx: int, pad token index
        """
        if self.tokenizer is None:
            if self.lm_model_name not in self.model_names:
                raise ValueError(f"lm_model_name {self.lm_model_name} not supported")
            self.tokenizer = EsmTokenizer.from_pretrained(f"{self.model_path}/codon_LM/{self.lm_model_name}/")  

        pad_token_id = self.tokenizer.pad_token_id
        
        return pad_token_id
    
    def codon2aa(self, codon):
        return codon2aminoacid[codon]
    
    def get_high_freq_codon(self, amino_acid):
        return high_frequency_aa2codon_dict[amino_acid][0]
    
    def random_map_codon_sequence(self, prot_seq):
        input_text = ''
        for aa in prot_seq:
            codons = aminoacid2codon[aa]
            input_text += random.choice(codons)
        codons = [input_text[i:i+3] for i in range(0, len(input_text), 3)]
        codon_seq = ' '.join(codons)
        return codon_seq
        
    
    def codon_optimized_pp(self, codon_seq):
        problem = DnaOptimizationProblem(
                sequence=codon_seq,
                constraints=[
                    EnforceGCContent(mini=0.3, maxi=0.7, window=50),
                    EnforceTranslation(),
                ],
                objectives=[CodonOptimize(species="e_coli")],
                logger=None
                )
        problem.resolve_constraints()
        problem.optimize()
        final_sequence = problem.sequence  # string
        
        return final_sequence
