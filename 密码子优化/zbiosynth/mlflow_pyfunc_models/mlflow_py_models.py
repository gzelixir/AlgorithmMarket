import numpy as np
import torch
from torch import nn
import mlflow
from zbiosynth import core, data, utils
from zbiosynth.core import Registry as R
from tqdm import tqdm


class SoluProtPyModel(mlflow.pyfunc.PythonModel):

    def __init__(self, predictor, featurize, signature):
    
        self.predictor = predictor
        self.featurize = featurize
        self.signature = signature
        self.input_names, self.output_names = signature.inputs.input_names(), signature.outputs.input_names()
    
    def predict(self, context, model_input):
        
        '''
        context:
            a instance of PythonModelContext
            PythonModelContext对象由save_model() 和log_model()持久化方法隐式创建, 使用这些方法的artifacts参数指定的内容
        model_input:
            if request from flask, it will be a dataframe format
            model_input: [pandas.DataFrame, numpy.ndarray, scipy.sparse.(csc.csc_matrix | csr.csr_matrix),
            
        return 
            -> [numpy.ndarray | pandas.(Series | DataFrame) | List]
        '''
        outputs = {}
        inputs = model_input[self.input_names]
        preds = []
        with torch.no_grad():
            for idx in tqdm(range(len(inputs))):
                row = inputs.iloc[idx]
                data_dict = self.featurize_mlflow(row)
                for key in data_dict:
                    data_dict[key] = data_dict[key].to(self.predictor.device)
                pred = self.predictor.predict(data_dict, postprocess=True).tolist()[0]
                preds.append(pred)
        preds = np.array(preds).reshape(len(self.output_names), -1)
        for idx, name in enumerate(self.output_names):
            outputs[name] = preds[idx]
        return outputs
    
    def featurize_mlflow(self, row):
        seq = row['seq']
        data_dict = {}
        data = self.featurize.featurize(seq)
        if self.featurize.lm_model_name:
            data_dict = data
        else:
            data_dict['input'] = torch.FloatTensor(data).unsqueeze(0)
        return data_dict
    

# ProteinMutationDdgPrediction
class DDGPyModel(mlflow.pyfunc.PythonModel):

    def __init__(self, predictor, featurize, signature):
    
        self.predictor = predictor
        self.featurize = featurize
        self.signature = signature
        self.input_names, self.output_names = signature.inputs.input_names(), signature.outputs.input_names()
    
    def predict(self, context, model_input):
        
        '''
        context:
            a instance of PythonModelContext
            PythonModelContext对象由save_model() 和log_model()持久化方法隐式创建, 使用这些方法的artifacts参数指定的内容
        model_input:
            if request from flask, it will be a dataframe format
            model_input: [pandas.DataFrame, numpy.ndarray, scipy.sparse.(csc.csc_matrix | csr.csr_matrix),
            
        return 
            -> [numpy.ndarray | pandas.(Series | DataFrame) | List]
        '''
        outputs = {}
        inputs = model_input[self.input_names]
        preds = []
        with torch.no_grad():
            for idx in tqdm(range(len(inputs))):
                row = inputs.iloc[idx]
                data_dict = self.featurize_mlflow(row)
                for key in data_dict:
                    data_dict[key] = data_dict[key].to(self.predictor.device)
                pred = self.predictor.predict(data_dict).tolist()[0]
                preds.append(pred)
        preds = np.array(preds).reshape(len(self.output_names), -1)
        for idx, name in enumerate(self.output_names):
            outputs[name] = preds[idx]
        return outputs
    
    def featurize_mlflow(self, row):
        wt_seq, mut_seq = row['wt_seq'], row['mut_seq']
        position = int(row['position']) - 1
        position = torch.LongTensor([position]).view(-1)
        data_dict = {}
        wt_data = self.featurize.featurize(wt_seq)
        mut_data = self.featurize.featurize(mut_seq)
        
        data_dict['wt_input_ids'] = wt_data['input_ids']
        data_dict['wt_attention_mask'] = wt_data['attention_mask']
        data_dict['mut_input_ids'] = mut_data['input_ids']
        data_dict['mut_attention_mask'] = mut_data['attention_mask']
        data_dict['position'] = position
        
        return data_dict
    
    
class EcNumBerPyModel(mlflow.pyfunc.PythonModel):

    def __init__(self, predictor, featurize, signature):
    
        self.predictor = predictor
        self.featurize = featurize
        self.signature = signature
        self.input_names, self.output_names = signature.inputs.input_names(), signature.outputs.input_names()
    
    def predict(self, context, model_input):
        
        '''
        context:
            a instance of PythonModelContext
            PythonModelContext对象由save_model() 和log_model()持久化方法隐式创建, 使用这些方法的artifacts参数指定的内容
        model_input:
            if request from flask, it will be a dataframe format
            model_input: [pandas.DataFrame, numpy.ndarray, scipy.sparse.(csc.csc_matrix | csr.csr_matrix),
            
        return 
            -> [numpy.ndarray | pandas.(Series | DataFrame) | List]
        '''
        outputs = {}
        inputs = model_input[self.input_names]
        preds = []
        with torch.no_grad():
            for idx in tqdm(range(len(inputs))):
                row = inputs.iloc[idx]
                data_dict = self.featurize_mlflow(row)
                for key in data_dict:
                    data_dict[key] = data_dict[key].to(self.predictor.device)
                pred = self.predictor.predict(data_dict, postprocess=True).tolist()[0]
                preds.append(pred)
        preds = np.array(preds).reshape(len(self.output_names), -1)
        for idx, name in enumerate(self.output_names):
            outputs[name] = preds[idx]
        return outputs
    
    def featurize_mlflow(self, row):
        seq = row['seq']
        data_dict = {}
        data = self.featurize.featurize(seq)
        if self.featurize.lm_model_name:
            data_dict = data
        else:
            data_dict['input'] = torch.FloatTensor(data).unsqueeze(0)
        return data_dict
    
    
    
class KcatPyModel(mlflow.pyfunc.PythonModel):

    def __init__(self, predictor, prot_featurize, mol_featurize, signature):
    
        self.predictor = predictor
        self.prot_featurize = prot_featurize
        self.mol_featurize = mol_featurize
        self.signature = signature
        self.input_names, self.output_names = signature.inputs.input_names(), signature.outputs.input_names()
    
    def predict(self, context, model_input):
        
        '''
        context:
            a instance of PythonModelContext
            PythonModelContext对象由save_model() 和log_model()持久化方法隐式创建, 使用这些方法的artifacts参数指定的内容
        model_input:
            if request from flask, it will be a dataframe format
            model_input: [pandas.DataFrame, numpy.ndarray, scipy.sparse.(csc.csc_matrix | csr.csr_matrix),
            
        return 
            -> [numpy.ndarray | pandas.(Series | DataFrame) | List]
        '''
        outputs = {}
        inputs = model_input[self.input_names]
        preds = []
        with torch.no_grad():
            for idx in tqdm(range(len(inputs))):
                row = inputs.iloc[idx]
                data_dict = self.featurize_mlflow(row)
                for key in data_dict:
                    data_dict[key] = data_dict[key].to(self.predictor.device)
                pred = self.predictor.predict(data_dict).tolist()[0]
                preds.append(pred)
        preds = np.array(preds).reshape(len(self.output_names), -1)
        for idx, name in enumerate(self.output_names):
            outputs[name] = preds[idx]
        return outputs
    
    def featurize_mlflow(self, row):
        prot_seq, mol_seq = row['seq'],row['smile']
        data_dict = {}
        
        prot_data = self.prot_featurize.featurize(prot_seq)
        mol_data = self.mol_featurize.featurize(mol_seq)

        data_dict['prot_input_ids'] = prot_data['input_ids']
        data_dict['prot_attention_mask'] = prot_data['attention_mask']
        data_dict['mol_input_ids'] = mol_data['input_ids']
        data_dict['mol_attention_mask'] = mol_data['attention_mask']
        
        return data_dict
    
    
class EnhancerActivityPyModel(mlflow.pyfunc.PythonModel):

    def __init__(self, predictor, featurize, signature):
    
        self.predictor = predictor
        self.featurize = featurize
        self.signature = signature
        self.input_names, self.output_names = signature.inputs.input_names(), signature.outputs.input_names()
    
    def predict(self, context, model_input):
        
        '''
        context:
            a instance of PythonModelContext
            PythonModelContext对象由save_model() 和log_model()持久化方法隐式创建, 使用这些方法的artifacts参数指定的内容
        model_input:
            if request from flask, it will be a dataframe format
            model_input: [pandas.DataFrame, numpy.ndarray, scipy.sparse.(csc.csc_matrix | csr.csr_matrix),
            
        return 
            -> [numpy.ndarray | pandas.(Series | DataFrame) | List]
        '''
        outputs = {}
        inputs = model_input[self.input_names]
        preds = []
        with torch.no_grad():
            for idx in tqdm(range(len(inputs))):
                row = inputs.iloc[idx]
                data_dict = self.featurize_mlflow(row)
                for key in data_dict:
                    data_dict[key] = data_dict[key].to(self.predictor.device)
                pred = self.predictor.predict(data_dict).tolist()[0]
                preds.append(pred)
        preds = np.array(preds).reshape(len(self.output_names), -1)
        for idx, name in enumerate(self.output_names):
            outputs[name] = preds[idx]
        return outputs
    
    def featurize_mlflow(self, row):
        seq = row['seq']
        data_dict = {}
        data = self.featurize.featurize(seq)
        if self.featurize.lm_model_name:
            data_dict = data
        else:
            data_dict['input'] = torch.FloatTensor(data).unsqueeze(0)
        return data_dict
    
    

class CodonOptimizedPyModel(mlflow.pyfunc.PythonModel):

    def __init__(self, predictor, featurize, signature):
    
        self.predictor = predictor
        self.featurize = featurize
        self.signature = signature
        self.input_names, self.output_names = signature.inputs.input_names(), signature.outputs.input_names()
    
    
    def predict(self, context, model_input):
        
        '''
        context:
            a instance of PythonModelContext
            PythonModelContext对象由save_model() 和log_model()持久化方法隐式创建, 使用这些方法的artifacts参数指定的内容
        model_input:
            if request from flask, it will be a dataframe format
            model_input: [pandas.DataFrame, numpy.ndarray, scipy.sparse.(csc.csc_matrix | csr.csr_matrix),
            
        return 
            -> [numpy.ndarray | pandas.(Series | DataFrame) | List]
        '''
        outputs = {}
        inputs = model_input[self.input_names]
        pred_codon_seqs = []
        with torch.no_grad():
            for idx in tqdm(range(len(inputs))):
                row = inputs.iloc[idx]
                data_dict = self.featurize_mlflow(row)
                for key in data_dict:
                    data_dict[key] = data_dict[key].to(self.predictor.device)
                pred = self.predictor.predict(data_dict)
                preds = torch.argmax(pred, dim=2)[:, 1:-1]
                preds = self.featurize.tokenizer.batch_decode(preds)
                pred_codon_seqs.extend(preds)   
         
        pred_codon_seqs = self.fix_mutate(pred_codon_seqs, inputs.prot_seq.values)
        preds = np.array(pred_codon_seqs).reshape(len(self.output_names), -1)
        for idx, name in enumerate(self.output_names):
            outputs[name] = preds[idx]
        return outputs
    
    def fix_mutate(self, pred_codon_seqs, prot_seqs):
        fix_codon_seqs = []
        for pred_codon_seq,  prot_seq in zip(pred_codon_seqs, prot_seqs):
            fix_codon_seq = ''
            pred_codon_seq = pred_codon_seq.split(' ')
            prot_seq = prot_seq + '*'
            
            for codon, aa in zip(pred_codon_seq, prot_seq):
                pred_aa = self.featurize.codon2aa(codon)
                if aa == pred_aa:
                    fix_codon_seq += codon
                else:
                    high_freq_codon = self.featurize.get_high_freq_codon(aa)  
                    fix_codon_seq += high_freq_codon 
            fix_codon_seqs.append(fix_codon_seq)  
        
        ## post_process
        fix_codon_seqs = [self.featurize.codon_optimized_pp(codon_seq) for codon_seq in fix_codon_seqs]
        return fix_codon_seqs     
            
            
             
        counts = 0
        for pred_codon_seq,  truth_codon_seq in zip(pred_codon_seqs, truth_condon_seqs):
            pred_codon_seq = pred_codon_seq.split(' ')
            truth_codon_seq = [truth_codon_seq[i:i+3] for i in range(0, len(truth_codon_seq), 3)]
            pred_aa_seq, truth_aa_seq = '', ''
            try:
                for pred_codon,  truth_codon in zip(pred_codon_seq, truth_codon_seq):
                    pred_aa_seq += codon2aminoacid[pred_codon]
                    truth_aa_seq += codon2aminoacid[truth_codon]
            except:
                counts += 1
                continue
            
            if pred_aa_seq != truth_aa_seq:
                counts += 1
        return counts / len(truth_condon_seqs)

    
    def featurize_mlflow(self, row):
        seq = row['prot_seq'] + '*'
        data_dict = {}
        
        codon_seq = self.featurize.random_map_codon_sequence(seq)
        data = self.featurize.featurize(codon_seq)
        
        if self.featurize.lm_model_name:
            data_dict = data
        else:
            data_dict['input'] = torch.FloatTensor(data).unsqueeze(0)
        return data_dict
    
    # def predict(self, context, model_input):
        
    #     '''
    #     context:
    #         a instance of PythonModelContext
    #         PythonModelContext对象由save_model() 和log_model()持久化方法隐式创建, 使用这些方法的artifacts参数指定的内容
    #     model_input:
    #         if request from flask, it will be a dataframe format
    #         model_input: [pandas.DataFrame, numpy.ndarray, scipy.sparse.(csc.csc_matrix | csr.csr_matrix),
            
    #     return 
    #         -> [numpy.ndarray | pandas.(Series | DataFrame) | List]
    #     '''
    #     outputs = {}
    #     inputs = model_input[self.input_names]
    #     losses = []
    #     loss_fct = nn.CrossEntropyLoss() 
        
    #     for idx in tqdm(range(len(inputs))):
    #         row = inputs.iloc[idx]
    #         data_dict = self.featurize_mlflow(row)
    #         for key in data_dict:
    #             data_dict[key] = data_dict[key].to(self.predictor.device)
    #             # print(key, data_dict[key].shape)
    #         pred = self.predictor.predict(data_dict)
    #         loss = loss_fct(pred.view(-1, len(self.featurize.tokenizer)), data_dict['label'].view(-1))
    #         losses.append(loss.item())
        
    #     print(np.mean(losses))
    #     return outputs
    
    # def featurize_mlflow(self, row):
    #     print(row)
    #     prot_seq = row['prot_seq'] + '*'
    #     dna_seq = row['dna_seq']
    #     dna_seq = [dna_seq[i:i+3] for i in range(0, len(dna_seq), 3)]
    #     dna_seq = ' '.join(dna_seq)
        
    #     data_dict = {}
        
    #     codon_seq = self.featurize.random_map_codon_sequence(prot_seq)
    #     data = self.featurize.featurize(codon_seq)
    #     data2 = self.featurize.featurize(dna_seq)
        
    #     data2['input_ids'][:, 0] = -100
    #     data2['input_ids'][:, -1] = -100
        
    #     if self.featurize.lm_model_name:
    #         data_dict = data
    #         data_dict['label'] = data2['input_ids']
    #     else:
    #         data_dict['input'] = torch.FloatTensor(data).unsqueeze(0)
    #     return data_dict
    

class PromoterPyModel(mlflow.pyfunc.PythonModel):

    def __init__(self, predictor, featurize, signature):
    
        self.predictor = predictor
        self.featurize = featurize
        self.signature = signature
        self.input_names, self.output_names = signature.inputs.input_names(), signature.outputs.input_names()
    
    def predict(self, context, model_input):
        
        '''
        context:
            a instance of PythonModelContext
            PythonModelContext对象由save_model() 和log_model()持久化方法隐式创建, 使用这些方法的artifacts参数指定的内容
        model_input:
            if request from flask, it will be a dataframe format
            model_input: [pandas.DataFrame, numpy.ndarray, scipy.sparse.(csc.csc_matrix | csr.csr_matrix),
            
        return 
            -> [numpy.ndarray | pandas.(Series | DataFrame) | List]
        '''
        outputs = {}
        inputs = model_input[self.input_names]
        preds = []
        with torch.no_grad():
            for idx in tqdm(range(len(inputs))):
                row = inputs.iloc[idx]
                data_dict = self.featurize_mlflow(row)
                for key in data_dict:
                    data_dict[key] = data_dict[key].to(self.predictor.device)
                pred = self.predictor.predict(data_dict, postprocess=True).tolist()[0]
                preds.append(pred)
        preds = np.array(preds).reshape(len(self.output_names), -1)
        for idx, name in enumerate(self.output_names):
            outputs[name] = preds[idx]
        return outputs
    
    def featurize_mlflow(self, row):
        seq = row['seq']
        data_dict = {}
        data = self.featurize.featurize(seq)
        if self.featurize.lm_model_name:
            data_dict = data
        else:
            data_dict['input'] = torch.FloatTensor(data).unsqueeze(0)
        return data_dict
    

class TerminatorPyModel(mlflow.pyfunc.PythonModel):

    def __init__(self, predictor, featurize, signature):
    
        self.predictor = predictor
        self.featurize = featurize
        self.signature = signature
        self.input_names, self.output_names = signature.inputs.input_names(), signature.outputs.input_names()
    
    def predict(self, context, model_input):
        
        '''
        context:
            a instance of PythonModelContext
            PythonModelContext对象由save_model() 和log_model()持久化方法隐式创建, 使用这些方法的artifacts参数指定的内容
        model_input:
            if request from flask, it will be a dataframe format
            model_input: [pandas.DataFrame, numpy.ndarray, scipy.sparse.(csc.csc_matrix | csr.csr_matrix),
            
        return 
            -> [numpy.ndarray | pandas.(Series | DataFrame) | List]
        '''
        outputs = {}
        inputs = model_input[self.input_names]
        preds = []
        with torch.no_grad():
            for idx in tqdm(range(len(inputs))):
                row = inputs.iloc[idx]
                data_dict = self.featurize_mlflow(row)
                for key in data_dict:
                    data_dict[key] = data_dict[key].to(self.predictor.device)
                pred = self.predictor.predict(data_dict, postprocess=True).tolist()[0]
                preds.append(pred)
        preds = np.array(preds).reshape(len(self.output_names), -1)
        for idx, name in enumerate(self.output_names):
            outputs[name] = preds[idx]
        return outputs
    
    def featurize_mlflow(self, row):
        seq = row['seq']
        data_dict = {}
        data = self.featurize.featurize(seq)
        if self.featurize.lm_model_name:
            data_dict = data
        else:
            data_dict['input'] = torch.FloatTensor(data).unsqueeze(0)
        return data_dict
    
    
class SgrnaofftargetPyModel(mlflow.pyfunc.PythonModel):

    def __init__(self, predictor, featurize, signature):
    
        self.predictor = predictor
        self.featurize = featurize
        self.signature = signature
        self.input_names, self.output_names = signature.inputs.input_names(), signature.outputs.input_names()
    
    def predict(self, context, model_input):
        
        '''
        context:
            a instance of PythonModelContext
            PythonModelContext对象由save_model() 和log_model()持久化方法隐式创建, 使用这些方法的artifacts参数指定的内容
        model_input:
            if request from flask, it will be a dataframe format
            model_input: [pandas.DataFrame, numpy.ndarray, scipy.sparse.(csc.csc_matrix | csr.csr_matrix),
            
        return 
            -> [numpy.ndarray | pandas.(Series | DataFrame) | List]
        '''
        outputs = {}
        inputs = model_input[self.input_names]
        preds = []
        with torch.no_grad():
            for idx in tqdm(range(len(inputs))):
                row = inputs.iloc[idx]
                data_dict = self.featurize_mlflow(row)
                for key in data_dict:
                    data_dict[key] = data_dict[key].to(self.predictor.device)
                pred = self.predictor.predict(data_dict, postprocess=True).tolist()[0]
                preds.append(pred)
        preds = np.array(preds).reshape(len(self.output_names), -1)
        for idx, name in enumerate(self.output_names):
            outputs[name] = preds[idx]
        return outputs
    
    def featurize_mlflow(self, row):
        # seq = row['seq']
        
        sgrna_seq = row.sgrna_seq.upper()
        dna_seq = row.dna_seq.upper()
        seq = sgrna_seq + dna_seq
        seq = seq.replace('U', 'T')
        
        data_dict = {}
        data = self.featurize.featurize(seq)
        if self.featurize.lm_model_name:
            data_dict = data
        else:
            data_dict['input'] = torch.FloatTensor(data).unsqueeze(0)
        return data_dict
    
    
class TFBSPyModel(mlflow.pyfunc.PythonModel):

    def __init__(self, predictor, featurize, signature):
    
        self.predictor = predictor
        self.featurize = featurize
        self.signature = signature

        self.input_names, self.output_names = signature.inputs.input_names(), signature.outputs.input_names()
    
    def predict(self, context, model_input):
        
        '''
        context:
            a instance of PythonModelContext
            PythonModelContext对象由save_model() 和log_model()持久化方法隐式创建, 使用这些方法的artifacts参数指定的内容
        model_input:
            if request from flask, it will be a dataframe format
            model_input: [pandas.DataFrame, numpy.ndarray, scipy.sparse.(csc.csc_matrix | csr.csr_matrix),
            
        return 
            -> [numpy.ndarray | pandas.(Series | DataFrame) | List]
        '''
        outputs = {}
        inputs = model_input[self.input_names]
        preds = []
        with torch.no_grad():
            for idx in tqdm(range(len(inputs))):
                row = inputs.iloc[idx]
                data_dict = self.featurize_mlflow(row)
                for key in data_dict:
                    data_dict[key] = data_dict[key].to(self.predictor.device)
                pred = self.predictor.predict(data_dict, postprocess=True).tolist()[0]
                preds.append(pred)
        preds = np.array(preds).reshape(len(self.output_names), -1)
        for idx, name in enumerate(self.output_names):
            outputs[name] = preds[idx]
        return outputs
    
    def featurize_mlflow(self, row):
        seq = row['seq'].upper()    
        seq = seq.replace('U', 'T')
        
        data_dict = {}
        data = self.featurize.featurize(seq)
        if self.featurize.lm_model_name:
            data_dict = data
        else:
            data_dict['input'] = torch.FloatTensor(data).unsqueeze(0)
        return data_dict
    
class ProtGOSinglePyModel(mlflow.pyfunc.PythonModel):

    def __init__(self, predictor, featurize, threshold, go2id, signature):
    
        self.predictor = predictor
        self.featurize = featurize
        self.signature = signature
        self.threshold = threshold
        self.go2id = go2id
        self.id2go = {v:k for k, v in go2id.items()}
        self.input_names, self.output_names = signature.inputs.input_names(), signature.outputs.input_names()

    def predict(self, context, model_input):
        
        '''
        context:
            a instance of PythonModelContext
            PythonModelContext对象由save_model() 和log_model()持久化方法隐式创建, 使用这些方法的artifacts参数指定的内容
        model_input:
            if request from flask, it will be a dataframe format
            model_input: [pandas.DataFrame, numpy.ndarray, scipy.sparse.(csc.csc_matrix | csr.csr_matrix),
            
        return 
            -> [numpy.ndarray | pandas.(Series | DataFrame) | List]
        '''
        outputs = {}
        inputs = model_input[self.input_names]
        preds = []
        with torch.no_grad():
            for idx in tqdm(range(len(inputs))):
                row = inputs.iloc[idx]
                data_dict = self.featurize_mlflow(row)
                for key in data_dict:
                    data_dict[key] = data_dict[key].to(self.predictor.device)
                pred = self.predictor.predict(data_dict, postprocess=True)
                preds.append(pred.cpu())
        preds = np.concatenate(preds, axis=0)
        preds = (preds > self.threshold).astype(int)
        predictions = []
        for pred in preds:
            prediction = []
            pred_idx = np.where(pred>0)[0]
            for idx in pred_idx:
                prediction.append(self.id2go[idx])
            prediction = ','.join(prediction)
            predictions.append(prediction)
        outputs[self.output_names[0]] = predictions        
        return outputs
    
    def featurize_mlflow(self, row):
        seq = row['seq']
        data_dict = {}
        data = self.featurize.featurize(seq)
        if self.featurize.lm_model_name:
            data_dict = data
        else:
            data_dict['input'] = torch.FloatTensor(data).unsqueeze(0)
        return data_dict
    
class ProtGOPyModel(mlflow.pyfunc.PythonModel):

    def __init__(self, predictor, featurize, thresholds, bp_go2id, mf_go2id, cc_go2id, signature):
    
        self.predictor = predictor
        self.featurize = featurize
        self.signature = signature
        self.thresholds = thresholds
        self.bp_go2id = bp_go2id
        self.mf_go2id = mf_go2id
        self.cc_go2id = cc_go2id
        self.bp_id2go = {v:k for k, v in bp_go2id.items()}
        self.mf_id2go = {v:k for k, v in mf_go2id.items()}
        self.cc_id2go = {v:k for k, v in cc_go2id.items()}
        self.id2go = [self.bp_id2go, self.mf_id2go, self.cc_id2go]
        self.input_names, self.output_names = signature.inputs.input_names(), signature.outputs.input_names()
    
    def predict(self, context, model_input):
        
        '''
        context:
            a instance of PythonModelContext
            PythonModelContext对象由save_model() 和log_model()持久化方法隐式创建, 使用这些方法的artifacts参数指定的内容
        model_input:
            if request from flask, it will be a dataframe format
            model_input: [pandas.DataFrame, numpy.ndarray, scipy.sparse.(csc.csc_matrix | csr.csr_matrix),
            
        return 
            -> [numpy.ndarray | pandas.(Series | DataFrame) | List]
        '''
        outputs = {}
        inputs = model_input[self.input_names]
        preds_bp, preds_mf, preds_cc = [], [], []
        with torch.no_grad():
            for idx in tqdm(range(len(inputs))):
                row = inputs.iloc[idx]
                data_dict = self.featurize_mlflow(row)
                for key in data_dict:
                    data_dict[key] = data_dict[key].to(self.predictor.device)
                pred_bp, pred_mf, pred_cc = self.predictor.predict(data_dict, postprocess=True)
                
                preds_bp.append(pred_bp.cpu())
                preds_mf.append(pred_mf.cpu())
                preds_cc.append(pred_cc.cpu())
                
        preds_bp = np.concatenate(preds_bp, axis=0)
        preds_bp = (preds_bp > self.thresholds[0]).astype(int)
        
        preds_mf = np.concatenate(preds_mf, axis=0)
        preds_mf = (preds_mf > self.thresholds[1]).astype(int)
        
        preds_cc = np.concatenate(preds_cc, axis=0)
        preds_cc = (preds_cc > self.thresholds[2]).astype(int)
                                                
        predictions = [[], [], []]
        for task_idx, preds in enumerate([preds_bp, preds_mf, preds_cc]):
            for pred in preds:
                prediction = []
                pred_idx = np.where(pred>0)[0]
                for idx in pred_idx:
                    prediction.append(self.id2go[task_idx][idx])
                prediction = ','.join(prediction)
                predictions[task_idx].append(prediction)
            outputs[self.output_names[task_idx]] = predictions[task_idx]
        
        return outputs     
    
    def featurize_mlflow(self, row):
        seq = row['seq']
        data_dict = {}
        data = self.featurize.featurize(seq)
        if self.featurize.lm_model_name:
            data_dict = data
        else:
            data_dict['input'] = torch.FloatTensor(data).unsqueeze(0)
        return data_dict
    

model2pymodel = {
    'ProteinSolubilityPrediction': SoluProtPyModel,
    'ProteinMutationDdgPrediction': DDGPyModel,
    'EnzymeECNumberPrediction': EcNumBerPyModel,
    'KcatPrediction': KcatPyModel,
    'EnhancerActivityPrediction': EnhancerActivityPyModel,
    'CodonOptimization': CodonOptimizedPyModel,
    'PromoterPrediction': PromoterPyModel,
    'TerminatorPrediction': TerminatorPyModel,
    'SgrnaofftargetPrediction': SgrnaofftargetPyModel,
    'TFBSPrediction': TFBSPyModel,
    'ProteinGOPrediction': ProtGOPyModel,
    'ProteinGOSinglePrediction': ProtGOSinglePyModel,
}
