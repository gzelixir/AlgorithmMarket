import json
from mlflow.models import ModelSignature



##  expected instance of DataType or one of ['boolean', 'integer', 'long', 'float', 'double', 'string', 'binary', 'datetime']
def get_signature(task): 
    if task == 'ProteinSolubilityPrediction':
        inp = json.dumps([{'name': 'seq','type':'string'}])
        oup = json.dumps([{'name': 'score','type':'double'}])
        signature = ModelSignature.from_dict({'inputs': inp, 'outputs': oup})
    elif task == 'ProteinMutationDdgPrediction':
        inp = json.dumps([{'name': 'wt_seq','type':'string'}, {'name': 'mut_seq','type':'string'}, {'name': 'position','type':'integer'}])
        oup = json.dumps([{'name': 'ddg','type':'double'}])
        signature = ModelSignature.from_dict({'inputs': inp, 'outputs': oup})
    elif task == 'EnzymeECNumberPrediction':
        inp = json.dumps([{'name': 'seq','type':'string'}])
        oup = json.dumps([{'name': 'prediction','type':'integer'}])
        signature = ModelSignature.from_dict({'inputs': inp, 'outputs': oup})
    elif task == 'KcatPrediction':
        inp = json.dumps([{'name': 'seq','type':'string'}, {'name': 'smile','type':'string'}])
        oup = json.dumps([{'name': 'kcat','type':'float'}])
        signature = ModelSignature.from_dict({'inputs': inp, 'outputs': oup})
    elif task == 'EnhancerActivityPrediction':
        inp = json.dumps([{'name': 'seq','type':'string'}])
        oup = json.dumps([{'name': 'activity','type':'double'}])
        signature = ModelSignature.from_dict({'inputs': inp, 'outputs': oup})
    elif task == 'CodonOptimization':
        inp = json.dumps([{'name': 'prot_seq','type':'string'}])
        # inp = json.dumps([{'name': 'prot_seq','type':'string'}, {'name': 'dna_seq','type':'string'}])
        oup = json.dumps([{'name': 'dna_seq','type':'string'}])
        signature = ModelSignature.from_dict({'inputs': inp, 'outputs': oup})
    elif task == 'PromoterPrediction':
        inp = json.dumps([{'name': 'seq','type':'string'}])
        oup = json.dumps([{'name': 'score','type':'double'}])
        signature = ModelSignature.from_dict({'inputs': inp, 'outputs': oup})
    elif task == 'TerminatorPrediction':
        inp = json.dumps([{'name': 'seq','type':'string'}])
        oup = json.dumps([{'name': 'score','type':'double'}])
        signature = ModelSignature.from_dict({'inputs': inp, 'outputs': oup})
    elif task == 'SgrnaofftargetPrediction':
        inp = json.dumps([{'name': 'sgrna_seq','type':'string'}, {'name': 'dna_seq','type':'string'}])
        oup = json.dumps([{'name': 'score','type':'double'}])
        signature = ModelSignature.from_dict({'inputs': inp, 'outputs': oup})
    elif task == 'TFBSPrediction':
        inp = json.dumps([{'name': 'seq','type':'string'}])
        oup = json.dumps([{'name': 'score','type':'double'}])
        signature = ModelSignature.from_dict({'inputs': inp, 'outputs': oup}) 
    elif task == 'ProteinGOSinglePrediction':
        inp = json.dumps([{'name': 'seq','type':'string'}])
        oup = json.dumps([{'name': 'prediction','type':'string'}])
        signature = ModelSignature.from_dict({'inputs': inp, 'outputs': oup}) 
    elif task == 'ProteinGOPrediction':
        inp = json.dumps([{'name': 'seq','type':'string'}])
        oup = json.dumps([{'name': 'BP','type':'string'}, {'name': 'MF','type':'string'}, {'name': 'CC','type':'string'}])
        signature = ModelSignature.from_dict({'inputs': inp, 'outputs': oup}) 
    else:
        raise ValueError(f"Task {task} is not supported")
    return signature
    